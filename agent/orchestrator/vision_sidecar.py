"""ArduCam vision: rpicam smoke test, overlay recording to disk, person streak logging."""

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import cv2

from agent.orchestrator.camera_manager import CameraManager
from agent.orchestrator.inference.detection_manager import DetectionManager
from agent.orchestrator.inference.yolo_onnx import Detection, scale_detections_xyxy
from agent.orchestrator.vision_overlay import annotate_frame

log = logging.getLogger(__name__)

_DEFAULT_VIDEO_DIR = "/arducamvideos"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def arducam_vision_enabled() -> bool:
    return _env_flag("ARDUCAM_VISION")


def configure_vision_environment() -> None:
    """When ArduCam vision is on, default to Hailo unless the operator set YOLO_BACKEND."""
    if not arducam_vision_enabled():
        return
    if os.environ.get("YOLO_BACKEND", "").strip() == "":
        os.environ["YOLO_BACKEND"] = "hailo"


def run_rpicam_health_check(*, timeout_s: float = 60.0) -> None:
    """Fail fast if libcamera CLI cannot capture (camera unplugged, permissions, etc.)."""
    cmd: list[str] | None = None
    if shutil.which("rpicam-still"):
        cmd = [
            "rpicam-still",
            "-t",
            "500",
            "--nopreview",
            "-o",
            "/dev/null",
        ]
    elif shutil.which("rpicam-hello"):
        cmd = ["rpicam-hello", "--timeout", "500", "--nopreview"]
    elif shutil.which("libcamera-still"):
        cmd = [
            "libcamera-still",
            "-t",
            "500",
            "--nopreview",
            "-o",
            "/dev/null",
        ]
    elif shutil.which("libcamera-hello"):
        cmd = ["libcamera-hello", "--timeout", "500", "--nopreview"]

    if cmd is None:
        log.error("No rpicam/libcamera CLI found (install libcamera-apps / rpicam-apps).")
        raise RuntimeError("rpicam_health_check: no libcamera CLI")

    log.info("Camera smoke test: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log.error("rpicam health check failed: %s stderr=%s", e, e.stderr)
        raise RuntimeError("rpicam_health_check_failed") from e
    except subprocess.TimeoutExpired as e:
        log.error("rpicam health check timed out")
        raise RuntimeError("rpicam_health_check_timeout") from e


class _PersonPresenceLogger:
    """Log once when person confidence meets threshold for N consecutive recorder frames."""

    def __init__(self, min_confidence: float, consecutive_frames: int) -> None:
        self._min_confidence = min_confidence
        self._consecutive_frames = max(1, consecutive_frames)
        self._streak = 0
        self._armed = True

    def tick(self, dets: list[Detection]) -> None:
        best = 0.0
        for d in dets:
            if d.class_id != 0 and d.class_name != "person":
                continue
            if d.confidence > best:
                best = float(d.confidence)

        qualified = best >= self._min_confidence
        if qualified:
            self._streak += 1
            if self._streak >= self._consecutive_frames and self._armed:
                log.info(
                    "Person detected: confidence=%.2f (threshold=%.2f consecutive_frames=%d)",
                    best,
                    self._min_confidence,
                    self._consecutive_frames,
                )
                self._armed = False
        else:
            self._streak = 0
            self._armed = True


class _OverlayRecorder:
    """Background thread writes annotated RGB frames to MP4 (downsampled stream size)."""

    def __init__(
        self,
        cam: CameraManager,
        detector: DetectionManager,
        *,
        out_dir: Path,
        fps: float,
        person_conf: float,
        person_frames: int,
    ) -> None:
        self._cam = cam
        self._detector = detector
        self._out_dir = out_dir
        self._fps = max(1.0, fps)
        self._person = _PersonPresenceLogger(person_conf, person_frames)
        self._running = False
        self._thread: threading.Thread | None = None
        self._out_path: Path | None = None

    def output_path(self) -> Path | None:
        return self._out_path

    def start(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_path = self._out_dir / f"arducam_{ts}.mp4"
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Overlay recorder started → %s", self._out_path)

    def _loop(self) -> None:
        writer: cv2.VideoWriter | None = None
        period = 1.0 / self._fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while self._running:
            t0 = time.perf_counter()
            with self._cam.lock:
                frame = (
                    None
                    if self._cam.latest_frame is None
                    else self._cam.latest_frame.copy()
                )
            if frame is None:
                time.sleep(0.02)
                continue

            with self._detector.lock:
                dets = list(self._detector.latest)

            scale = self._detector.overlay_scale_for_preview()
            if scale is not None:
                sx, sy = scale
                dets = scale_detections_xyxy(dets, sx, sy)

            self._person.tick(dets)
            annotate_frame(frame, dets)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]

            if writer is None:
                assert self._out_path is not None
                writer = cv2.VideoWriter(
                    str(self._out_path),
                    fourcc,
                    self._fps,
                    (w, h),
                )
                if not writer.isOpened():
                    log.error("VideoWriter failed to open for %s", self._out_path)
                    return

            writer.write(bgr)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

        if writer is not None:
            writer.release()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


@dataclass
class VisionRuntime:
    camera: CameraManager
    detector: DetectionManager
    recorder: _OverlayRecorder

    def stop(self) -> None:
        self.recorder.stop()
        self.detector.stop()
        self.camera.stop()


def start_arducam_vision() -> VisionRuntime:
    """Run rpicam smoke test, Picamera2, Hailo/ONNX detection, and overlay recording."""
    configure_vision_environment()
    run_rpicam_health_check()

    video_dir = (
        Path(os.environ.get("ARDUCAM_VIDEO_DIR", _DEFAULT_VIDEO_DIR))
        .expandvars()
        .expanduser()
    )
    fps = float(os.environ.get("ARDUCAM_RECORD_FPS", "15"))
    person_conf = float(os.environ.get("ARDUCAM_PERSON_CONF", "0.5"))
    person_frames = int(os.environ.get("ARDUCAM_PERSON_FRAMES", "3"))

    cam = CameraManager()
    cam.start()
    if cam.startup_error is not None:
        cam.stop()
        raise RuntimeError(f"Picamera2 failed: {cam.startup_error}")

    detector = DetectionManager(
        model_path=os.environ.get("YOLO_ONNX_PATH"),
        hef_path=os.environ.get("YOLO_HEF_PATH"),
    )
    if detector.detector is None:
        cam.stop()
        raise RuntimeError("YOLO detector failed to load (check YOLO_BACKEND / HEF / hailo_platform).")

    detector.start(cam)
    recorder = _OverlayRecorder(
        cam,
        detector,
        out_dir=video_dir,
        fps=fps,
        person_conf=person_conf,
        person_frames=person_frames,
    )
    recorder.start()
    return VisionRuntime(camera=cam, detector=detector, recorder=recorder)


def stop_arducam_vision(rt: VisionRuntime | None) -> None:
    if rt is None:
        return
    try:
        rt.stop()
    except Exception:
        log.exception("Error stopping ArduCam vision")
