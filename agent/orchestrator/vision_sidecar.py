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
import numpy as np

from agent.orchestrator.camera_manager import CameraManager
from agent.orchestrator.inference.detection_manager import DetectionManager
from agent.orchestrator.inference.yolo_common import Detection, scale_detections_xyxy
from agent.orchestrator.rescue.dispatcher import RescueMissionDispatcher
from agent.orchestrator.rescue.snapshots import save_rescue_snapshots
from agent.orchestrator.rescue.trigger import PersonRescueTrigger
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
    """When ArduCam vision is on, normalize legacy env (inference is Hailo-only)."""
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
    """Logs a single INFO message when a person is seen for N consecutive recorder frames.

    Re-arms itself once the streak breaks (i.e. a miss resets state), so it will
    log again on the next qualifying run. This is separate from ``PersonRescueTrigger``
    which manages the cooldown and mission dispatch; this class is purely for console
    visibility during live flights.
    """

    def __init__(self, min_confidence: float, consecutive_frames: int) -> None:
        """Initialise the logger.

        Args:
            min_confidence: Minimum person confidence score (0–1) to count a frame.
            consecutive_frames: Required uninterrupted qualifying frames before logging.
        """
        self._min_confidence = min_confidence
        self._consecutive_frames = max(1, consecutive_frames)
        self._streak = 0
        self._armed = True

    def tick(self, dets: list[Detection]) -> None:
        """Advance state for one recorder frame; emit a log line on first qualifying run.

        Args:
            dets: Detections from the current frame.
        """
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

        # Rescue trigger — armed lazily via set_rescue_dispatcher().
        self._rescue_trigger: PersonRescueTrigger | None = None
        self._rescue_dispatcher: RescueMissionDispatcher | None = None
        self._rescue_photos_dir: Path | None = None
        self._rescue_save_person_crop: bool = True
        self._rescue_lock = threading.Lock()

    def set_rescue_dispatcher(
        self,
        *,
        dispatcher: RescueMissionDispatcher,
        rescue_person_conf: float,
        rescue_person_frames: int,
        rescue_arm_delay_s: float,
        rescue_photos_dir: Path,
        rescue_save_person_crop: bool = True,
    ) -> None:
        """Arm the rescue trigger.  Safe to call from any thread after start().

        The trigger starts in SUPPRESSED state.  Call ``notify_mission_sent()`` once
        the first operator mission is confirmed uploaded to transition to the arming
        countdown.

        Args:
            dispatcher: Live rescue dispatcher connected to the asyncio event loop.
            rescue_person_conf: Minimum YOLO person confidence to count a frame.
            rescue_person_frames: Consecutive qualifying frames before the trigger fires.
            rescue_arm_delay_s: Seconds after the first mission is sent before the
                trigger becomes active (gives the drone time to fly away from the operator).
            rescue_photos_dir: Directory where annotated frame (and optionally crop) are saved.
            rescue_save_person_crop: When false, only the annotated full-frame JPEG is written.
        """
        trigger = PersonRescueTrigger(
            min_confidence=rescue_person_conf,
            consecutive_frames=rescue_person_frames,
            arm_delay_s=rescue_arm_delay_s,
        )
        with self._rescue_lock:
            self._rescue_trigger = trigger
            self._rescue_dispatcher = dispatcher
            self._rescue_photos_dir = rescue_photos_dir
            self._rescue_save_person_crop = rescue_save_person_crop
        log.info(
            "Rescue trigger armed (conf=%.2f frames=%d arm_delay=%.0fs photos_dir=%s save_person_crop=%s)",
            rescue_person_conf,
            rescue_person_frames,
            rescue_arm_delay_s,
            rescue_photos_dir,
            rescue_save_person_crop,
        )

    def notify_mission_sent(self) -> None:
        """Notify the rescue trigger that the first operator mission has been sent.

        Starts the arm-delay countdown inside ``PersonRescueTrigger``.  Safe to call
        from any thread; no-op if the rescue trigger has not been armed yet or if
        called more than once.
        """
        with self._rescue_lock:
            trigger = self._rescue_trigger
        if trigger is not None:
            trigger.notify_mission_sent()

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
            self._tick_rescue(frame, dets)

            annotate_frame(frame, dets, image_is_bgr=self._cam.buffer_is_bgr())
            if self._cam.buffer_is_bgr():
                to_write = frame
            else:
                to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = to_write.shape[:2]

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

            writer.write(to_write)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

        if writer is not None:
            writer.release()

    def _tick_rescue(self, frame: np.ndarray, dets: list[Detection]) -> None:
        """Evaluate rescue trigger and dispatch a rescue if it fires.

        Called on every recorder frame from ``_loop``. Reads trigger/dispatcher refs
        under a lock so ``set_rescue_dispatcher`` can be called from another thread.

        Args:
            frame: Current video frame as a NumPy array (RGB or BGR depending on
                ``CameraManager.buffer_is_bgr``).
            dets: Scaled detections for this frame.
        """
        with self._rescue_lock:
            trigger = self._rescue_trigger
            dispatcher = self._rescue_dispatcher
            photos_dir = self._rescue_photos_dir
            save_person_crop = self._rescue_save_person_crop

        if trigger is None or dispatcher is None or photos_dir is None:
            return

        fire = trigger.tick(dets)
        if fire is None:
            return

        # Convert to BGR for saving if needed.
        is_bgr = self._cam.buffer_is_bgr()
        frame_bgr = frame if is_bgr else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        log.info(
            "Rescue trigger fired: conf=%.2f bbox=%s",
            fire.best_detection.confidence,
            fire.best_detection.xyxy,
        )

        try:
            full_path, crop_path = save_rescue_snapshots(
                frame_bgr,
                dets,
                out_dir=photos_dir,
                save_person_crop=save_person_crop,
            )
        except Exception:
            log.exception("save_rescue_snapshots failed")
            return

        log.info(
            "Rescue snapshots saved (initial filenames): full=%s crop=%s | paths: full=%s crop=%s",
            full_path.name,
            crop_path.name if crop_path is not None else "(none)",
            full_path,
            crop_path if crop_path is not None else "(none)",
        )

        dispatcher.request_rescue(
            bbox_xyxy=fire.best_detection.xyxy,
            image_wh=(w, h),
            full_path=full_path,
            crop_path=crop_path,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


@dataclass
class VisionRuntime:
    """Container for the three vision pipeline components.

    Attributes:
        camera: Live Picamera2 frame source.
        detector: Background Hailo YOLO inference thread.
        recorder: Overlay annotation and MP4 recording thread.
    """

    camera: CameraManager
    detector: DetectionManager
    recorder: _OverlayRecorder

    def notify_mission_sent(self) -> None:
        """Propagate first-mission notification to the rescue trigger.

        Call once immediately after the first operator mission is confirmed sent via
        gRPC so the rescue trigger starts its arm-delay countdown.
        """
        self.recorder.notify_mission_sent()

    def stop(self) -> None:
        self.recorder.stop()
        self.detector.stop()
        self.camera.stop()


def start_arducam_vision() -> VisionRuntime:
    """Run rpicam smoke test, Picamera2, Hailo/ONNX detection, and overlay recording."""
    configure_vision_environment()
    run_rpicam_health_check()

    raw_video_dir = os.environ.get("ARDUCAM_VIDEO_DIR", _DEFAULT_VIDEO_DIR)
    video_dir = Path(os.path.expandvars(os.path.expanduser(raw_video_dir)))
    fps = float(os.environ.get("ARDUCAM_RECORD_FPS", "15"))
    person_conf = float(os.environ.get("ARDUCAM_PERSON_CONF", "0.5"))
    person_frames = int(os.environ.get("ARDUCAM_PERSON_FRAMES", "3"))

    cam = CameraManager()
    cam.start()
    if cam.startup_error is not None:
        cam.stop()
        raise RuntimeError(f"Picamera2 failed: {cam.startup_error}")

    detector = DetectionManager(hef_path=os.environ.get("YOLO_HEF_PATH"))
    if detector.detector is None:
        cam.stop()
        raise RuntimeError(
            "YOLO Hailo detector failed to load (check YOLO_HEF_PATH, HEF on disk, and hailo_platform)."
        )

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
