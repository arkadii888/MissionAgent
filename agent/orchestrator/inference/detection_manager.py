import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .yolo_hailo import hailo_platform_available
from .yolo_onnx import Detection, Yolo26OnnxDetector


def _default_yolo_hef_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "models" / "yolo26n_b8.hef")


def _effective_hef_path() -> str:
    raw = os.environ.get("YOLO_HEF_PATH", "").strip()
    return raw if raw else _default_yolo_hef_path()

logger = logging.getLogger(__name__)

_HISTORY_MAX = 100
_FPS_EMA_ALPHA = 0.15
_LOG_INTERVAL_SEC = 1.0


@dataclass
class DetectionSnapshot:
    """One recorded inference result for debugging."""
    at: str 
    count: int
    detections: list[dict[str, Any]]  # class_name, confidence, class_id


@dataclass
class PipelineDebugStats:
    running: bool
    backend: str  # hailo | onnx | none
    detector_loaded: bool
    frames_processed: int
    frames_waited_none: int
    last_inference_ms: float | None
    inference_fps_ema: float
    last_frame_shape: list[int] | None
    last_frame_mean_rgb: list[float] | None
    last_frame_std_rgb: list[float] | None
    last_detection_count: int
    last_inference_at_iso: str | None
    last_error: str | None


def _create_detector(
    model_path: str | None,
) -> tuple[Any | None, str]:
    """
    YOLO_BACKEND=auto|onnx|hailo (default auto).
    Hailo: YOLO_HEF_PATH (default repo agent/models/yolo26n_b8.hef). ONNX: YOLO_ONNX_PATH / model_path.
    """
    backend_pref = os.environ.get("YOLO_BACKEND", "auto").strip().lower()
    hef_path = _effective_hef_path()

    if backend_pref == "onnx":
        try:
            return Yolo26OnnxDetector(model_path=model_path), "onnx"
        except FileNotFoundError as e:
            print(f"YOLO ONNX disabled (no model): {e}")
            return None, "none"

    if backend_pref == "hailo":
        from .yolo_hailo import YoloHailoDetector

        try:
            return YoloHailoDetector(hef_path=hef_path), "hailo"
        except Exception as e:
            print(f"YOLO Hailo failed to initialize: {e}")
            return None, "none"

    if backend_pref not in ("auto", "", "default"):
        print(f"Unknown YOLO_BACKEND={backend_pref!r}; using auto.")

    if hailo_platform_available() and os.path.isfile(hef_path):
        from .yolo_hailo import YoloHailoDetector

        try:
            return YoloHailoDetector(hef_path=hef_path), "hailo"
        except Exception as e:
            print(f"Hailo unavailable ({e}); falling back to ONNX.")

    try:
        return Yolo26OnnxDetector(model_path=model_path), "onnx"
    except FileNotFoundError as e:
        print(f"YOLO ONNX disabled (no model): {e}")
        return None, "none"


class DetectionManager:
    """Background thread: runs YOLO on latest camera frame (drops frames if inference is slower)."""

    def __init__(self, model_path: str | None = None) -> None:
        self.latest: list[Detection] = []
        self.lock = threading.Lock()
        self.running = False
        self.detector: Any | None = None
        self._backend: str = "none"
        self._thread: threading.Thread | None = None

        # Debug, telemetry (updated under self.lock from inference thread)
        self._frames_processed = 0
        self._frames_waited_none = 0
        self._last_inference_ms: float | None = None
        self._inference_fps_ema = 0.0
        self._last_frame_shape: tuple[int, ...] | None = None
        self._last_frame_mean_rgb: tuple[float, float, float] | None = None
        self._last_frame_std_rgb: tuple[float, float, float] | None = None
        self._last_detection_count = 0
        self._last_inference_at_iso: str | None = None
        self._last_error: str | None = None
        self._history: deque[DetectionSnapshot] = deque(maxlen=_HISTORY_MAX)
        self._last_console_log = 0.0

        self._debug_log = os.environ.get("DETECTION_DEBUG_LOG", "").lower() in (
            "1",
            "true",
            "yes",
        )

        self.detector, self._backend = _create_detector(model_path)

    def get_debug_stats(self) -> PipelineDebugStats:
        with self.lock:
            return PipelineDebugStats(
                running=self.running,
                backend=self._backend,
                detector_loaded=self.detector is not None,
                frames_processed=self._frames_processed,
                frames_waited_none=self._frames_waited_none,
                last_inference_ms=self._last_inference_ms,
                inference_fps_ema=round(self._inference_fps_ema, 2),
                last_frame_shape=list(self._last_frame_shape)
                if self._last_frame_shape
                else None,
                last_frame_mean_rgb=list(self._last_frame_mean_rgb)
                if self._last_frame_mean_rgb
                else None,
                last_frame_std_rgb=list(self._last_frame_std_rgb)
                if self._last_frame_std_rgb
                else None,
                last_detection_count=self._last_detection_count,
                last_inference_at_iso=self._last_inference_at_iso,
                last_error=self._last_error,
            )

    def get_detection_history(self) -> list[DetectionSnapshot]:
        with self.lock:
            return list(self._history)

    def start(self, cam: Any) -> None:
        if self.detector is None:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, args=(cam,), daemon=True)
        self._thread.start()
        print("Detection inference thread started.")

    def _append_history(self, dets: list[Detection]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
            }
            for d in dets
        ]
        self._history.append(
            DetectionSnapshot(at=now, count=len(dets), detections=payload)
        )

    def _maybe_console_log(self) -> None:
        if not self._debug_log:
            return
        t = time.monotonic()
        if t - self._last_console_log < _LOG_INTERVAL_SEC:
            return
        self._last_console_log = t
        fps = self._inference_fps_ema
        shape = self._last_frame_shape
        mean = self._last_frame_mean_rgb
        n = self._last_detection_count
        ms = self._last_inference_ms
        msg = (
            f"[detections] fps≈{fps:.2f} infer_ms={ms:.1f} "
            f"shape={shape} mean_rgb={tuple(round(x, 1) for x in mean) if mean else None} "
            f"last_count={n}"
        )
        logger.info(msg)
        print(msg, flush=True)

    def _loop(self, cam: Any) -> None:
        assert self.detector is not None
        while self.running:
            with cam.lock:
                frame = cam.latest_frame
            if frame is None:
                with self.lock:
                    self._frames_waited_none += 1
                time.sleep(0.01)
                continue

            shape = tuple(int(x) for x in frame.shape)
            mean_rgb: tuple[float, float, float] | None = None
            std_rgb: tuple[float, float, float] | None = None
            if frame.ndim == 3 and frame.shape[2] == 3:
                f = frame.astype(np.float64)
                m = f.reshape(-1, 3).mean(axis=0)
                s = f.reshape(-1, 3).std(axis=0)
                mean_rgb = (float(m[0]), float(m[1]), float(m[2]))
                std_rgb = (float(s[0]), float(s[1]), float(s[2]))

            t0 = time.perf_counter()
            try:
                dets = self.detector.detect(frame)
            except Exception as e:
                with self.lock:
                    self._last_error = f"{type(e).__name__}: {e}"
                logger.exception("Detection inference failed")
                time.sleep(0.05)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            inst_fps = 1000.0 / elapsed_ms if elapsed_ms > 1e-6 else 0.0
            now_iso = datetime.now(timezone.utc).isoformat()

            with self.lock:
                self._frames_processed += 1
                self._last_frame_shape = shape
                self._last_frame_mean_rgb = mean_rgb
                self._last_frame_std_rgb = std_rgb
                self._last_inference_ms = elapsed_ms
                if self._inference_fps_ema <= 0:
                    self._inference_fps_ema = inst_fps
                else:
                    self._inference_fps_ema = (1 - _FPS_EMA_ALPHA) * self._inference_fps_ema + _FPS_EMA_ALPHA * inst_fps
                self._last_detection_count = len(dets)
                self._last_inference_at_iso = now_iso
                self._last_error = None
                self.latest = dets
                self._append_history(dets)

            self._maybe_console_log()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        det = self.detector
        if det is not None and hasattr(det, "close"):
            det.close()
