"""Picamera2 capture thread for ArduCam (dual-stream lores preview or single-stream fallback)."""

import logging
import os
import threading
import time

log = logging.getLogger(__name__)


def parse_size_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = raw.lower().replace(" ", "").split("x", 1)
    if len(parts) != 2:
        return default
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return default


CAPTURE_SIZE = parse_size_env("CAPTURE_SIZE", (4624, 3472))
STREAM_DISPLAY_SIZE = parse_size_env("STREAM_DISPLAY_SIZE", (1280, 960))


class CameraManager:
    """Picamera2 wrapper; imports ``picamera2`` only when :meth:`start` runs (optional ``raspi`` extra)."""

    def __init__(self) -> None:
        self.picam2: object | None = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.capture_size = CAPTURE_SIZE
        self.stream_display_size = STREAM_DISPLAY_SIZE
        self.startup_error: str | None = None
        self._capture_stream_name = "lores"
        self._stream_mode: str | None = None
        self.thread: threading.Thread | None = None

    def _reset_camera(self) -> None:
        from picamera2 import Picamera2

        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
        self.picam2 = Picamera2()

    def start(self) -> None:
        from libcamera import controls
        from picamera2 import Picamera2

        self.startup_error = None
        self.picam2 = Picamera2()
        assert self.picam2 is not None
        try:
            cfg = self.picam2.create_preview_configuration(
                main={"size": self.capture_size, "format": "RGB888"},
                lores={"size": self.stream_display_size, "format": "RGB888"},
                buffer_count=4,
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            self._capture_stream_name = "lores"
            self._stream_mode = "dual"
        except Exception as e:
            log.warning(
                "Dual-stream (main+lores) failed: %r; falling back to single-stream preview.",
                e,
            )
            self._reset_camera()
            try:
                cfg = self.picam2.create_preview_configuration(
                    main={"size": self.stream_display_size, "format": "RGB888"},
                    buffer_count=4,
                )
                self.picam2.configure(cfg)
                self.picam2.start()
                self._capture_stream_name = "main"
                self._stream_mode = "single_preview"
            except Exception as e2:
                self.startup_error = f"{type(e2).__name__}: {e2}"
                self._stream_mode = None
                log.error("Camera start failed: %s", self.startup_error)
                return

        try:
            self.picam2.set_controls({
                "AfMode": controls.AfModeEnum.Continuous,
                "AwbMode": controls.AwbModeEnum.Auto,
            })
        except Exception as e:
            log.warning("Camera controls (AF/AWB) skipped: %s", e)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        log.info(
            "Camera capture thread started (mode=%s, dequeue=%r).",
            self._stream_mode,
            self._capture_stream_name,
        )

    def _capture_loop(self) -> None:
        assert self.picam2 is not None
        while self.running:
            try:
                frame = self.picam2.capture_array(self._capture_stream_name)
                with self.lock:
                    self.latest_frame = frame
            except Exception as e:
                log.error("Capture error: %s", e)
                break
            time.sleep(0.01)

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
            self.thread = None
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
