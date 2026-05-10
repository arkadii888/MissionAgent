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
        #: True when ``capture_array`` numpy is OpenCV-style BGR (channel 0 = blue).
        #: Picamera2 maps stream **RGB888** → BGR memory and **BGR888** → RGB memory
        #: (see ``picamera2`` helpers ``_get_pil_mode`` / JPEG ``FORMAT_TABLE``).
        self.frames_are_bgr: bool = False
        self._pixel_format: str = "RGB888"

    def _pixel_format_try_order(self) -> list[str]:
        """Prefer ``RGB888``: ``capture_array`` is BGR in memory (native for :class:`~cv2.VideoWriter`)."""
        if os.environ.get("ARDUCAM_FORCE_BGR888", "").strip().lower() in ("1", "true", "yes"):
            return ["BGR888", "RGB888"]
        if os.environ.get("ARDUCAM_FORCE_RGB888", "").strip().lower() in ("1", "true", "yes"):
            return ["RGB888", "BGR888"]
        return ["RGB888", "BGR888"]

    def buffer_is_bgr(self) -> bool:
        """
        Whether ``capture_array`` frames are BGR for OpenCV / VideoWriter.

        Override with ``ARDUCAM_PIXEL_LAYOUT=bgr|rgb`` if colors are wrong for your ISP build.
        """
        raw = os.environ.get("ARDUCAM_PIXEL_LAYOUT", "").strip().lower()
        if raw in ("bgr", "bgr888", "opencv"):
            return True
        if raw in ("rgb", "rgb888"):
            return False
        return self.frames_are_bgr

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
        self.frames_are_bgr = False
        self.picam2 = Picamera2()
        assert self.picam2 is not None

        dual_ok = False
        last_dual_err: Exception | None = None
        for px in self._pixel_format_try_order():
            try:
                cfg = self.picam2.create_preview_configuration(
                    main={"size": self.capture_size, "format": px},
                    lores={"size": self.stream_display_size, "format": px},
                    buffer_count=4,
                )
                self.picam2.configure(cfg)
                self.picam2.start()
                self._capture_stream_name = "lores"
                self._stream_mode = "dual"
                self._pixel_format = px
                self.frames_are_bgr = px == "RGB888"
                dual_ok = True
                log.info("Picamera2 dual-stream using pixel format %s.", px)
                break
            except Exception as e:
                last_dual_err = e
                self._reset_camera()

        if not dual_ok:
            log.warning(
                "Dual-stream failed (%r); falling back to single-stream preview.",
                last_dual_err,
            )
            self._reset_camera()
            assert self.picam2 is not None
            single_ok = False
            last_single_err: Exception | None = None
            for px in self._pixel_format_try_order():
                try:
                    cfg = self.picam2.create_preview_configuration(
                        main={"size": self.stream_display_size, "format": px},
                        buffer_count=4,
                    )
                    self.picam2.configure(cfg)
                    self.picam2.start()
                    self._capture_stream_name = "main"
                    self._stream_mode = "single_preview"
                    self._pixel_format = px
                    self.frames_are_bgr = px == "RGB888"
                    single_ok = True
                    log.info("Picamera2 single-stream using pixel format %s.", px)
                    break
                except Exception as e2:
                    last_single_err = e2
                    self._reset_camera()

            if not single_ok:
                self.startup_error = (
                    f"{type(last_single_err).__name__}: {last_single_err}"
                    if last_single_err is not None
                    else "unknown"
                )
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
            "Camera capture thread started (mode=%s, dequeue=%r, format=%s, buffer_is_bgr=%s).",
            self._stream_mode,
            self._capture_stream_name,
            self._pixel_format,
            self.buffer_is_bgr(),
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

    def capture_for_detection(self):
        """
        Full-resolution frame for inference (dual-stream: main). Preview thread uses lores.
        """
        assert self.picam2 is not None
        with self.lock:
            if self._stream_mode == "dual":
                try:
                    return self.picam2.capture_array("main")
                except Exception as e:
                    log.error("capture_array(main) failed: %s", e)
                    return None
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

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
