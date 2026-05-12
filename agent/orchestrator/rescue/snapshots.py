"""Save rescue snapshot images: annotated full frame and optional cropped person region."""

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from agent.orchestrator.inference.yolo_common import Detection
from agent.orchestrator.vision_overlay import annotate_frame


def _best_person_xyxy(
    dets: Iterable[Detection],
) -> tuple[float, float, float, float] | None:
    """Return the bounding box of the highest-confidence person detection, or None.

    Args:
        dets: Iterable of Detection objects from a single frame.

    Returns:
        (x1, y1, x2, y2) pixel coordinates, or None if no person was detected.
    """
    best_conf = 0.0
    best_xyxy: tuple[float, float, float, float] | None = None
    for d in dets:
        if d.class_id != 0 and d.class_name != "person":
            continue
        if d.confidence > best_conf:
            best_conf = float(d.confidence)
            best_xyxy = d.xyxy
    return best_xyxy


def save_rescue_snapshots(
    frame_bgr: np.ndarray,
    dets: list[Detection],
    *,
    out_dir: Path,
    save_person_crop: bool = True,
) -> tuple[Path, Path | None]:
    """Save an annotated full frame and optionally a cropped person region to *out_dir*.

    The full frame is always written as a JPEG with a timestamp-based filename and
    receives all detection overlays. When ``save_person_crop`` is true, a second JPEG
    is written: the crop is tightly bounded to the highest-confidence person bounding
    box (with a small margin) and is used for multimodal LLM analysis.

    If no person bbox is available (e.g. detections have already aged out) the crop
    falls back to the central quarter of the frame (only when ``save_person_crop``).

    Args:
        frame_bgr: Source BGR frame as a NumPy array (shape H×W×3).
        dets: Detection list for this frame; used to draw overlays and pick the
            person crop when enabled.
        out_dir: Directory for JPEG output. Created automatically if missing.
        save_person_crop: When false, only the annotated full frame is written.

    Returns:
        ``(full_frame_path, crop_path)`` where ``crop_path`` is ``None`` if no crop
        was saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame_bgr.shape[:2]

    # Annotated full frame — drawn on a copy so the caller's array is untouched.
    full = frame_bgr.copy()
    annotate_frame(full, dets, image_is_bgr=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = out_dir / f"{ts}_full.jpg"
    cv2.imwrite(str(full_path), full)

    if not save_person_crop:
        return full_path, None

    # Person crop: highest-confidence person bbox with a small margin, or centre fallback.
    xyxy = _best_person_xyxy(dets)
    if xyxy is None:
        cx, cy = w // 2, h // 2
        size = min(w, h) // 4
        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(w - 1, cx + size)
        y2 = min(h - 1, cy + size)
    else:
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w - 1, x2 + margin)
        y2 = min(h - 1, y2 + margin)

    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_path = out_dir / f"{ts}_person.jpg"
    cv2.imwrite(str(crop_path), crop)

    return full_path, crop_path
