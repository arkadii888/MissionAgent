"""Draw detection overlays on RGB frames (Picamera2 / OpenCV)."""

import cv2
import numpy as np

from agent.orchestrator.inference.yolo_onnx import Detection


def color_rgb_for_class(class_id: int) -> tuple[int, int, int]:
    x = (class_id * 7919 + 17) & 0xFFFFFF
    r, g, b = x & 255, (x >> 8) & 255, (x >> 16) & 255
    return (max(r, 50), max(g, 50), max(b, 50))


def annotate_frame(rgb: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw overlays in place on rgb (same channel order as Picamera2)."""
    h, w = rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    counts: dict[str, int] = {}

    for d in dets:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
        x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_rgb_for_class(d.class_id)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), bl = cv2.getTextSize(label, font, 0.5, 1)
        if y1 > th + bl + 8:
            ty = y1 - 4
        else:
            ty = min(h - 2, y2 + th + 6)
        cv2.rectangle(
            rgb,
            (x1, ty - th - 2),
            (x1 + tw + 4, ty + bl + 2),
            color,
            -1,
        )
        cv2.putText(
            rgb,
            label,
            (x1 + 2, ty),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    summary_lines = [f"Total: {len(dets)}"] + [
        f"{name}: {n}" for name, n in sorted(counts.items())
    ]
    x0, y_line = 8, 24
    fs, thk = 0.55, 1
    for line in summary_lines:
        (tw, th), bl = cv2.getTextSize(line, font, fs, thk)
        cv2.rectangle(
            rgb,
            (x0 - 2, y_line - th - 2),
            (x0 + tw + 8, y_line + bl + 2),
            (32, 32, 32),
            -1,
        )
        cv2.putText(
            rgb,
            line,
            (x0 + 4, y_line),
            font,
            fs,
            (255, 255, 255),
            thk,
            cv2.LINE_AA,
        )
        y_line += th + bl + 8

    return rgb
