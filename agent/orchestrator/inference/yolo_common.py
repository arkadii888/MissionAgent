"""Shared YOLO types and image/box helpers used by the Hailo detector."""

from dataclasses import dataclass

import cv2
import numpy as np

_DEFAULT_IMGSZ = 640


def _letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int] = (_DEFAULT_IMGSZ, _DEFAULT_IMGSZ),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize image with unchanged aspect ratio, padding to new_shape with RGB uint8."""
    shape = im.shape[:2]  # current [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw_half, dh_half = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh_half - 0.1)), int(round(dh_half + 0.1))
    left, right = int(round(dw_half - 0.1)), int(round(dw_half + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (float(left), float(top))


def _scale_boxes_back(
    boxes_xyxy: np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Map boxes from letterboxed image coords to original image (h, w)."""
    gain = ratio_pad[0][0]
    pad = ratio_pad[1]
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels


def scale_detections_xyxy(
    dets: list[Detection], sx: float, sy: float
) -> list[Detection]:
    """Scale box coordinates (e.g. full-frame → preview stream)."""
    out: list[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        out.append(
            Detection(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
            )
        )
    return out


def nms_xyxy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return indices to keep after single-class NMS. boxes_xyxy Nx4, scores N."""
    n = int(boxes_xyxy.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64)
    rects: list[list[float]] = []
    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        rects.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    idx = cv2.dnn.NMSBoxes(
        rects,
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )
    if idx is None or len(idx) == 0:
        return np.array([], dtype=np.int64)
    flat = np.asarray(idx).reshape(-1)
    return flat.astype(np.int64, copy=False)
