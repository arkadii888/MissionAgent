"""
YOLO26 ONNX Runtime inference for RGB uint8 frames (Picamera2 capture_array).

Expects Ultralytics export with end-to-end head (default): output shape (1, max_det, 6)
per row: x1, y1, x2, y2, confidence, class_id (in letterboxed input space).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from .coco_names import COCO_CLASSES
from .paths import resolve_model_file

# Input Params
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
    # ratio_pad format matches scale_boxes: gain, (pad_left, pad_top)
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


class Yolo26OnnxDetector:
    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.25,
        class_names: Sequence[str] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        path = model_path or os.environ.get("YOLO_ONNX_PATH", "models/yolo26n.onnx")
        p = Path(path)
        if not p.is_absolute():
            p = resolve_model_file(p)
        path = str(p)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ONNX model not found: {path}. Set YOLO_ONNX_PATH or place model in models/."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            path,
            sess_options=sess_opts,
            providers=providers or ["CPUExecutionProvider"],
        )

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape
        # static export is commonly [1, 3, 640, 640], but some models are fixed-batch
        # (e.g. [8, 3, 640, 640]). Keep track of required input batch for fallback safety.
        self._in_batch = 1
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self._in_h, self._in_w = shape[2], shape[3]
            if isinstance(shape[0], int) and shape[0] > 0:
                self._in_batch = int(shape[0])
        else:
            self._in_h = self._in_w = _DEFAULT_IMGSZ

        self._conf_threshold = conf_threshold
        self._class_names = tuple(class_names) if class_names is not None else COCO_CLASSES

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """
        Run detection on a single RGB uint8 frame (H, W, 3).
        Returns boxes in original image pixel coordinates.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")

        h0, w0 = rgb.shape[0], rgb.shape[1]
        im, ratio_pad_gain, pad_xy = _letterbox(rgb, (self._in_h, self._in_w))
        im = im.transpose((2, 0, 1))  # HWC -> CHW
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255.0
        batch = np.expand_dims(im, axis=0)
        if self._in_batch > 1:
            # Some exported ONNX graphs require fixed batch (e.g. 8). Replicate the
            # same frame to satisfy the static shape and parse detections for slot 0.
            batch = np.repeat(batch, self._in_batch, axis=0)

        outs = self._session.run(None, {self._input_name: batch})
        if not outs:
            return []

        raw = outs[0]
        dets = self._parse_e2e(raw, batch_index=0)
        if dets is None:
            raise ValueError(
                "Unexpected ONNX output shape; expected YOLO26 end-to-end (N, 300, 6) or compatible."
            )

        dets = dets[dets[:, 4] >= self._conf_threshold]
        if dets.size == 0:
            return []

        boxes = dets[:, :4].astype(np.float32)
        boxes = _scale_boxes_back(boxes, (h0, w0), (ratio_pad_gain, pad_xy))

        out: list[Detection] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            conf = float(dets[i, 4])
            cid = int(round(float(dets[i, 5])))
            name = self._class_names[cid] if 0 <= cid < len(self._class_names) else str(cid)
            out.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    def _parse_e2e(self, raw: np.ndarray, batch_index: int = 0) -> np.ndarray | None:
        """
        YOLO26 end-to-end: (1, max_det, 6) with [x1,y1,x2,y2,conf,class].
        Rows may be padded; filter by confidence later.
        """
        if raw.ndim != 3 or raw.shape[2] < 6:
            return None
        # (B, max_det, 6+) -> (max_det, 6+) for one frame.
        if raw.shape[0] <= 0:
            return None
        bi = min(max(int(batch_index), 0), raw.shape[0] - 1)
        d = raw[bi]
        return d.astype(np.float32, copy=False)