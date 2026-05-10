"""
YOLO on Hailo-8 / Hailo-8L via HailoRT (HEF). RGB uint8 frames from Picamera2.

Requires Raspberry Pi OS packages that provide `hailo_platform` and a
compatible .hef for your accelerator (Hailo-8 vs Hailo-8L).

Supports:
- End-to-end export: single output (B, N, 6+) with x1,y1,x2,y2,conf,class in letterbox space
  (same parsing as inference.yolo_onnx). B≥1 (e.g. batch-8 HEF uses B=8; decoded batch slot 0).
- Multi-head export (6 outputs): cls/reg at 80/40/20 grids — Python postprocess (logit threshold).
  Shapes may be (H,W,C), (1,H,W,C), or (B,H,W,C) when B matches the network input batch.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coco_names import COCO_CLASSES
from .yolo_onnx import Detection, _letterbox, _scale_boxes_back

try:
    from hailo_platform import (
        ConfigureParams,
        FormatType,
        HEF,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
except ImportError:
    HEF = None  # type: ignore[misc, assignment]


def hailo_platform_available() -> bool:
    return HEF is not None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


# Typical YOLO26 split-head Hailo builds: semantic layout is (grid_h, grid_w, C).
# HailoRT may omit batch ((H,W,C)), use batch 1 ((1,H,W,C)), or batch B e.g. 8 ((8,H,W,C)).
_TAIL_HWC_TO_HEAD: dict[tuple[int, int, int], str] = {
    (80, 80, 80): "cls_80",
    (40, 40, 80): "cls_40",
    (20, 20, 80): "cls_20",
    (80, 80, 4): "reg_80",
    (40, 40, 4): "reg_40",
    (20, 20, 4): "reg_20",
}

# Back-compat: older code matched full reported shapes including leading 1.
_SHAPE_TO_HEAD: dict[tuple[int, ...], str] = {
    (1, 80, 80, 80): "cls_80",
    (1, 40, 40, 80): "cls_40",
    (1, 20, 20, 80): "cls_20",
    (1, 80, 80, 4): "reg_80",
    (1, 40, 40, 4): "reg_40",
    (1, 20, 20, 4): "reg_20",
    **{((1,) + k): v for k, v in _TAIL_HWC_TO_HEAD.items()},
    **_TAIL_HWC_TO_HEAD,
}


def _split_batch_hwc(full: tuple[int, ...]) -> tuple[int, tuple[int, int, int]]:
    """Return (batch_dim, (H,W,C)) for output vstream shapes."""
    if len(full) == 4:
        b, h, w, c = full
        return int(b), (int(h), int(w), int(c))
    if len(full) == 3:
        h, w, c = full
        return 1, (int(h), int(w), int(c))
    raise ValueError(f"Expected HWC or NHWC feature map meta-shape, got {full}")


def _hwc_feature_map(x: np.ndarray, batch_index: int = 0) -> np.ndarray:
    """Select one batch slice; accept (H, W, C), (N, H, W, C) with N>=1."""
    if x.ndim == 4:
        if x.shape[0] <= batch_index:
            raise ValueError(
                f"Output batch dimension {x.shape[0]} too small for index {batch_index}"
            )
        return np.ascontiguousarray(x[batch_index])
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected feature map NHWC or HWC, got shape {x.shape}")


@dataclass
class _OutputMode:
    kind: str  # "e2e" | "multi_head"
    e2e_name: str | None = None
    #: Network input/output batch compiled into the HEF (e.g. 8 for yolo26n_b8.hef).
    hef_batch: int = 1


class YoloHailoDetector:
    """Run HEF inference on the Hailo NPU; returns `Detection` in original image coordinates."""

    def __init__(
        self,
        hef_path: str | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        if not hailo_platform_available():
            raise ImportError(
                "hailo_platform not installed. On Raspberry Pi install Hailo software "
                "(e.g. sudo apt install hailo-all) so Python can import hailo_platform."
            )

        path = hef_path or os.environ.get("YOLO_HEF_PATH", "").strip()
        if not path:
            path = str(Path(__file__).resolve().parents[2] / "models" / "yolo26n_b8.hef")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Hailo HEF not found: {path}. Set YOLO_HEF_PATH or place yolo26n_b8.hef under agent/models/."
            )

        self._conf_threshold = conf_threshold
        self._class_names = COCO_CLASSES
        self._hef_path = path

        self._target = VDevice()
        self._hef = HEF(path)
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._target.configure(self._hef, configure_params)[0]

        self._input_vstream_params = InputVStreamParams.make(self._network_group)
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        self._in_info = self._hef.get_input_vstream_infos()
        self._out_info = self._hef.get_output_vstream_infos()
        if not self._in_info:
            raise RuntimeError("HEF has no input vstreams")

        self._input_name = self._in_info[0].name
        shp = tuple(self._in_info[0].shape)
        # Typical static HEF: [B, 640, 640, 3] (NHWC) or [B, 3, 640, 640] (NCHW)
        self._in_hef_batch: int = int(shp[0]) if len(shp) == 4 else 1
        if len(shp) == 4:
            if shp[1] == 3 and isinstance(shp[2], int) and isinstance(shp[3], int):
                self._in_nchw = True
                self._in_h, self._in_w = int(shp[2]), int(shp[3])
            elif shp[3] == 3:
                self._in_nchw = False
                self._in_h, self._in_w = int(shp[1]), int(shp[2])
            else:
                self._in_nchw = False
                self._in_h, self._in_w = 640, 640
        else:
            self._in_nchw = False
            self._in_hef_batch = 1
            self._in_h, self._in_w = 640, 640

        self._out_mode = self._detect_output_mode()

    def _detect_output_mode(self) -> _OutputMode:
        outs = self._out_info
        if len(outs) == 1:
            name = outs[0].name
            shp_full = tuple(outs[0].shape)
            hef_b = self._infer_e2e_batch(shp_full)
            return _OutputMode(kind="e2e", e2e_name=name, hef_batch=hef_b)

        by_head: dict[str, tuple[int, tuple[int, int, int]]] = {}
        for v in outs:
            raw = tuple(v.shape)
            b, hwc = _split_batch_hwc(raw)
            key = _TAIL_HWC_TO_HEAD.get(hwc)
            if key is None:
                # Fall back to legacy dict (includes (1,H,W,C) keys)
                key = _SHAPE_TO_HEAD.get(raw)
                if key is None:
                    raise ValueError(
                        f"Unknown output tensor shape {raw}; expected one of the YOLO26 "
                        f"cls/reg heads at 80²/40²/20² (optionally leading batch)."
                    )
            if key in by_head:
                raise ValueError(f"Duplicate head {key} in HEF outputs")
            by_head[key] = (b, hwc)

        required_heads = frozenset(
            ("cls_80", "cls_40", "cls_20", "reg_80", "reg_40", "reg_20")
        )
        if set(by_head) != required_heads:
            raise ValueError(
                "Unsupported HEF output layout: expected 6 split heads "
                f"{sorted(required_heads)}, got {sorted(by_head)}"
            )

        batches = {b for b, _ in by_head.values()}
        if len(batches) != 1:
            raise ValueError(
                f"Inconsistent batch dimension across output tensors: {sorted(batches)}"
            )
        tail_b = next(iter(batches))
        net_b = self._in_hef_batch
        if tail_b != net_b and tail_b != 1:
            raise ValueError(
                f"Output batch {tail_b} does not match input batch {net_b}"
            )

        hb = tail_b if tail_b > 1 else net_b

        return _OutputMode(kind="multi_head", hef_batch=max(hb, 1))

    def _infer_e2e_batch(self, shp_full: tuple[int, ...]) -> int:
        """Deduce compiled batch from e2e output shape."""
        if len(shp_full) == 3 and shp_full[2] >= 6:
            return int(shp_full[0])
        if len(shp_full) == 2 and shp_full[1] >= 6:
            return 1
        raise ValueError(
            f"Unexpected e2e output rank/shape metadata {shp_full}; "
            "expected (B, N, K) with K>=6 or (N, K)."
        )

    def _effective_hef_batch(self) -> int:
        b = self._out_mode.hef_batch
        if b > 1:
            return b
        return max(self._in_hef_batch, 1)

    def _preprocess(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int], Any]:
        h0, w0 = rgb.shape[0], rgb.shape[1]
        im_lb, ratio_pad_gain, pad_xy = _letterbox(rgb, (self._in_h, self._in_w))
        hb = self._effective_hef_batch()
        if self._in_nchw:
            im = im_lb.transpose((2, 0, 1))
            im = np.ascontiguousarray(im, dtype=np.float32)
            im /= 255.0
            one = np.expand_dims(im, axis=0)
        else:
            one = np.expand_dims(im_lb, axis=0).astype(np.uint8, copy=False)
        if hb <= 1:
            batch = one
        else:
            batch = np.repeat(one, hb, axis=0)
        return batch, (h0, w0), (ratio_pad_gain, pad_xy)

    def _infer_raw(self, input_batch: np.ndarray) -> dict[str, np.ndarray]:
        feed: dict[str, Any]
        if isinstance(input_batch, dict):
            feed = input_batch
        else:
            feed = {self._input_name: input_batch}

        # Match hailo-rpi5 / yolo26_hailo examples: activate() then InferVStreams
        with self._network_group.activate():
            with InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params,
            ) as pipeline:
                out = pipeline.infer(feed)
        if not isinstance(out, dict):
            raise RuntimeError(f"Unexpected infer() return type: {type(out)}")
        return {k: np.asarray(v) for k, v in out.items()}

    def _parse_e2e(
        self,
        raw: np.ndarray,
        img0_hw: tuple[int, int],
        ratio_pad: Any,
    ) -> list[Detection]:
        if raw.ndim != 3 or raw.shape[2] < 6:
            raise ValueError(f"Unexpected e2e output shape: {raw.shape}")
        bi = int(os.environ.get("YOLO_HAILO_BATCH_INDEX", "0").strip())
        if bi < 0 or bi >= raw.shape[0]:
            raise ValueError(
                f"YOLO_HAILO_BATCH_INDEX={bi} out of range for e2e batch dim "
                f"{raw.shape[0]}"
            )
        d = raw[bi].astype(np.float32, copy=False)
        d = d[d[:, 4] >= self._conf_threshold]
        if d.size == 0:
            return []
        boxes = d[:, :4].astype(np.float32)
        boxes = _scale_boxes_back(boxes, (img0_hw[0], img0_hw[1]), ratio_pad)
        out: list[Detection] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            conf = float(d[i, 4])
            cid = int(round(float(d[i, 5])))
            name = (
                self._class_names[cid] if 0 <= cid < len(self._class_names) else str(cid)
            )
            out.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    def _parse_multi_head(
        self,
        tensors_by_name: dict[str, np.ndarray],
        img0_hw: tuple[int, int],
        ratio_pad: Any,
    ) -> list[Detection]:
        tensors: dict[str, np.ndarray] = {}
        bi = int(os.environ.get("YOLO_HAILO_BATCH_INDEX", "0").strip())
        hb = max(self._effective_hef_batch(), 1)
        if bi < 0 or bi >= hb:
            raise ValueError(
                f"YOLO_HAILO_BATCH_INDEX={bi} out of range for hef_batch={hb}"
            )

        for name, data in tensors_by_name.items():
            raw = tuple(int(x) for x in data.shape)
            if len(raw) == 4:
                _, hwc = _split_batch_hwc(raw)
                key = _TAIL_HWC_TO_HEAD.get(hwc) or _SHAPE_TO_HEAD.get(raw)
            else:
                key = _SHAPE_TO_HEAD.get(raw)
            if key:
                tensors[key] = data

        required = [
            "cls_80",
            "cls_40",
            "cls_20",
            "reg_80",
            "reg_40",
            "reg_20",
        ]
        missing = [t for t in required if t not in tensors]
        if missing:
            found = [tuple(tensors_by_name[k].shape) for k in tensors_by_name]
            raise ValueError(
                "HEF outputs do not match the expected 6-head YOLO layout. "
                f"Missing mapped heads {missing}. Output shapes: {found}"
            )

        logit_thr = -np.log(1.0 / self._conf_threshold - 1.0)
        strides = (8, 16, 32)
        grids = (80, 40, 20)
        cand: list[tuple[float, float, float, float, float, int]] = []

        for stride, g in zip(strides, grids, strict=True):
            cls_data = _hwc_feature_map(tensors[f"cls_{g}"], batch_index=bi)
            reg_data = _hwc_feature_map(tensors[f"reg_{g}"], batch_index=bi)
            cls_flat = cls_data.reshape(-1, 80)
            reg_flat = reg_data.reshape(-1, 4)
            max_logits = cls_flat.max(axis=1)
            class_ids = cls_flat.argmax(axis=1)
            mask = max_logits > logit_thr
            if not mask.any():
                continue
            indices = np.where(mask)[0]
            scores = _sigmoid(max_logits[indices])
            cls = class_ids[indices]
            rows = indices // g
            cols = indices % g
            l = reg_flat[indices, 0]
            t = reg_flat[indices, 1]
            r_ = reg_flat[indices, 2]
            b = reg_flat[indices, 3]
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r_) * stride
            y2 = (rows + 0.5 + b) * stride
            for j in range(len(indices)):
                cand.append(
                    (float(x1[j]), float(y1[j]), float(x2[j]), float(y2[j]), float(scores[j]), int(cls[j]))
                )

        if not cand:
            return []

        arr = np.array(cand, dtype=np.float32)
        boxes = arr[:, :4]
        boxes = _scale_boxes_back(boxes, (img0_hw[0], img0_hw[1]), ratio_pad)

        out: list[Detection] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            conf = float(arr[i, 4])
            cid = int(arr[i, 5])
            name = (
                self._class_names[cid] if 0 <= cid < len(self._class_names) else str(cid)
            )
            out.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")

        batch, img0_hw, ratio_pad = self._preprocess(rgb)
        outs = self._infer_raw(batch)

        if self._out_mode.kind == "e2e":
            name = self._out_mode.e2e_name
            if name is None or name not in outs:
                if len(outs) == 1:
                    raw = next(iter(outs.values()))
                else:
                    raise ValueError(f"e2e mode but unexpected outputs: {list(outs)}")
            else:
                raw = outs[name]
            return self._parse_e2e(raw, img0_hw, ratio_pad)

        return self._parse_multi_head(outs, img0_hw, ratio_pad)

    def close(self) -> None:
        try:
            self._target.release()
        except Exception:
            pass
