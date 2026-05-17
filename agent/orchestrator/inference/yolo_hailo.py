"""
YOLO on Hailo-8 / Hailo-8L via HailoRT (HEF). RGB uint8 frames from Picamera2.

Requires Raspberry Pi OS packages that provide `hailo_platform` and a
compatible .hef for your accelerator (Hailo-8 vs Hailo-8L).

Supports:
- End-to-end export: single output (1, N, 6+) with x1,y1,x2,y2,conf,class in letterbox space.
- Multi-head export (6 outputs): cls/reg at 80/40/20 grids — Python postprocess (logit threshold).

**Default:** **`YOLO_NUMBER_TILES=1`** — the full frame is letterboxed once to the HEF input
(e.g. 640×640), one inference, boxes mapped back with the same letterbox metadata.

**Multi-tile:** set **`YOLO_NUMBER_TILES`** > 1 to split each frame into a **rows×cols**
grid with `rows*cols == N` (layout chosen to match frame aspect). Each cell is cropped from the
full-res frame, **letterboxed** to the HEF input (e.g. 640²), inferred, then boxes are mapped back
to crop pixels and **offset** into full-frame coordinates; **global NMS** merges duplicates at
seams. **`N` is clamped** per frame to at most **`ceil(W/model_w) * ceil(H/model_h)`** (the
finest non-overlapping 640-style cover of the frame).

Env: **YOLO_NUMBER_TILES**, YOLO_HAILO_FRAME_MODE (informational / health), PERSON_NMS_IOU,
YOLO_PERSON_ONLY. Runtime: get_runtime_info() / GET /health.

Speed: lower YOLO_NUMBER_TILES or CAPTURE_SIZE, skip frames in the app loop,
or use a HEF with true input batch>1. Stacking 8 inputs only works when the HEF input tensor
actually supports batch 8.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coco_names import COCO_CLASSES
from .paths import resolve_model_file
from .yolo_common import Detection, _letterbox, _scale_boxes_back, nms_xyxy

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

logger = logging.getLogger(__name__)


def hailo_platform_available() -> bool:
    """Return whether ``hailo_platform`` imported successfully.

    Returns:
        ``True`` when Hailo Python bindings are available.
    """
    return HEF is not None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _local_peak_mask(x: np.ndarray) -> np.ndarray:
    """3x3 local-max mask for 2D logits/probability maps."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D map for peak mask, got shape {x.shape}")
    p = np.pad(x, ((1, 1), (1, 1)), mode="constant", constant_values=-np.inf)
    neighborhoods = [
        p[0:-2, 0:-2],
        p[0:-2, 1:-1],
        p[0:-2, 2:],
        p[1:-1, 0:-2],
        p[1:-1, 1:-1],
        p[1:-1, 2:],
        p[2:, 0:-2],
        p[2:, 1:-1],
        p[2:, 2:],
    ]
    local_max = neighborhoods[0]
    for n in neighborhoods[1:]:
        local_max = np.maximum(local_max, n)
    return x >= local_max


# Typical YOLO26 split-head Hailo builds. HailoRT may report NHWC with batch in the
# shape (1, H, W, C) or omit batch and report (H, W, C) only — include both.
_SHAPE_TO_HEAD: dict[tuple[int, ...], str] = {
    (1, 80, 80, 80): "cls_80",
    (1, 40, 40, 80): "cls_40",
    (1, 20, 20, 80): "cls_20",
    (1, 80, 80, 4): "reg_80",
    (1, 40, 40, 4): "reg_40",
    (1, 20, 20, 4): "reg_20",
    (80, 80, 80): "cls_80",
    (40, 40, 80): "cls_40",
    (20, 20, 80): "cls_20",
    (80, 80, 4): "reg_80",
    (40, 40, 4): "reg_40",
    (20, 20, 4): "reg_20",
}


def _head_key_from_shape(shape: tuple[int, ...]) -> str | None:
    if len(shape) == 5 and shape[0] >= 1:
        inner = tuple(shape[1:])
        k = _SHAPE_TO_HEAD.get(inner)
        if k is not None:
            return k
    if len(shape) == 4 and shape[0] > 1:
        inner = tuple(shape[1:])
        k = _SHAPE_TO_HEAD.get(inner)
        if k is not None:
            return k
    return _SHAPE_TO_HEAD.get(shape)


def _hwc_feature_map(x: np.ndarray) -> np.ndarray:
    """Drop a leading batch of 1; accept (H, W, C) as-is."""
    if x.ndim == 4 and x.shape[0] == 1:
        return x[0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected feature map NHWC (N=1) or HWC, got shape {x.shape}")


def _hwc_feature_map_b(x: np.ndarray, b: int) -> np.ndarray:
    """Batch slice B,H,W,C -> H,W,C."""
    if x.ndim == 4:
        return x[b]
    raise ValueError(f"Expected NHWC batch, got shape {x.shape}")


@dataclass
class _OutputMode:
    kind: str  # "e2e" | "multi_head" | "three_head_cls"
    e2e_name: str | None = None


@dataclass(frozen=True)
class _TileSpec:
    tx: int
    ty: int
    y0: int
    x0: int
    crop_h: int
    crop_w: int


def _max_tiles_for_frame(
    frame_h: int, frame_w: int, model_in_h: int, model_in_w: int
) -> int:
    """
    Upper bound on tile count: minimum number of model-sized rectangles needed to cover the frame
    (same count as a non-overlapping grid of model_in_w × model_in_h windows).
    """
    if frame_h <= 0 or frame_w <= 0:
        return 1
    cw = max(1, int(model_in_w))
    ch = max(1, int(model_in_h))
    return max(1, math.ceil(frame_w / cw) * math.ceil(frame_h / ch))


def _grid_shapes_with_n_tiles(n: int) -> list[tuple[int, int]]:
    """Unique (rows, cols) with rows * cols == n, rows >= 1, cols >= 1."""
    if n < 1:
        return [(1, 1)]
    if n == 1:
        return [(1, 1)]
    pairs: set[tuple[int, int]] = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            j = n // i
            pairs.add((i, j))
            pairs.add((j, i))
        i += 1
    return sorted(pairs)


def _best_grid_shape(n: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Pick rows, cols with rows * cols == n so each tile's aspect ratio (w/h) is closest to the
    full frame's aspect. Tie: prefer more columns on landscape, more rows on portrait.
    """
    if n <= 1 or frame_w <= 0 or frame_h <= 0:
        return (1, 1)
    target_log_ar = math.log(frame_w / frame_h)
    best: tuple[int, int] = (1, n)
    best_score = float("inf")
    landscape = frame_w >= frame_h
    for rows, cols in _grid_shapes_with_n_tiles(n):
        tile_ar = (frame_w * rows) / (frame_h * cols)
        if tile_ar <= 0:
            continue
        score = abs(math.log(tile_ar) - target_log_ar)
        if score < best_score - 1e-12:
            best_score = score
            best = (rows, cols)
        elif abs(score - best_score) <= 1e-12:
            br, bc = best
            if landscape and cols > bc:
                best = (rows, cols)
            elif not landscape and rows > br:
                best = (rows, cols)
    return best


def _grid_specs(frame_h: int, frame_w: int, rows: int, cols: int) -> list[_TileSpec]:
    """Non-overlapping partition: integer slice bounds cover the full frame."""
    if frame_h <= 0 or frame_w <= 0 or rows < 1 or cols < 1:
        return []
    specs: list[_TileSpec] = []
    for ty in range(rows):
        y0 = ty * frame_h // rows
        y1 = (ty + 1) * frame_h // rows
        for tx in range(cols):
            x0 = tx * frame_w // cols
            x1 = (tx + 1) * frame_w // cols
            crop_h = y1 - y0
            crop_w = x1 - x0
            if crop_h <= 0 or crop_w <= 0:
                continue
            specs.append(_TileSpec(tx, ty, y0, x0, crop_h, crop_w))
    return specs


def _static_dim(d: Any) -> int | None:
    if isinstance(d, int) and d > 0:
        return d
    return None


class YoloHailoDetector:
    """Run HEF inference on the Hailo NPU; returns `Detection` in original image coordinates."""

    def __init__(
        self,
        hef_path: str | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        """Load a Hailo HEF and configure input/output vstreams.

        Args:
            hef_path: Path to ``.hef`` file; defaults to ``YOLO_HEF_PATH`` env.
            conf_threshold: Minimum detection confidence (0–1).

        Raises:
            ImportError: If ``hailo_platform`` is not installed.
            FileNotFoundError: If the HEF file is missing.
            ValueError: If ``YOLO_NUMBER_TILES`` is invalid.
            RuntimeError: If the HEF has no input vstreams.
        """
        if not hailo_platform_available():
            raise ImportError(
                "hailo_platform not installed. On Raspberry Pi install Hailo software "
                "(e.g. sudo apt install hailo-all) so Python can import hailo_platform."
            )

        path = hef_path or os.environ.get("YOLO_HEF_PATH", "models/yolo26n_b8.hef")
        p = Path(path)
        if not p.is_absolute():
            p = resolve_model_file(p)
        path = str(p)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Hailo HEF not found: {path}. Set YOLO_HEF_PATH or place the file under models/."
            )

        self._conf_threshold = conf_threshold
        self._class_names = COCO_CLASSES
        self._hef_path = path
        nt_raw = os.environ.get("YOLO_NUMBER_TILES", "").strip()
        if nt_raw:
            try:
                self._number_tiles_requested = max(1, int(nt_raw))
            except ValueError as e:
                raise ValueError(
                    f"YOLO_NUMBER_TILES must be a positive integer, got {nt_raw!r}"
                ) from e
        else:
            self._number_tiles_requested = 1
        self._nms_iou = float(os.environ.get("PERSON_NMS_IOU", "0.35"))
        self._max_candidates = max(100, int(os.environ.get("YOLO_MAX_CANDIDATES", "3000")))
        self._max_detections = max(1, int(os.environ.get("YOLO_MAX_DETECTIONS", "40")))
        # For class-only 3-head outputs (no box regression), use larger anchor-like boxes.
        self._three_head_box_scale = float(
            os.environ.get("YOLO_THREE_HEAD_BOX_SCALE", "5.0")
        )
        self._person_only = os.environ.get("YOLO_PERSON_ONLY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "",
        )
        # letterbox: whole frame -> model input (e.g. 640²) once; tile: sliding windows (legacy).
        _fm = os.environ.get("YOLO_HAILO_FRAME_MODE", "letterbox").strip().lower()
        if _fm in ("tile", "tiled", "multi", "windows"):
            self._frame_mode = "tile"
        else:
            self._frame_mode = "letterbox"

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
        shp = self._in_info[0].shape
        self._in_batch = 1
        # Typical static HEF: [1, 640, 640, 3] (NHWC) or [1, 3, 640, 640] (NCHW).
        # Model Zoo HEFs often report [H, W, C] with implicit batch 1 (see get_runtime_info).
        if len(shp) == 4:
            b0 = _static_dim(shp[0])
            if shp[1] == 3 and isinstance(shp[2], int) and isinstance(shp[3], int):
                self._in_nchw = True
                self._in_h, self._in_w = int(shp[2]), int(shp[3])
                if b0 is not None:
                    self._in_batch = b0
            elif shp[3] == 3:
                self._in_nchw = False
                self._in_h, self._in_w = int(shp[1]), int(shp[2])
                if b0 is not None:
                    self._in_batch = b0
            else:
                self._in_nchw = False
                self._in_h, self._in_w = 640, 640
        elif len(shp) == 3 and shp[2] == 3:
            self._in_nchw = False
            self._in_h, self._in_w = int(shp[0]), int(shp[1])
        else:
            self._in_nchw = False
            self._in_h, self._in_w = 640, 640

        self._out_mode = self._detect_output_mode()
        self._last_detect_tiling: dict[str, Any] | None = None

    def get_runtime_info(self) -> dict[str, Any]:
        """Introspection for /health: batch size, tiling, and how tiles are scheduled."""
        if self._frame_mode == "letterbox":
            pattern = "single_full_frame_letterbox_to_model_input"
        else:
            pattern = (
                "batched_input_chunks"
                if self._in_batch > 1
                else "serial_infer_per_tile_one_session"
            )
        info: dict[str, Any] = {
            "hef_path": self._hef_path,
            "input_name": self._input_name,
            "input_shape": list(self._in_info[0].shape) if self._in_info else None,
            "input_layout_nchw": self._in_nchw,
            "input_hw": [self._in_h, self._in_w],
            "model_input_hw": [self._in_h, self._in_w],
            "hef_batch_size": self._in_batch,
            "frame_mode": self._frame_mode,
            "yolo_number_tiles_requested": self._number_tiles_requested,
            "yolo_number_tiles_max_formula": (
                "per frame: ceil(frame_w / model_input_w) * ceil(frame_h / model_input_h)"
            ),
            "tiling": "partition_into_rows_times_cols_then_letterbox_each_crop",
            "output_mode": self._out_mode.kind,
            "inference_scheduling": pattern,
            "model_zoo_batch_column_note": (
                "Hailo Model Zoo tables often list throughput at different effective batch "
                "settings; those numbers do not always imply a HEF with explicit tensor "
                "batch > 1. Runtime batching in this app depends on the HEF input shape "
                "(see hef_batch_size above)."
            ),
            "parallelism_note": (
                "The Hailo NPU runs one compiled network per infer() call. "
                "Multiple tiles are either stacked when hef_batch_size>1 (batched infer) "
                "or run as repeated infer() in one InferVStreams session (still one tile "
                "per NPU invocation when batch is 1). True parallel tile execution would "
                "require multiple devices/contexts or a HEF compiled with an explicit batch "
                "dimension in the input vstream (recompile / different artifact)."
            ),
        }
        if self._last_detect_tiling is not None:
            info["last_frame_tiling"] = dict(self._last_detect_tiling)
        return info

    def _detect_output_mode(self) -> _OutputMode:
        outs = self._out_info
        if len(outs) == 1:
            return _OutputMode(kind="e2e", e2e_name=outs[0].name)

        if len(outs) == 3:
            shapes = [tuple(v.shape) for v in outs]
            normalized: list[tuple[int, int, int]] = []
            for s in shapes:
                if len(s) == 4 and isinstance(s[0], int) and s[0] > 0:
                    normalized.append((int(s[1]), int(s[2]), int(s[3])))
                elif len(s) == 3:
                    normalized.append((int(s[0]), int(s[1]), int(s[2])))
            if sorted(normalized) == [(20, 20, 80), (40, 40, 80), (80, 80, 80)]:
                return _OutputMode(kind="three_head_cls")

        shapes = [tuple(v.shape) for v in outs]
        if len(shapes) == 6:
            ok = True
            for s in shapes:
                key = _head_key_from_shape(s)
                if key is None:
                    ok = False
                    break
            if ok:
                return _OutputMode(kind="multi_head")

        raise ValueError(
            f"Unsupported HEF output layout: {len(outs)} outputs with "
            f"{[(v.name, tuple(v.shape)) for v in outs]}. "
            "Expected either 1 end-to-end tensor (..., 6+), 6 split-head tensors "
            "(cls/reg at 80², 40², 20²), or 3 class-only heads "
            "(80²/40²/20² with 80 classes)."
        )

    def _parse_three_head_cls(
        self,
        tensors_by_name: dict[str, np.ndarray],
        img0_hw: tuple[int, int],
        ratio_pad: Any,
        batch_index: int | None = None,
    ) -> list[Detection]:
        """
        Decode 3-head class-only maps (H, W, C=80) by turning hot cells into coarse boxes.
        This supports HEFs exposing conv heads like conv61/conv74/conv85.
        """
        heads: dict[int, np.ndarray] = {}
        for _, data in tensors_by_name.items():
            sh = tuple(data.shape)
            if len(sh) == 4:
                h, w, c = int(sh[1]), int(sh[2]), int(sh[3])
            elif len(sh) == 3:
                h, w, c = int(sh[0]), int(sh[1]), int(sh[2])
            else:
                continue
            if h == w and h in (80, 40, 20) and c == 80:
                heads[h] = data

        missing = [g for g in (80, 40, 20) if g not in heads]
        if missing:
            raise ValueError(
                f"3-head decode expected grids 80/40/20, missing {missing}. "
                f"Got shapes: {[tuple(v.shape) for v in tensors_by_name.values()]}"
            )

        # Threshold in logit space; avoids accepting almost all cells when conf<0.5.
        conf = min(max(float(self._conf_threshold), 1e-6), 1.0 - 1e-6)
        logit_thr = float(np.log(conf / (1.0 - conf)))
        cand: list[tuple[float, float, float, float, float, int]] = []
        for g in (80, 40, 20):
            t = heads[g]
            if batch_index is not None and t.ndim == 4 and t.shape[0] > 1:
                cls_map = _hwc_feature_map_b(t, batch_index)
            else:
                cls_map = _hwc_feature_map(t)
            # Keep decode robust even when class count differs from COCO.
            cdim = int(cls_map.shape[2])
            if self._person_only:
                if cdim <= 0:
                    continue
                logit0 = cls_map[:, :, 0]
                mask = (logit0 > logit_thr) & _local_peak_mask(logit0)
                if not mask.any():
                    continue
                ys, xs = np.where(mask)
                scores = _sigmoid(logit0[ys, xs])
                class_ids = np.zeros_like(ys, dtype=np.int64)
            else:
                max_logits = cls_map.max(axis=2)
                class_ids_map = cls_map.argmax(axis=2)
                mask = (max_logits > logit_thr) & _local_peak_mask(max_logits)
                if not mask.any():
                    continue
                ys, xs = np.where(mask)
                scores = _sigmoid(max_logits[ys, xs])
                class_ids = class_ids_map[ys, xs]

            stride = self._in_w / float(g)
            box_size = max(stride, stride * self._three_head_box_scale)
            for y, x, s, cid in zip(ys, xs, scores, class_ids, strict=True):
                cx = (float(x) + 0.5) * stride
                cy = (float(y) + 0.5) * stride
                half = 0.5 * box_size
                x1 = max(0.0, cx - half)
                y1 = max(0.0, cy - half)
                x2 = min(float(self._in_w), cx + half)
                y2 = min(float(self._in_h), cy + half)
                cand.append((x1, y1, x2, y2, float(s), int(cid)))

        if not cand:
            return []

        arr = np.array(cand, dtype=np.float32)
        # Safety cap to keep endpoint latency stable on dense heads.
        if arr.shape[0] > self._max_candidates:
            keep = np.argsort(arr[:, 4])[::-1][: self._max_candidates]
            arr = arr[keep]
        boxes = arr[:, :4]
        boxes = _scale_boxes_back(boxes, (img0_hw[0], img0_hw[1]), ratio_pad)
        out: list[Detection] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            conf = float(arr[i, 4])
            cid = int(arr[i, 5])
            if self._person_only:
                cid = 0
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

    def _preprocess(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int], Any]:
        h0, w0 = rgb.shape[0], rgb.shape[1]
        im_lb, ratio_pad_gain, pad_xy = _letterbox(rgb, (self._in_h, self._in_w))
        if self._in_nchw:
            im = im_lb.transpose((2, 0, 1))
            im = np.ascontiguousarray(im, dtype=np.float32)
            im /= 255.0
            batch = np.expand_dims(im, axis=0)
        else:
            batch = np.expand_dims(im_lb, axis=0).astype(np.uint8, copy=False)
        return batch, (h0, w0), (ratio_pad_gain, pad_xy)

    def _stack_batch(self, batches: list[np.ndarray]) -> np.ndarray:
        """Stack N single-tile batches (1,...) -> (N,...)."""
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches, axis=0)

    def _infer_raw(self, input_batch: np.ndarray) -> dict[str, np.ndarray]:
        feed: dict[str, Any]
        if isinstance(input_batch, dict):
            feed = input_batch
        else:
            feed = {self._input_name: input_batch}

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

    def _infer_many_serial(
        self, feeds: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]:
        """One activate + InferVStreams session; multiple infer() calls (serial tiles)."""
        outs_list: list[dict[str, np.ndarray]] = []
        with self._network_group.activate():
            with InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params,
            ) as pipeline:
                for fb in feeds:
                    feed = {self._input_name: fb}
                    out = pipeline.infer(feed)
                    if not isinstance(out, dict):
                        raise RuntimeError(f"Unexpected infer() return type: {type(out)}")
                    outs_list.append({k: np.asarray(v) for k, v in out.items()})
        return outs_list

    def _parse_e2e_rows(
        self,
        d2d: np.ndarray,
        img0_hw: tuple[int, int],
        ratio_pad: Any,
    ) -> list[Detection]:
        """d2d shape (N, 6+)."""
        if d2d.size == 0:
            return []
        d = d2d.astype(np.float32, copy=False)
        d = d[d[:, 4] >= self._conf_threshold]
        if self._person_only:
            d = d[np.abs(d[:, 5]) < 0.5]
        if d.size == 0:
            return []
        boxes = d[:, :4].astype(np.float32)
        boxes = _scale_boxes_back(boxes, (img0_hw[0], img0_hw[1]), ratio_pad)
        out: list[Detection] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            conf = float(d[i, 4])
            if self._person_only:
                cid = 0
                name = self._class_names[0]
            else:
                cid = int(round(float(d[i, 5])))
                name = (
                    self._class_names[cid]
                    if 0 <= cid < len(self._class_names)
                    else str(cid)
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

    def _parse_e2e(
        self,
        raw: np.ndarray,
        img0_hw: tuple[int, int],
        ratio_pad: Any,
        batch_index: int = 0,
    ) -> list[Detection]:
        if raw.ndim != 3 or raw.shape[2] < 6:
            raise ValueError(f"Unexpected e2e output shape: {raw.shape}")
        if raw.shape[0] == 1:
            d2d = raw[0]
        else:
            d2d = raw[batch_index]
        return self._parse_e2e_rows(d2d, img0_hw, ratio_pad)

    def _parse_multi_head(
        self,
        tensors_by_name: dict[str, np.ndarray],
        img0_hw: tuple[int, int],
        ratio_pad: Any,
        batch_index: int | None = None,
    ) -> list[Detection]:
        tensors: dict[str, np.ndarray] = {}
        for name, data in tensors_by_name.items():
            sh = tuple(data.shape)
            key = _head_key_from_shape(sh)
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
            cls_tensor = tensors[f"cls_{g}"]
            reg_tensor = tensors[f"reg_{g}"]
            if batch_index is not None and cls_tensor.ndim == 4 and cls_tensor.shape[0] > 1:
                cls_data = _hwc_feature_map_b(cls_tensor, batch_index)
                reg_data = _hwc_feature_map_b(reg_tensor, batch_index)
            else:
                cls_data = _hwc_feature_map(cls_tensor)
                reg_data = _hwc_feature_map(reg_tensor)
            cls_flat = cls_data.reshape(-1, 80)
            reg_flat = reg_data.reshape(-1, 4)
            if self._person_only:
                logit0 = cls_flat[:, 0]
                mask = logit0 > logit_thr
                if not mask.any():
                    continue
                indices = np.where(mask)[0]
                scores = _sigmoid(logit0[indices])
                cls = np.zeros(len(indices), dtype=np.int64)
            else:
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
                    (
                        float(x1[j]),
                        float(y1[j]),
                        float(x2[j]),
                        float(y2[j]),
                        float(scores[j]),
                        int(cls[j]),
                    )
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
            if self._person_only:
                cid = 0
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

    def _merge_tile_offset(
        self, dets: list[Detection], x0: int, y0: int
    ) -> list[Detection]:
        out: list[Detection] = []
        for d in dets:
            xa, ya, xb, yb = d.xyxy
            out.append(
                Detection(
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    xyxy=(xa + x0, ya + y0, xb + x0, yb + y0),
                )
            )
        return out

    def _nms_global(self, dets: list[Detection]) -> list[Detection]:
        if len(dets) <= 1:
            return dets
        xy = np.array([[d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3]] for d in dets], dtype=np.float32)
        sc = np.array([d.confidence for d in dets], dtype=np.float32)
        keep = nms_xyxy(xy, sc, self._nms_iou)
        out = [dets[int(i)] for i in keep]
        if len(out) > self._max_detections:
            out = sorted(out, key=lambda d: d.confidence, reverse=True)[: self._max_detections]
        return out

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run inference on an RGB frame and return detections in image coordinates.

        Args:
            rgb: uint8 array with shape ``(H, W, 3)`` in RGB order.

        Returns:
            Detections after tiling (if any), merge, and global NMS.

        Raises:
            ValueError: If ``rgb`` is not a 3-channel image.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")

        fh, fw = rgb.shape[0], rgb.shape[1]
        n_max = _max_tiles_for_frame(fh, fw, self._in_h, self._in_w)
        n_req = self._number_tiles_requested
        n_eff = min(n_req, n_max)
        if n_eff < n_req:
            logger.debug(
                "YOLO_NUMBER_TILES=%s exceeds max %s for frame %sx%s; using %s.",
                n_req,
                n_max,
                fw,
                fh,
                n_eff,
            )
        rows, cols = _best_grid_shape(n_eff, fw, fh)
        specs = _grid_specs(fh, fw, rows, cols)
        self._last_detect_tiling = {
            "frame_wh": [int(fw), int(fh)],
            "requested": int(n_req),
            "effective": int(n_eff),
            "max_for_frame": int(n_max),
            "rows": int(rows),
            "cols": int(cols),
            "model_input_hw": [int(self._in_h), int(self._in_w)],
        }
        if not specs:
            return []

        all_dets: list[Detection] = []

        bsz = max(1, self._in_batch)
        if bsz > 1:
            i = 0
            while i < len(specs):
                chunk = specs[i : i + bsz]
                if len(chunk) < bsz:
                    feeds_p: list[np.ndarray] = []
                    meta_p: list[tuple[_TileSpec, tuple[int, int], Any]] = []
                    for spec in chunk:
                        crop = rgb[spec.y0 : spec.y0 + spec.crop_h, spec.x0 : spec.x0 + spec.crop_w]
                        batch, img0_hw, ratio_pad = self._preprocess(crop)
                        feeds_p.append(batch)
                        meta_p.append((spec, img0_hw, ratio_pad))
                    for out, (spec, img0_hw, ratio_pad) in zip(
                        self._infer_many_serial(feeds_p), meta_p, strict=True
                    ):
                        if self._out_mode.kind == "e2e":
                            name = self._out_mode.e2e_name
                            if name is None or name not in out:
                                raw = next(iter(out.values())) if len(out) == 1 else None
                            else:
                                raw = out[name]
                            if raw is None:
                                raise ValueError(f"e2e mode but unexpected outputs: {list(out)}")
                            dets = self._parse_e2e(raw, img0_hw, ratio_pad)
                        elif self._out_mode.kind == "multi_head":
                            dets = self._parse_multi_head(out, img0_hw, ratio_pad)
                        else:
                            dets = self._parse_three_head_cls(out, img0_hw, ratio_pad)
                        dets = self._merge_tile_offset(dets, spec.x0, spec.y0)
                        all_dets.extend(dets)
                    i += len(chunk)
                    continue
                batches_in: list[np.ndarray] = []
                metas: list[tuple[_TileSpec, tuple[int, int], Any]] = []
                for spec in chunk:
                    crop = rgb[spec.y0 : spec.y0 + spec.crop_h, spec.x0 : spec.x0 + spec.crop_w]
                    batch, img0_hw, ratio_pad = self._preprocess(crop)
                    batches_in.append(batch)
                    metas.append((spec, img0_hw, ratio_pad))
                stacked = self._stack_batch(batches_in)
                outs = self._infer_raw(stacked)
                for bi, (spec, img0_hw, ratio_pad) in enumerate(metas):
                    if self._out_mode.kind == "e2e":
                        name = self._out_mode.e2e_name
                        if name is None or name not in outs:
                            if len(outs) == 1:
                                raw = next(iter(outs.values()))
                            else:
                                raise ValueError(f"e2e mode but unexpected outputs: {list(outs)}")
                        else:
                            raw = outs[name]
                        if raw.ndim != 3 or raw.shape[2] < 6:
                            raise ValueError(f"Unexpected e2e batched shape: {raw.shape}")
                        d2d = raw[bi] if raw.shape[0] > 1 else raw[0]
                        dets = self._parse_e2e_rows(d2d, img0_hw, ratio_pad)
                    elif self._out_mode.kind == "multi_head":
                        dets = self._parse_multi_head(
                            outs, img0_hw, ratio_pad, batch_index=bi
                        )
                    else:
                        dets = self._parse_three_head_cls(
                            outs, img0_hw, ratio_pad, batch_index=bi
                        )
                    dets = self._merge_tile_offset(dets, spec.x0, spec.y0)
                    all_dets.extend(dets)
                i += len(chunk)
        else:
            feeds: list[np.ndarray] = []
            meta_list: list[tuple[_TileSpec, tuple[int, int], Any]] = []
            for spec in specs:
                crop = rgb[spec.y0 : spec.y0 + spec.crop_h, spec.x0 : spec.x0 + spec.crop_w]
                batch, img0_hw, ratio_pad = self._preprocess(crop)
                feeds.append(batch)
                meta_list.append((spec, img0_hw, ratio_pad))
            all_outs = self._infer_many_serial(feeds)
            for out, (spec, img0_hw, ratio_pad) in zip(all_outs, meta_list, strict=True):
                if self._out_mode.kind == "e2e":
                    name = self._out_mode.e2e_name
                    if name is None or name not in out:
                        raw = next(iter(out.values())) if len(out) == 1 else None
                    else:
                        raw = out[name]
                    if raw is None:
                        raise ValueError(f"e2e mode but unexpected outputs: {list(out)}")
                    dets = self._parse_e2e(raw, img0_hw, ratio_pad)
                elif self._out_mode.kind == "multi_head":
                    dets = self._parse_multi_head(out, img0_hw, ratio_pad)
                else:
                    dets = self._parse_three_head_cls(out, img0_hw, ratio_pad)
                dets = self._merge_tile_offset(dets, spec.x0, spec.y0)
                all_dets.extend(dets)

        return self._nms_global(all_dets)

    def close(self) -> None:
        """Release the Hailo device handle."""
        try:
            self._target.release()
        except Exception:
            pass
