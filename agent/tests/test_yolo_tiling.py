"""Tests for Hailo YOLO grid tiling helpers (no hailo_platform required)."""

from agent.orchestrator.inference.yolo_hailo import (
    _best_grid_shape,
    _grid_shapes_with_n_tiles,
    _grid_specs,
    _max_tiles_for_frame,
)


def test_max_tiles_16mp_style() -> None:
    # ~16MP landscape from .env example
    fh, fw = 3472, 4624
    mh, mw = 640, 640
    n = _max_tiles_for_frame(fh, fw, mh, mw)
    assert n == 8 * 6


def test_max_tiles_small_frame() -> None:
    assert _max_tiles_for_frame(100, 100, 640, 640) == 1


def test_grid_specs_partition_full_frame() -> None:
    fh, fw = 3472, 4624
    rows, cols = 2, 2
    specs = _grid_specs(fh, fw, rows, cols)
    assert len(specs) == 4
    assert specs[0].y0 == 0 and specs[0].x0 == 0
    # last tile reaches bottom-right
    last = specs[-1]
    assert last.y0 + last.crop_h == fh
    assert last.x0 + last.crop_w == fw
    # no gaps: row-major adjacent
    assert specs[1].x0 == specs[0].x0 + specs[0].crop_w


def test_best_grid_shape_four_tiles_landscape() -> None:
    rows, cols = _best_grid_shape(4, 4624, 3472)
    assert rows * cols == 4
    assert (rows, cols) == (2, 2)


def test_grid_shapes_unique() -> None:
    s = _grid_shapes_with_n_tiles(6)
    assert (1, 6) in s and (6, 1) in s and (2, 3) in s and (3, 2) in s


def test_best_grid_shape_six_tiles_prefers_landscape_columns() -> None:
    rows, cols = _best_grid_shape(6, 4624, 3472)
    assert rows * cols == 6
    # 2×3 gives wider tiles than 3×2 on a wide frame
    assert (rows, cols) == (2, 3)


def test_number_tiles_one() -> None:
    assert _best_grid_shape(1, 1000, 500) == (1, 1)
    specs = _grid_specs(500, 1000, 1, 1)
    assert len(specs) == 1
    assert specs[0].crop_w == 1000 and specs[0].crop_h == 500
