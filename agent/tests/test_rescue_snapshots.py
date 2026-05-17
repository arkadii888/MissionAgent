"""Tests for save_rescue_snapshots: writes two non-empty JPEG files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.orchestrator.inference.yolo_common import Detection
from agent.orchestrator.rescue.snapshots import save_rescue_snapshots


def _bgr_frame(w: int = 320, h: int = 240) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _person_det() -> Detection:
    return Detection(class_id=0, class_name="person", confidence=0.90, xyxy=(50.0, 40.0, 150.0, 180.0))


class TestSaveRescueSnapshots:
    def test_writes_two_jpeg_files(self, tmp_path: Path) -> None:
        full, crop = save_rescue_snapshots(_bgr_frame(), [_person_det()], out_dir=tmp_path)
        assert full.exists(), f"full frame not written: {full}"
        assert crop.exists(), f"person crop not written: {crop}"

    def test_files_are_non_empty(self, tmp_path: Path) -> None:
        full, crop = save_rescue_snapshots(_bgr_frame(), [_person_det()], out_dir=tmp_path)
        assert full.stat().st_size > 0
        assert crop.stat().st_size > 0

    def test_filenames_have_jpeg_extension(self, tmp_path: Path) -> None:
        full, crop = save_rescue_snapshots(_bgr_frame(), [_person_det()], out_dir=tmp_path)
        assert full.suffix == ".jpg"
        assert crop.suffix == ".jpg"

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        save_rescue_snapshots(_bgr_frame(), [_person_det()], out_dir=nested)
        assert nested.is_dir()

    def test_full_only_skips_person_jpeg(self, tmp_path: Path) -> None:
        full, crop = save_rescue_snapshots(
            _bgr_frame(), [_person_det()], out_dir=tmp_path, save_person_crop=False
        )
        assert full.exists() and full.stat().st_size > 0
        assert crop is None
        assert list(tmp_path.glob("*_person.jpg")) == []
        assert len(list(tmp_path.glob("*.jpg"))) == 1

    def test_works_with_no_detections(self, tmp_path: Path) -> None:
        full, crop = save_rescue_snapshots(_bgr_frame(), [], out_dir=tmp_path)
        assert full.exists()
        assert crop is not None and crop.exists()
