"""Tests for estimate_person_offset (pinhole camera geometry)."""

import math

import pytest

from agent.orchestrator.rescue.geo import estimate_person_offset


class TestEstimatePersonOffset:
    IMG = (1280, 960)
    ALT = 30.0
    HFOV = 66.0
    VFOV = 41.0
    PITCH = 90.0  # nadir

    def _center_bbox(self) -> tuple[float, float, float, float]:
        w, h = self.IMG
        return (w * 0.4, h * 0.4, w * 0.6, h * 0.6)

    def test_center_bbox_nadir_near_zero(self) -> None:
        off = estimate_person_offset(
            self._center_bbox(), self.IMG, self.ALT, self.PITCH, self.HFOV, self.VFOV
        )
        assert abs(off.forward_m) < 1.0
        assert abs(off.right_m) < 1.0

    def test_right_bbox_gives_positive_right_m(self) -> None:
        w, h = self.IMG
        bbox = (w * 0.7, h * 0.4, w * 0.9, h * 0.6)
        off = estimate_person_offset(bbox, self.IMG, self.ALT, self.PITCH, self.HFOV, self.VFOV)
        assert off.right_m > 0.0

    def test_left_bbox_gives_negative_right_m(self) -> None:
        w, h = self.IMG
        bbox = (w * 0.1, h * 0.4, w * 0.3, h * 0.6)
        off = estimate_person_offset(bbox, self.IMG, self.ALT, self.PITCH, self.HFOV, self.VFOV)
        assert off.right_m < 0.0

    def test_top_bbox_gives_positive_forward_m(self) -> None:
        """In a nadir camera, image-top (small y) corresponds to the forward direction."""
        w, h = self.IMG
        bbox = (w * 0.4, h * 0.1, w * 0.6, h * 0.3)
        off = estimate_person_offset(bbox, self.IMG, self.ALT, self.PITCH, self.HFOV, self.VFOV)
        assert off.forward_m > 0.0

    def test_bottom_bbox_gives_negative_forward_m(self) -> None:
        """In a nadir camera, image-bottom (large y) corresponds to behind the drone."""
        w, h = self.IMG
        bbox = (w * 0.4, h * 0.7, w * 0.6, h * 0.9)
        off = estimate_person_offset(bbox, self.IMG, self.ALT, self.PITCH, self.HFOV, self.VFOV)
        assert off.forward_m < 0.0

    def test_altitude_scales_offset(self) -> None:
        w, h = self.IMG
        bbox = (w * 0.7, h * 0.4, w * 0.9, h * 0.6)
        off_low = estimate_person_offset(bbox, self.IMG, 10.0, self.PITCH, self.HFOV, self.VFOV)
        off_high = estimate_person_offset(bbox, self.IMG, 40.0, self.PITCH, self.HFOV, self.VFOV)
        # Higher altitude → larger slant range → larger lateral offset.
        assert off_high.right_m > off_low.right_m

    def test_zero_size_image_returns_zero(self) -> None:
        off = estimate_person_offset((0.0, 0.0, 10.0, 10.0), (0, 0), 30.0, 90.0, 66.0, 41.0)
        assert off.forward_m == 0.0
        assert off.right_m == 0.0

    def test_tilted_camera_shifts_forward(self) -> None:
        """A forward-tilted camera (pitch < 90) should project the centre pixel further ahead."""
        w, h = self.IMG
        center = self._center_bbox()
        off_nadir = estimate_person_offset(center, self.IMG, self.ALT, 90.0, self.HFOV, self.VFOV)
        off_tilted = estimate_person_offset(center, self.IMG, self.ALT, 60.0, self.HFOV, self.VFOV)
        # Nadir centre ≈ 0 forward; tilt 60° points 30° below horizontal → positive forward distance.
        assert off_tilted.forward_m > off_nadir.forward_m
        assert off_tilted.forward_m > 0.0
