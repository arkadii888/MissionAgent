"""Tests for rescue person geo: body rotation and lat/lon projection."""

import math

import pytest

from agent.orchestrator.rescue.geo import (
    body_forward_right_to_north_east_m,
    estimate_person_lat_lon,
    estimate_person_offset,
)


def test_body_forward_right_north_facing() -> None:
    n, e = body_forward_right_to_north_east_m(10.0, 3.0, yaw_deg=0.0)
    assert n == pytest.approx(10.0)
    assert e == pytest.approx(3.0)


def test_body_forward_right_east_facing() -> None:
    # Facing east: 10 m forward → +E; 3 m right → −N (south).
    n, e = body_forward_right_to_north_east_m(10.0, 3.0, yaw_deg=90.0)
    assert n == pytest.approx(-3.0)
    assert e == pytest.approx(10.0)


def test_estimate_person_lat_lon_yaw_zero_forward_only() -> None:
    # Nadir camera, bbox dead centre → ray straight down → forward=0, right=0 at any AGL with 90° pitch.
    bbox = (400.0, 300.0, 400.0, 300.0)
    wh = (800, 600)
    lat0, lon0 = 47.3977419, 8.5455938
    off, geo = estimate_person_lat_lon(
        bbox,
        wh,
        drone_latitude_deg=lat0,
        drone_longitude_deg=lon0,
        drone_altitude_m=20.0,
        yaw_deg=0.0,
        camera_pitch_deg=90.0,
        hfov_deg=66.0,
        vfov_deg=41.0,
    )
    assert off.forward_m == pytest.approx(0.0, abs=1e-6)
    assert off.right_m == pytest.approx(0.0, abs=1e-6)
    assert geo.latitude_deg == pytest.approx(lat0, abs=1e-9)
    assert geo.longitude_deg == pytest.approx(lon0, abs=1e-9)


def test_estimate_person_lat_lon_rotates_with_yaw() -> None:
    """Pure right offset in body frame becomes +east when yaw is 0."""
    bbox = (600.0, 300.0, 600.0, 300.0)  # centre-right of 800×600
    wh = (800, 600)
    lat0, lon0 = 47.3977419, 8.5455938
    _, geo0 = estimate_person_lat_lon(
        bbox,
        wh,
        drone_latitude_deg=lat0,
        drone_longitude_deg=lon0,
        drone_altitude_m=15.0,
        yaw_deg=0.0,
        camera_pitch_deg=90.0,
        hfov_deg=66.0,
        vfov_deg=41.0,
    )
    _, geo90 = estimate_person_lat_lon(
        bbox,
        wh,
        drone_latitude_deg=lat0,
        drone_longitude_deg=lon0,
        drone_altitude_m=15.0,
        yaw_deg=90.0,
        camera_pitch_deg=90.0,
        hfov_deg=66.0,
        vfov_deg=41.0,
    )
    # Same pixel offset → different N/E; at least one component should differ materially.
    d0 = math.hypot(geo0.latitude_deg - lat0, geo0.longitude_deg - lon0)
    d90 = math.hypot(geo90.latitude_deg - lat0, geo90.longitude_deg - lon0)
    assert d0 > 1e-8 and d90 > 1e-8
    assert not (geo0.latitude_deg == pytest.approx(geo90.latitude_deg) and geo0.longitude_deg == pytest.approx(geo90.longitude_deg))


def test_estimate_person_offset_matches_lat_lon_pipeline_centre() -> None:
    bbox = (400.0, 300.0, 450.0, 350.0)
    wh = (800, 600)
    off_only = estimate_person_offset(
        bbox,
        wh,
        drone_altitude_m=12.0,
        camera_pitch_deg=90.0,
        hfov_deg=66.0,
        vfov_deg=41.0,
    )
    off_geo, _ = estimate_person_lat_lon(
        bbox,
        wh,
        drone_latitude_deg=10.0,
        drone_longitude_deg=20.0,
        drone_altitude_m=12.0,
        yaw_deg=33.0,
        camera_pitch_deg=90.0,
        hfov_deg=66.0,
        vfov_deg=41.0,
    )
    assert off_geo.forward_m == pytest.approx(off_only.forward_m)
    assert off_geo.right_m == pytest.approx(off_only.right_m)
