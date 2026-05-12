"""Tests for RescueMissionDispatcher._build_rescue_plan round-trip through expand_intents_to_mission."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.orchestrator.mission_intents.expand import expand_intents_to_mission
from agent.orchestrator.rescue.dispatcher import RescueMissionDispatcher


def _telemetry() -> dict[str, float]:
    return {
        "latitude_deg": 47.3977419,
        "longitude_deg": 8.5455938,
        "relative_altitude_m": 15.0,
        "absolute_altitude_m": 503.0,
    }


def _make_dispatcher(**kwargs: Any) -> RescueMissionDispatcher:
    defaults = dict(
        loop=MagicMock(),
        client=AsyncMock(),
        llm=AsyncMock(),
        cache=AsyncMock(),
        home_state=MagicMock(),
        json_logger=MagicMock(),
        min_rth_alt_m=10.0,
        camera_mount_pitch_deg=90.0,
        camera_hfov_deg=66.0,
        camera_vfov_deg=41.0,
    )
    defaults.update(kwargs)
    return RescueMissionDispatcher(**defaults)


class TestBuildRescuePlan:
    def test_plan_has_correct_structure(self) -> None:
        d = _make_dispatcher()
        plan = d._build_return_home_plan(home_lat=47.3977, home_lon=8.5455, rth_alt=15.0)
        assert plan["mission_name"] == "rescue_rth"
        assert len(plan["intents"]) == 2
        assert plan["intents"][0]["type"] == "goto_lat_lon"
        assert plan["intents"][1]["type"] == "land"

    def test_goto_lat_lon_carries_home_coords(self) -> None:
        d = _make_dispatcher()
        plan = d._build_return_home_plan(home_lat=47.1234, home_lon=8.9876, rth_alt=20.0)
        intent = plan["intents"][0]
        assert intent["latitude_deg"] == pytest.approx(47.1234)
        assert intent["longitude_deg"] == pytest.approx(8.9876)

    def test_rth_alt_passed_through(self) -> None:
        d = _make_dispatcher(min_rth_alt_m=10.0)
        plan = d._build_return_home_plan(home_lat=47.0, home_lon=8.0, rth_alt=25.0)
        assert plan["intents"][0]["altitude_m"] == pytest.approx(25.0)

    def test_plan_expands_to_two_mission_items(self) -> None:
        d = _make_dispatcher()
        tel = _telemetry()
        plan = d._build_return_home_plan(
            home_lat=tel["latitude_deg"],
            home_lon=tel["longitude_deg"],
            rth_alt=tel["relative_altitude_m"],
        )
        proto = expand_intents_to_mission(plan, tel)
        assert len(proto.items) == 2

    def test_last_item_is_land(self) -> None:
        d = _make_dispatcher()
        tel = _telemetry()
        plan = d._build_return_home_plan(
            home_lat=tel["latitude_deg"],
            home_lon=tel["longitude_deg"],
            rth_alt=tel["relative_altitude_m"],
        )
        proto = expand_intents_to_mission(plan, tel)
        assert proto.items[-1].vehicle_action == 2  # land

    def test_home_at_different_location_than_current_telemetry(self) -> None:
        """goto_lat_lon must fly to stored home, not current position."""
        d = _make_dispatcher()
        tel = _telemetry()
        home_lat = tel["latitude_deg"] + 0.01  # ~1 km north
        home_lon = tel["longitude_deg"] + 0.01
        plan = d._build_return_home_plan(home_lat=home_lat, home_lon=home_lon, rth_alt=15.0)
        proto = expand_intents_to_mission(plan, tel)
        goto_item = proto.items[0]
        assert goto_item.latitude_deg == pytest.approx(home_lat, abs=1e-6)
        assert goto_item.longitude_deg == pytest.approx(home_lon, abs=1e-6)
