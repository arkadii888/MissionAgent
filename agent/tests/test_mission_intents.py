import math

import pytest

from agent.orchestrator.mission_intents.expand import build_default_registry, expand_intents_to_mission


def _telemetry() -> dict[str, float]:
    return {
        "latitude_deg": 47.3977419,
        "longitude_deg": 8.5455938,
        "relative_altitude_m": 0.0,
        "absolute_altitude_m": 488.0,
    }


def test_expand_intents_to_mission_basic_flow() -> None:
    plan = {
        "mission_name": "basic flow",
        "intents": [
            {"type": "takeoff", "altitude_m": 10},
            {"type": "move", "north_m": 0, "east_m": 0, "up_m": 10},
            {"type": "move", "north_m": 10, "east_m": 0, "up_m": 0},
            {"type": "return_to_home"},
            {"type": "land"},
        ],
    }

    out = expand_intents_to_mission(plan, _telemetry())
    assert len(out.items) == 5
    assert out.items[-1].vehicle_action == 2

    # After return_to_home, waypoint should be very close to origin.
    origin_lat = _telemetry()["latitude_deg"]
    origin_lon = _telemetry()["longitude_deg"]
    assert abs(out.items[-2].latitude_deg - origin_lat) < 1e-5
    assert abs(out.items[-2].longitude_deg - origin_lon) < 1e-5

    # North 10m latitude increment check.
    expected_lat_inc = math.degrees(10.0 / 6_378_137.0)
    actual_lat_inc = out.items[2].latitude_deg - origin_lat
    assert abs(actual_lat_inc - expected_lat_inc) < 1e-7


def test_registry_unknown_intent_fails() -> None:
    plan = {"mission_name": "bad", "intents": [{"type": "unknown_intent"}]}
    with pytest.raises(ValueError, match="unsupported intent type"):
        expand_intents_to_mission(plan, _telemetry(), registry=build_default_registry())


def test_phase1_world_frame_directional_and_vertical_and_turn() -> None:
    plan = {
        "mission_name": "phase1 movement",
        "intents": [
            {"type": "takeoff", "altitude_m": 15},
            {"type": "move_directional", "direction": "northeast", "distance_m": 20},
            {"type": "move_vertical", "direction": "down", "distance_m": 3},
            {"type": "turn_relative", "maneuver": "turn_around"},
            {"type": "land"},
        ],
    }
    out = expand_intents_to_mission(plan, _telemetry())
    assert len(out.items) == 5
    move_item = out.items[1]
    lat_inc = move_item.latitude_deg - _telemetry()["latitude_deg"]
    lon_inc = move_item.longitude_deg - _telemetry()["longitude_deg"]
    assert lat_inc > 0.0
    assert lon_inc > 0.0
    assert out.items[2].relative_altitude_m == pytest.approx(12.0)


def test_direction_matrix_all_compass_outputs() -> None:
    directions = [
        "north",
        "south",
        "east",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    ]
    for direction in directions:
        plan = {
            "mission_name": f"dir-{direction}",
            "intents": [
                {"type": "takeoff", "altitude_m": 10},
                {"type": "move_directional", "direction": direction, "distance_m": 10},
                {"type": "land"},
            ],
        }
        out = expand_intents_to_mission(plan, _telemetry())
        assert len(out.items) == 3


def test_drone_relative_phrases_rejected() -> None:
    plan = {
        "mission_name": "no relative",
        "intents": [
            {"type": "takeoff", "altitude_m": 10},
            {"type": "move_directional", "direction": "forward", "distance_m": 10},
            {"type": "land"},
        ],
    }
    with pytest.raises(ValueError, match="unsupported world-frame direction"):
        expand_intents_to_mission(plan, _telemetry())


def test_safety_preempts_following_movement() -> None:
    plan = {
        "mission_name": "safety preempts",
        "intents": [
            {"type": "takeoff", "altitude_m": 10},
            {"type": "safety_control", "action": "hold"},
            {"type": "move_directional", "direction": "north", "distance_m": 30},
            {"type": "land"},
        ],
    }
    out = expand_intents_to_mission(plan, _telemetry())
    assert len(out.items) == 3
    assert out.items[1].loiter_time_s == pytest.approx(5.0)
    assert out.items[2].vehicle_action == 2


def test_comb_square_area_implemented() -> None:
    plan = {
        "mission_name": "comb",
        "intents": [
            {"type": "takeoff", "altitude_m": 20},
            {
                "type": "comb_square_area",
                "side_m": 40,
                "lane_spacing_m": 5,
                "altitude_m": 15,
                "start_corner": "south_west",
            },
            {"type": "land"},
        ],
    }
    out = expand_intents_to_mission(plan, _telemetry())
    # 9 north/south passes + 8 east connectors + takeoff + land
    assert len(out.items) == 19
    assert out.items[-1].vehicle_action == 2
