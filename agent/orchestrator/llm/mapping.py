import math
from collections.abc import Mapping
from typing import Any

from agent.orchestrator.protoc import internal_communication_pb2


def _nan() -> float:
    return float("nan")


def _as_float(item: Mapping[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    return float(value)


def _as_int(item: Mapping[str, Any], key: str, default: int) -> int:
    value = item.get(key, default)
    return int(value)


def mission_plan_to_proto(mission_plan: Mapping[str, Any]) -> internal_communication_pb2.MissionItemList:
    items = mission_plan.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("mission_plan.items must be a non-empty list")

    result = internal_communication_pb2.MissionItemList()
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("each mission item must be an object")

        proto_item = internal_communication_pb2.MissionItem()
        proto_item.latitude_deg = _as_float(raw_item, "latitude_deg", 0.0)
        proto_item.longitude_deg = _as_float(raw_item, "longitude_deg", 0.0)
        proto_item.relative_altitude_m = _as_float(raw_item, "relative_altitude_m", 10.0)
        proto_item.speed_m_s = _as_float(raw_item, "speed_m_s", 2.0)
        proto_item.is_fly_through = bool(raw_item.get("is_fly_through", False))
        proto_item.gimbal_pitch_deg = _nan()
        proto_item.gimbal_yaw_deg = _nan()
        proto_item.camera_action = _as_int(raw_item, "camera_action", 0)
        proto_item.loiter_time_s = _as_float(raw_item, "loiter_time_s", _nan())
        proto_item.camera_photo_interval_s = _as_float(raw_item, "camera_photo_interval_s", 0.1)
        proto_item.acceptance_radius_m = _as_float(raw_item, "acceptance_radius_m", 0.5)
        proto_item.yaw_deg = _as_float(raw_item, "yaw_deg", 0.0)
        proto_item.camera_photo_distance_m = _as_float(raw_item, "camera_photo_distance_m", _nan())
        proto_item.vehicle_action = _as_int(raw_item, "vehicle_action", 0)

        result.items.append(proto_item)

    # Guard against accidental NaN-elimination refactors.
    if not math.isnan(result.items[0].gimbal_pitch_deg):
        raise ValueError("expected gimbal_pitch_deg default to NaN")
    return result
