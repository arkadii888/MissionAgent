import math
from collections.abc import Mapping
from typing import Any

from agent.orchestrator.protoc import internal_communication_pb2


def _nan() -> float:
    return float("nan")


def _as_float(item: Mapping[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def _as_optional_float(item: Mapping[str, Any], key: str, default: float | None = None) -> float:
    if key not in item or item.get(key) is None:
        return _nan() if default is None else default
    out = float(item[key])
    if not math.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def _as_int(item: Mapping[str, Any], key: str, default: int) -> int:
    value = item.get(key, default)
    return int(value)


def _validate_mission_item(raw_item: Mapping[str, Any]) -> None:
    lat = _as_float(raw_item, "latitude_deg", 0.0)
    lon = _as_float(raw_item, "longitude_deg", 0.0)
    rel_alt = _as_float(raw_item, "relative_altitude_m", 10.0)
    speed = _as_float(raw_item, "speed_m_s", 2.0)
    loiter = _as_float(raw_item, "loiter_time_s", 0.0) if "loiter_time_s" in raw_item else None
    yaw = _as_float(raw_item, "yaw_deg", 0.0) if "yaw_deg" in raw_item else None

    camera_action = _as_int(raw_item, "camera_action", 0)
    vehicle_action = _as_int(raw_item, "vehicle_action", 0)

    if not (-90.0 <= lat <= 90.0):
        raise ValueError("latitude_deg must be in [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("longitude_deg must be in [-180, 180]")
    if not (0.0 <= rel_alt <= 120.0):
        raise ValueError("relative_altitude_m must be in [0, 120]")
    if not (0.0 <= speed <= 15.0):
        raise ValueError("speed_m_s must be in [0, 15]")
    if loiter is not None and loiter < 0.0:
        raise ValueError("loiter_time_s must be >= 0")
    if yaw is not None and not (-360.0 <= yaw <= 360.0):
        raise ValueError("yaw_deg must be in [-360, 360]")
    if camera_action not in {0, 1, 2, 3, 4, 5, 6, 7}:
        raise ValueError("camera_action must be in {0,1,2,3,4,5,6,7}")
    if vehicle_action not in {0, 1, 2, 3, 4}:
        raise ValueError("vehicle_action must be in {0,1,2,3,4}")


def mission_plan_to_proto(mission_plan: Mapping[str, Any]) -> internal_communication_pb2.MissionItemList:
    items = mission_plan.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("mission_plan.items must be a non-empty list")

    result = internal_communication_pb2.MissionItemList()
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("each mission item must be an object")
        _validate_mission_item(raw_item)

        proto_item = internal_communication_pb2.MissionItem()
        proto_item.latitude_deg = _as_float(raw_item, "latitude_deg", 0.0)
        proto_item.longitude_deg = _as_float(raw_item, "longitude_deg", 0.0)
        proto_item.relative_altitude_m = _as_float(raw_item, "relative_altitude_m", 10.0)
        proto_item.speed_m_s = _as_float(raw_item, "speed_m_s", 2.0)
        proto_item.is_fly_through = bool(raw_item.get("is_fly_through", False))
        proto_item.gimbal_pitch_deg = _nan()
        proto_item.gimbal_yaw_deg = _nan()
        proto_item.camera_action = _as_int(raw_item, "camera_action", 0)
        proto_item.loiter_time_s = _as_optional_float(raw_item, "loiter_time_s")
        proto_item.camera_photo_interval_s = _as_float(raw_item, "camera_photo_interval_s", 0.1)
        proto_item.acceptance_radius_m = _as_float(raw_item, "acceptance_radius_m", 0.5)
        proto_item.yaw_deg = _as_float(raw_item, "yaw_deg", 0.0)
        proto_item.camera_photo_distance_m = _as_optional_float(raw_item, "camera_photo_distance_m")
        proto_item.vehicle_action = _as_int(raw_item, "vehicle_action", 0)

        result.items.append(proto_item)

    # Guard against accidental NaN-elimination refactors.
    if not math.isnan(result.items[0].gimbal_pitch_deg):
        raise ValueError("expected gimbal_pitch_deg default to NaN")
    return result
