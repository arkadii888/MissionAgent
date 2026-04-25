import math
import re
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


def compute_lat_long_from_offset(
    base_latitude_deg: float,
    base_longitude_deg: float,
    north_offset_m: float,
    east_offset_m: float,
) -> tuple[float, float]:
    """Compute destination lat/lon from local N/E meter offsets."""
    earth_radius_m = 6_378_137.0
    d_lat = north_offset_m / earth_radius_m
    lat_rad = math.radians(base_latitude_deg)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-12:
        raise ValueError("cannot compute longitude offset at poles")
    d_lon = east_offset_m / (earth_radius_m * cos_lat)
    out_lat = base_latitude_deg + math.degrees(d_lat)
    out_lon = base_longitude_deg + math.degrees(d_lon)
    return out_lat, out_lon


def _normalize_yaw(yaw_deg: float) -> float:
    """Normalize yaw to [-360, 360] while keeping common values unchanged."""
    if yaw_deg > 360.0 or yaw_deg < -360.0:
        yaw_deg = math.fmod(yaw_deg, 360.0)
    return yaw_deg


def _bearing_to_yaw_deg(north_m: float, east_m: float) -> float:
    """Convert N/E movement vector to yaw degrees (0=north, 90=east, 180=south)."""
    if abs(north_m) < 1e-9 and abs(east_m) < 1e-9:
        return 0.0
    yaw = math.degrees(math.atan2(east_m, north_m))
    if yaw < 0.0:
        yaw += 360.0
    return _normalize_yaw(yaw)


def _extract_prompt_segments(user_prompt: str) -> tuple[list[tuple[float, float]], bool]:
    """
    Extract sequential movement segments from user prompt as N/E offsets in meters.
    Returns (segments, should_return_to_takeoff).
    """
    pattern = re.compile(
        r"\b(?:go|fly|move)\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?\s*(?:to\s+the\s+)?"
        r"(north|south|east|west)\b",
        re.IGNORECASE,
    )
    segments: list[tuple[float, float]] = []
    for m in pattern.finditer(user_prompt):
        dist_m = float(m.group(1))
        direction = m.group(2).lower()
        if direction == "north":
            segments.append((dist_m, 0.0))
        elif direction == "south":
            segments.append((-dist_m, 0.0))
        elif direction == "east":
            segments.append((0.0, dist_m))
        elif direction == "west":
            segments.append((0.0, -dist_m))

    lower = user_prompt.lower()
    should_return = any(
        phrase in lower
        for phrase in (
            "come back",
            "return to takeoff",
            "return to take off",
            "back to takeoff",
            "back to take off",
        )
    )
    return segments, should_return


def _build_items_from_prompt_geometry(
    *,
    user_prompt: str,
    base_latitude_deg: float,
    base_longitude_deg: float,
    cruise_altitude_m: float,
) -> list[dict[str, float | int | bool]]:
    """Build deterministic intermediate waypoints from natural-language movement segments."""
    segments, should_return = _extract_prompt_segments(user_prompt)
    if not segments:
        return []

    out: list[dict[str, float | int | bool]] = []
    north_total = 0.0
    east_total = 0.0
    prev_north = 0.0
    prev_east = 0.0

    for dn, de in segments:
        north_total += dn
        east_total += de
        lat, lon = compute_lat_long_from_offset(
            base_latitude_deg,
            base_longitude_deg,
            north_total,
            east_total,
        )
        yaw_deg = _bearing_to_yaw_deg(north_total - prev_north, east_total - prev_east)
        out.append(
            {
                "latitude_deg": lat,
                "longitude_deg": lon,
                "relative_altitude_m": cruise_altitude_m,
                "speed_m_s": 1.0,
                "is_fly_through": True,
                "loiter_time_s": 0.0,
                "yaw_deg": yaw_deg,
                "camera_action": 0,
                "vehicle_action": 0,
            }
        )
        prev_north = north_total
        prev_east = east_total

    if should_return and (abs(north_total) > 1e-9 or abs(east_total) > 1e-9):
        yaw_deg = _bearing_to_yaw_deg(-north_total, -east_total)
        out.append(
            {
                "latitude_deg": base_latitude_deg,
                "longitude_deg": base_longitude_deg,
                "relative_altitude_m": cruise_altitude_m,
                "speed_m_s": 1.0,
                "is_fly_through": True,
                "loiter_time_s": 0.0,
                "yaw_deg": yaw_deg,
                "camera_action": 0,
                "vehicle_action": 0,
            }
        )

    return out


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
    if speed != 1.0:
        raise ValueError("speed_m_s must be 1.0")
    if loiter is not None and loiter < 0.0:
        raise ValueError("loiter_time_s must be >= 0")
    if yaw is not None and not (-360.0 <= yaw <= 360.0):
        raise ValueError("yaw_deg must be in [-360, 360]")
    if camera_action != 0:
        raise ValueError("camera_action must be 0")
    if vehicle_action not in {0, 1, 2, 3, 4}:
        raise ValueError("vehicle_action must be in {0,1,2,3,4}")


def _build_proto_item(
    *,
    latitude_deg: float,
    longitude_deg: float,
    relative_altitude_m: float,
    speed_m_s: float,
    is_fly_through: bool,
    vehicle_action: int,
    loiter_time_s: float,
    yaw_deg: float,
) -> internal_communication_pb2.MissionItem:
    proto_item = internal_communication_pb2.MissionItem()
    proto_item.latitude_deg = latitude_deg
    proto_item.longitude_deg = longitude_deg
    proto_item.relative_altitude_m = relative_altitude_m
    proto_item.speed_m_s = speed_m_s
    proto_item.is_fly_through = is_fly_through
    proto_item.gimbal_pitch_deg = _nan()
    proto_item.gimbal_yaw_deg = _nan()
    proto_item.camera_action = 0
    proto_item.loiter_time_s = 1.0
    proto_item.camera_photo_interval_s = 0.1
    proto_item.acceptance_radius_m = 0.5
    proto_item.yaw_deg = yaw_deg
    proto_item.camera_photo_distance_m = _nan()
    proto_item.vehicle_action = vehicle_action
    return proto_item


def mission_plan_to_proto(
    mission_plan: Mapping[str, Any],
    telemetry: Mapping[str, float],
    user_prompt: str | None = None,
) -> internal_communication_pb2.MissionItemList:
    items = mission_plan.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("mission_plan.items must be a non-empty list")
    current_lat = _as_float(telemetry, "latitude_deg", 0.0)
    current_lon = _as_float(telemetry, "longitude_deg", 0.0)
    if not (-90.0 <= current_lat <= 90.0):
        raise ValueError("telemetry latitude_deg must be in [-90, 90]")
    if not (-180.0 <= current_lon <= 180.0):
        raise ValueError("telemetry longitude_deg must be in [-180, 180]")

    cruise_altitude_m = _as_float(items[0], "relative_altitude_m", 10.0)
    if user_prompt:
        prompt_items = _build_items_from_prompt_geometry(
            user_prompt=user_prompt,
            base_latitude_deg=current_lat,
            base_longitude_deg=current_lon,
            cruise_altitude_m=cruise_altitude_m,
        )
        if prompt_items:
            items = prompt_items

    result = internal_communication_pb2.MissionItemList()
    # Deterministic takeoff point: always first, independent from model output.
    result.items.append(
        _build_proto_item(
            latitude_deg=current_lat,
            longitude_deg=current_lon,
            relative_altitude_m=10.0,
            speed_m_s=1.0,
            is_fly_through=False,
            vehicle_action=1,
            loiter_time_s=_nan(),
            yaw_deg=0.0,
        )
    )

    last_lat = current_lat
    last_lon = current_lon
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("each mission item must be an object")
        _validate_mission_item(raw_item)

        lat = _as_float(raw_item, "latitude_deg", 0.0)
        lon = _as_float(raw_item, "longitude_deg", 0.0)
        last_lat = lat
        last_lon = lon
        # Intermediate waypoints come from model geometry, but camera/speed are fixed.
        proto_item = _build_proto_item(
            latitude_deg=lat,
            longitude_deg=lon,
            relative_altitude_m=_as_float(raw_item, "relative_altitude_m", 10.0),
            speed_m_s=1.0,
            is_fly_through=False,
            vehicle_action=0,
            loiter_time_s=1.0,
            yaw_deg=_normalize_yaw(_as_float(raw_item, "yaw_deg", 0.0)),
        )
        result.items.append(proto_item)

    # Deterministic final point: always land at last modeled coordinate.
    result.items.append(
        _build_proto_item(
            latitude_deg=last_lat,
            longitude_deg=last_lon,
            relative_altitude_m=0.0,
            speed_m_s=1.0,
            is_fly_through=False,
            vehicle_action=2,
            loiter_time_s=_nan(),
            yaw_deg=0.0,
        )
    )

    # Guard against accidental NaN-elimination refactors.
    if not math.isnan(result.items[0].gimbal_pitch_deg):
        raise ValueError("expected gimbal_pitch_deg default to NaN")
    return result
