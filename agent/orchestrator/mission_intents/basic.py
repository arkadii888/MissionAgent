from collections.abc import Mapping
from typing import Any

from .context import ExpansionContext
from .geometry import bearing_to_yaw_deg, clamp_relative_altitude_m, compute_lat_long_from_offset, normalize_yaw
from .proto import build_proto_item

_DEFAULT_DIRECTIONAL_DISTANCE_M = 10.0
_DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "n": (1.0, 0.0),
    "north": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "south": (-1.0, 0.0),
    "e": (0.0, 1.0),
    "east": (0.0, 1.0),
    "w": (0.0, -1.0),
    "west": (0.0, -1.0),
    "ne": (1.0, 1.0),
    "northeast": (1.0, 1.0),
    "north-east": (1.0, 1.0),
    "nw": (1.0, -1.0),
    "northwest": (1.0, -1.0),
    "north-west": (1.0, -1.0),
    "se": (-1.0, 1.0),
    "southeast": (-1.0, 1.0),
    "south-east": (-1.0, 1.0),
    "sw": (-1.0, -1.0),
    "southwest": (-1.0, -1.0),
    "south-west": (-1.0, -1.0),
}

_TURN_AROUND_SYNONYMS = {"turn_around", "around", "u_turn", "u-turn", "reverse_heading"}
_DESCEND_SYNONYMS = {"down", "descend", "sink", "lower"}
_SAFETY_ACTION_SYNONYMS: dict[str, str] = {
    "stop": "stop",
    "halt": "stop",
    "hold": "hold",
    "pause": "hold",
    "abort": "abort",
    "cancel": "abort",
    "return_home": "return_home",
    "return_to_home": "return_home",
    "rtl": "return_home",
}


def _as_float(intent: Mapping[str, Any], key: str) -> float:
    if key not in intent:
        raise ValueError(f"intent field {key!r} is required")
    value = float(intent[key])
    if value != value:  # NaN check
        raise ValueError(f"intent field {key!r} must be finite")
    return value


def _append_waypoint(
    ctx: ExpansionContext,
    *,
    vehicle_action: int,
    is_fly_through: bool,
    loiter_time_s: float = 1.0,
    north_delta_m: float = 0.0,
    east_delta_m: float = 0.0,
) -> None:
    yaw_deg = (
        ctx.pending_yaw_deg
        if ctx.pending_yaw_deg is not None
        else bearing_to_yaw_deg(north_delta_m, east_delta_m)
    )
    lat, lon = compute_lat_long_from_offset(
        ctx.base_latitude_deg,
        ctx.base_longitude_deg,
        ctx.north_total_m,
        ctx.east_total_m,
    )
    item = build_proto_item(
        latitude_deg=lat,
        longitude_deg=lon,
        relative_altitude_m=ctx.current_altitude_m,
        speed_m_s=1.0,
        is_fly_through=is_fly_through,
        vehicle_action=vehicle_action,
        loiter_time_s=loiter_time_s,
        yaw_deg=yaw_deg,
    )
    ctx.items.append(item)
    ctx.pending_yaw_deg = None


def handle_takeoff(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    ctx.current_altitude_m = clamp_relative_altitude_m(_as_float(intent, "altitude_m"))
    _append_waypoint(ctx, vehicle_action=1, is_fly_through=False)


def handle_move(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    north_m = _as_float(intent, "north_m")
    east_m = _as_float(intent, "east_m")
    up_m = _as_float(intent, "up_m")
    ctx.north_total_m += north_m
    ctx.east_total_m += east_m
    ctx.current_altitude_m = clamp_relative_altitude_m(ctx.current_altitude_m + up_m)
    _append_waypoint(
        ctx,
        vehicle_action=0,
        is_fly_through=True,
        north_delta_m=north_m,
        east_delta_m=east_m,
    )


def handle_move_directional(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    direction_raw = str(intent.get("direction", "")).strip().lower()
    if direction_raw not in _DIRECTION_VECTORS:
        raise ValueError(f"unsupported world-frame direction: {direction_raw!r}")
    north_unit, east_unit = _DIRECTION_VECTORS[direction_raw]
    distance_m = float(intent.get("distance_m", _DEFAULT_DIRECTIONAL_DISTANCE_M))
    if distance_m != distance_m or distance_m <= 0.0:
        raise ValueError("distance_m must be > 0")
    north_m = north_unit * distance_m
    east_m = east_unit * distance_m
    handle_move(ctx, {"north_m": north_m, "east_m": east_m, "up_m": 0.0})


def handle_move_vertical(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    direction_raw = str(intent.get("direction", "down")).strip().lower()
    if direction_raw not in _DESCEND_SYNONYMS:
        raise ValueError("move_vertical only supports descending direction in phase 1")
    distance_m = float(intent.get("distance_m", 5.0))
    if distance_m != distance_m or distance_m <= 0.0:
        raise ValueError("distance_m must be > 0")
    handle_move(ctx, {"north_m": 0.0, "east_m": 0.0, "up_m": -distance_m})


def handle_turn_relative(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    maneuver = str(intent.get("maneuver", "turn_around")).strip().lower()
    degrees = float(intent.get("degrees", 180.0))
    if maneuver not in _TURN_AROUND_SYNONYMS and abs(degrees - 180.0) > 1e-9:
        raise ValueError("turn_relative only supports turn-around (180 degrees) in phase 1")
    handle_yaw(ctx, {"degrees": normalize_yaw((ctx.pending_yaw_deg or 0.0) + 180.0)})
    _append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=0.0)


def handle_safety_control(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    raw_action = str(intent.get("action", "")).strip().lower()
    action = _SAFETY_ACTION_SYNONYMS.get(raw_action)
    if action is None:
        raise ValueError(f"unsupported safety action: {raw_action!r}")
    if action == "return_home":
        handle_return_to_home(ctx, intent)
    elif action == "hold":
        _append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=5.0)
    elif action in {"stop", "abort"}:
        _append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=0.0)
    ctx.preempted = True


def handle_loiter(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    seconds = _as_float(intent, "seconds")
    if not ctx.items:
        _append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=seconds)
        return
    ctx.items[-1].loiter_time_s = seconds


def handle_yaw(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    ctx.pending_yaw_deg = normalize_yaw(_as_float(intent, "degrees"))


def handle_return_to_home(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    del intent
    north_delta_m = -ctx.north_total_m
    east_delta_m = -ctx.east_total_m
    ctx.north_total_m = 0.0
    ctx.east_total_m = 0.0
    _append_waypoint(
        ctx,
        vehicle_action=0,
        is_fly_through=True,
        north_delta_m=north_delta_m,
        east_delta_m=east_delta_m,
    )


def handle_land(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    del intent
    _append_waypoint(ctx, vehicle_action=2, is_fly_through=False)
