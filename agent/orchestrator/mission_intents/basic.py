"""Core intent handlers: takeoff/move/safety/yaw/etc. Waypoints emit at cumulative N/E totals."""

from collections.abc import Mapping
from typing import Any

from .context import ExpansionContext
from .fields import optional_float, require_float
from .geometry import (
    bearing_to_yaw_deg,
    clamp_relative_altitude_m,
    compute_lat_long_from_offset,
    compute_north_east_offset_m,
    north_east_m_from_bearing_deg,
    normalize_yaw,
)
from .proto import build_proto_item

_DEFAULT_DIRECTIONAL_DISTANCE_M = 10.0
# Matches ``move_directional`` JSON schema ``direction`` enum.
_DIRECTION_UNIT: dict[str, tuple[float, float]] = {
    "north": (1.0, 0.0),
    "south": (-1.0, 0.0),
    "east": (0.0, 1.0),
    "west": (0.0, -1.0),
    "northeast": (1.0, 1.0),
    "northwest": (1.0, -1.0),
    "southeast": (-1.0, 1.0),
    "southwest": (-1.0, -1.0),
}
_SAFETY_ACTION_MAP: dict[str, str] = {
    "stop": "stop",
    "hold": "hold",
    "abort": "abort",
    "return": "return_home",
    "return_home": "return_home",
}


def append_waypoint(
    ctx: ExpansionContext,
    *,
    vehicle_action: int,
    is_fly_through: bool,
    loiter_time_s: float = 1.0,
    north_delta_m: float = 0.0,
    east_delta_m: float = 0.0,
) -> None:
    """Emit a ``MissionItem`` at current totals; yaw from pending intent or move direction.

    ``ctx.north_total_m`` / ``ctx.east_total_m`` must already include this leg—callers typically
    add deltas before invoking this helper.
    """
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
        speed_m_s=1.75,
        is_fly_through=is_fly_through,
        vehicle_action=vehicle_action,
        loiter_time_s=loiter_time_s,
        yaw_deg=yaw_deg,
    )
    ctx.items.append(item)
    ctx.pending_yaw_deg = None


def handle_takeoff(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Set cruise altitude and emit a takeoff ``MissionItem``."""
    ctx.current_altitude_m = clamp_relative_altitude_m(require_float(intent, "altitude_m"))
    append_waypoint(ctx, vehicle_action=1, is_fly_through=False)


def handle_move(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Apply a north/east/up offset in metres and emit a transit waypoint."""
    north_m = require_float(intent, "north_m")
    east_m = require_float(intent, "east_m")
    up_m = require_float(intent, "up_m")
    ctx.north_total_m += north_m
    ctx.east_total_m += east_m
    ctx.current_altitude_m = clamp_relative_altitude_m(ctx.current_altitude_m + up_m)
    append_waypoint(
        ctx,
        vehicle_action=0,
        is_fly_through=False,
        north_delta_m=north_m,
        east_delta_m=east_m,
    )


def handle_goto_lat_lon(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Fly to absolute WGS84 lat/lon in the same local frame as other intents (telemetry origin base)."""
    lat = require_float(intent, "latitude_deg")
    lon = require_float(intent, "longitude_deg")
    north_target, east_target = compute_north_east_offset_m(
        ctx.base_latitude_deg,
        ctx.base_longitude_deg,
        lat,
        lon,
    )
    north_delta_m = north_target - ctx.north_total_m
    east_delta_m = east_target - ctx.east_total_m
    ctx.north_total_m = north_target
    ctx.east_total_m = east_target
    if "altitude_m" in intent:
        ctx.current_altitude_m = clamp_relative_altitude_m(require_float(intent, "altitude_m"))
    append_waypoint(
        ctx,
        vehicle_action=0,
        is_fly_through=False,
        north_delta_m=north_delta_m,
        east_delta_m=east_delta_m,
    )


def handle_move_directional(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Move along a named compass direction (north, southeast, …)."""
    direction_raw = str(intent.get("direction", "")).strip().lower()
    if direction_raw not in _DIRECTION_UNIT:
        raise ValueError(f"unsupported world-frame direction: {direction_raw!r}")
    north_unit, east_unit = _DIRECTION_UNIT[direction_raw]
    distance_m = optional_float(intent, "distance_m", _DEFAULT_DIRECTIONAL_DISTANCE_M)
    if distance_m <= 0.0:
        raise ValueError("distance_m must be > 0")
    north_m = north_unit * distance_m
    east_m = east_unit * distance_m
    handle_move(ctx, {"north_m": north_m, "east_m": east_m, "up_m": 0.0})


def handle_move_bearing(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """World-frame move along a compass bearing: ``bearing_deg`` clockwise from north (0°=N, 90°=E)."""
    distance_m = require_float(intent, "distance_m")
    bearing_deg = require_float(intent, "bearing_deg")
    if distance_m <= 0.0:
        raise ValueError("distance_m must be > 0")
    north_m, east_m = north_east_m_from_bearing_deg(distance_m, bearing_deg)
    handle_move(ctx, {"north_m": north_m, "east_m": east_m, "up_m": 0.0})


def handle_move_vertical(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Change relative altitude (down only) without horizontal displacement."""
    direction_raw = str(intent.get("direction", "down")).strip().lower()
    if direction_raw != "down":
        raise ValueError("move_vertical only supports direction=down")
    distance_m = optional_float(intent, "distance_m", 5.0)
    if distance_m <= 0.0:
        raise ValueError("distance_m must be > 0")
    handle_move(ctx, {"north_m": 0.0, "east_m": 0.0, "up_m": -distance_m})


def handle_turn_relative(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Rotate in place (``turn_around`` → 180° yaw) and emit a waypoint."""
    maneuver = str(intent.get("maneuver", "turn_around")).strip().lower()
    if maneuver != "turn_around":
        raise ValueError("turn_relative only supports maneuver=turn_around")
    handle_yaw(ctx, {"degrees": normalize_yaw((ctx.pending_yaw_deg or 0.0) + 180.0)})
    append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=0.0)


def handle_safety_control(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Handle stop, hold, abort, or return-home safety actions."""
    raw_action = str(intent.get("action", "")).strip().lower()
    action = _SAFETY_ACTION_MAP.get(raw_action)
    if action is None:
        raise ValueError(f"unsupported safety action: {raw_action!r}")
    if action == "return_home":
        handle_return_to_home(ctx, intent)
    elif action == "hold":
        append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=5.0)
    elif action in {"stop", "abort"}:
        append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=0.0)
    ctx.preempted = True


def handle_loiter(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Set loiter time on the last waypoint, or create one if the plan is empty."""
    seconds = require_float(intent, "seconds")
    if not ctx.items:
        append_waypoint(ctx, vehicle_action=0, is_fly_through=False, loiter_time_s=seconds)
        return
    ctx.items[-1].loiter_time_s = seconds


def handle_yaw(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Set pending yaw for the next emitted waypoint (no item until a move)."""
    ctx.pending_yaw_deg = normalize_yaw(require_float(intent, "degrees"))


def handle_return_to_home(ctx: ExpansionContext, _intent: Mapping[str, Any]) -> None:
    """Fly back to the telemetry origin in one fly-through leg."""
    north_delta_m = -ctx.north_total_m
    east_delta_m = -ctx.east_total_m
    ctx.north_total_m = 0.0
    ctx.east_total_m = 0.0
    append_waypoint(
        ctx,
        vehicle_action=0,
        is_fly_through=True,
        north_delta_m=north_delta_m,
        east_delta_m=east_delta_m,
    )


def handle_land(ctx: ExpansionContext, _intent: Mapping[str, Any]) -> None:
    """Emit a land ``MissionItem`` at the current cumulative position."""
    append_waypoint(ctx, vehicle_action=2, is_fly_through=False)
