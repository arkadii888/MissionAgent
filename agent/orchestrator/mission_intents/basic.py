from collections.abc import Mapping
from typing import Any

from .context import ExpansionContext
from .geometry import bearing_to_yaw_deg, clamp_relative_altitude_m, compute_lat_long_from_offset, normalize_yaw
from .proto import build_proto_item


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
