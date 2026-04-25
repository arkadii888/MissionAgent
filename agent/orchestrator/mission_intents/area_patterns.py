from collections.abc import Mapping
from typing import Any

from .basic import _append_waypoint
from .context import ExpansionContext

_DEFAULT_SIDE_M = 40.0
_DEFAULT_LANE_SPACING_M = 5.0


def _as_float(intent: Mapping[str, Any], key: str, default: float) -> float:
    value = float(intent.get(key, default))
    if value != value:
        raise ValueError(f"{key} must be finite")
    return value


def _as_corner(intent: Mapping[str, Any]) -> str:
    raw = str(intent.get("start_corner", "south_west")).strip().lower()
    valid = {"south_west", "south_east", "north_west", "north_east"}
    if raw not in valid:
        raise ValueError(f"start_corner must be one of {sorted(valid)}")
    return raw


def handle_comb_square_area(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    side_m = _as_float(intent, "side_m", _DEFAULT_SIDE_M)
    lane_spacing_m = _as_float(intent, "lane_spacing_m", _DEFAULT_LANE_SPACING_M)
    if side_m <= 0.0:
        raise ValueError("side_m must be > 0")
    if lane_spacing_m <= 0.0:
        raise ValueError("lane_spacing_m must be > 0")

    if "altitude_m" in intent:
        altitude_m = _as_float(intent, "altitude_m", ctx.current_altitude_m)
        if altitude_m < 0.0:
            raise ValueError("altitude_m must be >= 0")
        ctx.current_altitude_m = altitude_m

    corner = _as_corner(intent)
    lanes = max(1, int(round(side_m / lane_spacing_m)))
    step_m = side_m / lanes

    north_sign = 1.0
    east_sign = 1.0
    if corner in {"north_west", "north_east"}:
        north_sign = -1.0
    if corner in {"south_east", "north_east"}:
        east_sign = -1.0

    for lane_idx in range(lanes + 1):
        north_delta = north_sign * (side_m if lane_idx % 2 == 0 else -side_m)
        ctx.north_total_m += north_delta
        _append_waypoint(
            ctx,
            vehicle_action=0,
            is_fly_through=True,
            north_delta_m=north_delta,
            east_delta_m=0.0,
        )
        if lane_idx == lanes:
            break
        east_delta = east_sign * step_m
        ctx.east_total_m += east_delta
        _append_waypoint(
            ctx,
            vehicle_action=0,
            is_fly_through=True,
            north_delta_m=0.0,
            east_delta_m=east_delta,
        )
