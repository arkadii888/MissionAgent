"""Area-coverage intents; expand into many fly-through segments (currently square comb pattern)."""

import math
from collections.abc import Mapping
from typing import Any

from .basic import append_waypoint
from .context import ExpansionContext
from .fields import optional_float

_DEFAULT_SIDE_M = 40.0
_DEFAULT_LANE_SPACING_M = 5.0

# Arducam 64 MP (and similar): nominal cross-track FOV (degrees, full angle).
_ARDUCAM64_CROSS_TRACK_FOV_DEG = 84.0
# Gimbal / mount: optical axis 45° below horizontal ⇒ 45° tilt from nadir on flat ground.
_ARDUCAM64_TILT_FROM_NADIR_DEG = 45.0
# Small sidelap between adjacent north–south legs when lane spacing is inferred from altitude.
_DEFAULT_SWATH_OVERLAP_FRACTION = 0.15


def comb_lane_spacing_from_altitude_m(
    altitude_m: float,
    *,
    cross_track_fov_deg: float = _ARDUCAM64_CROSS_TRACK_FOV_DEG,
    tilt_from_nadir_deg: float = _ARDUCAM64_TILT_FROM_NADIR_DEG,
    overlap_fraction: float = _DEFAULT_SWATH_OVERLAP_FRACTION,
) -> float:
    """Suggest east–west step between comb legs from AGL and fixed camera geometry.

    Models a pinhole looking obliquely at flat terrain: ground swath perpendicular to the optical
    axis scales with ``2 * h * tan(FOV/2) / cos(tilt)``. ``tilt_from_nadir_deg`` is the angle
    between nadir and the optical axis (45° mount looking 45° below the horizon). Lane spacing
    is a fraction ``(1 - overlap_fraction)`` of that swath so neighbouring passes overlap slightly.

    Raises:
        ValueError: Non-finite altitude, non-positive altitude, or overlap not in (0, 1).
    """
    if altitude_m != altitude_m:
        raise ValueError("altitude_m must be finite")
    if altitude_m <= 0.0:
        raise ValueError("altitude_m must be > 0 for camera-based lane spacing")
    if not (0.0 < overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in (0, 1)")
    half_fov = math.radians(cross_track_fov_deg) / 2.0
    tilt = math.radians(tilt_from_nadir_deg)
    cos_tilt = math.cos(tilt)
    if cos_tilt < 1e-9:
        raise ValueError("tilt_from_nadir_deg too close to ±90°")
    swath_m = 2.0 * altitude_m * math.tan(half_fov) / cos_tilt
    return float(swath_m * (1.0 - overlap_fraction))


def _as_corner(intent: Mapping[str, Any]) -> str:
    """Return validated ``start_corner`` key (defaults to south-west)."""
    raw = str(intent.get("start_corner", "south_west")).strip().lower()
    valid = {"south_west", "south_east", "north_west", "north_east"}
    if raw not in valid:
        raise ValueError(f"start_corner must be one of {sorted(valid)}")
    return raw


def handle_comb_square_area(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    """Back-and-forth north legs with east steps; footprint from ``side_m`` and lane count.

    Optional ``start_corner`` flips traversal direction inside the nominal square anchored
    at the current cumulative horizontal position.

    If ``lane_spacing_m`` is omitted, spacing is derived from ``ctx.current_altitude_m`` after any
    intent ``altitude_m`` update (Arducam-style 84° cross-track FOV, 45° below horizontal, ~15% overlap).
    """
    side_m = optional_float(intent, "side_m", _DEFAULT_SIDE_M)
    if side_m <= 0.0:
        raise ValueError("side_m must be > 0")

    if "altitude_m" in intent:
        altitude_m = optional_float(intent, "altitude_m", ctx.current_altitude_m)
        if altitude_m < 0.0:
            raise ValueError("altitude_m must be >= 0")
        ctx.current_altitude_m = altitude_m

    if "lane_spacing_m" in intent:
        lane_spacing_m = optional_float(intent, "lane_spacing_m", _DEFAULT_LANE_SPACING_M)
    elif ctx.current_altitude_m > 0.0:
        lane_spacing_m = comb_lane_spacing_from_altitude_m(ctx.current_altitude_m)
    else:
        lane_spacing_m = _DEFAULT_LANE_SPACING_M
    if lane_spacing_m <= 0.0:
        raise ValueError("lane_spacing_m must be > 0")

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
        append_waypoint(
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
        append_waypoint(
            ctx,
            vehicle_action=0,
            is_fly_through=True,
            north_delta_m=0.0,
            east_delta_m=east_delta,
        )
