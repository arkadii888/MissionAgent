"""Local flat-earth helpers: N/E offsets, yaw, altitude clamp for waypoint expansion."""

import math

MIN_RELATIVE_ALTITUDE_M = 0.0
MAX_RELATIVE_ALTITUDE_M = 50.0


def compute_lat_long_from_offset(
    base_latitude_deg: float,
    base_longitude_deg: float,
    north_offset_m: float,
    east_offset_m: float,
) -> tuple[float, float]:
    """Convert meters north/east into WGS84-like lat/lon using a spherical Earth model.

    Args:
        base_latitude_deg: Reference latitude (degrees).
        base_longitude_deg: Reference longitude (degrees).
        north_offset_m: Northward displacement in meters.
        east_offset_m: Eastward displacement in meters.

    Returns:
        ``(latitude_deg, longitude_deg)``.

    Raises:
        ValueError: Near poles where east scale collapses (``cos(lat) ≈ 0``).
    """
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


def normalize_yaw(yaw_deg: float) -> float:
    """Wrap yaw toward ``[-360, 360]`` without altering already in-range headings."""
    if yaw_deg > 360.0 or yaw_deg < -360.0:
        yaw_deg = math.fmod(yaw_deg, 360.0)
    return yaw_deg


def clamp_relative_altitude_m(relative_altitude_m: float) -> float:
    """Clamp relative altitude between ``MIN_RELATIVE_ALTITUDE_M`` and ``MAX_RELATIVE_ALTITUDE_M``."""
    return min(max(relative_altitude_m, MIN_RELATIVE_ALTITUDE_M), MAX_RELATIVE_ALTITUDE_M)


def bearing_to_yaw_deg(north_m: float, east_m: float) -> float:
    """Heading in degrees from a north/east displacement (0=north, 90=east).

    Uses ``atan2(east, north)`` and normalizes yaw.
    """
    if abs(north_m) < 1e-9 and abs(east_m) < 1e-9:
        return 0.0
    yaw = math.degrees(math.atan2(east_m, north_m))
    if yaw < 0.0:
        yaw += 360.0
    return normalize_yaw(yaw)
