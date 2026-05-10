"""Local flat-earth helpers: N/E offsets, yaw, altitude clamp for waypoint expansion."""

import math

EARTH_RADIUS_M = 6_378_137.0
MIN_RELATIVE_ALTITUDE_M = 0.0
MAX_RELATIVE_ALTITUDE_M = 50.0


def compute_lat_long_from_offset(
    base_latitude_deg: float,
    base_longitude_deg: float,
    north_offset_m: float,
    east_offset_m: float,
) -> tuple[float, float]:
    """Meters N/E to WGS84 lat/lon (spherical model). Raises near poles where E scale vanishes."""
    d_lat = north_offset_m / EARTH_RADIUS_M
    lat_rad = math.radians(base_latitude_deg)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-12:
        raise ValueError("cannot compute longitude offset at poles")
    d_lon = east_offset_m / (EARTH_RADIUS_M * cos_lat)
    out_lat = base_latitude_deg + math.degrees(d_lat)
    out_lon = base_longitude_deg + math.degrees(d_lon)
    return out_lat, out_lon


def compute_north_east_offset_m(
    base_latitude_deg: float,
    base_longitude_deg: float,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[float, float]:
    """Inverse of :func:`compute_lat_long_from_offset`; longitude delta wrapped for shortest E/W leg."""
    lat_rad = math.radians(base_latitude_deg)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-12:
        raise ValueError("cannot compute longitude offset at poles")
    d_lon_deg = longitude_deg - base_longitude_deg
    if d_lon_deg > 180.0:
        d_lon_deg -= 360.0
    elif d_lon_deg < -180.0:
        d_lon_deg += 360.0
    north_m = math.radians(latitude_deg - base_latitude_deg) * EARTH_RADIUS_M
    east_m = math.radians(d_lon_deg) * EARTH_RADIUS_M * cos_lat
    return north_m, east_m


def normalize_yaw(yaw_deg: float) -> float:
    """Wrap yaw toward ``[-360, 360]`` without changing values already in range."""
    if yaw_deg > 360.0 or yaw_deg < -360.0:
        yaw_deg = math.fmod(yaw_deg, 360.0)
    return yaw_deg


def clamp_relative_altitude_m(relative_altitude_m: float) -> float:
    """Clamp relative altitude between ``MIN_RELATIVE_ALTITUDE_M`` and ``MAX_RELATIVE_ALTITUDE_M``."""
    return min(max(relative_altitude_m, MIN_RELATIVE_ALTITUDE_M), MAX_RELATIVE_ALTITUDE_M)


def north_east_m_from_bearing_deg(distance_m: float, bearing_deg: float) -> tuple[float, float]:
    """World-frame offset for a compass bearing clockwise from north (0°=N, 90°=E, 180°=S, 270°=W).

    Matches the heading convention of :func:`bearing_to_yaw_deg` for a non-zero leg.
    """
    r = math.radians(bearing_deg)
    return distance_m * math.cos(r), distance_m * math.sin(r)


def bearing_to_yaw_deg(north_m: float, east_m: float) -> float:
    """Heading in degrees from N/E displacement (0=north, 90=east); uses ``atan2(east, north)``."""
    if abs(north_m) < 1e-9 and abs(east_m) < 1e-9:
        return 0.0
    yaw = math.degrees(math.atan2(east_m, north_m))
    if yaw < 0.0:
        yaw += 360.0
    return normalize_yaw(yaw)
