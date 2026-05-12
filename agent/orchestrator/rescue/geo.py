"""Body-frame person-offset estimation from camera geometry and lat/lon projection."""

import math
from dataclasses import dataclass

from agent.orchestrator.mission_intents.geometry import compute_lat_long_from_offset


@dataclass(frozen=True, slots=True)
class PersonOffset:
    """Drone body-frame offset estimate for a detected person.

    forward_m: metres ahead of the drone (positive = in front).
    right_m:   metres to the right of the drone (positive = right).
    """

    forward_m: float
    right_m: float


@dataclass(frozen=True, slots=True)
class PersonGeoEstimate:
    """WGS84 estimate for the detected person on flat ground.

    north_offset_m / east_offset_m are horizontal metres from the drone to the
    intersection point (same sign convention as :func:`compute_lat_long_from_offset`).
    """

    latitude_deg: float
    longitude_deg: float
    north_offset_m: float
    east_offset_m: float


def body_forward_right_to_north_east_m(
    forward_m: float,
    right_m: float,
    yaw_deg: float,
) -> tuple[float, float]:
    """Rotate body-frame offsets to North/East metres using vehicle yaw.

    Body axes: *forward* ahead of the drone, *right* to the drone's starboard.
    ``yaw_deg`` follows telemetry: 0° = north, 90° = east (clockwise from north).

    Returns:
        ``(north_m, east_m)`` horizontal offsets from drone to ground point.
    """
    r = math.radians(yaw_deg)
    c = math.cos(r)
    s = math.sin(r)
    north_m = forward_m * c - right_m * s
    east_m = forward_m * s + right_m * c
    return north_m, east_m


def estimate_person_lat_lon(
    bbox_xyxy: tuple[float, float, float, float],
    image_wh: tuple[int, int],
    *,
    drone_latitude_deg: float,
    drone_longitude_deg: float,
    drone_altitude_m: float,
    yaw_deg: float,
    camera_pitch_deg: float,
    hfov_deg: float,
    vfov_deg: float,
) -> tuple[PersonOffset, PersonGeoEstimate]:
    """Body-frame offset from the camera model, then world lat/lon using yaw and flat earth.

    Uses the same pinhole / flat-terrain assumptions as :func:`estimate_person_offset`.
    Longitude scale uses ``compute_lat_long_from_offset`` (spherical Earth at drone latitude).

    Args:
        bbox_xyxy: Person bounding box in full-frame pixels (x1, y1, x2, y2).
        image_wh: Frame width and height in pixels.
        drone_latitude_deg: Drone WGS84 latitude.
        drone_longitude_deg: Drone WGS84 longitude.
        drone_altitude_m: Relative altitude (AGL) in metres.
        yaw_deg: Vehicle heading, 0 = north, 90 = east.
        camera_pitch_deg: Camera depression from horizontal (90 = nadir).
        hfov_deg / vfov_deg: Camera horizontal and vertical field of view.

    Returns:
        ``(PersonOffset, PersonGeoEstimate)``. If latitude is unusable for the spherical
        model (near poles), ``PersonGeoEstimate`` falls back to the drone position with
        zero horizontal offsets.
    """
    offset = estimate_person_offset(
        bbox_xyxy=bbox_xyxy,
        image_wh=image_wh,
        drone_altitude_m=drone_altitude_m,
        camera_pitch_deg=camera_pitch_deg,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
    )
    north_m, east_m = body_forward_right_to_north_east_m(
        offset.forward_m, offset.right_m, yaw_deg
    )
    try:
        lat, lon = compute_lat_long_from_offset(
            drone_latitude_deg, drone_longitude_deg, north_m, east_m
        )
    except ValueError:
        lat, lon = drone_latitude_deg, drone_longitude_deg
        north_m, east_m = 0.0, 0.0
    geo = PersonGeoEstimate(
        latitude_deg=lat,
        longitude_deg=lon,
        north_offset_m=north_m,
        east_offset_m=east_m,
    )
    return offset, geo


def estimate_person_offset(
    bbox_xyxy: tuple[float, float, float, float],
    image_wh: tuple[int, int],
    drone_altitude_m: float,
    camera_pitch_deg: float,
    hfov_deg: float,
    vfov_deg: float,
) -> PersonOffset:
    """Approximate body-frame offset to a detected person using a pinhole camera model.

    Assumes flat terrain at `drone_altitude_m` relative altitude.
    Deliberately returns only body-frame offsets; no heading or lat/lon conversion.

    Args:
        bbox_xyxy: Bounding box in pixel coordinates (x1, y1, x2, y2).
        image_wh: Full image width and height in pixels.
        drone_altitude_m: Drone relative altitude in metres (AGL).
        camera_pitch_deg: Camera mount angle measured downward from horizontal
            (90° = nadir / straight down, 0° = forward-looking).
        hfov_deg: Camera horizontal field of view in degrees.
        vfov_deg: Camera vertical field of view in degrees.

    Returns:
        PersonOffset with forward_m and right_m.
    """
    x1, y1, x2, y2 = bbox_xyxy
    img_w, img_h = image_wh
    if img_w <= 0 or img_h <= 0:
        return PersonOffset(0.0, 0.0)

    # Bbox centre in pixels.
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    # Normalised image-plane coordinates in [-1, 1], origin at image centre.
    # nx: positive = right in image.
    # ny: positive = up in image (image y increases downward, so we negate).
    nx = (cx - img_w * 0.5) / (img_w * 0.5)
    ny = -((cy - img_h * 0.5) / (img_h * 0.5))

    hfov_rad = math.radians(hfov_deg)
    vfov_rad = math.radians(vfov_deg)
    pitch_rad = math.radians(camera_pitch_deg)

    # Angular offsets from camera optical axis.
    # theta_yaw: positive = right.
    # theta_pitch: subtracted because ny > 0 (upper image = forward) means less downward pitch.
    theta_yaw = nx * (hfov_rad * 0.5)
    theta_pitch = ny * (vfov_rad * 0.5)

    # Total downward pitch from horizontal.
    # Subtracting theta_pitch: upper-image pixels (ny > 0) are closer to horizontal → smaller
    # total pitch → ground intersection further ahead (larger forward_m).
    total_pitch = pitch_rad - theta_pitch

    # Project onto flat ground plane.
    if total_pitch <= 0.0:
        # Optical ray pointing at or above the horizon — cannot intersect ground.
        forward_m = 0.0
    else:
        forward_m = drone_altitude_m / math.tan(total_pitch)

    # Lateral offset scales with slant range to the ground point.
    slant_range_m = math.hypot(forward_m, drone_altitude_m)
    right_m = math.tan(theta_yaw) * slant_range_m

    return PersonOffset(forward_m=forward_m, right_m=right_m)
