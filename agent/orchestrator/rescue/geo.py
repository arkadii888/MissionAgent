"""Body-frame person-offset estimation from camera geometry.

Upgrade path (TODO):
  When drone yaw/heading is available in telemetry, add `drone_heading_deg` here,
  rotate the (forward_m, right_m) body-frame vector by that heading, then add the
  result to drone lat/lon to produce a world-frame estimate.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersonOffset:
    """Drone body-frame offset estimate for a detected person.

    forward_m: metres ahead of the drone (positive = in front).
    right_m:   metres to the right of the drone (positive = right).
    """

    forward_m: float
    right_m: float


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
