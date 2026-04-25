from collections.abc import Mapping


def build_system_prompt(max_waypoints: int = 16) -> str:
    return (
        "You are a drone mission planner. "
        "Return only valid JSON that matches the provided schema. "
        f"Create at most {max_waypoints} mission items. "
        "Mission constraints: vehicle_action values are "
        "0=None, 1=Takeoff, 2=Land, 3=TransitionToFw, 4=TransitionToMc. "
        "camera_action values are "
        "0=None, 1=TakePhoto, 2=StartPhotoInterval, 3=StopPhotoInterval, "
        "4=StartVideo, 5=StopVideo, 6=StartPhotoDistance, 7=StopPhotoDistance. "
        "Mission generation policy: camera_action must always be 0; speed_m_s must always be 1.0; "
        "relative_altitude_m must be in [0.0, 100.0] where 0.0 is ground level; "
        "is_fly_through must be false; loiter_time_s must be 1.0; yaw_deg must be in [-360, 360]. "
        "You must plan the full mission sequence end-to-end in your output, including takeoff/landing "
        "when required, by using vehicle_action values directly in mission items. "
        "Waypoint geometry rules: parse the user request into ordered movement steps; "
        "for horizontal movement, compute new coordinates from telemetry origin and cumulative offsets using "
        "compute_lat_long_from_offset(base_latitude_deg, base_longitude_deg, north_offset_m, east_offset_m). "
        "For vertical-only movement (up/down), keep latitude_deg and longitude_deg unchanged and change only relative_altitude_m. "
        "If user says return/fly back/come back, include a waypoint at the original telemetry latitude_deg and longitude_deg. "
        "Do not keep all waypoints at identical coordinates when horizontal movement is requested."
    )


def build_user_prompt(
    user_prompt: str,
    telemetry: Mapping[str, float],
    mission_status: str = "IDLE",
) -> str:
    return (
        f"User mission request: {user_prompt}\n"
        "Current telemetry:\n"
        f"- latitude_deg: {telemetry.get('latitude_deg')}\n"
        f"- longitude_deg: {telemetry.get('longitude_deg')}\n"
        f"- relative_altitude_m: {telemetry.get('relative_altitude_m')}\n"
        f"- absolute_altitude_m: {telemetry.get('absolute_altitude_m')}\n"
        "Planning checklist:\n"
        "1) Extract movement steps in order (north/south/east/west/up/down/return).\n"
        "2) Convert horizontal steps into cumulative north/east offsets.\n"
        "3) Compute each waypoint lat/lon from telemetry origin and offsets.\n"
        "4) Keep lat/lon unchanged for vertical-only steps.\n"
        "5) Output only JSON matching schema (no markdown, no comments).\n"
        f"Mission status: {mission_status}\n"
        "Generate a mission plan now."
    )
