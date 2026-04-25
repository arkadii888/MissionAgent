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
        "Use this function for every computed waypoint coordinate from route reasoning: "
        "compute_lat_long_from_offset(base_latitude_deg, base_longitude_deg, north_offset_m, east_offset_m). "
        "Mission generation policy: camera_action must always be 0; speed_m_s must always be 1.0 for every item; "
        "the orchestrator injects a deterministic first fly-up and final land item, "
        "so your items are only intermediate route endpoints."
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
        # TODO fill up the properties
        f"Mission status: {mission_status}\n"
        "Generate a mission plan now."
    )
