from collections.abc import Mapping


def build_system_prompt(max_waypoints: int = 16) -> str:
    return (
        "You are Gemma 4 E2B, a drone mission intent planner. "
        "Return only valid JSON that matches the provided schema. "
        f"Create at most {max_waypoints} intents. "
        "Do not compute latitude/longitude. "
        "Use only these intent types: takeoff, move, loiter, yaw, return_to_home, land. "
        "Use metric distances in meters and yaw in degrees. "
        "Prefer complete missions: include takeoff first and land last unless the user explicitly asks otherwise. "
        "Keep values realistic and concise."
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
        "Intent checklist:\n"
        "1) Convert the user request into an ordered list of mission intents.\n"
        "2) Use move intent with north_m/east_m/up_m for directional movement.\n"
        "3) Use return_to_home when user asks to come back/return.\n"
        "4) Output only JSON matching schema (no markdown, no comments).\n"
        "Examples:\n"
        '- Input: "Fly up 10m, then go north 20m and land."\n'
        '- Output: {"mission_name":"up and north","intents":[{"type":"takeoff","altitude_m":10},{"type":"move","north_m":0,"east_m":0,"up_m":10},{"type":"move","north_m":20,"east_m":0,"up_m":0},{"type":"land"}]}\n'
        '- Input: "Take off to 15m, fly east 30m, come back, then land."\n'
        '- Output: {"mission_name":"east and return","intents":[{"type":"takeoff","altitude_m":15},{"type":"move","north_m":0,"east_m":30,"up_m":0},{"type":"return_to_home"},{"type":"land"}]}\n'
        '- Input: "Take off to 10m, yaw 90 degrees, hover 5 seconds, land."\n'
        '- Output: {"mission_name":"yaw and loiter","intents":[{"type":"takeoff","altitude_m":10},{"type":"yaw","degrees":90},{"type":"loiter","seconds":5},{"type":"land"}]}\n'
        f"Mission status: {mission_status}\n"
        "Generate mission intents now."
    )
