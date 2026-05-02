"""Text templates for the mission-planning LLM (system + user messages)."""

from collections.abc import Mapping


def build_system_prompt(max_waypoints: int = 32) -> str:
    """System message: JSON-only output, intent limit, units, and mission shape.

    Args:
        max_waypoints: Upper bound on ``intents`` length (should match schema ``maxItems``).
    """
    return (
        "You are Gemma 4 E2B, a drone mission intent planner. "
        "Return only valid JSON that matches the provided schema. "
        f"Create at most {max_waypoints} intents. "
        "Do not compute latitude/longitude. "
        "Use only schema-defined intent types. "
        "Use metric distances in meters and yaw in degrees. "
        "Prefer complete missions: include takeoff first and land last unless the user explicitly asks otherwise. "
        "Keep values realistic and concise."
    )


def build_user_prompt(
    user_prompt: str,
    telemetry: Mapping[str, float],
    mission_status: str = "IDLE",
) -> str:
    """User message: request, live telemetry, checklist, examples, and current mission status line.

    Args:
        user_prompt: Natural-language mission request.
        telemetry: Keys ``latitude_deg``, ``longitude_deg``, ``relative_altitude_m``, ``absolute_altitude_m``.
        mission_status: Short status string from ``MissionState.prompt_mission_status()``.
    """
    return (
        f"User mission request: {user_prompt}\n"
        "Current telemetry:\n"
        f"- latitude_deg: {telemetry.get('latitude_deg')}\n"
        f"- longitude_deg: {telemetry.get('longitude_deg')}\n"
        f"- relative_altitude_m: {telemetry.get('relative_altitude_m')}\n"
        f"- absolute_altitude_m: {telemetry.get('absolute_altitude_m')}\n"
        "Intent checklist:\n"
        "1) Convert the user request into an ordered list of mission intents.\n"
        "2) Prefer move_directional for world-frame compass movement.\n"
        "3) Use move_vertical for descend/down requests.\n"
        "4) Use turn_relative for turn around (180 degrees).\n"
        "5) Use safety_control for stop/hold/abort/return-home requests.\n"
        "6) For comb_square_area: omit side_m to use the default footprint, or set side_m and optionally "
        "lane_spacing_m, altitude_m, start_corner. Omit lane_spacing_m for default spacing or use a positive "
        "fraction of side_m (e.g. side_m/4); never use 0 for lane spacing.\n"
        "7) Output only JSON matching schema (no markdown, no comments).\n"
        "Examples:\n"
        '- Input: "Take off to 20m, fly northeast 30m, then descend 5m and land."\n'
        '- Output: {"mission_name":"northeast descend","intents":[{"type":"takeoff","altitude_m":20},{"type":"move_directional","direction":"northeast","distance_m":30},{"type":"move_vertical","direction":"down","distance_m":5},{"type":"land"}]}\n'
        '- Input: "Take off, comb a square area, return home and land."\n'
        '- Output: {"mission_name":"square comb","intents":[{"type":"takeoff","altitude_m":15},{"type":"comb_square_area","side_m":40,"lane_spacing_m":5,"start_corner":"south_west"},{"type":"safety_control","action":"return_home"},{"type":"land"}]}\n'
        '- Input: "Take off, turn around, hold position, then land."\n'
        '- Output: {"mission_name":"turn and hold","intents":[{"type":"takeoff","altitude_m":10},{"type":"turn_relative","maneuver":"turn_around"},{"type":"safety_control","action":"hold"},{"type":"land"}]}\n'
        '- Input: "Take off, comb a 5m square, then land." (no lane spacing given)\n'
        '- Output: {"mission_name":"small square","intents":[{"type":"takeoff","altitude_m":10},{"type":"comb_square_area","side_m":5,"start_corner":"south_west"},{"type":"land"}]}\n'
        f"Mission status: {mission_status}\n"
        "Generate mission intents now."
    )
