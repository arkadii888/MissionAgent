"""Text templates for the mission-planning and rescue-analysis LLM prompts."""

from collections.abc import Mapping


def _telemetry_block(telemetry: Mapping[str, float]) -> str:
    return (
        "Current telemetry:\n"
        f"- latitude_deg: {telemetry.get('latitude_deg')}\n"
        f"- longitude_deg: {telemetry.get('longitude_deg')}\n"
        f"- relative_altitude_m: {telemetry.get('relative_altitude_m')}\n"
        f"- absolute_altitude_m: {telemetry.get('absolute_altitude_m')}\n"
        f"- yaw_deg: {telemetry.get('yaw_deg')}\n"
    )


def _simple_system_prompt(max_waypoints: int) -> str:
    return (
        "You are Gemma 4 E2B, a drone mission intent planner. "
        "Return only valid JSON that matches the provided schema. "
        f"Create at most {max_waypoints} intents. "
        "Use only schema-defined intent types. "
        "Use metric distances in meters and yaw in degrees. "
        "Prefer complete missions: include takeoff first and land last unless the user explicitly asks otherwise. "
        "Keep values realistic and concise."
    )


def _extended_system_prompt(max_waypoints: int) -> str:
    return (
        "You are Gemma 4 E2B, a drone mission intent planner. "
        "Return only valid JSON that matches the provided schema. "
        f"Create at most {max_waypoints} intents. "
        "Unless the user provides explicit geographic coordinates, plan legs with move/move_directional/move_bearing "
        "(offsets); do not invent lat/lon. When they supply WGS84 coordinates, use goto_lat_lon "
        "(latitude_deg, longitude_deg; optional altitude_m at that waypoint). "
        "Use only schema-defined intent types. "
        "Use metric distances in meters and yaw in degrees. "
        "Prefer complete missions: include takeoff first and land last unless the user explicitly asks otherwise. "
        "Shorthand verbs such as \"return\", \"RTL\", \"go home\", and \"come back\" mean navigate to launch/home "
        "before touchdown—emit safety_control with action return_home (see user-message examples); do not map that "
        "to land alone unless the user only asks to land."
        " Keep values realistic and concise."
    )


def build_system_prompt(max_waypoints: int = 32, *, extended: bool = False) -> str:
    """System message: JSON-only output, intent limit, units, and mission shape.

    Args:
        max_waypoints: Upper bound on ``intents`` length (should match schema ``maxItems``).
        extended: When true, include coordinate/RTL guidance in the system message.
    """
    if extended:
        return _extended_system_prompt(max_waypoints)
    return _simple_system_prompt(max_waypoints)


def _simple_user_body(mission_status: str) -> str:
    return (
        "Summary:\n"
        "Convert the request into ordered mission intents using only schema-defined types "
        "(takeoff, land, move_directional, move_bearing, goto_lat_lon, move_vertical, turn_relative, "
        "comb_square_area, safety_control). "
        "Use relative move intents unless the user supplies WGS84 coordinates (then goto_lat_lon). "
        "Treat return/RTL/go-home shorthand as safety_control return_home, not land alone. "
        "Output only valid JSON matching the schema (no markdown, no comments).\n"
        f"Mission status: {mission_status}\n"
        "Generate mission intents now."
    )


def _extended_user_body(mission_status: str) -> str:
    return (
        "Intent checklist:\n"
        "1) Convert the user request into an ordered list of mission intents.\n"
        "2) For world-frame compass legs: use move_directional for named directions (north, southeast, …); "
        "use move_bearing with distance_m and bearing_deg (clockwise from north, 0°=north) when the user "
        "gives a numeric heading (e.g. \"100 m at 30 degrees\", \"bear 045 for 200 meters\").\n"
        "2b) Use goto_lat_lon when the user names a latitude and longitude (degrees); optional altitude_m "
        "sets relative altitude at that waypoint.\n"
        "3) Use move_vertical for descend/down requests.\n"
        "4) Use turn_relative for turn around (180 degrees).\n"
        "5) Use safety_control for stop/hold/abort and for return-home: treat \"return\", \"RTL/RTB\", "
        "\"go home\", \"come back\" as return to launch—not as a synonym for land alone.\n"
        "6) For comb_square_area: omit side_m to use the default footprint, or set side_m and optionally "
        "lane_spacing_m, altitude_m, start_corner. Omit lane_spacing_m so spacing is inferred from flight; omit altitude_m inside comb_square_area unless you intentionally change altitude "
        "for the sweep—otherwise cumulative altitude after takeoff is used. Explicit lane_spacing_m should "
        "be a positive fraction of side_m never use 0 for lane spacing. "
        'If the user asks for an "WxH", "WxW", or "200x200" square in meters (same-edge search area), '
        "set side_m to that edge length—for a square never substitute takeoff or leg distance numbers "
        '(e.g. "20 m up" is altitude only, not the search square).\n'
        "7) Output only JSON matching schema (no markdown, no comments).\n"
        "Examples:\n"
        '- Input: "Take off to 20m, fly northeast 30m, then descend 5m and land."\n'
        '- Output: {"mission_name":"northeast descend","intents":[{"type":"takeoff","altitude_m":20},{"type":"move_directional","direction":"northeast","distance_m":30},{"type":"move_vertical","direction":"down","distance_m":5},{"type":"land"}]}\n'
        '- Input: "Take off, comb a square area, return home and land."\n'
        '- Output: {"mission_name":"square comb","intents":[{"type":"takeoff","altitude_m":15},{"type":"comb_square_area","side_m":40,"start_corner":"south_west"},{"type":"safety_control","action":"return_home"},{"type":"land"}]}\n'
        '- Input: "Take off to 20m, go 100m northwest, comb a 20m square, then return."\n'
        '- Output: {"mission_name":"northwest search return","intents":[{"type":"takeoff","altitude_m":20},{"type":"move_directional","direction":"northwest","distance_m":100},{"type":"comb_square_area","side_m":20},{"type":"safety_control","action":"return_home"},{"type":"land"}]}\n'
        '- Input: "Fly 20m up, go 100m northwest, comb a 200x200 m square, return."\n'
        '- Output: {"mission_name":"northwest wide search","intents":[{"type":"takeoff","altitude_m":20},{"type":"move_directional","direction":"northwest","distance_m":100},{"type":"comb_square_area","side_m":200},{"type":"safety_control","action":"return_home"},{"type":"land"}]}\n'
        '- Input: "Take off, turn around, hold position, then land."\n'
        '- Output: {"mission_name":"turn and hold","intents":[{"type":"takeoff","altitude_m":10},{"type":"turn_relative","maneuver":"turn_around"},{"type":"safety_control","action":"hold"},{"type":"land"}]}\n'
        '- Input: "Take off, comb a 5m square, then land." (no lane spacing given)\n'
        '- Output: {"mission_name":"small square","intents":[{"type":"takeoff","altitude_m":10},{"type":"comb_square_area","side_m":5,"start_corner":"south_west"},{"type":"land"}]}\n'
        '- Input: "Take off to 15m, fly to latitude 47.3980 longitude 8.5460, land."\n'
        '- Output: {"mission_name":"goto coords","intents":[{"type":"takeoff","altitude_m":15},'
        '{"type":"goto_lat_lon","latitude_deg":47.398,"longitude_deg":8.546},{"type":"land"}]}\n'
        '- Input: "Take off to 10m, go 100m at 30 degrees, land."\n'
        '- Output: {"mission_name":"bearing hop","intents":[{"type":"takeoff","altitude_m":10},'
        '{"type":"move_bearing","distance_m":100,"bearing_deg":30},{"type":"land"}]}\n'
        f"Mission status: {mission_status}\n"
        "Generate mission intents now."
    )


def build_user_prompt(
    user_prompt: str,
    telemetry: Mapping[str, float],
    mission_status: str = "IDLE",
    *,
    extended: bool = False,
) -> str:
    """User message: request, live telemetry, guidance, and current mission status line.

    Args:
        user_prompt: Natural-language mission request.
        telemetry: Keys ``latitude_deg``, ``longitude_deg``, ``relative_altitude_m``, ``absolute_altitude_m``,
            ``yaw_deg`` (heading in degrees when available).
        mission_status: Short status string from ``MissionState.prompt_mission_status()``.
        extended: When true, include the full intent checklist and few-shot examples.
    """
    body = _extended_user_body(mission_status) if extended else _simple_user_body(mission_status)
    return f"User mission request: {user_prompt}\n{_telemetry_block(telemetry)}{body}"


def build_rescue_analysis_system_prompt() -> str:
    """System message for the Gemma rescue situation-analysis call.

    Instructs the model to act as an aerial rescue assistant and focus on
    the three key outputs: posture, health concern level, and action plan.

    Returns:
        System prompt string ready to pass as ``system`` to ``LlamaClient.analyze_image``.
    """
    return (
        "You are a rescue assistant analysing an aerial image captured by a drone. "
        "Focus on: the person's posture (standing, sitting, lying down, or unknown), "
        "the surrounding terrain and any immediate hazards visible. "
        "Provide a health concern estimate (low / medium / high) with one or two "
        "sentences of reasoning, and a concise numbered action plan for rescuers."
    )


def build_rescue_analysis_user_prompt(
    *,
    latitude_deg: float,
    longitude_deg: float,
    forward_m: float,
    right_m: float,
    drone_alt_m: float,
    person_latitude_deg: float | None = None,
    person_longitude_deg: float | None = None,
) -> str:
    """User message for the Gemma rescue situation-analysis call.

    Embeds the drone's WGS-84 position from telemetry plus the body-frame offset
    estimate so the model has both geo context and approximate relative distance.
    When ``person_latitude_deg`` / ``person_longitude_deg`` are set, includes the
    flat-earth projection of that offset using heading and camera geometry.

    Args:
        latitude_deg: Drone latitude in decimal degrees (WGS-84) at trigger time.
        longitude_deg: Drone longitude in decimal degrees (WGS-84) at trigger time.
        forward_m: Estimated metres ahead of the drone (positive = in front).
        right_m: Estimated metres to the right of the drone (positive = right).
        drone_alt_m: Drone relative altitude above ground in metres at trigger time.
        person_latitude_deg: Optional estimated person latitude (degrees).
        person_longitude_deg: Optional estimated person longitude (degrees).

    Returns:
        User prompt string ready to pass as ``user_text`` to ``LlamaClient.analyze_image``.
    """
    geo_line = ""
    if person_latitude_deg is not None and person_longitude_deg is not None:
        geo_line = (
            f"The person's estimated ground position (WGS-84) is approximately "
            f"latitude {person_latitude_deg:.7f} deg, longitude {person_longitude_deg:.7f} deg. "
        )
    return (
        f"The drone's current position (WGS-84) is latitude {latitude_deg:.7f} deg, "
        f"longitude {longitude_deg:.7f} deg, at {drone_alt_m:.1f} m AGL. "
        + geo_line
        + "The detected person is estimated in the drone body frame as approximately "
        f"{forward_m:.1f} m ahead and {right_m:.1f} m to the right of the drone "
        "(positive right). Use the image together with these numbers. "
        "Analyse the image, estimate the person's condition, and propose an action plan."
    )
