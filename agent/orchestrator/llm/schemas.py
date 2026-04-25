MISSION_PLAN_SCHEMA_NAME = "MissionPlan"

# Subset of MissionItem fields expected from the LLM.
# Any omitted fields are populated with safe defaults in mapping.py.
MISSION_PLAN_SCHEMA: dict = {
    "type": "object",
    "required": ["mission_name", "items"],
    "properties": {
        "mission_name": {
            "type": "string",
            "maxLength": 64,
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "type": "object",
                "required": [
                    "latitude_deg",
                    "longitude_deg",
                    "relative_altitude_m",
                    "speed_m_s",
                    "is_fly_through",
                    "loiter_time_s",
                    "yaw_deg",
                    "camera_action",
                    "vehicle_action",
                ],
                "properties": {
                    "latitude_deg": {"type": "number"},
                    "longitude_deg": {"type": "number"},
                    "relative_altitude_m": {"type": "number", "minimum": 0, "maximum": 100},
                    "speed_m_s": {"type": "number", "const": 1.0},
                    "is_fly_through": {"type": "boolean"},
                    "loiter_time_s": {"type": "number", "minimum": 0},
                    "yaw_deg": {"type": "number"},
                    "camera_action": {"type": "integer", "const": 0},
                    "vehicle_action": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
