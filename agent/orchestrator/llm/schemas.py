MISSION_INTENT_SCHEMA_NAME = "MissionIntentPlan"

# Mission Intent DSL schema.
# LLM emits mission intents; Python deterministically expands them to MissionItem protobuf.
MISSION_INTENT_SCHEMA: dict = {
    "type": "object",
    "required": ["mission_name", "intents"],
    "properties": {
        "mission_name": {
            "type": "string",
            "maxLength": 64,
        },
        "intents": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "takeoff",
                            "move",
                            "loiter",
                            "yaw",
                            "return_to_home",
                            "land",
                        ],
                    },
                },
                "oneOf": [
                    {
                        "required": ["type", "altitude_m"],
                        "properties": {
                            "type": {"const": "takeoff"},
                            "altitude_m": {"type": "number", "minimum": 0, "maximum": 100},
                        },
                        "additionalProperties": False,
                    },
                    {
                        "required": ["type", "north_m", "east_m", "up_m"],
                        "properties": {
                            "type": {"const": "move"},
                            "north_m": {"type": "number", "minimum": -1000, "maximum": 1000},
                            "east_m": {"type": "number", "minimum": -1000, "maximum": 1000},
                            "up_m": {"type": "number", "minimum": -100, "maximum": 100},
                        },
                        "additionalProperties": False,
                    },
                    {
                        "required": ["type", "seconds"],
                        "properties": {
                            "type": {"const": "loiter"},
                            "seconds": {"type": "number", "minimum": 0, "maximum": 300},
                        },
                        "additionalProperties": False,
                    },
                    {
                        "required": ["type", "degrees"],
                        "properties": {
                            "type": {"const": "yaw"},
                            "degrees": {"type": "number", "minimum": -360, "maximum": 360},
                        },
                        "additionalProperties": False,
                    },
                    {
                        "required": ["type"],
                        "properties": {
                            "type": {"const": "return_to_home"},
                        },
                        "additionalProperties": False,
                    },
                    {
                        "required": ["type"],
                        "properties": {
                            "type": {"const": "land"},
                        },
                        "additionalProperties": False,
                    },
                ],
            },
        },
    },
    "additionalProperties": False,
}

# Backwards-compatible aliases for any external imports still using older names.
MISSION_PLAN_SCHEMA_NAME = MISSION_INTENT_SCHEMA_NAME
MISSION_PLAN_SCHEMA = MISSION_INTENT_SCHEMA
