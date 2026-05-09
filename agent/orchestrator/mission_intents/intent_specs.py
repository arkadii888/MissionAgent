"""Declare each mission intent once: LLM JSON schema branch + Python expansion handler.

Adding an entry to :data:`INTENT_SPECS` updates both ``MISSION_INTENT_SCHEMA`` and
:func:`expand.build_default_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.orchestrator.mission_intents.basic import (
    handle_land,
    handle_loiter,
    handle_move,
    handle_move_directional,
    handle_move_vertical,
    handle_return_to_home,
    handle_safety_control,
    handle_takeoff,
    handle_turn_relative,
    handle_yaw,
)
from agent.orchestrator.mission_intents.area_patterns import handle_comb_square_area
from agent.orchestrator.mission_intents.registry import IntentHandler, IntentRegistry

_ALT_M = {"minimum": 0, "maximum": 50}


@dataclass(frozen=True)
class IntentSpec:
    """Metadata for one ``type`` label: its ``oneOf`` JSON schema and expansion function.

    Attributes:
        type_name: String value of the intent ``type`` field.
        one_of_schema: JSON Schema object that must match when this intent is used.
        handler: Synchronous function ``(ExpansionContext, intent_dict) -> None``.
    """

    type_name: str
    one_of_schema: dict[str, Any]
    handler: IntentHandler


def _intent_item_schema_skeleton(type_enum: list[str], one_of: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the per-intent-object schema: discriminator enum plus mutually exclusive shapes."""
    return {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"type": "string", "enum": type_enum},
        },
        "oneOf": one_of,
    }


def build_root_mission_schema(*, max_intents: int = 32) -> dict[str, Any]:
    """Full JSON Schema document for the LLM (``mission_name`` + ``intents`` array).

    Args:
        max_intents: Maximum length of ``intents`` (``maxItems`` in schema).

    Returns:
        Draft-style dict suitable for llama-server structured output.
    """
    specs = INTENT_SPECS
    type_enum = [s.type_name for s in specs]
    items = _intent_item_schema_skeleton(type_enum, [s.one_of_schema for s in specs])
    return {
        "type": "object",
        "required": ["mission_name", "intents"],
        "properties": {
            "mission_name": {"type": "string", "maxLength": 64},
            "intents": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_intents,
                "items": items,
            },
        },
        "additionalProperties": False,
    }


def build_default_registry_from_specs(registry: IntentRegistry | None = None) -> IntentRegistry:
    """Populate a registry from :data:`INTENT_SPECS`.

    Args:
        registry: Existing registry to extend, or a new empty one when omitted.
    """
    reg = registry or IntentRegistry()
    for spec in INTENT_SPECS:
        reg.register(spec.type_name, spec.handler)
    return reg


# fmt: off
INTENT_SPECS: tuple[IntentSpec, ...] = (
    IntentSpec(
        type_name="takeoff",
        one_of_schema={
            "required": ["type", "altitude_m"],
            "properties": {
                "type": {"const": "takeoff"},
                "altitude_m": {"type": "number", **_ALT_M},
            },
            "additionalProperties": False,
        },
        handler=handle_takeoff,
    ),
    IntentSpec(
        type_name="move",
        one_of_schema={
            "required": ["type", "north_m", "east_m", "up_m"],
            "properties": {
                "type": {"const": "move"},
                "north_m": {"type": "number", "minimum": -1000, "maximum": 1000},
                "east_m": {"type": "number", "minimum": -1000, "maximum": 1000},
                "up_m": {"type": "number", "minimum": -50, "maximum": 50},
            },
            "additionalProperties": False,
        },
        handler=handle_move,
    ),
    IntentSpec(
        type_name="move_directional",
        one_of_schema={
            "required": ["type", "direction"],
            "properties": {
                "type": {"const": "move_directional"},
                "direction": {
                    "type": "string",
                    "enum": [
                        "north", "south", "east", "west",
                        "northeast", "northwest", "southeast", "southwest",
                    ],
                },
                "distance_m": {"type": "number", "minimum": 0.1, "maximum": 1000},
            },
            "additionalProperties": False,
        },
        handler=handle_move_directional,
    ),
    IntentSpec(
        type_name="move_vertical",
        one_of_schema={
            "required": ["type", "direction"],
            "properties": {
                "type": {"const": "move_vertical"},
                "direction": {"type": "string", "enum": ["down"]},
                "distance_m": {"type": "number", "minimum": 0.1, "maximum": 50},
            },
            "additionalProperties": False,
        },
        handler=handle_move_vertical,
    ),
    IntentSpec(
        type_name="turn_relative",
        one_of_schema={
            "required": ["type"],
            "properties": {
                "type": {"const": "turn_relative"},
                "maneuver": {"type": "string", "enum": ["turn_around"]},
                "degrees": {"type": "number", "minimum": 180, "maximum": 180},
            },
            "additionalProperties": False,
        },
        handler=handle_turn_relative,
    ),
    IntentSpec(
        type_name="safety_control",
        one_of_schema={
            "required": ["type", "action"],
            "properties": {
                "type": {"const": "safety_control"},
                "action": {
                    "type": "string",
                    "enum": ["stop", "hold", "abort", "return_home", "return"],
                },
            },
            "additionalProperties": False,
        },
        handler=handle_safety_control,
    ),
    IntentSpec(
        type_name="comb_square_area",
        one_of_schema={
            "required": ["type"],
            "properties": {
                "type": {"const": "comb_square_area"},
                "side_m": {"type": "number", "minimum": 1, "maximum": 1000},
                "lane_spacing_m": {"type": "number", "minimum": 0.5, "maximum": 100},
                "altitude_m": {"type": "number", **_ALT_M},
                "start_corner": {
                    "type": "string",
                    "enum": ["south_west", "south_east", "north_west", "north_east"],
                },
            },
            "additionalProperties": False,
        },
        handler=handle_comb_square_area,
    ),
    IntentSpec(
        type_name="loiter",
        one_of_schema={
            "required": ["type", "seconds"],
            "properties": {
                "type": {"const": "loiter"},
                "seconds": {"type": "number", "minimum": 0, "maximum": 300},
            },
            "additionalProperties": False,
        },
        handler=handle_loiter,
    ),
    IntentSpec(
        type_name="yaw",
        one_of_schema={
            "required": ["type", "degrees"],
            "properties": {
                "type": {"const": "yaw"},
                "degrees": {"type": "number", "minimum": -360, "maximum": 360},
            },
            "additionalProperties": False,
        },
        handler=handle_yaw,
    ),
    IntentSpec(
        type_name="return_to_home",
        one_of_schema={
            "required": ["type"],
            "properties": {"type": {"const": "return_to_home"}},
            "additionalProperties": False,
        },
        handler=handle_return_to_home,
    ),
    IntentSpec(
        type_name="land",
        one_of_schema={
            "required": ["type"],
            "properties": {"type": {"const": "land"}},
            "additionalProperties": False,
        },
        handler=handle_land,
    ),
)
# fmt: on


__all__ = [
    "INTENT_SPECS",
    "IntentSpec",
    "IntentHandler",
    "build_root_mission_schema",
    "build_default_registry_from_specs",
]
