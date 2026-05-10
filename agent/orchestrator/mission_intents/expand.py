"""Expand a JSON intent list into protobuf ``MissionItem`` waypoints."""

from collections.abc import Mapping
from typing import Any, Callable

from agent.orchestrator.protoc import internal_communication_pb2

from .context import ExpansionContext
from .fields import optional_float
from .intent_specs import build_default_registry_from_specs
from .proto import validate_proto_list
from .registry import IntentRegistry

HandlerLogFn = Callable[[str, Mapping[str, Any]], None] | None


def build_default_registry() -> IntentRegistry:
    """Registry with all built-in intent types from :data:`intent_specs.INTENT_SPECS`."""
    return build_default_registry_from_specs()


def expand_intents_to_mission(
    mission_plan: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    registry: IntentRegistry | None = None,
    on_handler_called: HandlerLogFn = None,
) -> internal_communication_pb2.MissionItemList:
    """Walk ``mission_plan["intents"]`` and append waypoints derived from telemetry origin.

    Args:
        mission_plan: Parsed JSON with ``intents`` (non-empty list of objects).
        telemetry: Degrees + ``relative_altitude_m`` used as the planar origin.
        registry: Override handler map (tests); default uses built-ins.
        on_handler_called: Optional hook ``(intent_type, intent_dict)`` for logging each step.

    Returns:
        Filled ``MissionItemList`` passing geometry and upload contract checks.

    Raises:
        ValueError: On invalid telemetry, unknown intent types, or contract violations.

    Notes:
        After ``safety_control`` preempts, later movement intents are skipped except
        ``land`` and repeat ``safety_control``.
    """
    intents = mission_plan.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("mission_plan.intents must be a non-empty list")
    current_lat = optional_float(telemetry, "latitude_deg", 0.0)
    current_lon = optional_float(telemetry, "longitude_deg", 0.0)
    current_alt = optional_float(telemetry, "relative_altitude_m", 0.0)
    if not (-90.0 <= current_lat <= 90.0):
        raise ValueError("telemetry latitude_deg must be in [-90, 90]")
    if not (-180.0 <= current_lon <= 180.0):
        raise ValueError("telemetry longitude_deg must be in [-180, 180]")

    chosen_registry = registry or build_default_registry()
    ctx = ExpansionContext(
        base_latitude_deg=current_lat,
        base_longitude_deg=current_lon,
        current_altitude_m=current_alt,
    )

    for raw_intent in intents:
        if not isinstance(raw_intent, Mapping):
            raise ValueError("each mission intent must be an object")
        intent = dict(raw_intent)
        intent_type = str(intent.get("type", "")).strip()
        if not intent_type:
            raise ValueError("intent.type must be a non-empty string")
        if ctx.preempted and intent_type not in {"land", "safety_control"}:
            continue
        if on_handler_called is not None:
            on_handler_called(intent_type, intent)
        handler = chosen_registry.resolve(intent_type)
        handler(ctx, intent)

    result = internal_communication_pb2.MissionItemList()
    result.items.extend(ctx.items)
    _validate_contract(result)
    return result


def _validate_contract(result: internal_communication_pb2.MissionItemList) -> None:
    """Raise if items fail geometry validation or executor upload rules (speed, camera)."""
    validate_proto_list(result)
    for item in result.items:
        if item.speed_m_s > 4.0:
            raise ValueError("contract violation: speed_m_s must be smaller 4.0")
        if item.camera_action != 0:
            raise ValueError("contract violation: camera_action must be 0")
