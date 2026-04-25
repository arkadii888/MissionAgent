from collections.abc import Mapping
from typing import Any, Callable

from agent.orchestrator.protoc import internal_communication_pb2

from .basic import (
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
from .area_patterns import handle_comb_square_area
from .context import ExpansionContext
from .proto import validate_proto_list
from .registry import IntentRegistry

HandlerLogFn = Callable[[str, Mapping[str, Any]], None] | None


def build_default_registry() -> IntentRegistry:
    registry = IntentRegistry()
    registry.register("takeoff", handle_takeoff)
    registry.register("move", handle_move)
    registry.register("move_directional", handle_move_directional)
    registry.register("move_vertical", handle_move_vertical)
    registry.register("turn_relative", handle_turn_relative)
    registry.register("safety_control", handle_safety_control)
    registry.register("comb_square_area", handle_comb_square_area)
    registry.register("loiter", handle_loiter)
    registry.register("yaw", handle_yaw)
    registry.register("return_to_home", handle_return_to_home)
    registry.register("land", handle_land)
    return registry


def _as_float(item: Mapping[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    out = float(value)
    if out != out:
        raise ValueError(f"{key} must be finite")
    return out


def _validate_contract(result: internal_communication_pb2.MissionItemList) -> None:
    validate_proto_list(result)
    for item in result.items:
        if item.speed_m_s != 1.0:
            raise ValueError("contract violation: speed_m_s must be 1.0")
        if item.camera_action != 0:
            raise ValueError("contract violation: camera_action must be 0")


def expand_intents_to_mission(
    mission_plan: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    registry: IntentRegistry | None = None,
    on_handler_called: HandlerLogFn = None,
) -> internal_communication_pb2.MissionItemList:
    intents = mission_plan.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("mission_plan.intents must be a non-empty list")
    current_lat = _as_float(telemetry, "latitude_deg", 0.0)
    current_lon = _as_float(telemetry, "longitude_deg", 0.0)
    current_alt = _as_float(telemetry, "relative_altitude_m", 0.0)
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
