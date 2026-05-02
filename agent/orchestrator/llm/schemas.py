"""JSON Schema for structured mission intents (llama-server structured output)."""

from agent.orchestrator.mission_intents.intent_specs import build_root_mission_schema

MISSION_INTENT_SCHEMA_NAME = "MissionIntentPlan"

MISSION_INTENT_SCHEMA: dict = build_root_mission_schema()

MISSION_PLAN_SCHEMA_NAME = MISSION_INTENT_SCHEMA_NAME
MISSION_PLAN_SCHEMA = MISSION_INTENT_SCHEMA
