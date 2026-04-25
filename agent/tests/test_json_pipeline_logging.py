import json
from pathlib import Path

from agent.orchestrator.logging import JsonPipelineLogger
from agent.orchestrator.mission_intents.expand import expand_intents_to_mission
from agent.orchestrator.mission_intents.proto import mission_list_to_ordered_dict


def test_json_pipeline_logging_records_all_events(tmp_path: Path) -> None:
    log_path = tmp_path / "mission_pipeline.jsonl"
    logger = JsonPipelineLogger(path=log_path, enabled=True)
    trace_id = logger.new_trace_id()

    telemetry = {
        "latitude_deg": 47.3977419,
        "longitude_deg": 8.5455938,
        "relative_altitude_m": 0.0,
        "absolute_altitude_m": 488.0,
    }
    plan = {
        "mission_name": "logging flow",
        "intents": [
            {"type": "takeoff", "altitude_m": 10},
            {"type": "move", "north_m": 10, "east_m": 0, "up_m": 0},
            {"type": "return_to_home"},
            {"type": "land"},
        ],
    }

    logger.log("prompt_received", trace_id, {"prompt_text": "go north and return", "telemetry": telemetry})
    logger.log("intents_generated", trace_id, {"mission_plan": plan})

    called_handlers: list[dict] = []

    def _on_handler(intent_type: str, intent_payload: dict) -> None:
        called_handlers.append({"intent_type": intent_type, "intent": intent_payload})
        logger.log(
            "intent_handler_called",
            trace_id,
            {"intent_type": intent_type, "intent": intent_payload},
        )

    mission = expand_intents_to_mission(plan, telemetry, on_handler_called=_on_handler)
    logger.log(
        "mission_converted",
        trace_id,
        {"mission_proto": mission_list_to_ordered_dict(mission)},
    )
    logger.log("mission_uploaded", trace_id, {"item_count": len(mission.items)})

    raw_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw_lines) >= 5

    records = [json.loads(line) for line in raw_lines]
    assert all(r["trace_id"] == trace_id for r in records)
    events = [r["event"] for r in records]
    for expected in (
        "prompt_received",
        "intents_generated",
        "intent_handler_called",
        "mission_converted",
        "mission_uploaded",
    ):
        assert expected in events

    converted_payloads = [r["payload"] for r in records if r["event"] == "mission_converted"]
    assert converted_payloads
    assert "mission_proto" in converted_payloads[-1]
