import pytest

from agent.orchestrator.protoc import internal_communication_pb2
from agent.orchestrator.state import MissionPhase, MissionState, TelemetryCache


@pytest.mark.asyncio
async def test_telemetry_cache_and_mission_state() -> None:
    cache = TelemetryCache()
    t = internal_communication_pb2.TelemetryResponse(
        latitude_deg=1.0,
        longitude_deg=2.0,
        relative_altitude_m=3.0,
        absolute_altitude_m=4.0,
    )
    await cache.update_from_telemetry(t)
    snap = await cache.get_snapshot()
    assert snap is not None
    m = await cache.get_for_prompt()
    assert m is not None and m["latitude_deg"] == pytest.approx(1.0)

    ms = MissionState()
    assert (await ms.get_phase()) is MissionPhase.IDLE

    plan = internal_communication_pb2.MissionItemList()
    it = plan.items.add()
    it.latitude_deg = 0.0
    it.longitude_deg = 0.0
    it.relative_altitude_m = 10.0
    it.vehicle_action = 0

    await ms.set_mission("test", plan)
    assert (await ms.get_phase()) is MissionPhase.UPLOADED
    assert "UPLOADED" in (await ms.prompt_mission_status())
    await ms.mark_flying()
    assert "FLYING" in (await ms.prompt_mission_status())

    copy = await ms.get_plan()
    assert copy is not None and len(copy.items) == 1
    idx, n = await ms.get_waypoint_progress()
    assert (idx, n) == (0, 1)
