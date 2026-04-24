import asyncio

import pytest

from agent.orchestrator.protoc import internal_communication_pb2
from agent.orchestrator.state import MissionPhase, MissionState, TelemetryCache


@pytest.mark.asyncio
async def test_telemetry_cache_roundtrip():
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
    assert snap.latitude_deg == pytest.approx(1.0)
    m = await cache.get_for_prompt()
    assert m is not None
    assert m["latitude_deg"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_mission_state_plan_and_status():
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
    st = await ms.prompt_mission_status()
    assert "UPLOADED" in st
    assert "1/1" in st

    await ms.mark_flying()
    await ms.advance_waypoint()
    st2 = await ms.prompt_mission_status()
    assert "FLYING" in st2

    copy = await ms.get_plan()
    assert copy is not None
    assert len(copy.items) == 1

    idx, n = await ms.get_waypoint_progress()
    assert n == 1
    assert idx == 0


@pytest.mark.asyncio
async def test_telemetry_cache_concurrent_writer_reader():
    cache = TelemetryCache()
    t = internal_communication_pb2.TelemetryResponse(
        latitude_deg=0.0,
        longitude_deg=0.0,
        relative_altitude_m=0.0,
        absolute_altitude_m=0.0,
    )

    async def writer() -> None:
        for _ in range(20):
            await cache.update_from_telemetry(t)
            await asyncio.sleep(0)

    async def reader() -> None:
        for _ in range(20):
            await cache.get_snapshot()
            await asyncio.sleep(0)

    await asyncio.gather(writer(), reader())
    s = await cache.get_snapshot()
    assert s is not None
    assert s.latitude_deg == pytest.approx(0.0)
