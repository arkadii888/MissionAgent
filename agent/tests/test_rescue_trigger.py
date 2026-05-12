"""Tests for PersonRescueTrigger state machine: suppression, activation, one-shot."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agent.orchestrator.inference.yolo_common import Detection
from agent.orchestrator.rescue.trigger import PersonRescueTrigger


def _person_det(conf: float) -> Detection:
    return Detection(class_id=0, class_name="person", confidence=conf, xyxy=(10.0, 10.0, 50.0, 80.0))


def _no_dets() -> list[Detection]:
    return []


def _dets(conf: float) -> list[Detection]:
    return [_person_det(conf)]


def _armed_trigger(
    *,
    conf: float = 0.75,
    frames: int = 5,
    arm_delay_s: float = 0.0,
) -> PersonRescueTrigger:
    """Return a trigger that has already received notify_mission_sent() with no delay."""
    t = PersonRescueTrigger(
        min_confidence=conf,
        consecutive_frames=frames,
        arm_delay_s=arm_delay_s,
    )
    t.notify_mission_sent()
    return t


class TestSuppression:
    def test_does_not_fire_before_mission_sent(self) -> None:
        """Trigger stays SUPPRESSED if notify_mission_sent() was never called."""
        t = PersonRescueTrigger(min_confidence=0.5, consecutive_frames=1, arm_delay_s=0.0)
        for _ in range(10):
            assert t.tick(_dets(0.99)) is None

    def test_does_not_fire_within_arm_delay(self) -> None:
        """Trigger stays SUPPRESSED during the arm-delay window after mission sent."""
        t = PersonRescueTrigger(min_confidence=0.5, consecutive_frames=1, arm_delay_s=60.0)
        t.notify_mission_sent()
        # Simulate 30 s elapsed — still inside the 60 s delay.
        fake_now = time.monotonic() + 30.0
        with patch("agent.orchestrator.rescue.trigger.time.monotonic", return_value=fake_now):
            for _ in range(10):
                assert t.tick(_dets(0.99)) is None

    def test_activates_after_arm_delay(self) -> None:
        """Trigger becomes ACTIVE once arm_delay_s have elapsed since mission sent."""
        t = PersonRescueTrigger(min_confidence=0.5, consecutive_frames=1, arm_delay_s=60.0)
        t.notify_mission_sent()
        fake_now = time.monotonic() + 61.0
        with patch("agent.orchestrator.rescue.trigger.time.monotonic", return_value=fake_now):
            result = t.tick(_dets(0.99))
        assert result is not None

    def test_second_notify_call_ignored(self) -> None:
        """Only the first notify_mission_sent() call starts the countdown."""
        t = PersonRescueTrigger(min_confidence=0.5, consecutive_frames=1, arm_delay_s=0.0)
        t.notify_mission_sent()
        # First call with 0-delay arms immediately — should fire.
        result = t.tick(_dets(0.99))
        assert result is not None
        # Second notify must not re-arm the already-fired trigger.
        t.notify_mission_sent()
        assert t.tick(_dets(0.99)) is None


class TestDetectionLogic:
    def test_does_not_fire_below_threshold(self) -> None:
        t = _armed_trigger(conf=0.75, frames=1)
        assert t.tick(_dets(0.70)) is None

    def test_does_not_fire_before_streak_complete(self) -> None:
        t = _armed_trigger(conf=0.75, frames=5)
        for _ in range(4):
            assert t.tick(_dets(0.80)) is None

    def test_fires_after_n_consecutive_frames(self) -> None:
        t = _armed_trigger(conf=0.75, frames=5)
        result = None
        for _ in range(5):
            result = t.tick(_dets(0.80))
        assert result is not None
        assert result.best_detection.confidence == pytest.approx(0.80)

    def test_streak_resets_on_miss(self) -> None:
        t = _armed_trigger(conf=0.75, frames=5)
        for _ in range(4):
            t.tick(_dets(0.80))
        t.tick(_no_dets())  # break streak
        for _ in range(4):
            assert t.tick(_dets(0.80)) is None
        fire = t.tick(_dets(0.80))  # fresh streak of 5 completes
        assert fire is not None

    def test_non_person_class_ignored(self) -> None:
        t = _armed_trigger(conf=0.5, frames=1)
        non_person = Detection(class_id=2, class_name="car", confidence=0.99, xyxy=(0.0, 0.0, 100.0, 100.0))
        assert t.tick([non_person]) is None


class TestOneShot:
    def test_permanently_disabled_after_first_fire(self) -> None:
        """After firing once the trigger never fires again regardless of detections."""
        t = _armed_trigger(conf=0.5, frames=1)
        first = t.tick(_dets(0.99))
        assert first is not None
        # Any number of subsequent qualified frames must all return None.
        for _ in range(20):
            assert t.tick(_dets(0.99)) is None

    def test_disabled_state_survives_mission_resent(self) -> None:
        """notify_mission_sent() cannot re-enable a DISABLED trigger."""
        t = _armed_trigger(conf=0.5, frames=1)
        t.tick(_dets(0.99))  # fire → DISABLED
        t.notify_mission_sent()  # must have no effect
        assert t.tick(_dets(0.99)) is None
