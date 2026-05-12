"""Person-rescue trigger: fires at most once when the drone is airborne and clear of the operator.

State machine (one-way, irreversible transitions):

    SUPPRESSED  →  ACTIVE  →  DISABLED

SUPPRESSED (initial state)
    The trigger does nothing regardless of detections.  It leaves this state only
    after ``notify_mission_sent()`` has been called AND ``arm_delay_s`` seconds have
    elapsed.  The delay prevents the operator standing beside the drone at takeoff
    from being detected as a rescue target.

ACTIVE
    Normal detection logic runs.  The trigger fires when a person is detected with
    confidence >= ``min_confidence`` for ``consecutive_frames`` frames in a row.

DISABLED (terminal state)
    Entered permanently after the trigger fires once.  No further rescues will be
    dispatched for the rest of the flight.
"""

import time
from dataclasses import dataclass

from agent.orchestrator.inference.yolo_common import Detection


@dataclass(frozen=True, slots=True)
class TriggerFire:
    """Payload returned when the rescue trigger fires.

    Attributes:
        best_detection: The highest-confidence person Detection in the qualifying frame.
        triggered_at_s: Monotonic timestamp (seconds) at the moment of firing.
    """

    best_detection: Detection
    triggered_at_s: float


class PersonRescueTrigger:
    """One-shot rescue trigger that arms automatically after the first mission is airborne.

    Construct once and pass to the vision recorder.  Call ``notify_mission_sent()``
    from the orchestrator loop as soon as the first operator mission is confirmed
    uploaded via gRPC.  After ``arm_delay_s`` seconds the trigger enters ACTIVE state
    and ``tick()`` will return a ``TriggerFire`` when the detection criteria are met.
    It fires at most once; after firing ``tick()`` always returns ``None``.
    """

    def __init__(
        self,
        *,
        min_confidence: float,
        consecutive_frames: int,
        arm_delay_s: float,
    ) -> None:
        """Initialise the trigger in SUPPRESSED state.

        Args:
            min_confidence: Minimum YOLO person confidence score (0–1) required for a
                frame to count as qualifying.
            consecutive_frames: Number of consecutive qualifying frames required before
                the trigger fires. Must be at least 1.
            arm_delay_s: Seconds to wait after ``notify_mission_sent()`` before the
                trigger becomes active. Set this to the time it takes the drone to fly
                away from the operator (typically 60 s).
        """
        self._min_confidence = float(min_confidence)
        self._consecutive_frames = max(1, int(consecutive_frames))
        self._arm_delay_s = float(arm_delay_s)
        self._streak = 0
        self._mission_sent_at_s: float | None = None
        self._fired: bool = False

    def notify_mission_sent(self) -> None:
        """Record that the first operator mission has been sent via gRPC.

        Starts the ``arm_delay_s`` countdown.  Only the first call has any effect;
        subsequent calls are silently ignored.
        """
        if self._mission_sent_at_s is None:
            self._mission_sent_at_s = time.monotonic()

    def tick(self, dets: list[Detection]) -> TriggerFire | None:
        """Advance trigger state for one recorder frame.

        Returns ``None`` while SUPPRESSED or DISABLED.  Returns a ``TriggerFire``
        exactly once when the ACTIVE detection criteria are met, and immediately
        transitions to DISABLED so no second fire can occur.

        Args:
            dets: All detections from the current frame (may be empty).

        Returns:
            A TriggerFire if the trigger just fired, otherwise None.
        """
        # Terminal state — permanently disabled after first fire.
        if self._fired:
            return None

        # Suppressed until mission is sent and arm delay has elapsed.
        now_s = time.monotonic()
        if (
            self._mission_sent_at_s is None
            or (now_s - self._mission_sent_at_s) < self._arm_delay_s
        ):
            self._streak = 0
            return None

        # ACTIVE: find highest-confidence person detection.
        best_det: Detection | None = None
        best_conf = 0.0
        for d in dets:
            if d.class_id != 0 and d.class_name != "person":
                continue
            if d.confidence > best_conf:
                best_conf = float(d.confidence)
                best_det = d

        qualified = best_conf >= self._min_confidence and best_det is not None

        if not qualified:
            self._streak = 0
            return None

        self._streak += 1
        if self._streak < self._consecutive_frames:
            return None

        # Streak complete — fire once and enter DISABLED state.
        self._fired = True
        self._streak = 0
        return TriggerFire(best_detection=best_det, triggered_at_s=now_s)  # type: ignore[arg-type]
