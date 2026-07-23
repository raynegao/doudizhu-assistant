from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from src.logic.action_validation import validate_observed_action
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import GameStateTracker, StateUpdateStatus
from src.state.observable_state import ObservableGameState
from src.vision.scene_recognizer import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualSignal,
)


class VisualTrackerMode(str, Enum):
    WAITING_FOR_ROUND = "waiting_for_round"
    TRACKING = "tracking"
    UNCERTAIN = "uncertain"
    FINISHED = "finished"


@dataclass(frozen=True)
class VisualTrackerUpdate:
    mode: VisualTrackerMode
    message: str
    state: ObservableGameState | None
    event: ObservedAction | None = None
    initialized: bool = False
    warnings: tuple[str, ...] = ()

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "state_update",
            "mode": self.mode.value,
            "message": self.message,
            "initialized": self.initialized,
            "observed_action": (
                self.event.to_log_payload() if self.event is not None else None
            ),
            "state": self.state.to_log_payload() if self.state is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass
class _StableValue:
    fingerprint: tuple[object, ...] | None = None
    count: int = 0

    def update(self, fingerprint: tuple[object, ...]) -> int:
        if fingerprint == self.fingerprint:
            self.count += 1
        else:
            self.fingerprint = fingerprint
            self.count = 1
        return self.count


RoundIdFactory = Callable[[SceneObservation], str]


class VisualEventTracker:
    """Convert stable scene observations into validated Phase 4 game events."""

    def __init__(
        self,
        *,
        stability_frames: int = 3,
        confidence_threshold: float = 0.70,
        round_id_factory: RoundIdFactory | None = None,
    ) -> None:
        if stability_frames <= 0:
            raise ValueError("stability_frames must be positive")
        self.stability_frames = stability_frames
        self.confidence_threshold = confidence_threshold
        self.round_id_factory = round_id_factory or (
            lambda scene: f"live-{int(scene.timestamp * 1000)}"
        )
        self.mode = VisualTrackerMode.WAITING_FOR_ROUND
        self._tracker: GameStateTracker | None = None
        self._initial_stable = _StableValue()
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._armed = {seat: False for seat in PlayerSeat}
        self._uncertain_reason: str | None = None

    @property
    def state(self) -> ObservableGameState | None:
        if self._tracker is None:
            return None
        state = self._tracker.state
        if self.mode is VisualTrackerMode.UNCERTAIN:
            warning = self._uncertain_reason or "visual event tracker is uncertain"
            return replace(
                state,
                phase=RoundPhase.UNCERTAIN,
                warnings=tuple(dict.fromkeys((*state.warnings, warning))),
            )
        return state

    def update(self, scene: SceneObservation) -> VisualTrackerUpdate:
        initial = _initial_state_payload(scene)
        if initial is not None:
            stable_count = self._initial_stable.update(initial.fingerprint)
            should_initialize = self._tracker is None or self.mode in {
                VisualTrackerMode.UNCERTAIN,
                VisualTrackerMode.FINISHED,
            }
            if should_initialize and stable_count >= self.stability_frames:
                return self._initialize(scene, initial)
        else:
            self._initial_stable.update(("not_initial",))

        if self._tracker is None:
            return VisualTrackerUpdate(
                mode=VisualTrackerMode.WAITING_FOR_ROUND,
                message="等待完整新局：地主、17/20 张初始手牌和三家初始余牌必须稳定",
                state=None,
                warnings=scene.warnings,
            )
        if self.mode is VisualTrackerMode.UNCERTAIN:
            return VisualTrackerUpdate(
                mode=self.mode,
                message="当前牌局已不确定；为避免伪胜率，等待下一局完整初始化",
                state=self.state,
                warnings=(self._uncertain_reason or "uncertain",),
            )
        if self.mode is VisualTrackerMode.FINISHED:
            return VisualTrackerUpdate(
                mode=self.mode,
                message="本局已结束，等待下一局",
                state=self.state,
            )

        state = self._tracker.state
        expected = state.current_actor
        observation = scene.seat(expected)
        fingerprint = _seat_fingerprint(observation)
        stable_count = self._seat_stable[expected].update(fingerprint)

        if observation.signal is VisualSignal.NEUTRAL:
            self._armed[expected] = True
            return VisualTrackerUpdate(
                mode=self.mode,
                message=f"等待 {expected.value} 出牌或过牌",
                state=state,
                warnings=scene.warnings,
            )
        if observation.signal is VisualSignal.UNKNOWN:
            if stable_count >= self.stability_frames:
                return self._mark_uncertain(
                    f"{expected.value} action remained low-confidence for "
                    f"{stable_count} frames"
                )
            return VisualTrackerUpdate(
                mode=self.mode,
                message=f"等待 {expected.value} 视觉结果稳定",
                state=state,
                warnings=scene.warnings,
            )
        if not self._armed[expected]:
            return VisualTrackerUpdate(
                mode=self.mode,
                message=f"忽略 {expected.value} 的旧场面提示，等待空白到动作的新变化",
                state=state,
            )
        if stable_count < self.stability_frames:
            return VisualTrackerUpdate(
                mode=self.mode,
                message=(
                    f"稳定 {expected.value} 动作 "
                    f"{stable_count}/{self.stability_frames}"
                ),
                state=state,
            )

        cards = observation.card_set
        confidence = _action_confidence(observation)
        if confidence < self.confidence_threshold:
            return self._mark_uncertain(
                f"{expected.value} action confidence {confidence:.3f} is below "
                f"{self.confidence_threshold:.3f}"
            )
        event = ObservedAction(
            event_id=f"{state.round_id}:{state.last_sequence_no + 1}:{expected.value}",
            sequence_no=state.last_sequence_no + 1,
            actor=expected,
            cards=cards,
            confidence=confidence,
            source="live_visual",
        )
        count_error = _remaining_count_error(state, observation, event)
        if count_error is not None:
            return self._mark_uncertain(count_error, event=event)

        result = self._tracker.apply(event)
        if result.status is not StateUpdateStatus.APPLIED:
            return self._mark_uncertain(
                f"state rejected visual event: {result.message}",
                event=event,
            )

        self._armed[expected] = False
        next_actor = result.state.current_actor
        if scene.seat(next_actor).signal is VisualSignal.NEUTRAL:
            self._armed[next_actor] = True
        if result.state.phase is RoundPhase.FINISHED:
            self.mode = VisualTrackerMode.FINISHED
        message = (
            f"{expected.value} 过牌"
            if event.is_pass
            else f"{expected.value} 出牌：{' '.join(event.cards.cards)}"
        )
        return VisualTrackerUpdate(
            mode=self.mode,
            message=message,
            state=self.state,
            event=event,
            warnings=result.warnings,
        )

    def handle_window_unavailable(self, reason: str) -> VisualTrackerUpdate:
        if self._tracker is not None and self.mode is VisualTrackerMode.TRACKING:
            return self._mark_uncertain(
                f"{reason}；可能漏过场上事件，等待下一局完整初始化"
            )
        return VisualTrackerUpdate(
            mode=self.mode,
            message=reason,
            state=self.state,
            warnings=(reason,),
        )

    def _initialize(
        self,
        scene: SceneObservation,
        initial: "_InitialStatePayload",
    ) -> VisualTrackerUpdate:
        try:
            state = ObservableGameState.from_inputs(
                initial.hand,
                round_id=self.round_id_factory(scene),
                landlord=initial.landlord,
                current_actor=initial.landlord,
                remaining_cards=initial.remaining,
                state_confidence=initial.confidence,
            )
        except ValueError as exc:
            return self._mark_uncertain(f"cannot initialize visual round: {exc}")
        self._tracker = GameStateTracker(
            state,
            validator=validate_observed_action,
            confidence_threshold=self.confidence_threshold,
        )
        self.mode = VisualTrackerMode.TRACKING
        self._uncertain_reason = None
        self._seat_stable = {seat: _StableValue() for seat in PlayerSeat}
        self._armed = {
            seat: scene.seat(seat).signal is VisualSignal.NEUTRAL
            for seat in PlayerSeat
        }
        return VisualTrackerUpdate(
            mode=self.mode,
            message=(
                f"新局已初始化：地主={initial.landlord.value}，"
                f"当前行动者={initial.landlord.value}"
            ),
            state=state,
            initialized=True,
            warnings=scene.warnings,
        )

    def _mark_uncertain(
        self,
        reason: str,
        *,
        event: ObservedAction | None = None,
    ) -> VisualTrackerUpdate:
        self.mode = VisualTrackerMode.UNCERTAIN
        self._uncertain_reason = reason
        return VisualTrackerUpdate(
            mode=self.mode,
            message=reason,
            state=self.state,
            event=event,
            warnings=(reason,),
        )


@dataclass(frozen=True)
class _InitialStatePayload:
    landlord: PlayerSeat
    hand: tuple[str, ...]
    remaining: dict[PlayerSeat, int]
    confidence: float

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.landlord.value,
            self.hand,
            tuple((seat.value, self.remaining[seat]) for seat in PlayerSeat),
        )


def _initial_state_payload(scene: SceneObservation) -> _InitialStatePayload | None:
    landlords = [
        observation.seat
        for observation in scene.seats
        if observation.role is SeatRole.LANDLORD
    ]
    farmers = [
        observation.seat
        for observation in scene.seats
        if observation.role is SeatRole.FARMER
    ]
    if len(landlords) != 1 or len(farmers) != 2:
        return None
    landlord = landlords[0]
    expected = {
        seat: 20 if seat is landlord else 17
        for seat in PlayerSeat
    }
    if any(
        observation.signal is not VisualSignal.NEUTRAL
        for observation in scene.seats
    ):
        return None
    remaining: dict[PlayerSeat, int] = {}
    for observation in scene.seats:
        if observation.remaining_count is None:
            if observation.seat is not landlord:
                return None
            remaining[observation.seat] = expected[observation.seat]
            continue
        remaining[observation.seat] = observation.remaining_count
    if remaining != expected:
        return None
    hand = tuple(card.rank for card in scene.self_hand)
    if len(hand) != expected[PlayerSeat.SELF]:
        return None
    try:
        hand_set = scene.self_hand_set
    except ValueError:
        return None
    if len(hand_set) != len(hand):
        return None
    confidences = [
        scene.confidence,
        *(card.confidence for card in scene.self_hand),
        *(seat.role_confidence for seat in scene.seats),
        *(seat.remaining_confidence for seat in scene.seats),
    ]
    positive_confidences = [value for value in confidences if value > 0]
    if not positive_confidences:
        return None
    confidence = min(positive_confidences)
    if confidence < 0.70:
        return None
    return _InitialStatePayload(
        landlord=landlord,
        hand=hand,
        remaining=remaining,
        confidence=confidence,
    )


def _seat_fingerprint(observation: SeatObservation) -> tuple[object, ...]:
    return (
        observation.signal.value,
        tuple(card.rank for card in observation.cards),
    )


def _action_confidence(observation: SeatObservation) -> float:
    if observation.signal is VisualSignal.PASS:
        return observation.pass_confidence
    if observation.signal is VisualSignal.PLAY:
        return min(
            (card.confidence for card in observation.cards),
            default=observation.confidence,
        )
    return observation.confidence


def _remaining_count_error(
    state: ObservableGameState,
    observation: SeatObservation,
    event: ObservedAction,
) -> str | None:
    # The current Mac client does not expose a dedicated self count in all layouts;
    # self remaining is exactly derived from the tracked hand.
    if event.actor is PlayerSeat.SELF or observation.remaining_count is None:
        return None
    expected = state.remaining_for(event.actor) - len(event.cards)
    if observation.remaining_count != expected:
        return (
            f"{event.actor.value} remaining count mismatch: screen="
            f"{observation.remaining_count}, expected={expected}"
        )
    return None


__all__ = [
    "VisualEventTracker",
    "VisualTrackerMode",
    "VisualTrackerUpdate",
]
