from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from .cards import CardSet, sort_cards, validate_card_counts
from .events import ObservedAction, PlayerSeat, RoundPhase
from .observable_state import ObservableGameState


class StateUpdateStatus(str, Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class GameStateTransitionError(ValueError):
    pass


ActionValidator = Callable[[ObservableGameState, ObservedAction], None]


@dataclass(frozen=True)
class StateUpdateResult:
    status: StateUpdateStatus
    state: ObservableGameState
    event: ObservedAction
    message: str
    warnings: tuple[str, ...] = ()


def reduce_observed_action(
    state: ObservableGameState,
    event: ObservedAction,
    *,
    validator: ActionValidator,
    confidence_threshold: float = 0.70,
) -> StateUpdateResult:
    fingerprints = dict(state.processed_events)
    if event.event_id in fingerprints:
        if fingerprints[event.event_id] != event.fingerprint:
            raise GameStateTransitionError(
                f"event_id conflict: {event.event_id} was already used for different content"
            )
        return StateUpdateResult(
            status=StateUpdateStatus.DUPLICATE,
            state=state,
            event=event,
            message="duplicate event ignored",
        )
    if state.phase is not RoundPhase.PLAYING:
        raise GameStateTransitionError("cannot apply actions after the round is finished")
    if event.sequence_no != state.last_sequence_no + 1:
        raise GameStateTransitionError(
            f"out-of-order sequence: expected {state.last_sequence_no + 1}, got {event.sequence_no}"
        )
    if event.actor is not state.current_actor:
        raise GameStateTransitionError(
            f"out-of-turn action: expected {state.current_actor.value}, got {event.actor.value}"
        )
    if event.confidence < confidence_threshold:
        return StateUpdateResult(
            status=StateUpdateStatus.DEFERRED,
            state=state,
            event=event,
            message="low-confidence event deferred",
            warnings=(
                f"event {event.event_id} confidence {event.confidence:.3f} is below {confidence_threshold:.3f}",
            ),
        )
    validator(state, event)

    remaining = state.remaining_by_player
    played_cards = state.played_cards
    hand = state.self_hand
    history = (*state.history, event)
    processed = (*state.processed_events, (event.event_id, event.fingerprint))
    current_actor = state.next_player(event.actor)
    trick_target = state.trick_target
    trick_leader = state.trick_leader
    passes = state.consecutive_passes
    phase = state.phase

    if event.is_pass:
        if not trick_target:
            raise GameStateTransitionError("cannot pass when leading a new trick")
        passes += 1
        if passes == 2:
            if trick_leader is None:
                raise GameStateTransitionError("active trick is missing its leader")
            current_actor = trick_leader
            trick_target = CardSet(())
            trick_leader = None
            passes = 0
    else:
        if remaining[event.actor] < len(event.cards):
            raise GameStateTransitionError(
                f"{event.actor.value} cannot play more cards than remain"
            )
        if event.actor is PlayerSeat.SELF:
            missing = Counter(event.cards.cards) - Counter(hand.cards)
            if missing:
                raise GameStateTransitionError("self action contains cards not present in hand")
            updated = Counter(hand.cards) - Counter(event.cards.cards)
            hand = CardSet(sort_cards(card for rank in updated for card in [rank] * updated[rank]))
        else:
            missing = Counter(event.cards.cards) - Counter(state.unknown_cards.cards)
            if missing:
                raise GameStateTransitionError("opponent action contains cards outside the unknown pool")

        played_cards = sort_cards((*played_cards, *event.cards.cards))
        validate_card_counts((*hand.cards, *played_cards))
        remaining[event.actor] -= len(event.cards)
        trick_target = event.cards
        trick_leader = event.actor
        passes = 0
        if remaining[event.actor] == 0:
            phase = RoundPhase.FINISHED
            current_actor = event.actor

    next_state = replace(
        state,
        revision=state.revision + 1,
        phase=phase,
        self_hand=hand,
        remaining_cards=tuple((seat, remaining[seat]) for seat in PlayerSeat),
        current_actor=current_actor,
        trick_target=trick_target,
        trick_leader=trick_leader,
        consecutive_passes=passes,
        played_cards=played_cards,
        history=history,
        processed_events=processed,
        last_sequence_no=event.sequence_no,
        state_confidence=min(state.state_confidence, event.confidence),
    )
    return StateUpdateResult(
        status=StateUpdateStatus.APPLIED,
        state=next_state,
        event=event,
        message="event applied",
    )


class GameStateTracker:
    def __init__(
        self,
        initial_state: ObservableGameState,
        *,
        validator: ActionValidator,
        confidence_threshold: float = 0.70,
    ) -> None:
        self._state = initial_state
        self._pending: dict[int, ObservedAction] = {}
        self.confidence_threshold = confidence_threshold
        self.validator = validator

    @property
    def state(self) -> ObservableGameState:
        if self._pending:
            pending = ", ".join(str(sequence) for sequence in sorted(self._pending))
            warning = f"unresolved low-confidence event sequences: {pending}"
            return replace(
                self._state,
                phase=RoundPhase.UNCERTAIN,
                warnings=tuple(dict.fromkeys((*self._state.warnings, warning))),
            )
        return self._state

    @property
    def pending_events(self) -> tuple[ObservedAction, ...]:
        return tuple(self._pending[sequence] for sequence in sorted(self._pending))

    def apply(self, event: ObservedAction) -> StateUpdateResult:
        try:
            result = reduce_observed_action(
                self._state,
                event,
                confidence_threshold=self.confidence_threshold,
                validator=self.validator,
            )
        except GameStateTransitionError as exc:
            return StateUpdateResult(
                status=StateUpdateStatus.REJECTED,
                state=self.state,
                event=event,
                message=str(exc),
                warnings=(str(exc),),
            )
        if result.status is StateUpdateStatus.APPLIED:
            self._state = result.state
            self._pending.pop(event.sequence_no, None)
        elif result.status is StateUpdateStatus.DEFERRED:
            self._pending[event.sequence_no] = event
            result = replace(result, state=self.state)
        return result


__all__ = [
    "ActionValidator",
    "GameStateTracker",
    "GameStateTransitionError",
    "StateUpdateResult",
    "StateUpdateStatus",
    "reduce_observed_action",
]
