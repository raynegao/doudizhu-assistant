from __future__ import annotations

from collections import Counter

import pytest

from src.logic.action_validation import validate_observed_action
from src.state.cards import FULL_DECK
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import (
    GameStateTracker,
    StateUpdateStatus,
)
from src.state.observable_state import ObservableGameState


def _event(
    event_id: str,
    sequence_no: int,
    actor: str,
    cards: list[str] | None = None,
    confidence: float = 1.0,
) -> ObservedAction:
    return ObservedAction.from_payload({
        "event": "play_observed" if cards else "pass_observed",
        "event_id": event_id,
        "sequence_no": sequence_no,
        "actor": actor,
        "cards": cards or [],
        "confidence": confidence,
    })


def test_observable_state_preserves_full_deck_invariant(
    phase4_initial_state: ObservableGameState,
) -> None:
    state = phase4_initial_state
    assert len(state.unknown_cards) == 34
    assert state.decision_ready
    assert Counter((*state.self_hand.cards, *state.unknown_cards.cards)) == Counter(FULL_DECK)


def test_tracker_applies_turns_and_two_pass_reset(
    phase4_ready_state: ObservableGameState,
) -> None:
    tracker = GameStateTracker(phase4_ready_state, validator=validate_observed_action)
    tracker.apply(_event("turn-004", 4, "self", ["5"]))
    tracker.apply(_event("turn-005", 5, "right"))
    result = tracker.apply(_event("turn-006", 6, "left"))

    assert result.status is StateUpdateStatus.APPLIED
    assert tracker.state.current_actor is PlayerSeat.SELF
    assert not tracker.state.trick_target
    assert tracker.state.trick_leader is None
    assert tracker.state.consecutive_passes == 0
    assert tracker.state.remaining_for(PlayerSeat.SELF) == 18
    assert tracker.state.decision_ready


def test_tracker_duplicate_is_idempotent_and_conflict_is_rejected(
    phase4_initial_state: ObservableGameState,
) -> None:
    tracker = GameStateTracker(phase4_initial_state, validator=validate_observed_action)
    event = _event("turn-001", 1, "self", ["3"])
    first = tracker.apply(event)
    duplicate = tracker.apply(event)

    assert first.status is StateUpdateStatus.APPLIED
    assert duplicate.status is StateUpdateStatus.DUPLICATE
    assert duplicate.state is first.state

    conflict = tracker.apply(_event("turn-001", 1, "self", ["4"]))
    assert conflict.status is StateUpdateStatus.REJECTED
    assert "event_id conflict" in conflict.message
    assert tracker.state is first.state


def test_tracker_defers_low_confidence_without_mutating_state(
    phase4_initial_state: ObservableGameState,
) -> None:
    tracker = GameStateTracker(phase4_initial_state, validator=validate_observed_action)
    result = tracker.apply(_event("turn-001", 1, "self", ["3"], confidence=0.40))

    assert result.status is StateUpdateStatus.DEFERRED
    assert tracker.state.phase is RoundPhase.UNCERTAIN
    assert not tracker.state.decision_ready
    assert tracker.state.revision == 0

    confirmed = tracker.apply(_event("turn-001-confirmed", 1, "self", ["3"]))
    assert confirmed.status is StateUpdateStatus.APPLIED
    assert tracker.state.phase is RoundPhase.PLAYING
    assert not tracker.pending_events


def test_tracker_rejects_out_of_turn_and_non_beating_actions(
    phase4_initial_state: ObservableGameState,
) -> None:
    tracker = GameStateTracker(phase4_initial_state, validator=validate_observed_action)
    rejected = tracker.apply(_event("turn-001", 1, "right", ["3"]))
    assert rejected.status is StateUpdateStatus.REJECTED
    assert "out-of-turn" in rejected.message
    assert tracker.state is phase4_initial_state

    tracker.apply(_event("turn-001", 1, "self", ["3"]))
    rejected = tracker.apply(_event("turn-002", 2, "right", ["3"]))
    assert rejected.status is StateUpdateStatus.REJECTED
    assert "cannot beat" in rejected.message


def test_incomplete_state_is_not_decision_ready() -> None:
    state = ObservableGameState.from_inputs("3 4 5", landlord="self")
    assert not state.decision_ready
    assert state.warnings


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_observed_action_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        ObservedAction.from_payload({
            "event": "play_observed",
            "event_id": "bad-confidence",
            "sequence_no": 1,
            "actor": "self",
            "cards": ["3"],
            "confidence": confidence,
        })


@pytest.mark.parametrize(
    ("event_type", "cards"),
    [("play_observed", []), ("pass_observed", ["3"])],
)
def test_observed_action_rejects_event_card_mismatch(
    event_type: str,
    cards: list[str],
) -> None:
    with pytest.raises(ValueError):
        ObservedAction.from_payload({
            "event": event_type,
            "event_id": "bad-cards",
            "sequence_no": 1,
            "actor": "self",
            "cards": cards,
        })


def test_observable_state_enforces_role_limits_and_finished_phase() -> None:
    with pytest.raises(ValueError, match="role maximum"):
        ObservableGameState.from_inputs(
            "3",
            landlord="self",
            remaining_cards={"self": 1, "right": 18, "left": 17},
        )

    remaining_deck = list(FULL_DECK)
    remaining_deck.remove("3")
    remaining_deck.remove("4")
    state = ObservableGameState.from_inputs(
        "3",
        landlord="self",
        remaining_cards={"self": 1, "right": 0, "left": 1},
        played_cards=remaining_deck,
    )
    assert state.phase is RoundPhase.FINISHED
    assert state.winner is PlayerSeat.RIGHT
    assert not state.decision_ready
