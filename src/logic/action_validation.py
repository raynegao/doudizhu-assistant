from __future__ import annotations

from src.logic.rules import Play, PlayType, can_beat
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import GameStateTransitionError
from src.state.observable_state import ObservableGameState


def validate_observed_action(state: ObservableGameState, event: ObservedAction) -> None:
    if event.is_pass:
        if not state.trick_target:
            raise GameStateTransitionError("cannot pass when leading a new trick")
        return

    play = Play.parse(event.cards.cards)
    if play.type is PlayType.INVALID:
        raise GameStateTransitionError(f"invalid play type: {event.cards}")
    if state.trick_target:
        previous = Play.parse(state.trick_target.cards)
        if not can_beat(play, previous):
            raise GameStateTransitionError(
                f"play {play} cannot beat current trick target {previous}"
            )


def validate_decision_state(state: ObservableGameState) -> None:
    if state.phase is not RoundPhase.PLAYING:
        raise GameStateTransitionError(
            f"Phase 4 requires phase=playing, got {state.phase.value}"
        )
    if state.winner is not None:
        raise GameStateTransitionError("cannot recommend after the round has a winner")
    if state.current_actor is not PlayerSeat.SELF:
        raise GameStateTransitionError("Phase 4 recommendation requires current_actor=self")
    if state.state_confidence < 0.70:
        raise GameStateTransitionError(
            f"state confidence {state.state_confidence:.3f} is below 0.700"
        )
    if state.trick_target:
        target = Play.parse(state.trick_target.cards)
        if target.type is PlayType.INVALID:
            raise GameStateTransitionError(
                f"invalid trick target: {state.trick_target}"
            )
    if not state._turn_is_consistent():
        raise GameStateTransitionError(
            "current actor is inconsistent with trick leader and pass count"
        )


__all__ = ["validate_decision_state", "validate_observed_action"]
