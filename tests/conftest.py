from __future__ import annotations

import pytest

from src.logic.action_validation import validate_observed_action
from src.state.events import ObservedAction, PlayerSeat
from src.state.game_tracker import GameStateTracker
from src.state.observable_state import ObservableGameState


@pytest.fixture
def phase4_initial_state() -> ObservableGameState:
    return ObservableGameState.from_inputs(
        "3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 J Q K A 2",
        round_id="test-round",
        landlord="self",
        current_actor="self",
        remaining_cards={"self": 20, "right": 17, "left": 17},
    )


@pytest.fixture
def phase4_ready_state(phase4_initial_state: ObservableGameState) -> ObservableGameState:
    tracker = GameStateTracker(phase4_initial_state, validator=validate_observed_action)
    events = (
        ObservedAction.from_payload({
            "event": "play_observed",
            "event_id": "turn-001",
            "sequence_no": 1,
            "actor": "self",
            "cards": ["3"],
        }),
        ObservedAction.from_payload({
            "event": "play_observed",
            "event_id": "turn-002",
            "sequence_no": 2,
            "actor": "right",
            "cards": ["4"],
        }),
        ObservedAction.from_payload({
            "event": "pass_observed",
            "event_id": "turn-003",
            "sequence_no": 3,
            "actor": "left",
        }),
    )
    for event in events:
        tracker.apply(event)
    return tracker.state
