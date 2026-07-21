from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random

import pytest

from src.logic.opponent_model import OpponentModelError, UniformOpponentModel
from src.state.cards import FULL_DECK
from src.state.events import PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState


def test_uniform_opponent_sampling_is_deterministic_and_conserves_cards(
    phase4_ready_state: ObservableGameState,
) -> None:
    model = UniformOpponentModel()
    before = random.getstate()
    first = model.sample_many(phase4_ready_state, count=3, seed=11)
    second = model.sample_many(phase4_ready_state, count=3, seed=11)

    assert first == second
    assert random.getstate() == before
    for deal in first:
        right = deal.hand_for(PlayerSeat.RIGHT)
        left = deal.hand_for(PlayerSeat.LEFT)
        assert len(right) == 16
        assert len(left) == 17
        assert Counter((*right.cards, *left.cards)) == Counter(phase4_ready_state.unknown_cards.cards)


def test_uniform_opponent_estimate_has_bounded_probabilities(
    phase4_ready_state: ObservableGameState,
) -> None:
    model = UniformOpponentModel()
    deals = model.sample_many(phase4_ready_state, count=5, seed=5)
    estimate = model.summarize(phase4_ready_state, deals)

    assert estimate.sample_count == 5
    payload = estimate.to_log_payload()
    assert payload["model_version"] == "uniform-remaining-cards-v1"
    for section in ("bomb_probability", "rocket_probability", "can_beat_probability"):
        assert all(0.0 <= value <= 1.0 for value in payload[section].values())


def test_uniform_opponent_model_rejects_incomplete_state() -> None:
    state = ObservableGameState.from_inputs("3 4 5", landlord="self")
    with pytest.raises(OpponentModelError, match="not decision-ready"):
        UniformOpponentModel().sample_many(state, count=1, seed=1)


def test_uniform_opponent_model_rejects_low_confidence_state(
    phase4_ready_state: ObservableGameState,
) -> None:
    state = replace(phase4_ready_state, state_confidence=0.69)
    assert not state.decision_ready
    with pytest.raises(OpponentModelError, match="not decision-ready"):
        UniformOpponentModel().sample_many(state, count=1, seed=1)


def test_uniform_opponent_model_rejects_finished_state() -> None:
    played = list(FULL_DECK)
    played.remove("3")
    played.remove("4")
    state = ObservableGameState.from_inputs(
        "3",
        landlord="self",
        remaining_cards={"self": 1, "right": 0, "left": 1},
        played_cards=played,
    )
    assert state.phase is RoundPhase.FINISHED
    with pytest.raises(OpponentModelError, match="not decision-ready"):
        UniformOpponentModel().sample_many(state, count=1, seed=1)
