from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from src.logic.monte_carlo import (
    HeuristicRolloutPolicy,
    MonteCarloEvaluator,
    MonteCarloSettings,
    _simulate_candidate,
    recommend_phase4,
)
from src.logic.opponent_model import OpponentDeal, UniformOpponentModel
from src.logic.rules import Play, legal_actions
from src.state.cards import FULL_DECK
from src.state.cards import CardSet
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState


def _settings(**overrides) -> MonteCarloSettings:
    values = {
        "simulations": 4,
        "max_depth": 20,
        "time_budget_ms": 0,
        "seed": 17,
        "top_k": 3,
        "max_candidates": 8,
        "min_rollouts_per_action": 2,
    }
    values.update(overrides)
    return MonteCarloSettings(**values)


def test_phase4_top_k_is_legal_unique_and_deterministic(
    phase4_ready_state: ObservableGameState,
) -> None:
    first = recommend_phase4(phase4_ready_state, _settings())
    second = recommend_phase4(phase4_ready_state, _settings())
    legal = set(legal_actions(phase4_ready_state.self_hand, Play.parse("4")))

    assert first.action in legal
    assert [item.action for item in first.rankings] == [item.action for item in second.rankings]
    assert [item.strategy_score for item in first.rankings] == [item.strategy_score for item in second.rankings]
    assert len({item.action for item in first.rankings}) == len(first.rankings)
    assert all(item.action in legal for item in first.rankings)
    assert all(0.0 <= item.strategy_score <= 1.0 for item in first.rankings)
    assert all(item.simulations == 4 for item in first.rankings)


def test_phase4_exact_finish_is_preferred() -> None:
    pool = Counter(FULL_DECK)
    pool.subtract(["3", "SJ", "BJ"])
    played = tuple(card for rank in pool for card in [rank] * pool[rank])
    state = ObservableGameState.from_inputs(
        "3",
        landlord="self",
        remaining_cards={"self": 1, "right": 1, "left": 1},
        played_cards=played,
    )

    result = recommend_phase4(state, _settings(top_k=1, max_candidates=2))

    assert str(result.action) == "3"
    assert result.rankings[0].estimated_win_rate == 1.0
    assert "exact_finish" in result.rankings[0].reason_factors


def test_phase4_incomplete_state_falls_back_without_fake_win_rate() -> None:
    state = ObservableGameState.from_inputs("3 4 5", landlord="self")
    result = recommend_phase4(state, _settings())

    assert result.completed_simulations == 0
    assert result.opponent_estimate is None
    assert result.rankings[0].simulations == 0
    assert result.rankings[0].estimated_win_rate is None
    assert any("monte_carlo_unavailable" in warning for warning in result.warnings)


def test_phase4_time_budget_returns_fair_partial_batch(
    phase4_ready_state: ObservableGameState,
) -> None:
    value = 0.0

    def fake_clock() -> float:
        nonlocal value
        value += 0.01
        return value

    evaluator = MonteCarloEvaluator(clock=fake_clock)
    result = evaluator.evaluate(
        phase4_ready_state,
        _settings(simulations=10, time_budget_ms=1),
    )

    assert result.completed_simulations == 1
    assert all(item.simulations == 1 for item in result.rankings)
    assert any("time_budget_exhausted" in warning for warning in result.warnings)


def test_rollout_counts_farmer_teammate_finish_as_team_win() -> None:
    pool = Counter(FULL_DECK)
    pool.subtract(["5", "4", "SJ"])
    played = tuple(card for rank in pool for card in [rank] * pool[rank])
    state = ObservableGameState.from_inputs(
        "5",
        landlord="right",
        current_actor="self",
        remaining_cards={"self": 1, "right": 1, "left": 1},
        played_cards=played,
        last_play="2",
        last_player="left",
    )
    deal = OpponentDeal((
        (PlayerSeat.RIGHT, CardSet.parse("4")),
        (PlayerSeat.LEFT, CardSet.parse("SJ")),
    ))

    outcome = _simulate_candidate(
        state,
        deal,
        Play.parse(()),
        max_depth=10,
        policy=HeuristicRolloutPolicy(),
    )

    assert outcome.terminal_result == 1.0
    assert outcome.expected_result == 1.0


def test_depth_limited_estimate_is_normalized_across_roles() -> None:
    known = ["3", "4", "5", "6"]
    pool = list(FULL_DECK)
    for card in known:
        pool.remove(card)
    deal = OpponentDeal((
        (PlayerSeat.RIGHT, CardSet.parse("5")),
        (PlayerSeat.LEFT, CardSet.parse("6")),
    ))
    outcomes = []
    for landlord in (PlayerSeat.SELF, PlayerSeat.RIGHT):
        state = ObservableGameState.from_inputs(
            "3 4",
            landlord=landlord,
            remaining_cards={"self": 2, "right": 1, "left": 1},
            played_cards=pool,
        )
        outcomes.append(_simulate_candidate(
            state,
            deal,
            Play.parse("3"),
            max_depth=1,
            policy=HeuristicRolloutPolicy(),
        ))

    assert outcomes[0].terminal_result is None
    assert outcomes[0].expected_result == pytest.approx(0.75)
    assert outcomes[1].expected_result == pytest.approx(outcomes[0].expected_result)


def test_phase4_rejects_low_confidence_and_inconsistent_states(
    phase4_ready_state: ObservableGameState,
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        recommend_phase4(
            replace(phase4_ready_state, state_confidence=0.69),
            _settings(),
        )

    inconsistent = replace(
        phase4_ready_state,
        trick_leader=PlayerSeat.LEFT,
    )
    assert not inconsistent.decision_ready
    with pytest.raises(ValueError, match="inconsistent"):
        recommend_phase4(inconsistent, _settings())


def test_phase4_rejects_invalid_trick_target() -> None:
    pool = list(FULL_DECK)
    for card in ["5", "SJ", "BJ"]:
        pool.remove(card)
    state = ObservableGameState.from_inputs(
        "5",
        landlord="self",
        remaining_cards={"self": 1, "right": 1, "left": 1},
        played_cards=pool,
        last_play="3 3 4",
        last_player="left",
    )
    assert state.decision_ready
    with pytest.raises(ValueError, match="invalid trick target"):
        recommend_phase4(state, _settings())


def test_phase4_rejects_finished_round() -> None:
    pool = list(FULL_DECK)
    pool.remove("3")
    pool.remove("4")
    state = ObservableGameState.from_inputs(
        "3",
        landlord="self",
        remaining_cards={"self": 1, "right": 0, "left": 1},
        played_cards=pool,
    )
    with pytest.raises(ValueError, match="got finished"):
        recommend_phase4(state, _settings())


def test_rollout_rejects_illegal_policy_action(
    phase4_ready_state: ObservableGameState,
) -> None:
    sampled = UniformOpponentModel().sample_many(
        phase4_ready_state,
        count=1,
        seed=1,
    )[0]

    class IllegalPolicy:
        def choose_action(self, **_: object) -> Play:
            return Play.parse("3 3 3 3")

    candidate = next(
        action
        for action in legal_actions(
            phase4_ready_state.self_hand,
            Play.parse(phase4_ready_state.trick_target.cards),
        )
        if not action.is_pass
    )
    with pytest.raises(RuntimeError, match="illegal action"):
        _simulate_candidate(
            phase4_ready_state,
            sampled,
            candidate,
            max_depth=3,
            policy=IllegalPolicy(),
        )
