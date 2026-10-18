"""Successive halving spends the same rollout budget where the ranking is decided."""

from __future__ import annotations

import itertools

import pytest

from src.config.live_layout import LiveLayoutConfig, live_layout_from_dict
from src.logic.monte_carlo import (
    MonteCarloEvaluator,
    MonteCarloSettings,
    allocation_schedule,
    recommend_phase4,
)
from src.state.observable_state import ObservableGameState

# Twenty landlord cards with many legal leads (87 in the 2026-10-17 004 opening).
HAND = "BJ SJ 2 2 A K K J J 10 10 10 9 9 9 8 8 7 5 4"


def _state() -> ObservableGameState:
    return ObservableGameState.from_inputs(HAND, landlord="self", current_actor="self")


def test_schedule_halves_down_to_top_k_within_the_uniform_budget() -> None:
    settings = MonteCarloSettings(simulations=128, top_k=3, allocation="successive_halving")
    schedule = allocation_schedule(40, settings)
    assert [keep for keep, _ in schedule] == [40, 20, 10, 5, 3]
    assert sum(keep * worlds for keep, worlds in schedule) <= 128 * 40
    assert schedule[-1][1] > 4 * schedule[0][1]  # finalists see far more worlds
    assert allocation_schedule(40, MonteCarloSettings(simulations=128)) == ((40, 128),)
    assert allocation_schedule(3, settings) == ((3, 128),)
    with pytest.raises(ValueError):
        allocation_schedule(0, settings)
    with pytest.raises(ValueError):
        MonteCarloSettings(allocation="greedy")


def test_halving_ranks_finalists_first_on_more_samples() -> None:
    settings = MonteCarloSettings(
        simulations=12, max_depth=40, time_budget_ms=0, top_k=3,
        max_candidates=24, allocation="successive_halving", seed=7,
    )
    result = recommend_phase4(_state(), settings)
    counts = [evaluation.simulations for evaluation in result.rankings]
    assert len(result.rankings) == 3
    assert counts[0] == counts[1] == counts[2]
    assert counts[0] > 12  # finalists got more than the uniform per-candidate budget
    assert any(warning.startswith("successive_halving: rounds=") for warning in result.warnings)
    assert "candidate_pruned: evaluated 24 of" in " ".join(result.warnings)
    assert result.evaluated_candidate_count == 24
    assert result.completed_simulations == sum(
        worlds for _, worlds in allocation_schedule(24, settings)
    )
    assert f"{counts[0]} 组对手牌样本" in result.reason


def test_uniform_allocation_is_unchanged() -> None:
    settings = MonteCarloSettings(simulations=6, max_depth=30, time_budget_ms=0, max_candidates=8, seed=3)
    result = recommend_phase4(_state(), settings)
    assert {evaluation.simulations for evaluation in result.rankings} == {6}
    assert result.completed_simulations == 6
    assert not any(warning.startswith("successive_halving") for warning in result.warnings)


def test_deadline_inside_halving_keeps_the_current_survivors() -> None:
    ticks = itertools.count()

    def clock() -> float:
        # Each call advances 0.25 s; the deadline (1 s) falls in the first round.
        return next(ticks) * 0.25

    settings = MonteCarloSettings(
        simulations=20, max_depth=30, time_budget_ms=1000, top_k=2,
        max_candidates=16, allocation="successive_halving", seed=5,
        min_rollouts_per_action=1, enforce_min_rollouts=True,
    )
    result = MonteCarloEvaluator(clock=clock).evaluate(_state(), settings)
    assert any(warning.startswith("time_budget_exhausted") for warning in result.warnings)
    assert result.completed_simulations < 20
    assert len(result.rankings) == 2
    assert result.action in {evaluation.action for evaluation in result.rankings}


def test_live_layout_carries_the_allocation() -> None:
    assert LiveLayoutConfig().allocation == "successive_halving"
    config = live_layout_from_dict({"allocation": "uniform"})
    assert config.allocation == "uniform"
    assert config.to_payload()["allocation"] == "uniform"
    assert live_layout_from_dict({}).allocation == "successive_halving"
    with pytest.raises(ValueError):
        LiveLayoutConfig(allocation="random")
