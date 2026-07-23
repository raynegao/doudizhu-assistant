from __future__ import annotations

from src.logic.monte_carlo import (
    ActionEvaluation,
    MonteCarloEvaluator,
    MonteCarloSettings,
    _evaluation_sort_key,
)
from src.logic.rules import Play


def _evaluation(
    action: str,
    *,
    win_rate: float,
    strategy_score: float,
    terminal_rate: float = 1.0,
) -> ActionEvaluation:
    return ActionEvaluation(
        action=Play.parse(action),
        strategy_score=strategy_score,
        estimated_win_rate=win_rate,
        terminal_win_rate=win_rate,
        terminal_rate=terminal_rate,
        simulations=32,
        average_plies=10.0,
        average_margin=0.0,
        reason_factors=(),
        risk_flags=(),
    )


def test_phase6_ranks_estimated_win_rate_before_strategy_score() -> None:
    high_strategy = _evaluation("3", win_rate=0.40, strategy_score=0.95)
    high_win_rate = _evaluation("4", win_rate=0.60, strategy_score=0.50)

    ordered = sorted((high_strategy, high_win_rate), key=_evaluation_sort_key)

    assert ordered[0] is high_win_rate


def test_phase6_enforces_minimum_rollouts_before_soft_deadline(
    phase4_ready_state,
) -> None:
    value = 0.0

    def fake_clock() -> float:
        nonlocal value
        value += 0.01
        return value

    result = MonteCarloEvaluator(clock=fake_clock).evaluate(
        phase4_ready_state,
        MonteCarloSettings(
            simulations=10,
            max_depth=5,
            time_budget_ms=1,
            seed=7,
            top_k=1,
            max_candidates=2,
            min_rollouts_per_action=3,
            enforce_min_rollouts=True,
        ),
    )

    assert result.completed_simulations == 3
    assert result.rankings[0].simulations == 3
