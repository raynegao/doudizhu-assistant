from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from src.logic.monte_carlo import (
    ActionEvaluation,
    HeuristicRolloutPolicy,
    MonteCarloEvaluator,
    MonteCarloSettings,
    RolloutOutcome,
    _aggregate_action,
    _candidate_prescore,
    _simulate_candidate,
    recommend_phase4,
)
from src.logic.opponent_model import OpponentDeal, UniformOpponentModel
from src.logic.rules import Play, legal_actions
from src.state.cards import FULL_DECK, CardSet
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
    assert [item.strategy_score for item in first.rankings] == [
        item.strategy_score for item in second.rankings
    ]
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
    assert result.rankings[0].win_rate_ci95_low == 1.0
    assert result.rankings[0].win_rate_ci95_high == 1.0
    assert result.rankings[0].win_rate_interval_method == "rule_proven_finish"


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
    deal = OpponentDeal(
        (
            (PlayerSeat.RIGHT, CardSet.parse("4")),
            (PlayerSeat.LEFT, CardSet.parse("SJ")),
        )
    )

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
    deal = OpponentDeal(
        (
            (PlayerSeat.RIGHT, CardSet.parse("5")),
            (PlayerSeat.LEFT, CardSet.parse("6")),
        )
    )
    outcomes = []
    for landlord in (PlayerSeat.SELF, PlayerSeat.RIGHT):
        state = ObservableGameState.from_inputs(
            "3 4",
            landlord=landlord,
            remaining_cards={"self": 2, "right": 1, "left": 1},
            played_cards=pool,
        )
        outcomes.append(
            _simulate_candidate(
                state,
                deal,
                Play.parse("3"),
                max_depth=1,
                policy=HeuristicRolloutPolicy(),
            )
        )

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


def _aggregate_for_control_test(hand: str, action: str) -> ActionEvaluation:
    state = ObservableGameState.from_inputs(hand, landlord="self")
    return _aggregate_action(
        state,
        Play.parse(action),
        (RolloutOutcome(0.5, None, 1, 0.0),),
        global_risks=(),
    )


@pytest.mark.parametrize("count", (1, 2, 8))
@pytest.mark.parametrize("result", (0.0, 1.0))
def test_finite_identical_terminal_samples_do_not_imply_certainty(
    count: int, result: float,
) -> None:
    state = ObservableGameState.from_inputs("3 4", landlord="self")
    evaluation = _aggregate_action(
        state,
        Play.parse("3"),
        (RolloutOutcome(result, result, 2, 0.0),) * count,
        global_risks=(),
    )

    assert evaluation.estimated_win_rate == result
    assert evaluation.win_rate_interval_method == "wilson"
    assert evaluation.win_rate_estimate_kind == "terminal_rollout_win_rate"
    assert evaluation.win_rate_ci95_low is not None
    assert evaluation.win_rate_ci95_high is not None
    assert evaluation.win_rate_ci95_low < evaluation.win_rate_ci95_high
    if result == 1.0:
        assert 0.0 < evaluation.win_rate_ci95_low < 1.0
        assert evaluation.win_rate_ci95_high == 1.0
    else:
        assert evaluation.win_rate_ci95_low == 0.0
        assert 0.0 < evaluation.win_rate_ci95_high < 1.0


def test_mixed_binary_samples_use_known_wilson_interval() -> None:
    evaluation = _aggregate_action(
        ObservableGameState.from_inputs("3 4", landlord="self"),
        Play.parse("3"),
        (RolloutOutcome(0.0, 0.0, 2, 0.0), RolloutOutcome(1.0, 1.0, 2, 0.0)),
        global_risks=(),
    )

    assert evaluation.estimated_win_rate == 0.5
    assert evaluation.win_rate_ci95_low == pytest.approx(0.094531, abs=1e-6)
    assert evaluation.win_rate_ci95_high == pytest.approx(0.905469, abs=1e-6)


@pytest.mark.parametrize(
    "outcomes",
    (
        (RolloutOutcome(0.5, None, 1, 0.0),),
        (RolloutOutcome(1.0, 1.0, 2, 0.0), RolloutOutcome(0.8, None, 1, 0.0)),
        (RolloutOutcome(1.0, None, 1, 0.0),) * 8,
        (RolloutOutcome(0.0, None, 1, 0.0),) * 8,
    ),
)
def test_truncated_scores_use_non_degenerate_bounded_mean_interval(
    outcomes: tuple[RolloutOutcome, ...],
) -> None:
    evaluation = _aggregate_action(
        ObservableGameState.from_inputs("3 4", landlord="self"),
        Play.parse("3"),
        outcomes,
        global_risks=(),
    )

    assert evaluation.estimated_win_rate == pytest.approx(
        sum(value.expected_result for value in outcomes) / len(outcomes)
    )
    assert evaluation.win_rate_interval_method == "hoeffding_anytime"
    assert evaluation.win_rate_estimate_kind == "bounded_rollout_score"
    assert "heuristic_score_not_win_probability" in evaluation.risk_flags
    assert evaluation.win_rate_ci95_low is not None
    assert evaluation.win_rate_ci95_high is not None
    assert evaluation.win_rate_ci95_low < evaluation.win_rate_ci95_high
    payload = evaluation.to_log_payload()
    assert payload["win_rate_estimate_kind"] == "bounded_rollout_score"


def test_budget_stopping_uses_anytime_bound_even_for_terminal_outcomes() -> None:
    evaluation = _aggregate_action(
        ObservableGameState.from_inputs("3 4", landlord="self"),
        Play.parse("3"),
        (RolloutOutcome(1.0, 1.0, 2, 0.0),) * 8,
        global_risks=("time_budget_exhausted: completed 8 of 200 sampled worlds",),
    )

    assert evaluation.win_rate_interval_method == "hoeffding_anytime"
    assert evaluation.win_rate_ci95_low is not None
    assert 0.0 < evaluation.win_rate_ci95_low < 1.0


def test_only_a_legal_full_hand_finish_can_have_rule_proven_interval() -> None:
    state = ObservableGameState.from_inputs(
        "3", landlord="self", last_play="4", last_player="left",
    )
    evaluation = _aggregate_action(
        state, Play.parse("3"), (RolloutOutcome(1.0, 1.0, 1, 0.0),), global_risks=(),
    )

    assert "exact_finish" not in evaluation.reason_factors
    assert evaluation.win_rate_interval_method == "wilson"
    assert evaluation.win_rate_ci95_low != 1.0


def test_depth_limited_decision_exposes_score_warning(
    phase4_ready_state: ObservableGameState,
) -> None:
    result = recommend_phase4(phase4_ready_state, _settings(max_depth=1))

    assert "heuristic_score_not_win_probability" in result.warnings
    assert "模拟评分" in result.reason
    assert all(
        item.win_rate_estimate_kind == "bounded_rollout_score" for item in result.rankings
    )


@pytest.mark.parametrize(
    ("hand", "action"),
    (
        ("3 9 10 10 10 10 K", "3 9 10 10 10 10"),
        ("3 4 7 7 7 7", "7"),
        ("3 4 7 7 7 7", "7 7"),
        ("3 4 7 7 7 7", "7 7 7"),
        ("3 3 3 4 4 4 5 SJ BJ", "3 3 3 4 4 4 SJ BJ"),
    ),
    ids=(
        "four-with-two-consumes-bomb",
        "single-splits-bomb",
        "pair-splits-bomb",
        "triple-splits-bomb",
        "jokers-used-as-plane-wings",
    ),
)
def test_phase4_marks_control_commitment_from_residual_hand(
    hand: str,
    action: str,
) -> None:
    evaluation = _aggregate_for_control_test(hand, action)

    assert "preserves_bomb_or_rocket" not in evaluation.reason_factors
    assert "bomb_commitment" in evaluation.risk_flags


@pytest.mark.parametrize(
    ("hand", "action"),
    (
        ("3 4 7 7 7 7", "3"),
        ("3 4 SJ BJ", "3"),
    ),
    ids=("preserves-bomb", "preserves-rocket"),
)
def test_phase4_marks_control_preserved_only_when_asset_remains(
    hand: str,
    action: str,
) -> None:
    evaluation = _aggregate_for_control_test(hand, action)

    assert "preserves_bomb_or_rocket" in evaluation.reason_factors
    assert "bomb_commitment" not in evaluation.risk_flags


def test_phase4_does_not_claim_control_preservation_without_control_asset() -> None:
    evaluation = _aggregate_for_control_test("3 4 5", "3")

    assert evaluation.strategy_score == pytest.approx(0.458333)
    assert "preserves_bomb_or_rocket" not in evaluation.reason_factors
    assert "bomb_commitment" not in evaluation.risk_flags


def test_phase4_four_with_two_does_not_receive_control_bonus() -> None:
    evaluation = _aggregate_for_control_test(
        "3 9 10 10 10 10 K",
        "3 9 10 10 10 10",
    )

    assert evaluation.strategy_score == pytest.approx(0.535714)


@pytest.mark.parametrize(
    ("hand", "action", "expected"),
    (
        ("3 9 10 10 10 10 K", "3 9 10 10 10 10", 0.814286),
        ("3 4 7 7 7 7", "3", 0.44),
        ("3 4 5", "3", 0.35),
    ),
    ids=("consumes-control", "preserves-control", "no-control-to-preserve"),
)
def test_phase4_candidate_prescore_uses_residual_control_assets(
    hand: str,
    action: str,
    expected: float,
) -> None:
    state = ObservableGameState.from_inputs(hand, landlord="self")

    assert _candidate_prescore(state, Play.parse(action)) == pytest.approx(expected)
