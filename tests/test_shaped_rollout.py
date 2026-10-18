"""The shaped rollout policy keeps hands close to finishing and respects the table."""

from __future__ import annotations

import pytest

from src.config.live_layout import LiveLayoutConfig, live_layout_from_dict
from src.logic.monte_carlo import MonteCarloEvaluator, MonteCarloSettings, recommend_phase4
from src.logic.rollout_policies import ShapedRolloutPolicy, build_rollout_policy, hands_to_finish
from src.logic.rules import Play
from src.state.cards import CardSet
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState


@pytest.mark.parametrize("hand, expected", [
    ("3 4 5 6 7", 1),
    ("3 3 3 4", 1),
    ("BJ SJ", 1),
    ("3 3 4 4 5 5", 1),
    ("3 5 7 9", 4),
    ("3 3 3 4 4 4 5 6", 1),
    ("2 2 2 2", 1),
    ("A A K", 2),
    ("3 4 5 6 7 8 9 9", 2),
    ("3 3 3 4 4 4 5 5 7 7", 1),  # plane with pair wings
    ("BJ SJ 2 2 A K K J J 10 10 10 9 9 9 8 8 7 5 4", 8),
])
def test_hands_to_finish_counts_greedy_combinations(hand: str, expected: int) -> None:
    assert hands_to_finish(CardSet.parse(hand).cards) == expected


def _choose(hand: str, previous: str | None = None, *, player=PlayerSeat.SELF, last_player=None,
            landlord=PlayerSeat.SELF, remaining=None) -> Play:
    remaining = remaining or {PlayerSeat.SELF: len(CardSet.parse(hand)), PlayerSeat.RIGHT: 10, PlayerSeat.LEFT: 10}
    return ShapedRolloutPolicy().choose_action(
        player=player, hand=CardSet.parse(hand), previous=Play.parse(previous),
        last_player=last_player, landlord=landlord, remaining_cards=remaining,
    )


def test_leading_plays_the_shape_preserving_combination() -> None:
    assert _choose("3 4 5 6 7 8 9 K").cards == ("3", "4", "5", "6", "7", "8", "9")
    # Keeping the pair of nines instead of a lone nine is equally short and safer.
    assert _choose("3 4 5 6 7 8 9 9").cards == ("3", "4", "5", "6", "7", "8")
    lead = _choose("3 3 5 5 5 K")  # any lead that leaves a one-hand residual
    residual = CardSet.parse("3 3 5 5 5 K").counts
    residual.subtract(lead.cards)
    assert hands_to_finish([rank for rank, count in residual.items() for _ in range(count)]) == 1
    assert _choose("3 3 K K A").cards == ("3", "3")  # smaller pair first


def test_following_plays_the_smallest_sufficient_card_or_passes() -> None:
    assert _choose("5 9 K 2", "7").cards == ("9",)
    # Beating a single 7 would split the straight: pass instead.
    assert _choose("8 9 10 J Q", "7").is_pass
    # Bombs are kept while no enemy is close to finishing.
    assert _choose("8 8 8 8 3", "A").is_pass
    assert _choose("8 8 8 8 3", "A", remaining={PlayerSeat.SELF: 5, PlayerSeat.RIGHT: 1, PlayerSeat.LEFT: 10}).cards == ("8",) * 4


def test_exact_finish_and_partner_pass() -> None:
    assert _choose("5 5 5 9", "4 4 4 3").cards == ("5", "5", "5", "9")
    farmer = _choose("A A 3", "8", player=PlayerSeat.RIGHT, last_player=PlayerSeat.LEFT, landlord=PlayerSeat.SELF,
                     remaining={PlayerSeat.SELF: 10, PlayerSeat.RIGHT: 3, PlayerSeat.LEFT: 5})
    assert farmer.is_pass  # the partner leads and the landlord is not close to finishing
    farmer = _choose("A A 3", "8", player=PlayerSeat.RIGHT, last_player=PlayerSeat.LEFT, landlord=PlayerSeat.SELF,
                     remaining={PlayerSeat.SELF: 2, PlayerSeat.RIGHT: 3, PlayerSeat.LEFT: 5})
    assert not farmer.is_pass


def test_enemy_about_to_finish_changes_the_lead() -> None:
    hand = "3 4 4 K"
    one_card = {PlayerSeat.SELF: 4, PlayerSeat.RIGHT: 1, PlayerSeat.LEFT: 10}
    assert _choose(hand, remaining=one_card).cards == ("4", "4")
    two_cards = {PlayerSeat.SELF: 4, PlayerSeat.RIGHT: 2, PlayerSeat.LEFT: 10}
    assert _choose(hand, remaining=two_cards).cards != ("4", "4")
    assert _choose("3 K", remaining=one_card).cards == ("K",)  # a single must go: the biggest


def test_rollout_policy_is_selectable_and_legal_in_monte_carlo() -> None:
    state = ObservableGameState.from_inputs("3 3 4 5 6 7 8 K A 2", landlord="self")
    settings = MonteCarloSettings(simulations=6, max_depth=40, time_budget_ms=0, max_candidates=8,
                                  rollout_policy="shaped", seed=11)
    result = recommend_phase4(state, settings)
    assert result.action in {evaluation.action for evaluation in result.rankings}
    assert MonteCarloEvaluator(policy=ShapedRolloutPolicy()).policy is not None
    assert type(build_rollout_policy("shaped")).__name__ == "ShapedRolloutPolicy"
    assert type(build_rollout_policy("heuristic")).__name__ == "HeuristicRolloutPolicy"
    with pytest.raises(ValueError):
        build_rollout_policy("random")
    with pytest.raises(ValueError):
        MonteCarloSettings(rollout_policy="random")
    assert live_layout_from_dict({"rollout_policy": "shaped"}).rollout_policy == "shaped"
    assert LiveLayoutConfig().to_payload()["rollout_policy"] == "heuristic"
    with pytest.raises(ValueError):
        LiveLayoutConfig(rollout_policy="none")
