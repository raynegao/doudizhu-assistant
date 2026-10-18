"""The self-play arena plays legal, fully tracked games and reports paired results."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from src.logic.arena import (
    CONFIGURATIONS,
    AgentSpec,
    ArenaError,
    HeuristicAgent,
    MatchSettings,
    MonteCarloAgent,
    RandomAgent,
    RuleAgent,
    deal_cards,
    play_deal,
    play_game,
    relative_seat,
    run_match,
    summarize,
    wilson_interval,
)
from src.logic.monte_carlo import MonteCarloSettings
from src.logic.rules import Play
from src.state.cards import FULL_DECK
from src.state.events import PlayerSeat


def test_relative_seats_follow_the_turn_order() -> None:
    assert relative_seat(1, 1) is PlayerSeat.SELF
    assert relative_seat(2, 1) is PlayerSeat.RIGHT
    assert relative_seat(0, 1) is PlayerSeat.LEFT
    assert relative_seat(0, 2) is PlayerSeat.RIGHT


def test_deal_is_the_full_deck_with_three_extra_landlord_cards() -> None:
    deal = deal_cards(random.Random(7), landlord=2)
    assert [len(hand) for hand in deal.hands] == [17, 17, 20]
    assert Counter(card for hand in deal.hands for card in hand.cards) == Counter(FULL_DECK)
    assert deal_cards(random.Random(7), landlord=2) == deal


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_games_finish_with_consistent_trackers(seed: int) -> None:
    deal = deal_cards(random.Random(seed), landlord=seed % 3)
    record = play_game(deal, [RandomAgent()] * 3, rng=random.Random(seed), game_id=f"g{seed}")
    played = Counter(card for _, cards in record.actions for card in cards)
    dealt = Counter(card for hand in deal.hands for card in hand.cards)
    assert not (played - dealt)  # nothing was played that was not dealt
    assert record.landlord == deal.landlord
    assert record.landlord_won == (record.winner == deal.landlord)
    assert 0 < record.plies <= 300
    assert record.agents == ("random", "random", "random")
    assert sum(record.decisions) == record.plies
    # The winner played every card it was dealt.
    winner_played = Counter(
        card for seat, cards in record.actions if seat == record.winner for card in cards
    )
    assert winner_played == deal.hands[record.winner].counts


def test_illegal_agents_are_caught() -> None:
    class Cheater:
        name = "cheater"

        def choose(self, state, rng):
            return Play.parse("BJ SJ") if "BJ" not in state.self_hand.cards else Play.parse("3 3 3 3 3")

    deal = deal_cards(random.Random(11), landlord=0)
    with pytest.raises(ArenaError):
        play_game(deal, [Cheater()] * 3, rng=random.Random(0))


def test_rule_heuristic_and_monte_carlo_agents_play_legal_games() -> None:
    deal = deal_cards(random.Random(5), landlord=1)
    fast = MonteCarloAgent(
        settings=MonteCarloSettings(simulations=3, max_depth=12, time_budget_ms=0, max_candidates=4),
    )
    record = play_game(deal, [RuleAgent(), HeuristicAgent(), fast], rng=random.Random(5))
    assert record.agents == ("rule", "heuristic", "monte_carlo")
    assert record.winner in range(3)
    assert record.decision_ms[2] > 0.0


def test_match_is_paired_deterministic_and_summarised() -> None:
    settings = MatchSettings(games=2, seed=99, subject=AgentSpec("rule"), opponent=AgentSpec("random"))
    report, deals = run_match(settings)
    again, _ = run_match(settings)
    assert len(deals) == 2 and set(deals[0]) == set(CONFIGURATIONS)
    for deal in deals:
        hands = {deal[name].landlord for name in CONFIGURATIONS}
        assert len(hands) == 1  # same deal in every configuration
    assert report["schema_version"] == "decision-arena-v1"
    assert report["settings"]["subject"] == {"kind": "rule", "settings": {}}
    for name in CONFIGURATIONS:
        block = report["configurations"][name]
        assert block["games"] == 2
        assert 0.0 <= block["landlord_win_rate"] <= 1.0
        assert block["landlord_win_rate_ci95"][0] <= block["landlord_win_rate"] <= block["landlord_win_rate_ci95"][1]
    assert report["configurations"]["opponent_landlord_vs_opponent_farmers"]["subject_mean_decision_ms"] is None
    paired = report["paired_vs_reference"]["subject_as_landlord"]
    assert paired["n"] == 2 and -1.0 <= paired["mean"] <= 1.0
    strip = lambda item: {k: v for k, v in item.items() if "decision_ms" not in k}  # noqa: E731
    assert {k: strip(v) for k, v in report["configurations"].items()} == {
        k: strip(v) for k, v in again["configurations"].items()
    }


def test_agent_specs_rebuild_monte_carlo_settings() -> None:
    agent = AgentSpec("monte_carlo", {"simulations": 5, "max_depth": 9}).build()
    assert isinstance(agent, MonteCarloAgent)
    assert agent.settings.simulations == 5 and agent.settings.max_depth == 9
    assert agent.settings.time_budget_ms == 0
    with pytest.raises(ValueError):
        AgentSpec("nope").build()
    records = play_deal(MatchSettings(games=1, subject=AgentSpec("random"), opponent=AgentSpec("rule")), 0)
    assert set(records) == set(CONFIGURATIONS)
    assert summarize(MatchSettings(games=1), [records])["configurations"]["subject_landlord_vs_opponent_farmers"]["games"] == 1


def test_wilson_interval_is_bounded_and_centred() -> None:
    low, high = wilson_interval(50, 100)
    assert 0.40 < low < 0.5 < high < 0.60
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(10, 10)
    assert low < 0.75 and 0.99 < high <= 1.0
