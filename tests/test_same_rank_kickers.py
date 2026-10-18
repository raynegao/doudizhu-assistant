"""Four-with-two singles and plane wings may share a rank, as the real client allows.

Independent acceptance batch 2026-10-17: session 005 played ``A A A A J J``
and session 002's right seat played ``6 6 6 5 5 5 3 3``. The game accepted
both, the frozen engine typed neither, and the sealing/replay toolchain then
refused the games outright.
"""

from __future__ import annotations

import pytest

from src.logic.rules import Play, PlayType, can_beat, classify_play, legal_actions
from src.state.cards import CardSet


def _play(text: str) -> Play:
    return classify_play(CardSet.parse(text).cards)


def test_four_with_two_accepts_a_same_rank_kicker_pair() -> None:
    play = _play("A A A A J J")
    assert play.type is PlayType.FOUR_TWO_SINGLE
    assert play.main_rank == "A"
    assert play.length == 6 and play.combo_size == 1


def test_plane_wings_may_share_a_rank() -> None:
    play = _play("6 6 6 5 5 5 3 3")
    assert play.type is PlayType.PLANE_SINGLE
    assert play.main_rank == "6"
    assert play.length == 8 and play.combo_size == 2
    three = _play("3 3 3 4 4 4 5 5 5 7 7 8")
    assert three.type is PlayType.PLANE_SINGLE and three.combo_size == 3


def test_same_rank_kickers_compare_like_their_distinct_rank_forms() -> None:
    assert can_beat(_play("2 2 2 2 3 3"), _play("A A A A J J"))
    assert not can_beat(_play("K K K K Q Q"), _play("A A A A J 10"))
    assert can_beat(_play("7 7 7 8 8 8 4 4"), _play("6 6 6 5 5 5 3 3"))
    assert can_beat(_play("9 9 9 9"), _play("A A A A J J"))
    assert not can_beat(_play("A A A A J J"), _play("3 3 3 3"))


@pytest.mark.parametrize("text", [
    "A A A A 3 3 4",        # seven cards
    "6 6 6 6 3 3 3 3",      # two bombs are not four-with-two
    "3 3 3 4 4 4 5 5 5 5",  # a four-card kicker group does not make wings
    "3 3 3 5 5 5 7 7",      # non-consecutive triples
])
def test_unrelated_shapes_stay_invalid(text: str) -> None:
    assert _play(text).type is PlayType.INVALID


def test_pair_kicker_forms_are_generated_as_legal_actions() -> None:
    hand = CardSet.parse("A A A A J J 3")
    plays = {play.cards for play in legal_actions(hand)}
    assert tuple(CardSet.parse("A A A A J J").cards) in plays
    assert tuple(CardSet.parse("A A A A J 3").cards) in plays
    wings = CardSet.parse("6 6 6 5 5 5 3 3 K")
    wing_plays = {play.cards for play in legal_actions(wings)}
    assert tuple(CardSet.parse("6 6 6 5 5 5 3 3").cards) in wing_plays
    assert tuple(CardSet.parse("6 6 6 5 5 5 3 K").cards) in wing_plays
    # A bomb can only answer a four-with-two with a bigger bomb or a bigger four-with-two.
    answers = legal_actions(CardSet.parse("2 2 2 2 9 9"), _play("A A A A J J"))
    assert tuple(CardSet.parse("2 2 2 2 9 9").cards) in {p.cards for p in answers}
