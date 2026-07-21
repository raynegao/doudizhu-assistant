from __future__ import annotations

import json
import os
import subprocess
import sys

from src.logic.rules import Play, PlayType, can_beat, classify_play, legal_actions
from src.state.cards import CardSet


def assert_play(text: str, play_type: PlayType, main_rank: str | None = None) -> None:
    play = classify_play(CardSet.parse(text).cards)
    assert play.type is play_type
    assert play.main_rank == main_rank


def test_classifies_basic_play_types() -> None:
    assert_play("7", PlayType.SINGLE, "7")
    assert_play("8 8", PlayType.PAIR, "8")
    assert_play("9 9 9", PlayType.TRIPLE, "9")
    assert_play("4 4 4 6", PlayType.TRIPLE_SINGLE, "4")
    assert_play("5 5 5 7 7", PlayType.TRIPLE_PAIR, "5")
    assert_play("3 4 5 6 7", PlayType.STRAIGHT, "7")
    assert_play("10 J Q K A", PlayType.STRAIGHT, "A")
    assert_play("3 3 4 4 5 5", PlayType.PAIR_STRAIGHT, "5")
    assert_play("6 6 6 6 3 4", PlayType.FOUR_TWO_SINGLE, "6")
    assert_play("6 6 6 6 3 3 4 4", PlayType.FOUR_TWO_PAIR, "6")
    assert_play("6 6 6 6", PlayType.BOMB, "6")
    assert_play("SJ BJ", PlayType.ROCKET, "BJ")


def test_rejects_straight_with_two() -> None:
    assert classify_play(CardSet.parse("J Q K A 2").cards).type is PlayType.INVALID


def test_can_beat_same_type_and_rank() -> None:
    assert can_beat(Play.parse("8"), Play.parse("7"))
    assert not can_beat(Play.parse("8 8"), Play.parse("7"))


def test_bomb_and_rocket_rules() -> None:
    assert can_beat(Play.parse("6 6 6 6"), Play.parse("A A A"))
    assert can_beat(Play.parse("SJ BJ"), Play.parse("2 2 2 2"))
    assert not can_beat(Play.parse("2 2 2 2"), Play.parse("SJ BJ"))


def test_legal_actions_when_leading_do_not_force_pass() -> None:
    hand = CardSet.parse("3 3 4 4 5 5 6 6 7 8 9 SJ BJ")
    actions = legal_actions(hand, Play.parse(""))
    action_text = {str(action) for action in actions}
    assert "pass" not in action_text
    assert {"3", "3 3", "3 4 5 6 7", "3 3 4 4 5 5", "SJ BJ"} <= action_text


def test_legal_actions_when_responding_include_only_beating_actions_and_pass() -> None:
    hand = CardSet.parse("3 3 4 4 5 5 6 6 7 8 9 SJ BJ")
    actions = legal_actions(hand, Play.parse("5 5"))
    action_text = {str(action) for action in actions}
    assert "6 6" in action_text
    assert "pass" in action_text
    assert "SJ BJ" in action_text
    assert "3 3" not in action_text
    assert "4 4" not in action_text
    assert "5 5" not in action_text


def test_legal_actions_can_only_pass_when_no_action_beats_previous() -> None:
    hand = CardSet.parse("3 4 5 6 7")
    actions = legal_actions(hand, Play.parse("9 9"))
    assert [str(action) for action in actions] == ["pass"]


def test_legal_actions_include_four_with_two_variants() -> None:
    hand = CardSet.parse("6 6 6 6 3 3 4 4 5")
    action_text = {str(action) for action in legal_actions(hand, Play.parse(""))}
    assert "3 4 6 6 6 6" in action_text
    assert "3 3 4 4 6 6 6 6" in action_text


def test_legal_action_order_is_stable_across_hash_seeds() -> None:
    code = (
        "import json; "
        "from src.logic.rules import Play, legal_actions; "
        "from src.state.cards import CardSet; "
        "print(json.dumps([str(a) for a in legal_actions("
        "CardSet.parse('3 3 4 4 5 5 6 6 7 8 9 SJ BJ'), Play.parse(''))]))"
    )
    outputs = []
    for seed in ("1", "99"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
