from __future__ import annotations

import pytest

from src.state.cards import CardParseError, CardSet, parse_cards


def test_parse_cards_normalizes_case_and_aliases() -> None:
    assert parse_cards(" 3 10 j Q k A 2 sj BJ ") == (
        "3",
        "10",
        "J",
        "Q",
        "K",
        "A",
        "2",
        "SJ",
        "BJ",
    )


def test_parse_cards_supports_compact_input() -> None:
    assert parse_cards("334455") == ("3", "3", "4", "4", "5", "5")


def test_card_set_rejects_too_many_rank_cards() -> None:
    with pytest.raises(CardParseError):
        CardSet.parse("3 3 3 3 3")


def test_card_set_rejects_duplicate_joker() -> None:
    with pytest.raises(CardParseError):
        CardSet.parse("BJ BJ")


def test_card_set_rejects_unknown_rank() -> None:
    with pytest.raises(CardParseError):
        CardSet.parse("1 X")
