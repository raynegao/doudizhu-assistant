"""
Card model utilities shared across the assistant.
"""

from dataclasses import dataclass
from typing import List

RANK_ORDER: List[str] = [
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
    "2",
    "joker_small",
    "joker_big",
]


@dataclass(frozen=True)
class Card:
    """
    Lightweight representation of a single Dou Dizhu card.
    """

    rank: str


def build_standard_deck() -> List[Card]:
    """
    Construct a complete Dou Dizhu deck.
    """

    pass


def rank_to_id(rank: str) -> int:
    """
    Map a rank string to the YOLO class id.
    """

    pass


def id_to_rank(class_id: int) -> str:
    """
    Convert a YOLO class id back to a rank string.
    """

    pass
