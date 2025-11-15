"""
Rule engine helpers for Dou Dizhu card classification and comparison.
"""

from enum import Enum, auto
from typing import List, Optional

from .cards import Card


class HandType(Enum):
    """
    Enumerates the supported Dou Dizhu hand categories.
    """

    INVALID = auto()
    SINGLE = auto()
    PAIR = auto()
    TRIPLE = auto()
    TRIPLE_WITH_SINGLE = auto()
    TRIPLE_WITH_PAIR = auto()
    STRAIGHT = auto()
    DOUBLE_SEQUENCE = auto()
    AIRPLANE = auto()
    AIRPLANE_WITH_WINGS = auto()
    BOMB = auto()
    ROCKET = auto()


def classify_hand(cards: List[Card]) -> HandType:
    """
    Determine the hand type represented by the given cards.
    """

    pass


def can_beat(prev: List[Card], current: List[Card]) -> bool:
    """
    Evaluate whether the current play beats the previous play.
    """

    pass


def generate_all_legal_hands(hand: List[Card], prev: Optional[List[Card]] = None) -> List[List[Card]]:
    """
    Enumerate every legal play from the current hand.
    """

    pass
