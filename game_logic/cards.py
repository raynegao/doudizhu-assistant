"""
Card model utilities shared across the assistant.
"""

from dataclasses import dataclass
from typing import Dict, List

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

RANK_TO_ID: Dict[str, int] = {rank: idx for idx, rank in enumerate(RANK_ORDER)}
ID_TO_RANK: Dict[int, str] = {idx: rank for rank, idx in RANK_TO_ID.items()}


@dataclass(frozen=True)
class Card:
    """
    Lightweight representation of a single Dou Dizhu card.
    """

    rank: str


def build_standard_deck() -> List[Card]:
    """
    Construct a complete Dou Dizhu deck (54 cards).
    """

    deck: List[Card] = []
    for rank in RANK_ORDER:
        count = 4
        if rank in {"joker_small", "joker_big"}:
            count = 1
        deck.extend(Card(rank) for _ in range(count))
    return deck


def rank_to_id(rank: str) -> int:
    """
    Map a rank string to the YOLO class id.
    """

    if rank not in RANK_TO_ID:
        raise ValueError(f"Unknown rank: {rank}")
    return RANK_TO_ID[rank]


def id_to_rank(class_id: int) -> str:
    """
    Convert a YOLO class id back to a rank string.
    """

    if class_id not in ID_TO_RANK:
        raise ValueError(f"Unknown class id: {class_id}")
    return ID_TO_RANK[class_id]
