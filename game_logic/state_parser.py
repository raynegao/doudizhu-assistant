"""
Translate YOLO detections into structured Dou Dizhu game state.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .cards import Card

Detection = Tuple[float, float, float, float, int, float]


@dataclass
class GameState:
    """
    Snapshot of the current Dou Dizhu game.
    """

    my_hand: List[Card] = field(default_factory=list)
    left_opponent_count: int = 0
    right_opponent_count: int = 0
    last_play: List[Card] = field(default_factory=list)
    history: List[List[Card]] = field(default_factory=list)


def parse_game_state(detections: Sequence[Detection]) -> GameState:
    """
    Build a GameState from YOLO detections.
    """

    pass


def group_detections_by_row(detections: Sequence[Detection]) -> List[List[Detection]]:
    """
    Separate detections into logical rows to identify each player's hand.
    """

    pass


def detections_to_cards(detections: Sequence[Detection]) -> List[Card]:
    """
    Convert grouped detections to Card objects.
    """

    pass
