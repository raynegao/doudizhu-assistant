"""
Translate YOLO detections into structured Dou Dizhu game state.
"""

from dataclasses import dataclass, field
from statistics import median
from typing import List, Sequence, Tuple

from .cards import Card, id_to_rank

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

    rows = group_detections_by_row(detections)
    if not rows:
        return GameState()

    # 默认认为最下方一行是自己的手牌
    my_row = rows[-1]
    my_hand = detections_to_cards(my_row)

    left_opponent_count = 0
    right_opponent_count = 0
    if len(rows) == 2:
        # 只有一行对手信息，计为“左侧”
        left_opponent_count = len(rows[0])
    elif len(rows) >= 3:
        # 上方两行分别视为左右对手
        left_opponent_count = len(rows[0])
        right_opponent_count = len(rows[1])

    return GameState(
        my_hand=my_hand,
        left_opponent_count=left_opponent_count,
        right_opponent_count=right_opponent_count,
        last_play=[],
        history=[],
    )


def group_detections_by_row(detections: Sequence[Detection]) -> List[List[Detection]]:
    """
    Separate detections into logical rows to identify each player's hand.
    """

    if not detections:
        return []

    # 按 y 排序后分组，阈值基于框高的中位数
    dets = sorted(detections, key=lambda d: d[1])  # y_center
    heights = [d[3] for d in dets]  # h
    typical_h = median(heights) if heights else 40.0
    row_gap = max(40.0, typical_h * 1.2)

    rows: List[List[Detection]] = []
    current_row: List[Detection] = []
    current_row_y = dets[0][1]

    for det in dets:
        y_center = det[1]
        if abs(y_center - current_row_y) > row_gap and current_row:
            rows.append(current_row)
            current_row = []
            current_row_y = y_center
        current_row.append(det)
    if current_row:
        rows.append(current_row)

    # 按 y 从小到大排序行；行内按 x 排序便于稳定转换
    rows = sorted(rows, key=lambda row: sum(d[1] for d in row) / len(row))
    for row in rows:
        row.sort(key=lambda d: d[0])  # x_center
    return rows


def detections_to_cards(detections: Sequence[Detection]) -> List[Card]:
    """
    Convert grouped detections to Card objects.
    """

    cards: List[Card] = []
    for det in detections:
        _, _, _, _, cls_id, _ = det
        try:
            rank = id_to_rank(cls_id)
        except ValueError:
            continue
        cards.append(Card(rank))
    return cards
