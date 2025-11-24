"""
Rule engine helpers for Dou Dizhu card classification and comparison.
"""

from collections import Counter
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .cards import Card, RANK_ORDER, RANK_TO_ID


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

    n = len(cards)
    if n == 0:
        return HandType.INVALID

    ranks = [c.rank for c in cards]
    counts = Counter(ranks)
    count_values = sorted(counts.values(), reverse=True)

    # Rocket
    if n == 2 and set(ranks) == {"joker_small", "joker_big"}:
        return HandType.ROCKET

    # Bomb
    if n == 4 and 4 in count_values:
        return HandType.BOMB

    # Basic types
    if n == 1:
        return HandType.SINGLE
    if n == 2 and count_values == [2]:
        return HandType.PAIR
    if n == 3 and count_values == [3]:
        return HandType.TRIPLE
    if n == 4 and count_values == [3, 1]:
        return HandType.TRIPLE_WITH_SINGLE
    if n == 5 and count_values == [3, 2]:
        return HandType.TRIPLE_WITH_PAIR

    # Sequences
    unique_ranks = list(counts.keys())
    if n >= 5 and len(unique_ranks) == n and _is_straight(unique_ranks):
        return HandType.STRAIGHT

    if n >= 6 and n % 2 == 0 and all(v == 2 for v in counts.values()) and _is_straight(unique_ranks):
        return HandType.DOUBLE_SEQUENCE

    # Airplane (pure triple chain)
    if n >= 6 and n % 3 == 0 and all(v == 3 for v in counts.values()) and _is_straight(unique_ranks):
        return HandType.AIRPLANE

    # Airplane with single wings (简单版)
    triple_ranks = [r for r, c in counts.items() if c == 3]
    if triple_ranks:
        triple_ranks_sorted = sorted(triple_ranks, key=lambda r: RANK_TO_ID[r])
        triple_count = len(triple_ranks_sorted)
        singles_count = sum(1 for c in counts.values() if c == 1)
        pairs_count = sum(1 for c in counts.values() if c == 2)
        if (
            triple_count >= 2
            and _is_straight(triple_ranks_sorted)
            and n == triple_count * 4
            and singles_count == triple_count
        ):
            return HandType.AIRPLANE_WITH_WINGS
        if (
            triple_count >= 2
            and _is_straight(triple_ranks_sorted)
            and n == triple_count * 5
            and pairs_count == triple_count
        ):
            return HandType.AIRPLANE_WITH_WINGS

    return HandType.INVALID


def can_beat(prev: List[Card], current: List[Card]) -> bool:
    """
    Evaluate whether the current play beats the previous play.
    """

    prev_type = classify_hand(prev)
    curr_type = classify_hand(current)
    if curr_type == HandType.INVALID:
        return False
    if prev_type == HandType.INVALID:
        return True

    # Rocket beats everything
    if curr_type == HandType.ROCKET:
        return True
    if prev_type == HandType.ROCKET:
        return False

    # Bomb beats non-bomb
    if curr_type == HandType.BOMB and prev_type != HandType.BOMB:
        return True
    if prev_type == HandType.BOMB and curr_type != HandType.BOMB:
        return False

    # Type/length must match for the rest
    if curr_type != prev_type or len(prev) != len(current):
        return False

    prev_key = _hand_key(prev, prev_type)
    curr_key = _hand_key(current, curr_type)
    return curr_key > prev_key


def generate_all_legal_hands(hand: List[Card], prev: Optional[List[Card]] = None) -> List[List[Card]]:
    """
    Enumerate every legal play from the current hand.
    """

    counts = Counter([c.rank for c in hand])
    candidates: List[List[Card]] = []

    def add_cards(rank: str, num: int) -> List[Card]:
        return [Card(rank) for _ in range(num)]

    # Singles, pairs, triples, bombs
    for rank, cnt in counts.items():
        if cnt >= 1:
            candidates.append(add_cards(rank, 1))
        if cnt >= 2:
            candidates.append(add_cards(rank, 2))
        if cnt >= 3:
            candidates.append(add_cards(rank, 3))
        if cnt == 4:
            candidates.append(add_cards(rank, 4))

    # Rocket
    if counts.get("joker_small", 0) >= 1 and counts.get("joker_big", 0) >= 1:
        candidates.append([Card("joker_small"), Card("joker_big")])

    # Triple with attachments (简单版：单牌或对子)
    singles = [r for r, cnt in counts.items() if cnt >= 1]
    pairs = [r for r, cnt in counts.items() if cnt >= 2]
    triples = [r for r, cnt in counts.items() if cnt >= 3]
    for t in triples:
        for s in singles:
            if s == t:
                continue
            candidates.append(add_cards(t, 3) + add_cards(s, 1))
        for p in pairs:
            if p == t:
                continue
            candidates.append(add_cards(t, 3) + add_cards(p, 2))

    # Straights (5~12)
    straights = _enumerate_straights(counts, min_len=5, max_len=12, need_pairs=False)
    candidates.extend(straights)

    # Double sequences (6 张起且为偶数)
    double_straights = _enumerate_straights(counts, min_len=6, max_len=20, need_pairs=True)
    candidates.extend(double_straights)

    # Airplane (纯三顺)
    triple_chain = _enumerate_triple_chains(counts)
    candidates.extend(triple_chain)

    # 依据上家出牌过滤
    if prev is None:
        return candidates

    legal = [c for c in candidates if can_beat(prev, c)]
    # 排序：按牌数升序，再按关键牌点大小
    legal.sort(key=lambda cards: (len(cards), _hand_key(cards, classify_hand(cards))))
    return legal


# --------- 辅助函数 --------- #


def _is_straight(ranks: List[str]) -> bool:
    """判断是否为连续点数（不含 2/joker）。"""

    if not ranks:
        return False
    ids = sorted(RANK_TO_ID[r] for r in ranks)
    if any(rid >= RANK_TO_ID["2"] for rid in ids):
        return False
    return ids == list(range(ids[0], ids[0] + len(ids)))


def _hand_key(cards: List[Card], hand_type: HandType) -> int:
    """用于比较大小的关键点值（越大越大）。"""

    ranks = [c.rank for c in cards]
    counts = Counter(ranks)
    if hand_type in {HandType.SINGLE, HandType.PAIR, HandType.TRIPLE, HandType.BOMB}:
        key_rank = max(counts.items(), key=lambda kv: (kv[1], RANK_TO_ID[kv[0]]))[0]
        return RANK_TO_ID[key_rank]
    if hand_type == HandType.ROCKET:
        return RANK_TO_ID["joker_big"]
    if hand_type in {HandType.TRIPLE_WITH_SINGLE, HandType.TRIPLE_WITH_PAIR}:
        triple_rank = max((r for r, c in counts.items() if c == 3), key=lambda r: RANK_TO_ID[r])
        return RANK_TO_ID[triple_rank]
    if hand_type in {HandType.STRAIGHT, HandType.DOUBLE_SEQUENCE, HandType.AIRPLANE, HandType.AIRPLANE_WITH_WINGS}:
        seq_ranks = sorted((r for r, c in counts.items() if c >= (2 if hand_type == HandType.DOUBLE_SEQUENCE else 1)), key=lambda r: RANK_TO_ID[r])
        return RANK_TO_ID[seq_ranks[-1]]
    return -1


def _enumerate_straights(counts: Dict[str, int], min_len: int, max_len: int, need_pairs: bool) -> List[List[Card]]:
    """枚举顺子/连对。need_pairs 为 True 时表示连对。"""

    usable = [r for r in RANK_ORDER if r not in {"2", "joker_small", "joker_big"}]
    available = [r for r in usable if counts.get(r, 0) >= (2 if need_pairs else 1)]
    ids = [RANK_TO_ID[r] for r in available]
    straights: List[List[Card]] = []
    i = 0
    while i < len(ids):
        start = i
        while i + 1 < len(ids) and ids[i + 1] == ids[i] + 1:
            i += 1
        segment = available[start : i + 1]
        if len(segment) >= min_len:
            for length in range(min_len, min(len(segment), max_len) + 1):
                for j in range(0, len(segment) - length + 1):
                    seq = segment[j : j + length]
                    cards = [Card(r) for r in seq for _ in range(2 if need_pairs else 1)]
                    straights.append(cards)
        i += 1
    return straights


def _enumerate_triple_chains(counts: Dict[str, int]) -> List[List[Card]]:
    usable = [r for r in RANK_ORDER if r not in {"2", "joker_small", "joker_big"}]
    available = [r for r in usable if counts.get(r, 0) >= 3]
    ids = [RANK_TO_ID[r] for r in available]
    chains: List[List[Card]] = []
    i = 0
    while i < len(ids):
        start = i
        while i + 1 < len(ids) and ids[i + 1] == ids[i] + 1:
            i += 1
        segment = available[start : i + 1]
        if len(segment) >= 2:
            for length in range(2, len(segment) + 1):
                for j in range(0, len(segment) - length + 1):
                    seq = segment[j : j + length]
                    cards = [Card(r) for r in seq for _ in range(3)]
                    chains.append(cards)
        i += 1
    return chains
