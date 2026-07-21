from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Iterable

from src.state.cards import CHAIN_RANKS, RANK_VALUE, CardSet, parse_cards, sort_cards


class PlayType(str, Enum):
    PASS = "pass"
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    TRIPLE_SINGLE = "triple_single"
    TRIPLE_PAIR = "triple_pair"
    STRAIGHT = "straight"
    PAIR_STRAIGHT = "pair_straight"
    PLANE = "plane"
    PLANE_SINGLE = "plane_single"
    PLANE_PAIR = "plane_pair"
    FOUR_TWO_SINGLE = "four_two_single"
    FOUR_TWO_PAIR = "four_two_pair"
    BOMB = "bomb"
    ROCKET = "rocket"
    INVALID = "invalid"


@dataclass(frozen=True)
class Play:
    cards: tuple[str, ...]
    type: PlayType
    main_rank: str | None = None
    length: int = 0
    combo_size: int = 0

    @classmethod
    def parse(cls, value: str | Iterable[str] | None) -> "Play":
        return classify_play(parse_cards(value))

    @property
    def is_valid(self) -> bool:
        return self.type is not PlayType.INVALID

    @property
    def is_pass(self) -> bool:
        return self.type is PlayType.PASS

    def to_list(self) -> list[str]:
        return list(self.cards)

    def __str__(self) -> str:
        return " ".join(self.cards) if self.cards else "pass"


def _rank_groups(cards: tuple[str, ...]) -> Counter[str]:
    return Counter(cards)


def _is_consecutive(ranks: list[str]) -> bool:
    values = [RANK_VALUE[rank] for rank in ranks]
    return all(next_value == value + 1 for value, next_value in zip(values, values[1:]))


def _chain_main_rank(ranks: Iterable[str]) -> str:
    return max(ranks, key=RANK_VALUE.__getitem__)


def classify_play(cards: Iterable[str]) -> Play:
    normalized = sort_cards(cards)
    size = len(normalized)
    if size == 0:
        return Play(normalized, PlayType.PASS)

    counts = _rank_groups(normalized)
    count_values = sorted(counts.values(), reverse=True)
    ranks_by_count = {
        count: sorted([rank for rank, value in counts.items() if value == count], key=RANK_VALUE.__getitem__)
        for count in set(counts.values())
    }

    if size == 1:
        return Play(normalized, PlayType.SINGLE, normalized[0], size, 1)

    if size == 2:
        if set(normalized) == {"SJ", "BJ"}:
            return Play(normalized, PlayType.ROCKET, "BJ", size, 1)
        if count_values == [2]:
            return Play(normalized, PlayType.PAIR, normalized[0], size, 1)

    if size == 3 and count_values == [3]:
        return Play(normalized, PlayType.TRIPLE, normalized[0], size, 1)

    if size == 4:
        if count_values == [4]:
            return Play(normalized, PlayType.BOMB, normalized[0], size, 1)
        if count_values == [3, 1]:
            return Play(normalized, PlayType.TRIPLE_SINGLE, ranks_by_count[3][0], size, 1)

    if size == 5 and count_values == [3, 2]:
        return Play(normalized, PlayType.TRIPLE_PAIR, ranks_by_count[3][0], size, 1)

    if size == 6 and count_values == [4, 1, 1]:
        return Play(normalized, PlayType.FOUR_TWO_SINGLE, ranks_by_count[4][0], size, 1)

    if size == 8 and count_values == [4, 2, 2]:
        return Play(normalized, PlayType.FOUR_TWO_PAIR, ranks_by_count[4][0], size, 1)

    if _is_straight_counts(counts):
        return Play(normalized, PlayType.STRAIGHT, _chain_main_rank(counts), size, size)

    if _is_pair_straight_counts(counts):
        pair_ranks = list(counts.keys())
        return Play(normalized, PlayType.PAIR_STRAIGHT, _chain_main_rank(pair_ranks), size, len(pair_ranks))

    plane = _classify_plane(normalized, counts)
    if plane:
        return plane

    return Play(normalized, PlayType.INVALID)


def _is_straight_counts(counts: Counter[str]) -> bool:
    ranks = sorted(counts.keys(), key=RANK_VALUE.__getitem__)
    return (
        len(ranks) >= 5
        and all(count == 1 for count in counts.values())
        and all(rank in CHAIN_RANKS for rank in ranks)
        and _is_consecutive(ranks)
    )


def _is_pair_straight_counts(counts: Counter[str]) -> bool:
    ranks = sorted(counts.keys(), key=RANK_VALUE.__getitem__)
    return (
        len(ranks) >= 3
        and all(count == 2 for count in counts.values())
        and all(rank in CHAIN_RANKS for rank in ranks)
        and _is_consecutive(ranks)
    )


def _classify_plane(cards: tuple[str, ...], counts: Counter[str]) -> Play | None:
    triple_ranks = sorted([rank for rank, count in counts.items() if count == 3], key=RANK_VALUE.__getitem__)
    if len(triple_ranks) < 2 or not all(rank in CHAIN_RANKS for rank in triple_ranks):
        return None
    if not _is_consecutive(triple_ranks):
        return None

    triple_count = len(triple_ranks)
    size = len(cards)
    main_rank = _chain_main_rank(triple_ranks)
    if size == triple_count * 3:
        return Play(cards, PlayType.PLANE, main_rank, size, triple_count)

    attachments = Counter(count for rank, count in counts.items() if rank not in triple_ranks)
    if size == triple_count * 4 and attachments == Counter({1: triple_count}):
        return Play(cards, PlayType.PLANE_SINGLE, main_rank, size, triple_count)
    if size == triple_count * 5 and attachments == Counter({2: triple_count}):
        return Play(cards, PlayType.PLANE_PAIR, main_rank, size, triple_count)
    return None


def can_beat(candidate: Play, previous: Play) -> bool:
    if not candidate.is_valid or candidate.is_pass:
        return False
    if previous.is_pass:
        return True
    if candidate.type is PlayType.ROCKET:
        return previous.type is not PlayType.ROCKET
    if previous.type is PlayType.ROCKET:
        return False
    if candidate.type is PlayType.BOMB:
        if previous.type is not PlayType.BOMB:
            return True
        return _rank_greater(candidate.main_rank, previous.main_rank)
    if previous.type is PlayType.BOMB:
        return False
    if candidate.type is not previous.type:
        return False
    if candidate.length != previous.length or candidate.combo_size != previous.combo_size:
        return False
    return _rank_greater(candidate.main_rank, previous.main_rank)


def _rank_greater(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return RANK_VALUE[left] > RANK_VALUE[right]


def legal_actions(hand: CardSet, previous: Play | None = None, include_pass: bool = True) -> list[Play]:
    previous = previous or Play((), PlayType.PASS)
    candidates = _all_candidate_plays(hand)
    if previous.is_pass:
        actions = candidates
    else:
        actions = [candidate for candidate in candidates if can_beat(candidate, previous)]
        if include_pass:
            actions.append(Play((), PlayType.PASS))
    return sorted(actions, key=_play_sort_key)


def _all_candidate_plays(hand: CardSet) -> list[Play]:
    counts = hand.counts
    candidates: set[tuple[str, ...]] = set()

    for rank, count in counts.items():
        candidates.add((rank,))
        if count >= 2:
            candidates.add((rank, rank))
        if count >= 3:
            candidates.add((rank, rank, rank))
        if count == 4:
            candidates.add((rank, rank, rank, rank))

    if counts["SJ"] >= 1 and counts["BJ"] >= 1:
        candidates.add(("SJ", "BJ"))

    triple_ranks = [rank for rank, count in counts.items() if count >= 3]
    pair_ranks = [rank for rank, count in counts.items() if count >= 2]
    single_ranks = list(counts.keys())

    for triple in triple_ranks:
        for kicker in single_ranks:
            if kicker != triple:
                candidates.add(tuple(sorted((triple, triple, triple, kicker), key=RANK_VALUE.__getitem__)))
        for pair in pair_ranks:
            if pair != triple:
                candidates.add(tuple(sorted((triple, triple, triple, pair, pair), key=RANK_VALUE.__getitem__)))

    bomb_ranks = [rank for rank, count in counts.items() if count == 4]
    for bomb in bomb_ranks:
        single_ranks = [rank for rank in counts if rank != bomb]
        for kickers in combinations(single_ranks, 2):
            candidates.add(sort_cards((bomb, bomb, bomb, bomb, *kickers)))
        pair_ranks = [rank for rank, count in counts.items() if rank != bomb and count >= 2]
        for kickers in combinations(pair_ranks, 2):
            cards = [bomb, bomb, bomb, bomb]
            for rank in kickers:
                cards.extend([rank, rank])
            candidates.add(sort_cards(cards))

    _add_straights(candidates, counts)
    _add_pair_straights(candidates, counts)
    _add_planes(candidates, counts)

    plays = [classify_play(candidate) for candidate in candidates]
    return [play for play in plays if play.is_valid and not play.is_pass]


def _add_straights(candidates: set[tuple[str, ...]], counts: Counter[str]) -> None:
    available = [rank for rank in CHAIN_RANKS if counts[rank] >= 1]
    for chain in _consecutive_slices(available, min_len=5):
        candidates.add(tuple(chain))


def _add_pair_straights(candidates: set[tuple[str, ...]], counts: Counter[str]) -> None:
    available = [rank for rank in CHAIN_RANKS if counts[rank] >= 2]
    for chain in _consecutive_slices(available, min_len=3):
        cards: list[str] = []
        for rank in chain:
            cards.extend([rank, rank])
        candidates.add(tuple(cards))


def _add_planes(candidates: set[tuple[str, ...]], counts: Counter[str]) -> None:
    available = [rank for rank in CHAIN_RANKS if counts[rank] >= 3]
    for chain in _consecutive_slices(available, min_len=2):
        base: list[str] = []
        for rank in chain:
            base.extend([rank, rank, rank])
        candidates.add(tuple(base))

        excluded = set(chain)
        single_ranks = [rank for rank in counts if rank not in excluded]
        for kickers in combinations(single_ranks, len(chain)):
            candidates.add(sort_cards((*base, *kickers)))

        pair_ranks = [rank for rank, count in counts.items() if rank not in excluded and count >= 2]
        for kickers in combinations(pair_ranks, len(chain)):
            cards = list(base)
            for rank in kickers:
                cards.extend([rank, rank])
            candidates.add(sort_cards(cards))


def _consecutive_slices(ranks: list[str], min_len: int) -> list[list[str]]:
    chains: list[list[str]] = []
    start = 0
    while start < len(ranks):
        end = start + 1
        while end < len(ranks) and RANK_VALUE[ranks[end]] == RANK_VALUE[ranks[end - 1]] + 1:
            end += 1
        segment = ranks[start:end]
        for left in range(len(segment)):
            for right in range(left + min_len, len(segment) + 1):
                chains.append(segment[left:right])
        start = end
    return chains


def _play_sort_key(play: Play) -> tuple[int, int, int, int, tuple[int, ...]]:
    type_order = {
        PlayType.PASS: 0,
        PlayType.SINGLE: 1,
        PlayType.PAIR: 2,
        PlayType.TRIPLE: 3,
        PlayType.TRIPLE_SINGLE: 4,
        PlayType.TRIPLE_PAIR: 5,
        PlayType.STRAIGHT: 6,
        PlayType.PAIR_STRAIGHT: 7,
        PlayType.PLANE: 8,
        PlayType.PLANE_SINGLE: 9,
        PlayType.PLANE_PAIR: 10,
        PlayType.FOUR_TWO_SINGLE: 11,
        PlayType.FOUR_TWO_PAIR: 12,
        PlayType.BOMB: 13,
        PlayType.ROCKET: 14,
        PlayType.INVALID: 99,
    }
    main_value = RANK_VALUE[play.main_rank] if play.main_rank else -1
    card_values = tuple(RANK_VALUE[card] for card in play.cards)
    return (len(play.cards), type_order[play.type], main_value, play.combo_size, card_values)


__all__ = ["Play", "PlayType", "can_beat", "classify_play", "legal_actions"]
