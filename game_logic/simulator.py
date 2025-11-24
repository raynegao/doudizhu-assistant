"""
简单的 Monte-Carlo 模拟器，用于估算胜率（启发式玩法，非精确斗地主 AI）。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

from .cards import Card, RANK_ORDER, RANK_TO_ID, build_standard_deck
from .doudizhu_rules import HandType, can_beat, classify_hand, generate_all_legal_hands
from .state_parser import GameState


@dataclass
class SimulationConfig:
    """
    Simulation parameters for win-rate estimation.
    """

    num_samples: int = 200
    max_steps: int = 500


def simulate_round(state: GameState, config: SimulationConfig) -> bool:
    """
    模拟一局，返回英雄（索引 0）是否获胜。
    玩法策略较为贪心，仅用于粗略估计。
    """

    deck = build_standard_deck()
    pool = _remove_known_cards(deck, state)

    # 确定对手牌数
    remaining = len(pool)
    left_count = state.left_opponent_count or remaining // 2
    right_count = state.right_opponent_count or remaining - left_count

    random.shuffle(pool)
    left_hand = pool[:left_count]
    right_hand = pool[left_count : left_count + right_count]

    hands = [list(state.my_hand), left_hand, right_hand]
    prev_play: List[Card] = list(state.last_play)
    passes_in_row = 0
    player = 0

    for _ in range(config.max_steps):
        current_hand = hands[player]
        legal = generate_all_legal_hands(current_hand, prev_play if prev_play else None)
        play = _choose_play(prev_play, legal)

        if play:
            _remove_cards(current_hand, play)
            prev_play = play
            passes_in_row = 0
        else:
            passes_in_row += 1
            if passes_in_row >= 2:
                prev_play = []
                passes_in_row = 0

        if not current_hand:
            return player == 0

        player = (player + 1) % 3

    # 超过步数限制，保守返回未胜
    return False


def estimate_win_rate(state: GameState, num_samples: int = 200) -> float:
    """
    通过重复模拟估计当前玩家的胜率。
    """

    if not state.my_hand:
        return 0.0
    config = SimulationConfig(num_samples=num_samples)
    wins = 0
    for _ in range(config.num_samples):
        if simulate_round(state, config):
            wins += 1
    return wins / max(1, config.num_samples)


# --------- 辅助函数 --------- #


def _remove_known_cards(deck: List[Card], state: GameState) -> List[Card]:
    """
    从完整牌堆里移除已知的牌（自己的手牌 + 已出的牌）。
    """

    pool = deck.copy()
    known = list(state.my_hand) + list(state.last_play)
    for hist in state.history:
        known.extend(hist)
    known_counts = Counter([c.rank for c in known])
    filtered: List[Card] = []
    for card in pool:
        if known_counts[card.rank] > 0:
            known_counts[card.rank] -= 1
            continue
        filtered.append(card)
    return filtered


def _remove_cards(hand: List[Card], played: Sequence[Card]) -> None:
    counts = Counter([c.rank for c in played])
    new_hand: List[Card] = []
    for card in hand:
        if counts[card.rank] > 0:
            counts[card.rank] -= 1
        else:
            new_hand.append(card)
    hand.clear()
    hand.extend(new_hand)


def _choose_play(prev_play: List[Card], legal: List[List[Card]]) -> List[Card]:
    """
    贪心策略：出能压住的最小牌；若无上家牌则出最小单牌。
    """

    if not legal:
        return []

    if not prev_play:
        # 无上家牌，尽量出最小非炸弹
        legal.sort(key=_play_sort_key)
        return legal[0]

    legal_beating = [c for c in legal if can_beat(prev_play, c)]
    if not legal_beating:
        return []

    legal_beating.sort(key=_play_sort_key)
    return legal_beating[0]


def _play_sort_key(cards: List[Card]) -> tuple:
    """排序策略：非炸弹优先，牌数少优先，关键牌小优先。"""

    hand_type = classify_hand(cards)
    bomb_flag = 1 if hand_type in {HandType.BOMB, HandType.ROCKET} else 0
    strength = _hand_strength(cards, hand_type)
    return (bomb_flag, len(cards), strength)


def _hand_strength(cards: List[Card], hand_type: HandType) -> int:
    """用于排序的强度值，越小越优先。"""

    ranks = [c.rank for c in cards]
    counts = Counter(ranks)
    if hand_type in {HandType.SINGLE, HandType.PAIR, HandType.TRIPLE, HandType.BOMB}:
        key_rank = min(counts.items(), key=lambda kv: (-kv[1], RANK_TO_ID[kv[0]]))[0]
        return RANK_TO_ID[key_rank]
    if hand_type == HandType.ROCKET:
        return RANK_TO_ID["joker_small"]
    if hand_type in {HandType.TRIPLE_WITH_SINGLE, HandType.TRIPLE_WITH_PAIR}:
        triple_rank = min((r for r, c in counts.items() if c == 3), key=lambda r: RANK_TO_ID[r])
        return RANK_TO_ID[triple_rank]
    if hand_type in {HandType.STRAIGHT, HandType.DOUBLE_SEQUENCE, HandType.AIRPLANE, HandType.AIRPLANE_WITH_WINGS}:
        seq_ranks = sorted((r for r, c in counts.items() if c >= (2 if hand_type == HandType.DOUBLE_SEQUENCE else 1)), key=lambda r: RANK_TO_ID[r])
        return RANK_TO_ID[seq_ranks[0]]
    return RANK_TO_ID[ranks[0]]
