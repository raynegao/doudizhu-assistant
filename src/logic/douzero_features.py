"""DouZero observation encoding built from this project's observable state.

Re-implements the feature layout of DouZero (Kwai Inc., Apache-2.0,
https://github.com/kwai/DouZero, ``douzero/env/env.py``) so the published
checkpoints can score this project's legal actions:

* a card list becomes 54 values: a 4 x 13 count matrix (columns 3..A then 2)
  flattened column-major, then the small and big joker flags;
* the landlord sees 373 features per action, a farmer 484 (block order below);
* the last 15 actions (passes included, left-padded) form a (5, 162) sequence.

Only numpy is used here; the network lives in ``douzero_policy``. Positions
follow DouZero: play order landlord -> landlord_down -> landlord_up, so the
seat after the landlord is ``landlord_down``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.logic.rules import Play, PlayType, classify_play
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState

RANK_TO_CODE = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14, "2": 17, "SJ": 20, "BJ": 30,
}
CARD_TO_COLUMN = {code: index for index, code in enumerate((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17))}
NUM_ONES = {
    0: np.array([0, 0, 0, 0], dtype=np.int8),
    1: np.array([1, 0, 0, 0], dtype=np.int8),
    2: np.array([1, 1, 0, 0], dtype=np.int8),
    3: np.array([1, 1, 1, 0], dtype=np.int8),
    4: np.array([1, 1, 1, 1], dtype=np.int8),
}
POSITIONS = ("landlord", "landlord_down", "landlord_up")
LANDLORD_FEATURES = 373
FARMER_FEATURES = 484
SEQUENCE_LENGTH = 15


def cards_to_array(cards: Iterable[str]) -> np.ndarray:
    """DouZero ``_cards2array``: 52 rank slots (column-major) plus two joker flags."""
    counts = Counter(cards)
    if not counts:
        return np.zeros(54, dtype=np.int8)
    matrix: np.ndarray = np.zeros((4, 13), dtype=np.int8)
    jokers: np.ndarray = np.zeros(2, dtype=np.int8)
    for rank, count in counts.items():
        code = RANK_TO_CODE[rank]
        if code == 20:
            jokers[0] = 1
        elif code == 30:
            jokers[1] = 1
        else:
            matrix[:, CARD_TO_COLUMN[code]] = NUM_ONES[count]
    return np.concatenate((matrix.flatten("F"), jokers))


def one_hot(value: int, size: int) -> np.ndarray:
    """DouZero ``_get_one_hot_array``: index ``value - 1`` (a count, never zero)."""
    array: np.ndarray = np.zeros(size, dtype=np.int8)
    if 1 <= value <= size:
        array[value - 1] = 1
    return array


def bomb_one_hot(count: int) -> np.ndarray:
    array: np.ndarray = np.zeros(15, dtype=np.int8)
    array[min(max(count, 0), 14)] = 1
    return array


def action_sequence_array(actions: Sequence[Sequence[str]]) -> np.ndarray:
    """Last 15 actions (passes as empty lists), left-padded, as a (5, 162) block."""
    recent = list(actions)[-SEQUENCE_LENGTH:]
    padded: list[Sequence[str]] = [()] * (SEQUENCE_LENGTH - len(recent)) + recent
    stacked = np.stack([cards_to_array(action) for action in padded])
    return stacked.reshape(5, 162)


@dataclass(frozen=True)
class DouZeroView:
    """Everything DouZero's encoders read, in DouZero's own position names."""

    position: str
    hand: tuple[str, ...]
    other_hands: tuple[str, ...]
    last_move: tuple[str, ...]
    played_cards: dict[str, tuple[str, ...]]
    last_move_by_position: dict[str, tuple[str, ...]]
    cards_left: dict[str, int]
    bomb_count: int
    action_sequence: tuple[tuple[str, ...], ...]


def position_of(seat: PlayerSeat, state: ObservableGameState) -> str:
    if seat is state.landlord:
        return "landlord"
    if seat is state.next_player(state.landlord):
        return "landlord_down"
    return "landlord_up"


def view_from_state(state: ObservableGameState) -> DouZeroView:
    """Build the acting player's (SELF) DouZero view from the observable state.

    Played cards, last moves and the action sequence come from ``history``;
    a round modelled mid-game therefore has an incomplete sequence, exactly
    as the tracker's own ``hidden_played_count`` already acknowledges.
    """
    played: dict[str, list[str]] = {position: [] for position in POSITIONS}
    last_moves: dict[str, tuple[str, ...]] = {position: () for position in POSITIONS}
    sequence: list[tuple[str, ...]] = []
    bombs = 0
    for action in state.history:
        position = position_of(action.actor, state)
        cards = tuple(action.cards.cards)
        played[position].extend(cards)
        last_moves[position] = cards
        sequence.append(cards)
        if cards and classify_play(cards).type in {PlayType.BOMB, PlayType.ROCKET}:
            bombs += 1
    return DouZeroView(
        position=position_of(PlayerSeat.SELF, state),
        hand=tuple(state.self_hand.cards),
        other_hands=tuple(state.unknown_cards.cards),
        last_move=tuple(state.trick_target.cards),
        played_cards={position: tuple(cards) for position, cards in played.items()},
        last_move_by_position=last_moves,
        cards_left={position_of(seat, state): count for seat, count in state.remaining_cards},
        bomb_count=bombs,
        action_sequence=tuple(sequence),
    )


def encode(view: DouZeroView, actions: Sequence[Play]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(z_batch, x_batch)`` as float32 arrays, one row per legal action."""
    if not actions:
        raise ValueError("at least one legal action is required")
    if view.position == "landlord":
        blocks = [
            cards_to_array(view.hand),
            cards_to_array(view.other_hands),
            cards_to_array(view.last_move),
            cards_to_array(view.played_cards["landlord_up"]),
            cards_to_array(view.played_cards["landlord_down"]),
            one_hot(view.cards_left["landlord_up"], 17),
            one_hot(view.cards_left["landlord_down"], 17),
            bomb_one_hot(view.bomb_count),
        ]
        expected = LANDLORD_FEATURES
    else:
        teammate = "landlord_up" if view.position == "landlord_down" else "landlord_down"
        blocks = [
            cards_to_array(view.hand),
            cards_to_array(view.other_hands),
            cards_to_array(view.played_cards["landlord"]),
            cards_to_array(view.played_cards[teammate]),
            cards_to_array(view.last_move),
            cards_to_array(view.last_move_by_position["landlord"]),
            cards_to_array(view.last_move_by_position[teammate]),
            one_hot(view.cards_left["landlord"], 20),
            one_hot(view.cards_left[teammate], 17),
            bomb_one_hot(view.bomb_count),
        ]
        expected = FARMER_FEATURES
    base = np.concatenate(blocks)
    x_batch = np.stack([
        np.concatenate((base, cards_to_array(action.cards))) for action in actions
    ]).astype(np.float32)
    if x_batch.shape[1] != expected:
        raise AssertionError(f"{view.position} features are {x_batch.shape[1]}, expected {expected}")
    z = action_sequence_array(view.action_sequence)
    z_batch = np.repeat(z[np.newaxis, :, :], len(actions), axis=0).astype(np.float32)
    return z_batch, x_batch


__all__ = [
    "FARMER_FEATURES",
    "LANDLORD_FEATURES",
    "POSITIONS",
    "DouZeroView",
    "action_sequence_array",
    "bomb_one_hot",
    "cards_to_array",
    "encode",
    "one_hot",
    "position_of",
    "view_from_state",
]
