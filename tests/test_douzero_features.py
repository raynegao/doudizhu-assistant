"""The DouZero encoding reproduces the published feature layout from our state."""

from __future__ import annotations

import numpy as np
import pytest

from src.logic.action_validation import validate_observed_action
from src.logic.douzero_features import (
    FARMER_FEATURES,
    LANDLORD_FEATURES,
    action_sequence_array,
    bomb_one_hot,
    cards_to_array,
    encode,
    one_hot,
    position_of,
    view_from_state,
)
from src.logic.rules import Play, legal_actions
from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat
from src.state.game_tracker import GameStateTracker
from src.state.observable_state import ObservableGameState


def test_cards_to_array_matches_douzero_layout() -> None:
    assert cards_to_array([]).tolist() == [0] * 54
    vector = cards_to_array(["3", "3", "A", "2", "2", "2", "2", "SJ"])
    assert vector.shape == (54,) and vector.dtype == np.int8
    assert vector[0:4].tolist() == [1, 1, 0, 0]  # two threes in column 0
    assert vector[44:48].tolist() == [1, 0, 0, 0]  # one ace in column 11
    assert vector[48:52].tolist() == [1, 1, 1, 1]  # four twos in column 12
    assert vector[52] == 1 and vector[53] == 0  # small joker only
    assert cards_to_array(["BJ"])[52:54].tolist() == [0, 1]
    assert one_hot(1, 17).tolist()[:2] == [1, 0] and one_hot(17, 17)[16] == 1
    assert bomb_one_hot(0)[0] == 1 and bomb_one_hot(3)[3] == 1 and bomb_one_hot(40)[14] == 1


def test_action_sequence_is_left_padded_and_reshaped() -> None:
    block = action_sequence_array([("3",), (), ("4", "4")])
    assert block.shape == (5, 162)
    flat = block.reshape(15, 54)
    assert not flat[:12].any()  # twelve padded empty moves
    assert flat[12][0] == 1 and not flat[13].any() and flat[14][4:6].tolist() == [1, 1]
    assert action_sequence_array([("5",)] * 20).reshape(15, 54)[0][8] == 1  # keeps the last 15


def _tracked_state(hand: str, landlord: str, actions: list[tuple[str, str]]) -> ObservableGameState:
    counts = {"self": len(CardSet.parse(hand)), "right": 17, "left": 17}
    counts[landlord] = 20 if landlord != "self" else counts["self"]
    state = ObservableGameState.from_inputs(hand, landlord=landlord, current_actor=landlord, remaining_cards=counts)
    tracker = GameStateTracker(state, validator=validate_observed_action)
    for sequence_no, (actor, cards) in enumerate(actions, start=1):
        result = tracker.apply(ObservedAction(f"e{sequence_no}", sequence_no, PlayerSeat(actor), CardSet.parse(cards)))
        assert result.status.value == "applied", result.message
    return tracker.state


def test_positions_follow_douzero_turn_order() -> None:
    state = ObservableGameState.from_inputs("3 4 5", landlord="left")
    assert position_of(PlayerSeat.LEFT, state) == "landlord"
    assert position_of(PlayerSeat.SELF, state) == "landlord_down"  # SELF plays right after LEFT
    assert position_of(PlayerSeat.RIGHT, state) == "landlord_up"
    state = ObservableGameState.from_inputs("3 4 5", landlord="self")
    assert position_of(PlayerSeat.RIGHT, state) == "landlord_down"


def test_view_collects_history_per_position_and_counts_bombs() -> None:
    hand = "BJ SJ 2 2 A K K J J 10 10 10 9 9 9 8 8 7 5 4"
    state = _tracked_state(hand, "self", [
        ("self", "4"), ("right", "7"), ("left", "J"), ("self", "A"), ("right", "2"), ("left", ""),
        ("self", "BJ SJ"), ("right", ""), ("left", ""),
    ])
    view = view_from_state(state)
    assert view.position == "landlord" and state.current_actor is PlayerSeat.SELF
    assert view.played_cards["landlord"] == ("4", "A", "SJ", "BJ")  # CardSet order
    assert view.played_cards["landlord_down"] == ("7", "2")
    assert view.played_cards["landlord_up"] == ("J",)
    assert view.last_move == ()  # two passes: SELF leads freely
    assert view.last_move_by_position == {"landlord": ("SJ", "BJ"), "landlord_down": (), "landlord_up": ()}
    assert view.cards_left == {"landlord": 16, "landlord_down": 15, "landlord_up": 16}
    assert view.bomb_count == 1  # the rocket counts, as in DouZero
    assert view.action_sequence[-3:] == (("SJ", "BJ"), (), ())
    assert len(view.other_hands) == 31 and "BJ" not in view.other_hands


@pytest.mark.parametrize("landlord, expected", [("self", LANDLORD_FEATURES), ("right", FARMER_FEATURES), ("left", FARMER_FEATURES)])
def test_encode_shapes_per_position(landlord: str, expected: int) -> None:
    hand = "3 3 4 5 6 7 8 K A 2" if landlord != "self" else "3 3 4 5 6 7 8 K A 2 2 9 9 Q Q J 10 10 BJ 5"
    state = ObservableGameState.from_inputs(hand, landlord=landlord, current_actor="self")
    actions = legal_actions(state.self_hand, Play.parse(()))
    z_batch, x_batch = encode(view_from_state(state), actions)
    assert z_batch.shape == (len(actions), 5, 162) and z_batch.dtype == np.float32
    assert x_batch.shape == (len(actions), expected) and x_batch.dtype == np.float32
    # The action block is the last 54 columns and differs per action.
    assert not np.array_equal(x_batch[0, -54:], x_batch[1, -54:])
    assert np.array_equal(x_batch[0, :-54], x_batch[1, :-54])
    with pytest.raises(ValueError):
        encode(view_from_state(state), [])
