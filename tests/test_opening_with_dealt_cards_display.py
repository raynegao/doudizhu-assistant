"""A seat whose verified counter still equals its deal cannot have played: its cards are the dealt-cards display.

Independent acceptance session acceptance-20261017-002 (frozen ``b6029d5c``):
after bidding the client shows the landlord's three extra cards at the top
of the table, inside the left/right play ROIs, until the landlord's first
play. Self (the landlord) played within a second, so no frame ever had an
empty table with the idle 20-card hand, and the tracker had no initial
model until the first automatic rescan nine actions later.
"""

from __future__ import annotations

import pytest

from src.state.events import PlayerSeat
from src.tracking.visual_events import VisualEventTracker, VisualTrackerMode, _initial_state_payload
from src.vision.scene_types import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)

HAND = ("BJ", "SJ", "2", "2", "A", "K", "K", "J", "J", "10", "10", "10", "9", "9", "9", "8", "8", "7", "5", "4")


def _seat(seat: PlayerSeat, role: SeatRole, remaining: int, *, verified: bool = True,
          display: tuple[str, ...] = ()) -> SeatObservation:
    return SeatObservation(
        seat=seat, signal=VisualSignal.PLAY if display else VisualSignal.NEUTRAL,
        cards=tuple(VisualCard(rank, 0.97, (i * 40, 0, i * 40 + 30, 60)) for i, rank in enumerate(display)),
        remaining_count=remaining, role=role, confidence=0.97 if display else 0.8,
        remaining_confidence=0.98, role_confidence=0.999, remaining_verified=verified,
    )


def _scene(frame: int, *, left_display=("10",), right_display=("8",), left_remaining=17,
           right_remaining=17, verified=True, hand=HAND, self_play=()) -> SceneObservation:
    seats = (
        _seat(PlayerSeat.SELF, SeatRole.LANDLORD, len(hand), verified=False, display=self_play),
        _seat(PlayerSeat.RIGHT, SeatRole.FARMER, right_remaining, verified=verified, display=right_display),
        _seat(PlayerSeat.LEFT, SeatRole.FARMER, left_remaining, verified=verified, display=left_display),
    )
    return SceneObservation(
        frame, float(frame), (0, 0, 1000, 800),
        tuple(VisualCard(rank, 0.99, (i * 10, 0, i * 10 + 10, 20)) for i, rank in enumerate(hand)),
        seats, True, 0.99, 0.99,
    )


def test_dealt_cards_in_opponent_rois_do_not_block_the_opening() -> None:
    payload = _initial_state_payload(_scene(1), confidence_threshold=0.7)
    assert payload is not None
    assert payload.landlord is PlayerSeat.SELF
    assert payload.remaining == {PlayerSeat.SELF: 20, PlayerSeat.RIGHT: 17, PlayerSeat.LEFT: 17}
    assert payload.hand == HAND


def test_tracker_initialises_and_then_tracks_the_fast_first_play() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_stability_frames=2)
    first = tracker.update(_scene(1))
    second = tracker.update(_scene(2))
    assert first.state is None and second.state is not None
    assert second.state.landlord is PlayerSeat.SELF and second.state.current_actor is PlayerSeat.SELF
    assert tracker.mode is VisualTrackerMode.TRACKING


@pytest.mark.parametrize("fault", ["counter_decreased", "counter_unverified"])
def test_a_seat_that_really_played_still_goes_through_the_strict_rebuild_path(fault: str) -> None:
    kwargs = {"left_remaining": 16} if fault == "counter_decreased" else {"verified": False}
    assert _initial_state_payload(_scene(1, **kwargs), confidence_threshold=0.7) is None


def test_dealt_display_rule_is_off_while_a_round_is_actively_tracked() -> None:
    # While tracking, an opponent card with an unchanged verified counter is an
    # animated counter that has not decremented yet (see
    # test_visible_play_is_processed_before_turn_based_pass_inference); only an
    # untracked, uncertain or finished tracker may read it as the deal display.
    assert _initial_state_payload(_scene(1), confidence_threshold=0.7, allow_dealt_display=False) is None
    tracker = VisualEventTracker(stability_frames=1, initial_stability_frames=1)
    tracker.update(_scene(1, left_display=(), right_display=()))
    assert tracker.mode is VisualTrackerMode.TRACKING
    played = tracker.update(_scene(2, left_display=(), right_display=(), hand=HAND[1:], self_play=("BJ",)))
    assert played.event is not None and played.event.actor is PlayerSeat.SELF
    stale = tracker.update(_scene(3, left_display=(), right_display=("8",), hand=HAND[1:]))
    assert stale.event is None
    assert stale.state is not None and stale.state.revision == played.state.revision
    assert stale.state.current_actor is PlayerSeat.RIGHT
