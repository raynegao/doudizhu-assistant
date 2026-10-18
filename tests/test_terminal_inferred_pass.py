"""An opponent pass that only showed during a self-hand crop pause is proven by the self terminal play.

Independent acceptance session acceptance-20261017-001 (frozen ``b6029d5c``):
after the self bomb the right seat passed (recorded), the left seat's pass
label was visible only while the user raised the final cards (geometry
pause), and the game cleared it before geometry recovered. With the state
waiting for LEFT, the self terminal gate could never commit ``A A A A``.
The reducer chain pass(LEFT) -> terminal play(SELF) is the only legal
reading when LEFT's verified counter is unchanged and its seat is blank.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import VisualEventTracker, VisualTrackerMode
from src.vision.scene_types import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    SelfTableObservation,
    VisualCard,
    VisualSignal,
)

HAND = ("A", "A", "A", "A")


def _state(*, passes: int) -> ObservableGameState:
    """Self (landlord) led 7777; ``passes`` opponents have already passed."""
    state = ObservableGameState.from_inputs(
        HAND, round_id="terminal-pass", landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.LEFT if passes == 1 else PlayerSeat.RIGHT,
        remaining_cards={PlayerSeat.SELF: 4, PlayerSeat.RIGHT: 12, PlayerSeat.LEFT: 14},
        played_cards="7 7 7 7", last_play="7 7 7 7", last_player=PlayerSeat.SELF,
        consecutive_passes=passes, hidden_played_count=20,
    )
    lead = ObservedAction("terminal-pass:22:self", 22, PlayerSeat.SELF, CardSet.parse("7 7 7 7"))
    history = [lead]
    if passes == 1:
        history.append(ObservedAction("terminal-pass:23:right", 23, PlayerSeat.RIGHT, CardSet(())))
    return replace(state, revision=len(history) + 21, last_sequence_no=history[-1].sequence_no,
                   history=tuple(history), processed_events=tuple((e.event_id, e.fingerprint) for e in history))


def _seat(seat: PlayerSeat, signal: VisualSignal, remaining: int | None, *, verified: bool = True,
          cards: tuple[VisualCard, ...] = ()) -> SeatObservation:
    return SeatObservation(
        seat=seat, signal=signal, cards=cards, remaining_count=remaining,
        role=SeatRole.LANDLORD if seat is PlayerSeat.SELF else SeatRole.FARMER,
        confidence=0.99, pass_confidence=0.99 if signal is VisualSignal.PASS else 0.0,
        remaining_confidence=0.99, role_confidence=0.99, remaining_verified=verified,
    )


def _scene(frame: int, *, left: VisualSignal = VisualSignal.NEUTRAL, left_remaining: int | None = 14,
           left_verified: bool = True, right: VisualSignal = VisualSignal.PASS,
           left_cards: tuple[VisualCard, ...] = (), table: tuple[str, ...] = HAND) -> SceneObservation:
    seats = (
        _seat(PlayerSeat.SELF, VisualSignal.NEUTRAL, None, verified=False),
        _seat(PlayerSeat.RIGHT, right, 12),
        _seat(PlayerSeat.LEFT, left, left_remaining, verified=left_verified, cards=left_cards),
    )
    cards = tuple(VisualCard(rank, 0.95, (30 + i * 30, 40, 60 + i * 30, 100)) for i, rank in enumerate(table))
    return SceneObservation(
        frame, float(frame), (0, 0, 1000, 800), (), seats, None, 0.0, 0.99,
        self_hand_empty_verified=True, self_table=SelfTableObservation(cards, True, "closed"),
    )


def _drive(tracker: VisualEventTracker, scenes: list[SceneObservation]) -> list:
    events = []
    for scene in scenes:
        update = tracker.update(scene)
        if update.event is not None:
            events.append(update.event)
        if update.mode is VisualTrackerMode.FINISHED:
            break
    return events


def test_blank_left_seat_with_unchanged_counter_is_inferred_before_the_terminal_play() -> None:
    tracker = VisualEventTracker(initial_state=_state(passes=1), stability_frames=3)
    events = _drive(tracker, [_scene(f) for f in range(1, 13)])
    assert [(e.actor, e.cards.cards, e.source) for e in events] == [
        (PlayerSeat.LEFT, (), "live_terminal_inferred_pass"),
        (PlayerSeat.SELF, HAND, "live_self_table_terminal"),
    ]
    assert tracker.state is None or tracker.state.phase is RoundPhase.FINISHED or tracker.mode is VisualTrackerMode.FINISHED


def test_both_opponents_can_be_inferred_in_turn_order() -> None:
    tracker = VisualEventTracker(initial_state=_state(passes=0), stability_frames=3)
    events = _drive(tracker, [_scene(f, right=VisualSignal.NEUTRAL) for f in range(1, 16)])
    assert [(e.actor, e.cards.cards) for e in events] == [
        (PlayerSeat.RIGHT, ()), (PlayerSeat.LEFT, ()), (PlayerSeat.SELF, HAND),
    ]
    assert all(e.source == "live_terminal_inferred_pass" for e in events[:2])


@pytest.mark.parametrize("fault", ["left_played", "left_counter_changed", "left_counter_unverified",
                                   "table_incomplete", "wrong_cards"])
def test_inference_needs_a_blank_seat_an_unchanged_counter_and_the_full_terminal_proof(fault: str) -> None:
    tracker = VisualEventTracker(initial_state=_state(passes=1), stability_frames=3)
    kwargs: dict = {}
    if fault == "left_played":
        kwargs = {"left": VisualSignal.PLAY, "left_cards": (VisualCard("2", 0.99, (0, 0, 10, 20)),)}
    elif fault == "left_counter_changed":
        kwargs = {"left_remaining": 13}
    elif fault == "left_counter_unverified":
        kwargs = {"left_verified": False}
    elif fault == "table_incomplete":
        kwargs = {"table": ("A", "A", "A")}
    elif fault == "wrong_cards":
        kwargs = {"table": ("A", "A", "A", "K")}
    events = _drive(tracker, [_scene(f, **kwargs) for f in range(1, 13)])
    assert not [e for e in events if e.source in {"live_terminal_inferred_pass", "live_self_table_terminal"}]
    assert tracker.state is not None and tracker.state.revision == 23


def test_inference_is_not_a_shortcut_when_self_is_already_current() -> None:
    state = ObservableGameState.from_inputs(
        HAND, round_id="terminal-pass", landlord=PlayerSeat.SELF, current_actor=PlayerSeat.SELF,
        remaining_cards={PlayerSeat.SELF: 4, PlayerSeat.RIGHT: 12, PlayerSeat.LEFT: 14},
        played_cards="7 7 7 7", hidden_played_count=20,
    )
    tracker = VisualEventTracker(initial_state=state, stability_frames=3)
    events = _drive(tracker, [_scene(f, right=VisualSignal.NEUTRAL) for f in range(1, 8)])
    assert [e.source for e in events] == ["live_self_table_terminal"]
