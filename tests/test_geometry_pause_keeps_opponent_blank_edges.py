"""A self-hand crop pause must not erase opponents' blank edges seen on valid frames.

Independent acceptance session acceptance-20261014-001 (frozen ``04f83e90``):
the user raised the final bomb before the left seat passed. The crop pause reset
every seat's blank edge, so the pass visible after recovery was ignored as a
stale prompt, the state kept waiting for LEFT and the self terminal gate, which
requires SELF to be the current actor, could never commit ``7 7 7 7``.
"""

from __future__ import annotations

from dataclasses import replace

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

HAND = ("7", "7", "7", "7")
GEOMETRY_WARNING = "self_hand_geometry_unavailable: rank_top_outside_input_roi"


def _state() -> ObservableGameState:
    state = ObservableGameState.from_inputs(
        HAND, round_id="pause-keeps-edges", landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.LEFT,
        remaining_cards={PlayerSeat.SELF: 4, PlayerSeat.RIGHT: 9, PlayerSeat.LEFT: 8},
        played_cards="2", last_play="2", last_player=PlayerSeat.RIGHT,
        hidden_played_count=32,
    )
    event = ObservedAction("pause-keeps-edges:32:right", 32, PlayerSeat.RIGHT, CardSet.parse("2"))
    return replace(state, revision=32, last_sequence_no=32, history=(event,),
                   processed_events=((event.event_id, event.fingerprint),))


def _seat(seat: PlayerSeat, signal: VisualSignal, *, remaining: int | None,
          cards: tuple[VisualCard, ...] = ()) -> SeatObservation:
    return SeatObservation(
        seat=seat, signal=signal, cards=cards, remaining_count=remaining,
        role=SeatRole.LANDLORD if seat is PlayerSeat.SELF else SeatRole.FARMER,
        confidence=0.99, pass_confidence=0.99 if signal is VisualSignal.PASS else 0.0,
        remaining_confidence=0.99, role_confidence=0.99,
        remaining_verified=seat is not PlayerSeat.SELF,
    )


def _scene(frame: int, *, left: VisualSignal, hand: tuple[str, ...] = HAND,
           empty: bool = False, self_turn: bool | None = None,
           warnings: tuple[str, ...] = ()) -> SceneObservation:
    seats = (
        _seat(PlayerSeat.SELF, VisualSignal.NEUTRAL, remaining=None),
        # The already recorded trick target stays visible on the right seat.
        _seat(PlayerSeat.RIGHT, VisualSignal.PLAY, remaining=9,
              cards=(VisualCard("2", 0.98, (0, 0, 10, 20)),)),
        _seat(PlayerSeat.LEFT, left, remaining=8),
    )
    table = SelfTableObservation(
        tuple(VisualCard(rank, 0.95, (30 + i * 30, 40, 60 + i * 30, 100))
              for i, rank in enumerate(HAND)),
        True, "closed",
    ) if empty else None
    return SceneObservation(
        frame, float(frame), (0, 0, 1000, 800),
        () if empty else tuple(
            VisualCard(rank, 0.99, (i * 10, 0, i * 10 + 10, 20)) for i, rank in enumerate(hand)
        ),
        seats, self_turn, 0.99 if self_turn is not None else 0.0, 0.99,
        warnings=warnings, self_hand_empty_verified=empty, self_table=table,
    )


def _run(tracker: VisualEventTracker, first_left: VisualSignal) -> list:
    """Return the updates that carried an event; stop once the round finishes."""
    updates = []
    frames = [_scene(1, left=first_left)]
    frames += [_scene(f, left=VisualSignal.PASS, self_turn=True, warnings=(GEOMETRY_WARNING,))
               for f in (2, 3)]
    frames += [_scene(f, left=VisualSignal.PASS, empty=True) for f in range(4, 13)]
    for scene in frames:
        update = tracker.update(scene)
        if update.event is not None:
            updates.append(update)
        if update.mode is VisualTrackerMode.FINISHED:
            break
    return updates


def test_blank_edge_seen_before_the_pause_still_proves_the_recovered_pass() -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=3)
    updates = _run(tracker, VisualSignal.NEUTRAL)
    assert [(u.event.actor, u.event.cards.cards) for u in updates] == [
        (PlayerSeat.LEFT, ()), (PlayerSeat.SELF, HAND),
    ]
    # The pass needs fresh post-pause frames; an armed pass with an unchanged
    # verified counter is two independent observations, hence two frames.
    assert updates[0].event.source == "live_visual"
    assert updates[1].event.source == "live_self_table_terminal"
    finished = updates[-1].state
    assert finished is not None
    assert finished.phase is RoundPhase.FINISHED
    assert finished.winner is PlayerSeat.SELF
    assert finished.revision == 34
    assert tracker.mode is VisualTrackerMode.FINISHED


def test_pause_resets_only_the_self_blank_edge() -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=3)
    tracker.update(_scene(1, left=VisualSignal.NEUTRAL))
    # RIGHT keeps showing the recorded trick target, so it never showed a blank.
    assert tracker._armed == {PlayerSeat.SELF: True, PlayerSeat.LEFT: True, PlayerSeat.RIGHT: False}
    paused = tracker.update(_scene(2, left=VisualSignal.PASS, self_turn=True,
                                   warnings=(GEOMETRY_WARNING,)))
    assert paused.event is None and paused.state is not None
    assert paused.state.phase is RoundPhase.UNCERTAIN
    assert not tracker._recent_scenes
    assert tracker._armed == {PlayerSeat.SELF: False, PlayerSeat.LEFT: True, PlayerSeat.RIGHT: False}


def test_pass_label_already_visible_before_the_pause_remains_stale() -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=3)
    events = _run(tracker, VisualSignal.PASS)
    assert events == []
    assert tracker.state is not None
    assert tracker.state.current_actor is PlayerSeat.LEFT
    assert tracker.state.revision == 32


def test_recovered_pass_still_needs_fresh_stability() -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=3)
    tracker.update(_scene(1, left=VisualSignal.NEUTRAL))
    for frame in (2, 3):
        tracker.update(_scene(frame, left=VisualSignal.PASS, self_turn=True,
                              warnings=(GEOMETRY_WARNING,)))
    first = tracker.update(_scene(4, left=VisualSignal.PASS, empty=True))
    assert first.event is None
    assert tracker.state is not None and tracker.state.revision == 32
