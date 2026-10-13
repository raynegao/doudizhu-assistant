"""Transient visual failures must not mutate the accepted round or its log."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import VisualEventTracker, VisualTrackerMode
from src.vision.scene_types import (
    RoundResultObservation,
    RoundResultOutcome,
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)

HAND = tuple("4 5 7 8 9 10 J K A 2 2 SJ".split())


def _state() -> ObservableGameState:
    state = ObservableGameState.from_inputs(
        HAND, round_id="transient-test", landlord=PlayerSeat.RIGHT,
        current_actor=PlayerSeat.RIGHT,
        remaining_cards={PlayerSeat.SELF: 12, PlayerSeat.RIGHT: 16, PlayerSeat.LEFT: 17},
        played_cards="3 3 3 3 4 6 Q Q Q", last_play="4", last_player=PlayerSeat.SELF,
    )
    previous = ObservedAction("transient-test:6:self", 6, PlayerSeat.SELF, CardSet(("4",)))
    return replace(state, revision=6, last_sequence_no=6, history=(previous,),
                   processed_events=((previous.event_id, previous.fingerprint),))


def _scene(
    frame: int,
    ranks: tuple[str, ...] = ("3",),
    *,
    state: ObservableGameState | None = None,
    signal: VisualSignal | None = None,
    confidence: float = 0.99,
    remaining: int | None = 15,
    self_turn: bool = False,
) -> SceneObservation:
    state = state or _state()
    seats = tuple(SeatObservation(
        seat=seat,
        signal=(signal or (VisualSignal.PLAY if ranks else VisualSignal.NEUTRAL))
        if seat is PlayerSeat.RIGHT else VisualSignal.NEUTRAL,
        cards=tuple(VisualCard(rank, confidence, (0, 0, 10, 20)) for rank in ranks)
        if seat is PlayerSeat.RIGHT else (),
        remaining_count=remaining if seat is PlayerSeat.RIGHT else state.remaining_for(seat),
        role=SeatRole.LANDLORD if seat is state.landlord else SeatRole.FARMER,
        confidence=confidence if seat is PlayerSeat.RIGHT else 0.99,
        remaining_confidence=0.99,
        remaining_verified=True,
        role_confidence=0.99,
    ) for seat in PlayerSeat)
    return SceneObservation(
        frame, float(frame), (0, 0, 200, 100),
        tuple(VisualCard(rank, 0.99, (0, 0, 10, 20)) for rank in state.self_hand.cards),
        seats, self_turn, 0.99, 0.99,
    )


def _fresh_tracker() -> VisualEventTracker:
    return VisualEventTracker(initial_state=_state(), stability_frames=3, initial_stability_frames=2)


def _assert_unchanged(tracker: VisualEventTracker, original: ObservableGameState) -> None:
    assert tracker._tracker is not None
    assert tracker._tracker.state is original
    assert tracker._tracker.pending_events == ()
    assert tracker._tracker.state.history == original.history
    assert tracker._tracker.state.processed_events == original.processed_events
    assert tracker._tracker.state.remaining_cards == original.remaining_cards
    assert tracker._tracker.state.played_cards == original.played_cards


def test_bad_first_frame_recovers_next_frame_in_same_round() -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None
    paused = tracker.update(_scene(1))
    assert paused.mode is VisualTrackerMode.TRACKING
    assert tracker.mode is VisualTrackerMode.TRACKING
    assert paused.state is not None and paused.state.phase is RoundPhase.UNCERTAIN
    assert not paused.state.decision_ready
    assert paused.event is None
    _assert_unchanged(tracker, original)

    recovered = tracker.update(_scene(2, ("9",)))
    assert recovered.event is not None
    assert recovered.event.cards.cards == ("9",)
    assert recovered.event.sequence_no == 7
    assert recovered.state is not None
    assert recovered.state.round_id == original.round_id
    assert recovered.state.revision == 7
    assert recovered.state.remaining_for(PlayerSeat.RIGHT) == 15
    assert recovered.state.current_actor is PlayerSeat.LEFT
    assert tracker._invalid_play_evidence is None
    assert tracker.update(_scene(3, ("9",))).event is None


@pytest.mark.parametrize("signal,confidence", [(VisualSignal.PLAY, 0.99), (VisualSignal.UNKNOWN, 0.68)])
def test_legal_counter_supported_play_still_commits_in_one_frame(
    signal: VisualSignal, confidence: float,
) -> None:
    tracker = _fresh_tracker()
    update = tracker.update(_scene(1, ("9",), signal=signal, confidence=confidence))
    assert update.event is not None and update.event.cards.cards == ("9",)
    assert update.state is not None and update.state.revision == 7


@pytest.mark.parametrize("signal,confidence", [(VisualSignal.PLAY, 0.99), (VisualSignal.UNKNOWN, 0.68)])
def test_only_fresh_continuous_illegal_frames_reach_hard_stop(
    signal: VisualSignal, confidence: float,
) -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None
    for frame in (1, 2):
        update = tracker.update(_scene(frame, signal=signal, confidence=confidence))
        assert update.mode is VisualTrackerMode.TRACKING
        assert update.event is None
    update = tracker.update(_scene(3, signal=signal, confidence=confidence))
    assert update.mode is VisualTrackerMode.UNCERTAIN
    assert update.event is None
    _assert_unchanged(tracker, original)


@pytest.mark.parametrize("interruption", [
    "neutral", "low", "other_rank", "other_count", "unverified_count", "low_count_confidence",
])
def test_interrupted_illegal_evidence_cannot_accumulate(interruption: str) -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    if interruption == "neutral":
        middle = _scene(2, ())
    elif interruption == "low":
        middle = _scene(2, signal=VisualSignal.UNKNOWN, confidence=0.3)
    elif interruption == "other_rank":
        middle = _scene(2, ("4",))
    elif interruption == "other_count":
        middle = _scene(2, remaining=14)
    else:
        middle = _scene(2)
        middle = replace(middle, seats=tuple(
            replace(seat,
                    remaining_verified=interruption != "unverified_count",
                    remaining_confidence=0.4 if interruption == "low_count_confidence" else 0.99)
            if seat.seat is PlayerSeat.RIGHT else seat for seat in middle.seats
        ))
    tracker.update(middle)
    restarted = tracker.update(_scene(3))
    assert restarted.mode is VisualTrackerMode.TRACKING
    assert restarted.event is None
    assert tracker._invalid_play_evidence is not None
    assert tracker._invalid_play_evidence.count == 1
    assert tracker.update(_scene(4)).mode is VisualTrackerMode.TRACKING
    assert tracker.update(_scene(5)).mode is VisualTrackerMode.UNCERTAIN


def test_duplicate_and_out_of_order_source_frames_do_not_accumulate() -> None:
    tracker = _fresh_tracker()
    for frame in (10, 10, 10, 9, 10):
        update = tracker.update(_scene(frame))
        assert update.mode is VisualTrackerMode.TRACKING
        assert update.event is None
    assert tracker._invalid_play_evidence is not None
    assert tracker._invalid_play_evidence.count == 1


def test_old_buffered_illegal_frame_cannot_borrow_empty_frame_stability() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    for frame in range(2, 6):
        update = tracker.update(_scene(frame, ()))
        assert update.mode is VisualTrackerMode.TRACKING
        assert update.event is None
    assert tracker._invalid_play_evidence is not None
    assert tracker._invalid_play_evidence.count == 0
    recovered = tracker.update(_scene(6, ("9",)))
    assert recovered.event is not None and recovered.event.cards.cards == ("9",)


def test_successor_play_in_same_capture_is_still_recovered_from_buffer() -> None:
    tracker = _fresh_tracker()
    scene = _scene(1, ("9",))
    scene = replace(scene, seats=tuple(
        replace(seat, signal=VisualSignal.PLAY,
                cards=(VisualCard("J", 0.99, (0, 0, 10, 20)),), remaining_count=16)
        if seat.seat is PlayerSeat.LEFT else seat for seat in scene.seats
    ))
    first = tracker.update(scene)
    assert first.event is not None and first.event.actor is PlayerSeat.RIGHT
    second = tracker.update(_scene(2, ()))
    assert second.event is not None and second.event.actor is PlayerSeat.LEFT
    assert second.event.cards.cards == ("J",)
    assert second.state is not None and second.state.revision == 8


@pytest.mark.parametrize("ranks", [("3",), ("4", "5"), ("invalid-rank",), ("SJ", "SJ")])
def test_illegal_rule_or_cardset_candidate_is_only_soft_rejected(ranks: tuple[str, ...]) -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None
    update = tracker.update(_scene(1, ranks, remaining=16 - len(ranks)))
    assert update.event is None
    assert update.mode is VisualTrackerMode.TRACKING
    assert update.state is not None and update.state.phase is RoundPhase.UNCERTAIN
    _assert_unchanged(tracker, original)


def test_rank_that_can_beat_target_but_is_outside_unknown_pool_is_soft_rejected() -> None:
    state = ObservableGameState.from_inputs(
        "9 9 9 9 5", round_id="exhausted-nine", landlord=PlayerSeat.RIGHT,
        current_actor=PlayerSeat.RIGHT,
        remaining_cards={PlayerSeat.SELF: 5, PlayerSeat.RIGHT: 16, PlayerSeat.LEFT: 17},
        played_cards="4", last_play="4", last_player=PlayerSeat.SELF, hidden_played_count=15,
    )
    tracker = VisualEventTracker(initial_state=state)
    update = tracker.update(_scene(1, ("9",), state=state))
    assert update.event is None and update.mode is VisualTrackerMode.TRACKING
    assert "outside the unknown pool" in " ".join(update.warnings)
    _assert_unchanged(tracker, state)


def test_counter_mismatch_precheck_does_not_call_mutating_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None and tracker._tracker is not None
    def forbidden(_event: ObservedAction) -> None:
        raise AssertionError("invalid candidate reached the mutable tracker")
    monkeypatch.setattr(tracker._tracker, "apply", forbidden)
    scene = _scene(1, ("9",), remaining=14)
    event = ObservedAction("candidate-7", 7, PlayerSeat.RIGHT, CardSet(("9",)))
    update = tracker._apply_event(scene, scene.seat(PlayerSeat.RIGHT), event)
    assert update.mode is VisualTrackerMode.TRACKING and update.event is None
    assert "remaining count mismatch" in " ".join(update.warnings)
    _assert_unchanged(tracker, original)


def test_rule_rejection_precheck_does_not_call_mutating_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None and tracker._tracker is not None
    def forbidden(_event: ObservedAction) -> None:
        raise AssertionError("rule-invalid candidate reached the mutable tracker")
    monkeypatch.setattr(tracker._tracker, "apply", forbidden)
    update = tracker.update(_scene(1))
    assert update.event is None and update.mode is VisualTrackerMode.TRACKING
    assert "cannot beat" in " ".join(update.warnings)
    _assert_unchanged(tracker, original)


@pytest.mark.parametrize("failure", ["actor", "sequence", "event_id"])
def test_metadata_conflicts_still_hard_stop(failure: str) -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None
    scene = _scene(1, ("9",))
    event = ObservedAction(
        "transient-test:6:self" if failure == "event_id" else "candidate-7",
        8 if failure == "sequence" else 7,
        PlayerSeat.LEFT if failure == "actor" else PlayerSeat.RIGHT,
        CardSet(("9",)),
    )
    update = tracker._apply_event(scene, scene.seat(PlayerSeat.RIGHT), event)
    assert update.mode is VisualTrackerMode.UNCERTAIN
    _assert_unchanged(tracker, original)


@pytest.mark.parametrize("signal", [VisualSignal.PLAY, VisualSignal.UNKNOWN])
@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_nonfinite_or_out_of_range_visual_confidence_never_reaches_event(
    signal: VisualSignal, confidence: float,
) -> None:
    tracker = _fresh_tracker()
    original = tracker.state
    assert original is not None
    update = tracker.update(_scene(1, ("9",), signal=signal, confidence=confidence))
    assert update.event is None
    _assert_unchanged(tracker, original)


def test_unknown_bad_remaining_confidence_cannot_create_nonfinite_event() -> None:
    tracker = _fresh_tracker()
    scene = _scene(1, ("9",), signal=VisualSignal.UNKNOWN, confidence=0.68)
    scene = replace(scene, seats=tuple(
        replace(seat, remaining_confidence=float("nan"))
        if seat.seat is PlayerSeat.RIGHT else seat for seat in scene.seats
    ))
    assert tracker.update(scene).event is None
    assert tracker.state is not None and tracker.state.revision == 6


def test_active_rescan_cannot_steal_pending_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _fresh_tracker()
    def forbidden(*_args: object) -> None:
        raise AssertionError("pending visual candidate entered active rescan")
    monkeypatch.setattr(tracker, "_recover_drifted_state", forbidden)
    for frame in (1, 2):
        update = tracker.update(_scene(frame, self_turn=True))
        assert not update.initialized and update.event is None
        assert update.mode is VisualTrackerMode.TRACKING
    update = tracker.update(_scene(3, ("9",), self_turn=True))
    assert update.event is not None
    assert update.state is not None and update.state.round_id == "transient-test"


def _later_left_play(frame: int) -> SceneObservation:
    # Right's real play was missed. Left's later J and both reduced counters
    # are now reliable; no stale right-card crop can reconstruct the history.
    scene = _scene(frame, (), self_turn=True)
    return replace(scene, seats=tuple(
        replace(seat, signal=VisualSignal.PLAY, remaining_count=16,
                cards=(VisualCard("J", 0.99, (0, 0, 10, 20)),))
        if seat.seat is PlayerSeat.LEFT else seat for seat in scene.seats
    ))


def test_expired_illegal_source_allows_existing_stable_rescan_after_missed_play() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    updates = [tracker.update(_later_left_play(frame)) for frame in range(2, 13)]
    # The old six-frame buffer gives event recovery priority. Once it expires,
    # rescan must still meet the existing six-frame drift-confirmation gate.
    assert all(not update.initialized for update in updates[:-1])
    assert all(update.state is not None and update.state.round_id == "transient-test"
               for update in updates[:-1])
    recovered = updates[-1]
    assert recovered.initialized and recovered.event is None
    assert recovered.state is not None
    assert recovered.state.round_id == "auto-scan-12000"
    assert recovered.state.trick_target.cards == ("J",)
    assert recovered.state.hidden_played_count == 10
    assert tracker._invalid_play_evidence is None


def test_active_illegal_run_still_blocks_rescan_until_hard_stop() -> None:
    tracker = _fresh_tracker()
    updates = [tracker.update(_scene(frame, self_turn=True)) for frame in (1, 2, 3)]
    assert [update.mode for update in updates] == [
        VisualTrackerMode.TRACKING, VisualTrackerMode.TRACKING, VisualTrackerMode.UNCERTAIN,
    ]
    assert all(not update.initialized and update.event is None for update in updates)
    assert all(update.state is not None and update.state.round_id == "transient-test"
               and update.state.revision == 6 for update in updates)


def test_zero_count_retains_watermark_while_same_old_buffer_is_reused() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    for frame in range(2, 7):
        update = tracker.update(_later_left_play(frame))
        assert update.mode is VisualTrackerMode.TRACKING and update.event is None
        assert not update.initialized
        assert tracker._invalid_play_evidence is not None
        assert tracker._invalid_play_evidence.count == 0
        assert tracker._invalid_play_evidence.last_source_frame_id == 1


def test_legal_right_play_still_recovers_original_revision_during_rescan_wait() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    for frame in range(2, 9):
        assert not tracker.update(_later_left_play(frame)).initialized
    # A fresh valid right play arrives after the bad buffer expired, while
    # active rescan is accumulating support. Normal event recovery wins.
    scene = _later_left_play(9)
    scene = replace(scene, seats=tuple(
        replace(seat, signal=VisualSignal.PLAY,
                cards=(VisualCard("9", 0.99, (0, 0, 10, 20)),))
        if seat.seat is PlayerSeat.RIGHT else seat for seat in scene.seats
    ))
    right = tracker.update(scene)
    assert not right.initialized
    assert right.event is not None and right.event.cards.cards == ("9",)
    assert right.state is not None and right.state.round_id == "transient-test"
    assert right.state.revision == 7
    left = tracker.update(replace(scene, frame_id=10, timestamp=10.0))
    assert left.event is not None and left.event.cards.cards == ("J",)
    assert left.state is not None and left.state.round_id == "transient-test"
    assert left.state.revision == 8


def test_unchanged_counter_ignores_old_invalid_card_without_starting_pending() -> None:
    tracker = _fresh_tracker()
    for frame in range(1, 5):
        assert tracker.update(_scene(frame, remaining=16)).event is None
    assert tracker._invalid_play_evidence is None
    assert tracker.state is not None and tracker.state.revision == 6


@pytest.mark.parametrize("lifecycle", ["window", "geometry", "result", "manual", "new_round"])
def test_lifecycle_clears_transient_evidence(lifecycle: str) -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    assert tracker._invalid_play_evidence is not None
    if lifecycle == "window":
        tracker.handle_window_unavailable("test window loss")
    elif lifecycle == "geometry":
        tracker.update(replace(_scene(2), warnings=("self_hand_geometry_unavailable: test",)))
    elif lifecycle == "result":
        result = RoundResultObservation(RoundResultOutcome.DEFEAT, 0.99)
        tracker.update(replace(_scene(2), round_result=result))
        tracker.update(replace(_scene(3), round_result=result))
    elif lifecycle == "manual":
        scan = tracker.scan_current_scene(_scene(2, ("9",), self_turn=True))
        assert scan.initialized
    else:
        fresh = ObservableGameState.from_inputs(
            tuple("SJ 2 2 A K Q Q Q J 10 9 8 7 5 4 4 3".split()),
            landlord=PlayerSeat.RIGHT,
            remaining_cards={PlayerSeat.SELF: 17, PlayerSeat.RIGHT: 20, PlayerSeat.LEFT: 17},
        )
        tracker.update(_scene(2, (), state=fresh, remaining=20))
        update = tracker.update(_scene(3, (), state=fresh, remaining=20))
        assert update.initialized and update.state is not None
        assert update.state.round_id != "transient-test"
    assert tracker._invalid_play_evidence is None


def test_new_actor_does_not_inherit_previous_invalid_counter() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(1))
    tracker.update(_scene(2))
    tracker.update(_scene(3, ("9",)))
    scene = _scene(4, ("9",))
    scene = replace(scene, seats=tuple(
        replace(seat, signal=VisualSignal.PLAY, remaining_count=16,
                cards=(VisualCard("8", 0.99, (0, 0, 10, 20)),))
        if seat.seat is PlayerSeat.LEFT else seat for seat in scene.seats
    ))
    update = tracker.update(scene)
    assert update.mode is VisualTrackerMode.TRACKING and update.event is None
    assert tracker._invalid_play_evidence is not None
    assert tracker._invalid_play_evidence.count == 1


def test_terminal_play_with_disappeared_counter_keeps_one_frame_support() -> None:
    state = ObservableGameState.from_inputs(
        HAND, round_id="terminal", landlord=PlayerSeat.RIGHT, current_actor=PlayerSeat.RIGHT,
        remaining_cards={PlayerSeat.SELF: 12, PlayerSeat.RIGHT: 1, PlayerSeat.LEFT: 17},
        played_cards="4", last_play="4", last_player=PlayerSeat.SELF, hidden_played_count=23,
    )
    tracker = VisualEventTracker(initial_state=state)
    tracker.update(_scene(1, (), state=state, remaining=1))
    update = tracker.update(_scene(2, ("9",), state=state, remaining=None))
    assert update.event is not None
    assert update.mode is VisualTrackerMode.FINISHED
    assert update.state is not None and update.state.remaining_for(PlayerSeat.RIGHT) == 0


def test_runtime_does_not_log_soft_rejected_candidate_as_accepted_action(tmp_path: Path) -> None:
    from PIL import Image

    from src.capture.screen_geometry import CapturedWindow, ScreenGeometry
    from src.config.live_layout import LiveLayoutConfig
    from src.pipeline.calibration import WindowInfo
    from src.pipeline.live_runtime import LiveGameRuntime

    class Source:
        def capture(self, frame_id: int) -> CapturedWindow:
            return CapturedWindow(
                frame_id, float(frame_id), Image.new("RGB", (200, 100), "navy"),
                WindowInfo("test", "test", (0, 0, 200, 100)),
                (0, 0, 200, 100), ScreenGeometry((200, 100), (200, 100)),
            )

    class Recognizer:
        calls = 0

        def observe(self, frame: CapturedWindow) -> SceneObservation:
            self.calls += 1
            return _scene(frame.frame_id, ("3",) if self.calls == 1 else ("9",))

    config = LiveLayoutConfig(log_file=tmp_path / "events.jsonl", error_frames_dir=tmp_path / "errors")
    runtime = LiveGameRuntime(config, frame_source=Source(), recognizer=Recognizer(),
                              tracker=_fresh_tracker())
    try:
        first = runtime.run_once()
        second = runtime.run_once()
    finally:
        runtime.close()
    assert first.state is not None and first.state.revision == 6
    assert second.state is not None and second.state.revision == 7
    rows = [json.loads(line) for line in config.log_file.read_text().splitlines()]
    actions = [row for row in rows if row["event"] in ("play_observed", "pass_observed")]
    assert len(actions) == 1
    assert actions[0]["actor"] == "right"
    assert actions[0]["cards"] == ["9"]
    assert actions[0]["round_id"] == "transient-test"
