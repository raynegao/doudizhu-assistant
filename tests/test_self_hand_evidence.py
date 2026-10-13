"""Pure tracker regressions for complete hand evidence and actor freshness."""

from __future__ import annotations

from dataclasses import replace

from src.state.cards import CardSet
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import (
    VisualEventTracker,
    VisualTrackerMode,
    _current_self_hand_evidence,
    _self_hand_change,
    _self_hand_evidence_key,
)
from src.vision.scene_types import (
    RoundResultObservation,
    RoundResultOutcome,
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)

HAND = ("BJ", "Q", "J", "J", "J", "10", "10", "9", "8")
GEOMETRY_WARNING = "self_hand_geometry_unavailable: ambiguous_gray_background"


def _card(rank: str, confidence: float = 0.99) -> VisualCard:
    return VisualCard(rank=rank, confidence=confidence, box=(0, 0, 10, 20))


def _state(
    hand: tuple[str, ...] = HAND,
    *,
    target: tuple[str, ...] = (),
    current_actor: PlayerSeat = PlayerSeat.SELF,
    last_player: PlayerSeat = PlayerSeat.LEFT,
) -> ObservableGameState:
    return ObservableGameState.from_inputs(
        hand,
        round_id="self-hand-evidence",
        landlord=PlayerSeat.SELF,
        current_actor=current_actor,
        remaining_cards={
            PlayerSeat.SELF: len(hand), PlayerSeat.RIGHT: 8, PlayerSeat.LEFT: 9,
        },
        played_cards=target,
        last_play=target,
        last_player=last_player if target else None,
        hidden_played_count=54 - len(hand) - 8 - 9 - len(target),
    )


def _scene(
    frame: int,
    hand: tuple[str, ...] = HAND,
    *,
    self_turn: bool | None = True,
    self_signal: VisualSignal = VisualSignal.NEUTRAL,
    right_signal: VisualSignal = VisualSignal.PASS,
    left_signal: VisualSignal = VisualSignal.PASS,
    right_cards: tuple[str, ...] = (),
    left_cards: tuple[str, ...] = (),
    right_remaining: int = 8,
    left_remaining: int = 9,
    last_confidence: float = 0.99,
) -> SceneObservation:
    seats = []
    for seat, signal, cards, remaining in (
        (PlayerSeat.SELF, self_signal, (), len(hand)),
        (PlayerSeat.RIGHT, right_signal, right_cards, right_remaining),
        (PlayerSeat.LEFT, left_signal, left_cards, left_remaining),
    ):
        seats.append(SeatObservation(
            seat=seat,
            signal=signal,
            cards=tuple(_card(rank) for rank in cards),
            remaining_count=remaining,
            role=SeatRole.LANDLORD if seat is PlayerSeat.SELF else SeatRole.FARMER,
            confidence=0.99,
            pass_confidence=0.99 if signal is VisualSignal.PASS else 0.0,
            remaining_confidence=0.99,
            role_confidence=0.99,
            remaining_verified=True,
        ))
    return SceneObservation(
        frame_id=frame,
        timestamp=float(frame),
        window_pixel_box=(0, 0, 100, 100),
        self_hand=tuple(
            _card(rank, last_confidence if index == len(hand) - 1 else 0.99)
            for index, rank in enumerate(hand)
        ),
        seats=tuple(seats),
        self_turn=self_turn,
        self_turn_confidence=0.99 if self_turn is not None else 0.0,
        confidence=0.99,
    )


def _quiet_scene(
    frame: int,
    *,
    self_turn: bool = False,
    self_signal: VisualSignal = VisualSignal.NEUTRAL,
) -> SceneObservation:
    return _scene(
        frame,
        self_turn=self_turn,
        self_signal=self_signal,
        right_signal=VisualSignal.NEUTRAL,
        left_signal=VisualSignal.PLAY,
        left_cards=("4",),
    )


def _fresh_tracker() -> VisualEventTracker:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state())
    tracker.update(_scene(
        9, right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    ))
    return tracker


def test_restored_ninth_slot_blocks_old_eight_slot_candidate() -> None:
    tracker = _fresh_tracker()
    assert tracker.update(_scene(10, HAND[:-1])).event is None
    for frame, rank, confidence in (
        (11, "7", 0.946851), (12, "5", 0.470634), (13, "5", 0.470634),
    ):
        scene = _scene(frame, (*HAND[:-1], rank), last_confidence=confidence)
        assert len(scene.self_hand) == 9
        update = tracker.update(scene)
        assert update.event is None
        assert tracker.state is not None and tracker.state.revision == 0


def test_same_slot_count_different_ranks_use_latest_action() -> None:
    tracker = _fresh_tracker()
    assert tracker.update(_scene(10, HAND[:-1])).event is None
    no_nine = (*HAND[:-2], "8")
    # The second candidate requires its own fresh cycle evidence; no stale
    # eight-card post-hand or pass edge from the first candidate is reusable.
    assert tracker.update(_scene(
        11, no_nine, right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    )).event is None
    applied = tracker.update(_scene(12, no_nine))
    assert applied.event is not None and applied.event.cards == CardSet.parse(("9",))


def test_latest_low_confidence_cannot_borrow_old_high_confidence() -> None:
    tracker = _fresh_tracker()
    assert tracker.update(_scene(10, HAND[:-1])).event is None
    for frame in (11, 12):
        scene = _scene(frame, HAND[:-1], last_confidence=0.47)
        assert len(scene.self_hand) == 8
        assert tracker.update(scene).event is None
    assert tracker.state is not None and tracker.state.revision == 0


def test_first_good_frame_after_low_confidence_restarts_support() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(10, HAND[:-1], last_confidence=0.47))
    first_good = _scene(
        11, HAND[:-1], right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    )
    assert tracker.update(first_good).event is None
    applied = tracker.update(_scene(12, HAND[:-1]))
    assert applied.event is not None and applied.event.cards == CardSet.parse(("8",))


def test_event_confidence_comes_from_latest_matching_hand() -> None:
    tracker = _fresh_tracker()
    assert tracker.update(_scene(10, HAND[:-1], last_confidence=0.96)).event is None
    applied = tracker.update(_scene(11, HAND[:-1], last_confidence=0.97))
    assert applied.event is not None and applied.event.confidence == 0.97


def test_restored_hand_breaks_candidate_support_and_pass_edges() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(10, HAND[:-1]))
    assert tracker.update(_scene(11)).event is None
    assert tracker.update(_scene(
        12, HAND[:-1], right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    )).event is None
    applied = tracker.update(_scene(13, HAND[:-1]))
    assert applied.event is not None and applied.event.cards == CardSet.parse(("8",))


def test_accepted_single_outlier_is_still_a_physical_slot() -> None:
    tracker = _fresh_tracker()
    first = _scene(10, HAND[:-1], last_confidence=0.60)
    assert len(first.self_hand) == 8
    assert tracker.update(first).event is None
    applied = tracker.update(_scene(11, HAND[:-1], last_confidence=0.61))
    assert applied.event is not None and applied.event.cards == CardSet.parse(("8",))
    assert applied.state is not None and len(applied.state.self_hand) == 8


def test_slot_confidence_eligibility_change_restarts_support() -> None:
    tracker = _fresh_tracker()
    tracker.update(_scene(10, HAND[:-1], last_confidence=0.60))
    assert tracker.update(_scene(
        11, HAND[:-1], right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    )).event is None
    applied = tracker.update(_scene(12, HAND[:-1]))
    assert applied.event is not None and applied.event.cards == CardSet.parse(("8",))


def test_hand_evidence_is_actor_revision_and_distinct_frame_scoped() -> None:
    state = _state()
    first = _scene(10, HAND[:-1])
    latest = _scene(11, HAND[:-1])
    change = _self_hand_change(state, latest, confidence_threshold=0.7)
    assert change is not None and change.error is None
    evidence, support = _current_self_hand_evidence(
        state, latest, change, [first, latest], confidence_threshold=0.7, min_frame_id=11,
    )
    assert evidence is latest and support == 1
    _, support = _current_self_hand_evidence(
        state, latest, change, [latest, latest], confidence_threshold=0.7, min_frame_id=None,
    )
    assert support == 1
    key = _self_hand_evidence_key(
        state, latest, change, confidence_threshold=0.7, min_frame_id=11,
    )
    revised = _self_hand_evidence_key(
        replace(state, revision=1), latest, change, confidence_threshold=0.7, min_frame_id=11,
    )
    assert key != revised


def test_matching_post_hand_preserves_brief_successor_play_proof() -> None:
    state = _state()
    first = _scene(
        10, HAND[:-1], right_signal=VisualSignal.PLAY,
        right_cards=("2",), right_remaining=7,
    )
    latest = _scene(11, HAND[:-1])
    change = _self_hand_change(state, latest, confidence_threshold=0.7)
    assert change is not None and change.error is None
    evidence, support = _current_self_hand_evidence(
        state, latest, change, [first, latest], confidence_threshold=0.7, min_frame_id=10,
    )
    assert evidence is first and support == 2


def test_old_pass_labels_cannot_commit_gray_selection_or_rescan() -> None:
    hand = (*HAND, "3", "3", "3", "7")
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(hand))
    tracker.update(_scene(90, hand))
    for frame in range(91, 105):
        update = tracker.update(_scene(frame))
        assert update.event is None and not update.initialized
        assert tracker.state is not None and tracker.state.revision == 0
        assert tracker.state.self_hand == CardSet.parse(hand)


def test_same_epoch_pass_before_current_hand_run_is_not_fresh() -> None:
    hand = (*HAND, "3", "3", "3", "7")
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(hand))
    tracker.update(_scene(
        80, hand, right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    ))
    tracker.update(_scene(81, hand))
    for frame in range(82, 90):
        update = tracker.update(_scene(frame))
        assert update.event is None and not update.initialized
    assert tracker.state is not None and tracker.state.revision == 0


def test_one_fresh_opponent_pass_cannot_prove_full_cycle() -> None:
    hand = (*HAND, "3", "3", "3", "7")
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(hand))
    tracker.update(_scene(90, hand, right_signal=VisualSignal.NEUTRAL))
    for frame in range(91, 99):
        update = tracker.update(_scene(frame))
        assert update.event is None and not update.initialized


def test_fresh_low_confidence_pass_cannot_authorize_gray_candidate() -> None:
    hand = (*HAND, "3", "3", "3", "7")
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(hand))
    tracker.update(_scene(
        90, hand, right_signal=VisualSignal.NEUTRAL, left_signal=VisualSignal.NEUTRAL,
    ))
    for frame in range(91, 97):
        scene = _scene(frame)
        scene = replace(scene, seats=tuple(
            replace(seat, pass_confidence=0.4)
            if seat.seat is PlayerSeat.LEFT else seat
            for seat in scene.seats
        ))
        update = tracker.update(scene)
        assert update.event is None and not update.initialized


def test_controls_never_appearing_do_not_mean_disappeared() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(target=("4",)))
    for frame in range(1, 10):
        assert tracker.update(_quiet_scene(frame)).event is None
    assert tracker.state is not None and tracker.state.revision == 0


def test_low_confidence_active_hint_cannot_arm_self_pass() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(target=("4",)))
    tracker.update(replace(_quiet_scene(1, self_turn=True), self_turn_confidence=0.4))
    for frame in range(2, 9):
        assert tracker.update(_quiet_scene(frame)).event is None
    assert tracker.state is not None and tracker.state.revision == 0


def test_actor_entry_cannot_borrow_inactive_stability() -> None:
    state = _state(
        target=("4",), current_actor=PlayerSeat.LEFT, last_player=PlayerSeat.RIGHT,
    )
    tracker = VisualEventTracker(stability_frames=3, initial_state=state)
    for frame in range(1, 6):
        assert tracker.update(_scene(
            frame, right_signal=VisualSignal.PLAY, right_cards=("4",),
            left_signal=VisualSignal.NEUTRAL, self_turn=False,
        )).event is None
    left_play = _scene(
        6, right_signal=VisualSignal.PLAY, right_cards=("4",),
        left_signal=VisualSignal.PLAY, left_cards=("5",), left_remaining=8,
        self_turn=False,
    )
    applied = tracker.update(left_play)
    assert applied.event is not None and applied.event.actor is PlayerSeat.LEFT
    for frame in range(7, 12):
        assert tracker.update(replace(
            left_play, frame_id=frame, timestamp=float(frame),
        )).event is None
    assert tracker.update(replace(
        left_play, frame_id=12, timestamp=12.0, self_turn=True,
    )).event is None
    for frame in range(13, 16):
        update = tracker.update(replace(left_play, frame_id=frame, timestamp=float(frame)))
        assert (update.event is not None) is (frame == 15)
    assert update.event is not None and update.event.is_pass


def test_fresh_explicit_self_pass_survives_missed_active_controls() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(target=("4",)))
    assert tracker.update(_quiet_scene(1)).event is None
    for frame in range(2, 5):
        update = tracker.update(_quiet_scene(frame, self_signal=VisualSignal.PASS))
        assert (update.event is not None) is (frame == 4)
    assert update.event is not None and update.event.is_pass


def test_old_self_pass_cannot_arm_controls_never_seen() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(target=("4",)))
    for frame in range(1, 9):
        assert tracker.update(_quiet_scene(frame, self_signal=VisualSignal.PASS)).event is None


def test_control_appearance_proof_survives_long_current_turn() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(target=("4",)))
    for frame in range(1, 21):
        assert tracker.update(_quiet_scene(frame, self_turn=True)).event is None
    for frame in range(21, 24):
        update = tracker.update(_quiet_scene(frame))
        assert (update.event is not None) is (frame == 23)
    assert update.event is not None and update.event.is_pass


def test_strict_successor_play_preserves_pass_with_invalid_rank_slot() -> None:
    tracker = VisualEventTracker(stability_frames=1, initial_state=_state(target=("4",)))
    assert tracker.update(_quiet_scene(1, self_turn=True)).event is None
    scene = _scene(
        2, ("unreadable", *HAND[1:]), self_turn=False,
        right_signal=VisualSignal.PLAY, right_cards=("8",), right_remaining=7,
        left_signal=VisualSignal.PLAY, left_cards=("4",),
    )
    assert len(scene.self_hand) == len(HAND)
    update = tracker.update(scene)
    assert update.event is not None and update.event.actor is PlayerSeat.SELF
    assert update.event.is_pass


def test_unknown_geometry_is_not_empty_last_hand_and_recovery_uses_new_evidence() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(("8",)))
    tracker.update(_scene(1, ("8",)))
    unavailable = replace(
        _scene(2, (), self_turn=None), warnings=(GEOMETRY_WARNING,), confidence=0.0,
    )
    paused = tracker.update(unavailable)
    assert paused.event is None and not paused.initialized
    assert paused.mode is VisualTrackerMode.TRACKING
    assert paused.state is not None and not paused.state.decision_ready
    assert tracker.state is not None and tracker.state.self_hand == CardSet.parse(("8",))
    assert tracker.state.revision == 0
    assert tracker._self_turn_evidence is None
    assert not tracker._recent_scenes
    # Even an incorrectly confident inactive-control hint cannot override the
    # explicit geometry-unavailable warning and finish a legal final hand.
    contradictory = replace(unavailable, frame_id=3, self_turn=False, self_turn_confidence=0.99)
    assert tracker.update(contradictory).event is None
    recovered = tracker.update(_scene(4, ("8",)))
    assert recovered.event is None and recovered.state is not None
    assert recovered.state.decision_ready
    finished = tracker.update(_scene(5, (), self_turn=False))
    assert finished.event is not None and finished.event.cards == CardSet.parse(("8",))
    assert finished.mode is VisualTrackerMode.FINISHED


def test_unknown_geometry_blocks_bootstrap_and_clears_cycle_evidence() -> None:
    tracker = _fresh_tracker()
    assert tracker.update(_scene(10, HAND[:-1])).event is None
    unavailable = replace(
        _scene(11, (), self_turn=None), warnings=(GEOMETRY_WARNING,), confidence=0.0,
    )
    assert tracker.update(unavailable).event is None
    for frame in (12, 13, 14):
        assert tracker.update(_scene(frame, HAND[:-1])).event is None
    assert tracker.state is not None and tracker.state.revision == 0
    blank_tracker = VisualEventTracker(stability_frames=1)
    assert not blank_tracker.update(unavailable).initialized
    assert blank_tracker.state is None


def test_explicit_result_overlay_retains_priority_over_unknown_geometry() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_state=_state(("8",)))
    result = replace(
        _scene(1, (), self_turn=None),
        warnings=(GEOMETRY_WARNING,),
        round_result=RoundResultObservation(
            outcome=RoundResultOutcome.VICTORY, confidence=0.99,
        ),
    )
    first = tracker.update(result)
    assert first.completed_round_id is None
    confirmed = tracker.update(replace(result, frame_id=2, timestamp=2.0))
    assert confirmed.completed_round_id == "self-hand-evidence"
    assert confirmed.state is None


def test_manual_scan_refuses_unknown_geometry_even_with_confident_stale_hand() -> None:
    original = _state()
    tracker = VisualEventTracker(stability_frames=3, initial_state=original)
    misleading = replace(_scene(1), warnings=(GEOMETRY_WARNING,))
    assert misleading.self_turn is True and misleading.self_turn_confidence == 0.99
    assert len(misleading.self_hand) == 9
    rejected = tracker.scan_current_scene(misleading)
    assert not rejected.initialized and rejected.event is None
    assert rejected.state is not None and not rejected.state.decision_ready
    assert tracker.state == original
    assert GEOMETRY_WARNING in rejected.warnings
    new_tracker = VisualEventTracker(stability_frames=1)
    assert not new_tracker.scan_current_scene(misleading).initialized
    assert new_tracker.state is None
