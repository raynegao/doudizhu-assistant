"""Synthetic regressions for a lost opponent/self pass pair after a crop pause."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import (
    VisualEventTracker,
    VisualTrackerMode,
    _following_play_pass_evidence,
)
from src.vision.scene_types import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)

HAND = ("K", "K", "Q", "Q", "J", "J", "10", "10")
GEOMETRY_WARNING = "self_hand_geometry_unavailable: candidate_count_unconfirmed"


def _state() -> ObservableGameState:
    state = ObservableGameState.from_inputs(
        HAND, round_id="double-pass-recovery", landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.LEFT,
        remaining_cards={PlayerSeat.SELF: 8, PlayerSeat.RIGHT: 3, PlayerSeat.LEFT: 14},
        played_cards="A A", last_play="A A", last_player=PlayerSeat.RIGHT,
        hidden_played_count=27,
    )
    event = ObservedAction(
        "double-pass-recovery:26:right", 26, PlayerSeat.RIGHT, CardSet.parse("A A"),
    )
    return replace(state, revision=26, last_sequence_no=26, history=(event,),
                   processed_events=((event.event_id, event.fingerprint),))


def _scene(
    frame: int, *, cards: tuple[str, ...] = ("2", "2"), remaining: int | None = 1,
    self_turn: bool | None = False, hand: tuple[str, ...] = HAND,
    hand_confidence: float = 0.667232,
) -> SceneObservation:
    seats = tuple(SeatObservation(
        seat=seat,
        signal=VisualSignal.PLAY if seat is PlayerSeat.RIGHT and cards else VisualSignal.NEUTRAL,
        cards=tuple(VisualCard(rank, 0.99, (0, 0, 10, 20)) for rank in cards)
        if seat is PlayerSeat.RIGHT else (),
        remaining_count=remaining if seat is PlayerSeat.RIGHT else (len(hand) if seat is PlayerSeat.SELF else 14),
        role=SeatRole.LANDLORD if seat is PlayerSeat.SELF else SeatRole.FARMER,
        confidence=0.99, remaining_confidence=0.99, role_confidence=0.99,
        remaining_verified=remaining is not None if seat is PlayerSeat.RIGHT else True,
    ) for seat in PlayerSeat)
    return SceneObservation(
        frame, float(frame), (0, 0, 100, 100),
        tuple(VisualCard(rank, hand_confidence if index == 5 else 0.99, (0, 0, 10, 20))
              for index, rank in enumerate(hand)),
        seats, self_turn, 0.99 if self_turn is not None else 0.0, 0.99,
    )


def _seat_change(scene: SceneObservation, seat: PlayerSeat, **changes: object) -> SceneObservation:
    return replace(scene, seats=tuple(
        replace(observation, **changes) if observation.seat is seat else observation
        for observation in scene.seats
    ))


def _paused_tracker() -> VisualEventTracker:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=3)
    tracker.update(_scene(1, cards=(), remaining=3))
    paused = tracker.update(replace(_scene(2), warnings=(GEOMETRY_WARNING,)))
    assert paused.event is None and paused.state is not None
    assert paused.state.phase is RoundPhase.UNCERTAIN
    assert not any(tracker._armed.values())
    assert not tracker._recent_scenes
    return tracker


@pytest.mark.parametrize("self_turn", [False, None])
def test_stable_following_play_recovers_two_passes_without_controls(self_turn: bool | None) -> None:
    tracker = _paused_tracker()
    # Recovery alone is not a new pass edge.
    old_pass = _seat_change(_scene(3, cards=(), remaining=3, self_turn=None),
                            PlayerSeat.LEFT, signal=VisualSignal.PASS, pass_confidence=0.99)
    assert tracker.update(old_pass).event is None
    assert tracker.update(_scene(4, self_turn=self_turn)).event is None
    assert tracker.update(_scene(5, self_turn=self_turn)).event is None
    scenes = [_scene(frame, self_turn=self_turn) for frame in (6, 7, 8, 9)]
    before = [scene.to_log_payload() for scene in scenes]
    updates = [tracker.update(scene) for scene in scenes]
    assert [scene.to_log_payload() for scene in scenes] == before
    assert all(scene.self_turn is self_turn for scene in scenes)
    assert [(item.event.actor, item.event.cards.cards) for item in updates if item.event] == [
        (PlayerSeat.LEFT, ()), (PlayerSeat.SELF, ()), (PlayerSeat.RIGHT, ("2", "2")),
    ]
    assert tracker.state is not None and tracker.state.revision == 29
    assert tracker.state.round_id == "double-pass-recovery"
    assert tracker.state.remaining_for(PlayerSeat.RIGHT) == 1
    assert tracker.state.self_hand == CardSet.parse(HAND)


@pytest.mark.parametrize("fault", [
    "stale_count", "mismatched_count", "unverified_count", "low_count", "low_play",
    "illegal_group", "invalid_rank", "exhausted_rank", "hand_changed", "hand_short",
    "hand_low", "hand_nan", "left_count_changed", "left_count_low",
    "left_visible_play", "role_changed", "geometry_unknown",
    "self_count_unknown", "self_count_wrong", "self_count_unverified",
])
def test_no_pass_is_committed_from_incomplete_or_contradictory_chain(fault: str) -> None:
    tracker = _paused_tracker()
    original = tracker.state
    for frame in range(3, 6):
        scene = _scene(frame)
        if fault == "stale_count":
            scene = _scene(frame, remaining=3)
        elif fault == "mismatched_count":
            scene = _scene(frame, remaining=0)
        elif fault in {"unverified_count", "low_count", "low_play"}:
            changes = {"remaining_verified": False} if fault == "unverified_count" else (
                {"remaining_confidence": 0.3} if fault == "low_count" else
                {"cards": tuple(replace(card, confidence=0.3) for card in scene.seat(PlayerSeat.RIGHT).cards)}
            )
            scene = _seat_change(scene, PlayerSeat.RIGHT, **changes)
        elif fault == "illegal_group":
            scene = _scene(frame, cards=("3", "4"))
        elif fault == "invalid_rank":
            scene = _scene(frame, cards=("bad-rank", "bad-rank"))
        elif fault == "exhausted_rank":
            scene = _scene(frame, cards=("K", "K", "K"), remaining=0)
        elif fault == "hand_changed":
            scene = _scene(frame, hand=(*HAND[:-1], "9"))
        elif fault == "hand_short":
            scene = _scene(frame, hand=HAND[:-1])
        elif fault in {"hand_low", "hand_nan"}:
            scene = _scene(frame, hand_confidence=0.0 if fault == "hand_low" else float("nan"))
        elif fault == "left_count_changed":
            scene = _seat_change(scene, PlayerSeat.LEFT, remaining_count=13)
        elif fault == "left_count_low":
            scene = _seat_change(scene, PlayerSeat.LEFT, remaining_confidence=0.2)
        elif fault == "left_visible_play":
            scene = _seat_change(scene, PlayerSeat.LEFT, signal=VisualSignal.PLAY,
                                cards=(VisualCard("3", 0.2, (0, 0, 10, 20)),))
        elif fault == "role_changed":
            scene = _seat_change(scene, PlayerSeat.SELF, role=SeatRole.FARMER)
            scene = _seat_change(scene, PlayerSeat.LEFT, role=SeatRole.LANDLORD)
        elif fault.startswith("self_count_"):
            scene = _seat_change(scene, PlayerSeat.SELF,
                                remaining_count=None if fault == "self_count_unknown" else 8,
                                remaining_verified=fault != "self_count_unverified")
            if fault == "self_count_wrong":
                scene = _seat_change(scene, PlayerSeat.SELF, remaining_count=7)
        else:
            scene = replace(scene, warnings=(GEOMETRY_WARNING,))
        assert tracker.update(scene).event is None
    assert tracker._tracker is not None and tracker._tracker.state is original


def test_repeated_or_out_of_order_frames_cannot_supply_three_proofs() -> None:
    tracker = _paused_tracker()
    for frame in (3, 3, 3, 2, 3, 4):
        assert tracker.update(_scene(frame)).event is None
    assert tracker.update(_scene(5)).event is not None


def test_geometry_pause_drops_previously_accumulated_following_play() -> None:
    tracker = _paused_tracker()
    assert tracker.update(_scene(3)).event is None
    assert tracker.update(_scene(4)).event is None
    assert tracker.update(replace(_scene(5), warnings=(GEOMETRY_WARNING,))).event is None
    assert tracker.update(_scene(6)).event is None
    assert tracker.update(_scene(7)).event is None
    assert tracker.update(_scene(8)).event is not None


@pytest.mark.parametrize("fault", ["low_hand", "different_hand", "bad_following", "wrong_count"])
def test_first_pass_does_not_authorize_second_after_evidence_changes(fault: str) -> None:
    tracker = _paused_tracker()
    for frame in (3, 4, 5):
        first = tracker.update(_scene(frame))
    assert first.event is not None and first.event.actor is PlayerSeat.LEFT
    for frame in (6, 7, 8):
        scene = _scene(frame)
        if fault == "low_hand":
            scene = _scene(frame, hand_confidence=0.0)
        elif fault == "different_hand":
            scene = _scene(frame, hand=(*HAND[:-1], "9"))
        elif fault == "bad_following":
            scene = _scene(frame, cards=("3", "4"))
        else:
            scene = _scene(frame, remaining=2)
        assert tracker.update(scene).event is None
    assert tracker.state is not None and tracker.state.revision == 27


def test_final_single_card_uses_existing_terminal_rule_after_new_pass_pair() -> None:
    tracker = _paused_tracker()
    for frame in range(3, 9):
        tracker.update(_scene(frame))
    assert tracker.state is not None and tracker.state.revision == 29
    tracker.update(replace(_scene(9, cards=(), remaining=1), warnings=(GEOMETRY_WARNING,)))
    assert tracker.update(_scene(10, cards=(), remaining=1, self_turn=None)).event is None
    for frame in (11, 12):
        assert tracker.update(_scene(frame, cards=("K",), remaining=None)).event is None
    updates = [tracker.update(_scene(frame, cards=("K",), remaining=None)) for frame in (13, 14, 15, 16)]
    assert [(item.event.actor, item.event.cards.cards) for item in updates if item.event] == [
        (PlayerSeat.LEFT, ()), (PlayerSeat.SELF, ()), (PlayerSeat.RIGHT, ("K",)),
    ]
    assert updates[-1].mode is VisualTrackerMode.FINISHED
    assert updates[-1].state is not None and updates[-1].state.revision == 32
    assert updates[-1].state.remaining_for(PlayerSeat.RIGHT) == 0
    assert updates[-1].state.winner is PlayerSeat.RIGHT


def test_terminal_residual_without_fresh_blank_edge_cannot_prove_passes() -> None:
    state = replace(_state(), remaining_cards=((PlayerSeat.SELF, 8), (PlayerSeat.RIGHT, 1),
                                              (PlayerSeat.LEFT, 14)), hidden_played_count=29)
    tracker = VisualEventTracker(initial_state=state, stability_frames=3)
    tracker.update(replace(_scene(1), warnings=(GEOMETRY_WARNING,)))
    for frame in range(2, 7):
        assert tracker.update(_scene(frame, cards=("K",), remaining=None)).event is None
    assert tracker.state is state


def test_same_cards_from_previous_round_cannot_continue_an_inferred_chain() -> None:
    tracker = _paused_tracker()
    for frame in (3, 4, 5):
        tracker.update(_scene(frame))
    assert tracker.state is not None and tracker.state.revision == 27
    other_round = replace(tracker.state, round_id="different-round")
    assert _following_play_pass_evidence(
        other_round, tuple(_scene(frame) for frame in (4, 5, 6)),
        confidence_threshold=0.7, stability_frames=3, min_frame_id=3, terminal_armed=True,
    ) is None


def test_positive_controls_cannot_supply_following_play_recovery_evidence() -> None:
    assert _following_play_pass_evidence(
        _state(), tuple(_scene(frame, self_turn=True) for frame in (3, 4, 5)),
        confidence_threshold=0.7, stability_frames=3, min_frame_id=3, terminal_armed=True,
    ) is None


def test_one_bad_middle_frame_restarts_full_chain_support() -> None:
    tracker = _paused_tracker()
    assert tracker.update(_scene(3)).event is None
    assert tracker.update(_scene(4, hand_confidence=0.0)).event is None
    assert tracker.update(_scene(5)).event is None
    assert tracker.update(_scene(6)).event is None
    assert tracker.update(_scene(7)).event is not None


@pytest.mark.parametrize("field", ["hand", "play", "count", "role"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_evidence_never_proves_a_pass(field: str, value: float) -> None:
    tracker = _paused_tracker()
    for frame in (3, 4, 5):
        scene = _scene(frame)
        if field == "hand":
            scene = _scene(frame, hand_confidence=value)
        elif field == "play":
            scene = _seat_change(scene, PlayerSeat.RIGHT,
                                cards=tuple(replace(card, confidence=value)
                                            for card in scene.seat(PlayerSeat.RIGHT).cards))
        elif field == "count":
            scene = _seat_change(scene, PlayerSeat.RIGHT, remaining_confidence=value)
        else:
            scene = _seat_change(scene, PlayerSeat.SELF, role_confidence=value)
        assert tracker.update(scene).event is None
    assert tracker.state is not None and tracker.state.revision == 26


@pytest.mark.parametrize("interruption", ["low_hand", "geometry", "different_play"])
def test_second_pass_requires_three_new_good_frames_after_interruption(interruption: str) -> None:
    tracker = _paused_tracker()
    for frame in (3, 4, 5):
        tracker.update(_scene(frame))
    interrupted = _scene(6, hand_confidence=0.0)
    if interruption == "geometry":
        interrupted = replace(_scene(6), warnings=(GEOMETRY_WARNING,))
    elif interruption == "different_play":
        interrupted = _scene(6, cards=("3", "3"))
    assert tracker.update(interrupted).event is None
    assert tracker.update(_scene(7)).event is None
    assert tracker.update(_scene(8)).event is None
    recovered = tracker.update(_scene(9))
    assert recovered.event is not None and recovered.event.actor is PlayerSeat.SELF


def test_duplicate_and_out_of_order_frames_after_first_pass_do_not_advance() -> None:
    tracker = _paused_tracker()
    for frame in (3, 4, 5):
        tracker.update(_scene(frame))
    for frame in (5, 5, 3, 4, 5, 6):
        assert tracker.update(_scene(frame)).event is None
    assert tracker.update(_scene(7)).event is not None


@pytest.mark.parametrize("blank", ["low_confidence", "unknown", "before_pause"])
def test_terminal_needs_trusted_post_pause_blank(blank: str) -> None:
    state = replace(_state(), remaining_cards=((PlayerSeat.SELF, 8), (PlayerSeat.RIGHT, 1),
                                              (PlayerSeat.LEFT, 14)), hidden_played_count=29)
    tracker = VisualEventTracker(initial_state=state, stability_frames=3)
    tracker.update(_scene(1, cards=(), remaining=1))
    tracker.update(replace(_scene(2), warnings=(GEOMETRY_WARNING,)))
    neutral = _scene(3, cards=(), remaining=1)
    if blank == "low_confidence":
        neutral = _seat_change(neutral, PlayerSeat.RIGHT, confidence=0.3)
    elif blank == "unknown":
        neutral = _seat_change(neutral, PlayerSeat.RIGHT, signal=VisualSignal.UNKNOWN)
    else:
        neutral = _scene(3, cards=("K",), remaining=None)
    assert tracker.update(neutral).event is None
    for frame in range(4, 9):
        assert tracker.update(_scene(frame, cards=("K",), remaining=None)).event is None
    assert tracker.state is state


def test_real_current_scan_replaces_round_and_discards_old_proof_frames() -> None:
    tracker = _paused_tracker()
    for frame in (3, 4):
        assert tracker.update(_scene(frame)).event is None
    scanned = tracker.scan_current_scene(_scene(5, self_turn=True))
    assert scanned.initialized and scanned.state is not None
    assert scanned.state.round_id != "double-pass-recovery"
    assert scanned.state.revision == 0 and scanned.state.history == ()
    assert [scene.frame_id for scene in tracker._recent_scenes] == [5]


def _terminal_state() -> ObservableGameState:
    return replace(_state(), remaining_cards=((PlayerSeat.SELF, 8), (PlayerSeat.RIGHT, 1),
                                             (PlayerSeat.LEFT, 14)), hidden_played_count=29)


@pytest.mark.parametrize("stability", [1, 2, 3, 4, 5, 6, 8])
def test_configured_stability_preserves_terminal_blank_until_both_passes(stability: int) -> None:
    tracker = VisualEventTracker(initial_state=_terminal_state(), stability_frames=stability)
    assert tracker._recent_scenes.maxlen == max(6, 2 * stability)
    tracker.update(replace(_scene(1), warnings=(GEOMETRY_WARNING,)))
    assert tracker.update(_scene(2, cards=(), remaining=1)).event is None
    events = []
    for frame in range(3, 3 + 2 * stability + 3):
        update = tracker.update(_scene(frame, cards=("K",), remaining=None))
        if update.event is not None:
            events.append(update.event)
        if update.state is not None and update.state.winner is not None:
            assert update.state.winner is PlayerSeat.RIGHT
            break
    assert [(event.actor, event.cards.cards) for event in events] == [
        (PlayerSeat.LEFT, ()), (PlayerSeat.SELF, ()), (PlayerSeat.RIGHT, ("K",)),
    ]


@pytest.mark.parametrize("stability", [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("blank_frame", [9, 10])
def test_retransmitted_pre_pause_blank_never_proves_terminal(stability: int, blank_frame: int) -> None:
    tracker = VisualEventTracker(initial_state=_terminal_state(), stability_frames=stability)
    tracker.update(replace(_scene(10), warnings=(GEOMETRY_WARNING,)))
    assert tracker.update(_scene(blank_frame, cards=(), remaining=1)).event is None
    for frame in range(12, 16 + 2 * stability):
        assert tracker.update(_scene(frame, cards=("K",), remaining=None)).event is None
    assert tracker.state is not None and tracker.state.revision == 26


@pytest.mark.parametrize("stability", [1, 2, 3, 4, 5, 6, 8])
def test_one_untrusted_frame_breaks_configured_stable_chain(stability: int) -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=stability)
    tracker.update(replace(_scene(1), warnings=(GEOMETRY_WARNING,)))
    for frame in range(2, stability + 1):
        assert tracker.update(_scene(frame)).event is None
    bad_frame = stability + 1
    assert tracker.update(_scene(bad_frame, hand_confidence=0.0)).event is None
    for frame in range(bad_frame + 1, bad_frame + stability):
        assert tracker.update(_scene(frame)).event is None
    assert tracker.update(_scene(bad_frame + stability)).event is not None


@pytest.mark.parametrize("stability", [1, 2, 3, 4, 5, 6, 8])
def test_first_pass_retransmission_never_submits_second_with_any_stability(stability: int) -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=stability)
    tracker.update(replace(_scene(1), warnings=(GEOMETRY_WARNING,)))
    for frame in range(2, 2 + stability):
        update = tracker.update(_scene(frame))
    assert update.event is not None and update.event.actor is PlayerSeat.LEFT
    for frame in (stability + 1, stability, stability + 1):
        assert tracker.update(_scene(frame)).event is None
    assert tracker.state is not None and tracker.state.revision == 27


@pytest.mark.parametrize("stability", [1, 2, 3, 4, 5, 6, 8])
def test_scan_and_clear_recreate_configured_bounded_cache(stability: int) -> None:
    tracker = VisualEventTracker(initial_state=_state(), stability_frames=stability)
    tracker.scan_current_scene(_scene(50, self_turn=True))
    assert tracker._recent_scenes.maxlen == max(6, 2 * stability)
    assert tracker._scene_evidence_start_frame_id == 50
    tracker._clear_round()
    assert tracker._recent_scenes.maxlen == max(6, 2 * stability)
    assert tracker._scene_evidence_start_frame_id is None


def test_out_of_order_geometry_warning_cannot_rewind_new_evidence_floor() -> None:
    tracker = VisualEventTracker(initial_state=_terminal_state(), stability_frames=3)
    for frame in (10, 4):
        tracker.update(replace(_scene(frame), warnings=(GEOMETRY_WARNING,)))
    assert tracker._scene_evidence_start_frame_id == 11
    tracker.update(_scene(9, cards=(), remaining=1))
    for frame in range(12, 20):
        assert tracker.update(_scene(frame, cards=("K",), remaining=None)).event is None


@pytest.mark.parametrize("stability", [1, 2, 3, 4, 5, 6, 8])
def test_complete_initial_scene_uses_configured_cache_and_new_epoch(stability: int) -> None:
    tracker = VisualEventTracker(stability_frames=stability, initial_stability_frames=1)
    hand = ("3",) * 4 + ("4",) * 4 + ("5",) * 4 + ("6",) * 4 + ("7", "7", "SJ", "BJ")
    scene = _scene(20, cards=(), remaining=17, hand=hand, self_turn=True)
    scene = _seat_change(scene, PlayerSeat.LEFT, remaining_count=17)
    assert tracker.update(scene).initialized
    assert tracker._recent_scenes.maxlen == max(6, 2 * stability)
    assert tracker._scene_evidence_start_frame_id == 20
