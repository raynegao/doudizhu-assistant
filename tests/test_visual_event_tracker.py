from __future__ import annotations

from dataclasses import replace

from src.state.events import PlayerSeat
from src.tracking.visual_events import (
    VisualEventTracker,
    VisualTrackerMode,
)
from src.vision.scene_recognizer import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)


LANDLORD_HAND = (
    "3", "3", "3", "3",
    "4", "4", "4", "4",
    "5", "5", "5", "5",
    "6", "6", "6", "6",
    "7", "7", "7", "7",
)

FARMER_HAND = (
    "BJ", "SJ", "2", "2", "A", "Q", "J", "9", "9",
    "8", "8", "7", "7", "6", "6", "4", "3",
)

REAL_LANDLORD_HAND = (
    "2", "2", "2", "A", "K", "Q", "J", "J", "9", "9",
    "8", "7", "6", "6", "5", "5", "4", "4", "3", "3",
)


def _card(rank: str, confidence: float = 0.99) -> VisualCard:
    return VisualCard(rank=rank, confidence=confidence, box=(0, 0, 10, 20))


def _scene(
    *,
    frame_id: int,
    self_signal: VisualSignal = VisualSignal.NEUTRAL,
    self_cards: tuple[str, ...] = (),
    right_signal: VisualSignal = VisualSignal.NEUTRAL,
    right_cards: tuple[str, ...] = (),
    right_remaining: int = 17,
    right_remaining_verified: bool = True,
    left_signal: VisualSignal = VisualSignal.NEUTRAL,
    left_cards: tuple[str, ...] = (),
    left_remaining: int = 17,
    visible_hand: tuple[str, ...] | None = None,
    self_turn: bool | None = None,
) -> SceneObservation:
    hand = LANDLORD_HAND if visible_hand is None else visible_hand
    seats = (
        SeatObservation(
            seat=PlayerSeat.SELF,
            signal=self_signal,
            cards=tuple(_card(rank) for rank in self_cards),
            remaining_count=len(hand),
            role=SeatRole.LANDLORD,
            confidence=0.99,
            pass_confidence=0.99 if self_signal is VisualSignal.PASS else 0.0,
            remaining_confidence=0.99,
            role_confidence=0.99,
        ),
        SeatObservation(
            seat=PlayerSeat.RIGHT,
            signal=right_signal,
            cards=tuple(_card(rank) for rank in right_cards),
            remaining_count=right_remaining,
            role=SeatRole.FARMER,
            confidence=0.99,
            pass_confidence=0.99 if right_signal is VisualSignal.PASS else 0.0,
            remaining_confidence=0.99,
            role_confidence=0.99,
            remaining_verified=right_remaining_verified,
        ),
        SeatObservation(
            seat=PlayerSeat.LEFT,
            signal=left_signal,
            cards=tuple(_card(rank) for rank in left_cards),
            remaining_count=left_remaining,
            role=SeatRole.FARMER,
            confidence=0.99,
            pass_confidence=0.99 if left_signal is VisualSignal.PASS else 0.0,
            remaining_confidence=0.99,
            role_confidence=0.99,
        ),
    )
    return SceneObservation(
        frame_id=frame_id,
        timestamp=float(frame_id),
        window_pixel_box=(0, 0, 100, 100),
        self_hand=tuple(_card(rank) for rank in hand),
        seats=seats,
        self_turn=self_turn,
        self_turn_confidence=0.99 if self_turn is not None else 0.0,
        confidence=0.99,
    )


def _landlord_opening_scene(
    *,
    frame_id: int,
    landlord_remaining: int = 16,
    landlord_remaining_verified: bool = False,
) -> SceneObservation:
    opening = ("9", "8", "7", "6", "5", "4", "3")
    hand = tuple(
        _card(rank, 0.614 if rank == "A" else 0.916)
        for rank in FARMER_HAND
    )
    seats = (
        SeatObservation(
            seat=PlayerSeat.SELF,
            signal=VisualSignal.NEUTRAL,
            remaining_count=17,
            role=SeatRole.FARMER,
            confidence=0.812,
            remaining_confidence=0.614,
            role_confidence=0.999,
            remaining_verified=True,
        ),
        SeatObservation(
            seat=PlayerSeat.RIGHT,
            signal=VisualSignal.NEUTRAL,
            remaining_count=17,
            role=SeatRole.FARMER,
            confidence=1.0,
            remaining_confidence=0.949,
            role_confidence=0.971,
            remaining_verified=False,
        ),
        SeatObservation(
            seat=PlayerSeat.LEFT,
            signal=VisualSignal.PLAY,
            cards=tuple(
                _card(rank, 0.739 if rank == "9" else 0.98)
                for rank in opening
            ),
            remaining_count=landlord_remaining,
            role=SeatRole.LANDLORD,
            confidence=0.739,
            remaining_confidence=0.980,
            role_confidence=1.0,
            remaining_verified=landlord_remaining_verified,
        ),
    )
    return SceneObservation(
        frame_id=frame_id,
        timestamp=float(frame_id),
        window_pixel_box=(0, 0, 100, 100),
        self_hand=hand,
        seats=seats,
        self_turn=True,
        self_turn_confidence=1.0,
        confidence=0.614,
    )


def _scene_with_landlord(
    scene: SceneObservation,
    landlord: PlayerSeat,
) -> SceneObservation:
    return replace(
        scene,
        seats=tuple(
            replace(
                observation,
                role=(
                    SeatRole.LANDLORD
                    if observation.seat is landlord
                    else SeatRole.FARMER
                ),
                role_confidence=0.99,
            )
            for observation in scene.seats
        ),
    )


def test_visual_tracker_initializes_and_advances_play_pass_round() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        round_id_factory=lambda _: "round-1",
    )
    tracker.update(_scene(frame_id=1))
    initialized = tracker.update(_scene(frame_id=2))

    assert initialized.initialized is True
    assert initialized.state is not None
    assert initialized.state.current_actor is PlayerSeat.SELF

    tracker.update(_scene(
        frame_id=3,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))
    self_play = tracker.update(_scene(
        frame_id=4,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))
    assert self_play.event is not None
    assert self_play.state is not None
    assert self_play.state.remaining_for(PlayerSeat.SELF) == 19
    assert self_play.state.current_actor is PlayerSeat.RIGHT

    tracker.update(_scene(
        frame_id=5,
        self_signal=VisualSignal.NEUTRAL,
        right_signal=VisualSignal.PASS,
    ))
    right_pass = tracker.update(_scene(
        frame_id=6,
        self_signal=VisualSignal.NEUTRAL,
        right_signal=VisualSignal.PASS,
    ))
    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.state is not None
    assert right_pass.state.current_actor is PlayerSeat.LEFT

    tracker.update(_scene(
        frame_id=7,
        left_signal=VisualSignal.PASS,
    ))
    left_pass = tracker.update(_scene(
        frame_id=8,
        left_signal=VisualSignal.PASS,
    ))
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.state is not None
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert not left_pass.state.trick_target


def test_visual_tracker_pauses_on_transient_landlord_mismatch() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        round_id_factory=lambda _: "round-role-transient",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(frame_id=2))

    mismatch = tracker.update(_scene_with_landlord(
        _scene(frame_id=3),
        PlayerSeat.LEFT,
    ))
    recovered = tracker.update(_scene(frame_id=4))

    assert mismatch.mode is VisualTrackerMode.TRACKING
    assert mismatch.state is not None
    assert mismatch.state.decision_ready is False
    assert "地主位置已变化" in mismatch.message
    assert recovered.mode is VisualTrackerMode.TRACKING
    assert recovered.state is not None
    assert recovered.state.decision_ready is True


def test_visual_tracker_blocks_persistent_landlord_mismatch() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        round_id_factory=lambda _: "round-role-boundary",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(frame_id=2))

    tracker.update(_scene_with_landlord(
        _scene(frame_id=3),
        PlayerSeat.LEFT,
    ))
    boundary = tracker.update(_scene_with_landlord(
        _scene(frame_id=4),
        PlayerSeat.LEFT,
    ))

    assert boundary.mode is VisualTrackerMode.UNCERTAIN
    assert boundary.state is not None
    assert boundary.state.decision_ready is False
    assert "新牌局" in boundary.message


def test_visual_tracker_uses_faster_post_bidding_bootstrap_threshold() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=2,
        round_id_factory=lambda _: "round-post-bidding",
    )

    stabilizing = tracker.update(_scene(frame_id=1, self_turn=True))
    initialized = tracker.update(_scene(frame_id=2, self_turn=True))

    assert stabilizing.mode is VisualTrackerMode.WAITING_FOR_ROUND
    assert stabilizing.state is None
    assert "正在建立牌局 1/2" in stabilizing.message
    assert initialized.initialized is True
    assert initialized.state is not None
    assert initialized.state.current_actor is PlayerSeat.SELF
    assert initialized.state.decision_ready is True


def test_visual_tracker_derives_self_play_and_two_passes_from_scene_state() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        round_id_factory=lambda _: "round-hand-diff",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(frame_id=2))
    remaining_hand = LANDLORD_HAND[1:]

    stabilizing = tracker.update(_scene(
        frame_id=3,
        visible_hand=remaining_hand,
        self_turn=False,
    ))
    self_play = tracker.update(_scene(
        frame_id=4,
        visible_hand=remaining_hand,
        self_turn=False,
    ))

    assert stabilizing.event is None
    assert self_play.event is not None
    assert self_play.event.cards.cards == ("3",)
    assert self_play.event.source == "live_hand_diff"
    assert self_play.state is not None
    assert self_play.state.remaining_for(PlayerSeat.SELF) == 19
    assert self_play.state.current_actor is PlayerSeat.RIGHT

    right_pass = tracker.update(_scene(
        frame_id=5,
        visible_hand=remaining_hand,
        self_turn=True,
    ))
    left_pass = tracker.update(_scene(
        frame_id=6,
        visible_hand=remaining_hand,
        self_turn=True,
    ))

    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.event.actor is PlayerSeat.RIGHT
    assert right_pass.event.source == "live_turn_inferred_pass"
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.event.actor is PlayerSeat.LEFT
    assert left_pass.state is not None
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert not left_pass.state.trick_target
    assert left_pass.state.decision_ready is True


def test_visual_tracker_continues_after_self_play_opponent_bomb_and_pass() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=2,
        round_id_factory=lambda _: "round-real-regression",
    )
    initial = _scene(
        frame_id=1,
        visible_hand=REAL_LANDLORD_HAND,
        self_turn=True,
    )
    tracker.update(initial)
    tracker.update(_scene(
        frame_id=2,
        visible_hand=REAL_LANDLORD_HAND,
        self_turn=True,
    ))
    remaining_hand = tuple(
        rank
        for rank in REAL_LANDLORD_HAND
        if rank not in {"7", "2"}
    )
    current = dict(
        visible_hand=remaining_hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("10", "10", "10", "10"),
        right_remaining=13,
        left_remaining=17,
        self_turn=True,
    )

    tracker.update(_scene(frame_id=3, **current))
    self_play = tracker.update(_scene(frame_id=4, **current))
    tracker.update(_scene(frame_id=5, **current))
    tracker.update(_scene(frame_id=6, **current))
    right_play = tracker.update(_scene(frame_id=7, **current))
    left_pass = tracker.update(_scene(frame_id=8, **current))

    assert self_play.event is not None
    assert self_play.event.cards.cards == ("7", "2", "2", "2")
    assert right_play.event is not None
    assert right_play.event.cards.cards == ("10", "10", "10", "10")
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.state is not None
    assert left_pass.state.revision == 3
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert left_pass.state.trick_target.cards == ("10", "10", "10", "10")
    assert left_pass.state.decision_ready is True

    later = dict(
        visible_hand=remaining_hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("4",),
        right_remaining=12,
        left_signal=VisualSignal.PLAY,
        left_cards=("8",),
        left_remaining=16,
        self_turn=True,
    )
    inferred_self_pass = tracker.update(_scene(frame_id=9, **later))
    tracker.update(_scene(frame_id=10, **later))
    tracker.update(_scene(frame_id=11, **later))
    right_lead = tracker.update(_scene(frame_id=12, **later))
    tracker.update(_scene(frame_id=13, **later))
    tracker.update(_scene(frame_id=14, **later))
    left_play = tracker.update(_scene(frame_id=15, **later))

    assert inferred_self_pass.event is not None
    assert inferred_self_pass.event.actor is PlayerSeat.SELF
    assert inferred_self_pass.event.is_pass
    assert inferred_self_pass.event.source == "live_turn_inferred_pass"
    assert right_lead.event is not None
    assert right_lead.event.actor is PlayerSeat.RIGHT
    assert right_lead.event.cards.cards == ("4",)
    assert left_play.event is not None
    assert left_play.event.actor is PlayerSeat.LEFT
    assert left_play.event.cards.cards == ("8",)
    assert left_play.state is not None
    assert left_play.state.revision == 6
    assert left_play.state.current_actor is PlayerSeat.SELF
    assert left_play.state.trick_target.cards == ("8",)
    assert left_play.state.decision_ready is True


def test_visual_tracker_blocks_remaining_count_mismatch() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        round_id_factory=lambda _: "round-2",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(
        frame_id=2,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))
    update = tracker.update(_scene(
        frame_id=3,
        right_signal=VisualSignal.PLAY,
        right_cards=("8",),
        right_remaining=17,
    ))

    assert update.mode is VisualTrackerMode.UNCERTAIN
    assert update.state is not None
    assert update.state.phase.value == "uncertain"
    assert "remaining count mismatch" in update.message


def test_visual_tracker_ignores_unverified_remaining_template_mismatch() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        round_id_factory=lambda _: "round-unverified-count",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(
        frame_id=2,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))
    update = tracker.update(_scene(
        frame_id=3,
        right_signal=VisualSignal.PLAY,
        right_cards=("8",),
        right_remaining=17,
        right_remaining_verified=False,
    ))

    assert update.mode is VisualTrackerMode.TRACKING
    assert update.event is not None
    assert update.state is not None
    assert update.state.remaining_for(PlayerSeat.RIGHT) == 16


def test_visual_tracker_marks_active_round_uncertain_when_window_is_lost() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        round_id_factory=lambda _: "round-window-loss",
    )
    initialized = tracker.update(_scene(frame_id=1))
    assert initialized.mode is VisualTrackerMode.TRACKING

    update = tracker.handle_window_unavailable("斗地主窗口已最小化")

    assert update.mode is VisualTrackerMode.UNCERTAIN
    assert update.state is not None
    assert update.state.phase.value == "uncertain"
    assert "等待下一局" in update.message


def test_visual_tracker_refuses_mid_round_initialization() -> None:
    tracker = VisualEventTracker(stability_frames=1)

    update = tracker.update(_scene(frame_id=1, right_remaining=12))

    assert update.mode is VisualTrackerMode.WAITING_FOR_ROUND
    assert update.state is None


def test_visual_tracker_can_infer_missing_initial_landlord_20() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        round_id_factory=lambda _: "round-bootstrap",
    )
    hand = LANDLORD_HAND[:17]
    scene = SceneObservation(
        frame_id=1,
        timestamp=1.0,
        window_pixel_box=(0, 0, 100, 100),
        self_hand=tuple(_card(rank) for rank in hand),
        seats=(
            SeatObservation(
                seat=PlayerSeat.SELF,
                signal=VisualSignal.NEUTRAL,
                remaining_count=17,
                role=SeatRole.FARMER,
                confidence=0.99,
                remaining_confidence=0.99,
                role_confidence=0.99,
            ),
            SeatObservation(
                seat=PlayerSeat.RIGHT,
                signal=VisualSignal.NEUTRAL,
                remaining_count=17,
                role=SeatRole.FARMER,
                confidence=0.99,
                remaining_confidence=0.99,
                role_confidence=0.99,
            ),
            SeatObservation(
                seat=PlayerSeat.LEFT,
                signal=VisualSignal.NEUTRAL,
                remaining_count=None,
                role=SeatRole.LANDLORD,
                confidence=0.99,
                remaining_confidence=0.0,
                role_confidence=0.99,
            ),
        ),
        self_turn=False,
        self_turn_confidence=0.99,
        confidence=0.99,
    )

    update = tracker.update(scene)

    assert update.initialized is True
    assert update.state is not None
    assert update.state.remaining_for(PlayerSeat.LEFT) == 20


def test_visual_tracker_reconstructs_stable_landlord_opening_play() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        round_id_factory=lambda _: "round-opening-bootstrap",
    )

    first = tracker.update(_landlord_opening_scene(frame_id=1))
    initialized = tracker.update(_landlord_opening_scene(frame_id=2))

    assert first.mode is VisualTrackerMode.WAITING_FOR_ROUND
    assert initialized.mode is VisualTrackerMode.TRACKING
    assert initialized.initialized is True
    assert initialized.event is not None
    assert initialized.event.source == "live_visual_bootstrap"
    assert initialized.state is not None
    assert initialized.state.revision == 1
    assert initialized.state.remaining_for(PlayerSeat.LEFT) == 13
    assert initialized.state.remaining_for(PlayerSeat.RIGHT) == 17
    assert initialized.state.current_actor is PlayerSeat.SELF
    assert set(initialized.state.trick_target.cards) == {
        "3", "4", "5", "6", "7", "8", "9",
    }
    assert initialized.state.state_confidence == 0.739
    assert initialized.state.decision_ready is True
    assert any(
        warning.startswith("accepted_single_hand_confidence_outlier:A=")
        for warning in initialized.state.warnings
    )
    assert any(
        warning.startswith("accepted_single_hand_confidence_outlier:A=")
        for warning in initialized.warnings
    )
    assert "安全重建" in initialized.message

    same_opening = tracker.update(_landlord_opening_scene(frame_id=3))
    assert same_opening.initialized is False
    assert same_opening.state is not None
    assert same_opening.state.round_id == "round-opening-bootstrap"
    assert same_opening.state.revision == 1


def test_visual_tracker_refuses_opening_when_verified_count_conflicts() -> None:
    tracker = VisualEventTracker(stability_frames=1)

    update = tracker.update(_landlord_opening_scene(
        frame_id=1,
        landlord_remaining=16,
        landlord_remaining_verified=True,
    ))

    assert update.mode is VisualTrackerMode.WAITING_FOR_ROUND
    assert update.state is None


def test_visual_tracker_manual_scan_rebuilds_midgame_with_unknown_history() -> None:
    tracker = VisualEventTracker(stability_frames=3)
    visible_hand = REAL_LANDLORD_HAND[:16]

    update = tracker.scan_current_scene(_scene(
        frame_id=20,
        visible_hand=visible_hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("4",),
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    ))

    assert update.initialized is True
    assert update.mode is VisualTrackerMode.TRACKING
    assert update.state is not None
    assert update.state.current_actor is PlayerSeat.SELF
    assert update.state.trick_leader is PlayerSeat.LEFT
    assert update.state.trick_target.cards == ("4",)
    assert update.state.hidden_played_count == 18
    assert update.state.decision_ready is True
    assert "中途近似模型" in update.message
    assert "historical_played_cards_unknown=18" in update.state.warnings


def test_visual_tracker_auto_scans_stable_midgame_on_self_turn() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=2,
    )
    scene = _scene(
        frame_id=30,
        visible_hand=REAL_LANDLORD_HAND[:16],
        left_signal=VisualSignal.PLAY,
        left_cards=("4",),
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    )

    stabilizing = tracker.update(scene)
    initialized = tracker.update(replace(
        scene,
        frame_id=31,
        timestamp=31.0,
    ))

    assert stabilizing.state is None
    assert "正在自动扫描当前牌局 1/2" in stabilizing.message
    assert initialized.initialized is True
    assert initialized.mode is VisualTrackerMode.TRACKING
    assert initialized.state is not None
    assert initialized.state.round_id.startswith("auto-scan-")
    assert initialized.state.current_actor is PlayerSeat.SELF
    assert initialized.state.trick_target.cards == ("4",)
    assert "automatic_current_game_scan" in initialized.state.warnings


def test_visual_tracker_auto_recovers_after_window_loss() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
        round_id_factory=lambda _: "round-before-window-loss",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(frame_id=2))
    tracker.handle_window_unavailable("斗地主窗口已最小化")
    recovered_scene = _scene(
        frame_id=3,
        visible_hand=LANDLORD_HAND[1:],
        left_signal=VisualSignal.PLAY,
        left_cards=("8",),
        left_remaining=16,
        right_remaining=17,
        self_turn=True,
    )

    stabilizing = tracker.update(recovered_scene)
    recovered = tracker.update(replace(
        recovered_scene,
        frame_id=4,
        timestamp=4.0,
    ))

    assert stabilizing.mode is VisualTrackerMode.UNCERTAIN
    assert stabilizing.state is not None
    assert stabilizing.state.decision_ready is False
    assert recovered.initialized is True
    assert recovered.mode is VisualTrackerMode.TRACKING
    assert recovered.state is not None
    assert recovered.state.round_id.startswith("auto-scan-")
    assert recovered.state.trick_target.cards == ("8",)
    assert recovered.state.decision_ready is True


def test_auto_scanned_midgame_continues_through_full_trick_cycle() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
    )
    starting_hand = REAL_LANDLORD_HAND[:16]
    scan_scene = _scene(
        frame_id=40,
        visible_hand=starting_hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("4",),
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    )
    tracker.update(scan_scene)
    initialized = tracker.update(replace(
        scan_scene,
        frame_id=41,
        timestamp=41.0,
    ))
    assert initialized.initialized is True

    right_play_scene = _scene(
        frame_id=42,
        visible_hand=starting_hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("5",),
        right_remaining=11,
        left_signal=VisualSignal.PLAY,
        left_cards=("4",),
        left_remaining=7,
        self_turn=False,
    )
    self_pass = tracker.update(right_play_scene)
    tracker.update(replace(
        right_play_scene,
        frame_id=43,
        timestamp=43.0,
    ))
    right_play = tracker.update(replace(
        right_play_scene,
        frame_id=44,
        timestamp=44.0,
    ))

    assert self_pass.event is not None and self_pass.event.is_pass
    assert self_pass.event.actor is PlayerSeat.SELF
    assert right_play.event is not None
    assert right_play.event.actor is PlayerSeat.RIGHT
    assert right_play.event.cards.cards == ("5",)

    left_pass = tracker.update(_scene(
        frame_id=45,
        visible_hand=starting_hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("5",),
        right_remaining=11,
        left_remaining=7,
        self_turn=True,
    ))
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.event.actor is PlayerSeat.LEFT
    assert left_pass.state is not None
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert left_pass.state.trick_target.cards == ("5",)

    after_single_cards = list(starting_hand)
    after_single_cards.remove("6")
    after_single = tuple(after_single_cards)
    self_play_scene = _scene(
        frame_id=46,
        visible_hand=after_single,
        right_remaining=11,
        left_remaining=7,
        self_turn=False,
    )
    tracker.update(self_play_scene)
    self_play = tracker.update(replace(
        self_play_scene,
        frame_id=47,
        timestamp=47.0,
    ))
    assert self_play.event is not None
    assert self_play.event.actor is PlayerSeat.SELF
    assert self_play.event.cards.cards == ("6",)

    right_pass = tracker.update(_scene(
        frame_id=48,
        visible_hand=after_single,
        right_remaining=11,
        left_remaining=7,
        self_turn=True,
    ))
    left_pass = tracker.update(_scene(
        frame_id=49,
        visible_hand=after_single,
        right_remaining=11,
        left_remaining=7,
        self_turn=True,
    ))

    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.event.actor is PlayerSeat.RIGHT
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.event.actor is PlayerSeat.LEFT
    assert left_pass.state is not None
    assert left_pass.state.revision == 6
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert not left_pass.state.trick_target
    assert left_pass.state.decision_ready is True


def test_visual_tracker_manual_scan_requires_self_turn() -> None:
    tracker = VisualEventTracker(stability_frames=3)

    update = tracker.scan_current_scene(_scene(
        frame_id=21,
        visible_hand=REAL_LANDLORD_HAND,
        self_turn=False,
    ))

    assert update.initialized is False
    assert update.state is None
    assert "轮到自己" in update.message
