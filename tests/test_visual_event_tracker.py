from __future__ import annotations

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


def _card(rank: str) -> VisualCard:
    return VisualCard(rank=rank, confidence=0.99, box=(0, 0, 10, 20))


def _scene(
    *,
    frame_id: int,
    self_signal: VisualSignal = VisualSignal.NEUTRAL,
    self_cards: tuple[str, ...] = (),
    right_signal: VisualSignal = VisualSignal.NEUTRAL,
    right_cards: tuple[str, ...] = (),
    right_remaining: int = 17,
    left_signal: VisualSignal = VisualSignal.NEUTRAL,
    left_cards: tuple[str, ...] = (),
    left_remaining: int = 17,
) -> SceneObservation:
    seats = (
        SeatObservation(
            seat=PlayerSeat.SELF,
            signal=self_signal,
            cards=tuple(_card(rank) for rank in self_cards),
            remaining_count=20,
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
        self_hand=tuple(_card(rank) for rank in LANDLORD_HAND),
        seats=seats,
        self_turn=None,
        self_turn_confidence=0.0,
        confidence=0.99,
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
        right_cards=("4",),
        right_remaining=17,
    ))

    assert update.mode is VisualTrackerMode.UNCERTAIN
    assert update.state is not None
    assert update.state.phase.value == "uncertain"
    assert "remaining count mismatch" in update.message


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
