from __future__ import annotations

from dataclasses import replace

from src.state.cards import CardSet
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState
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

LIVE_SELECTION_HAND = (
    "BJ", "SJ", "2", "A", "A", "K", "K", "K", "J", "J",
    "10", "9", "9", "8", "7", "6", "6", "5", "5", "4",
)

LIVE_SELECTION_ANIMATION_MISREAD = (
    "BJ", "SJ", "2", "A", "A", "K", "K", "K", "J", "J",
    "10", "9", "9", "7", "7", "6", "5", "4", "3", "3",
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
    right_remaining: int | None = 17,
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


def test_verified_remaining_drop_confirms_opponent_play_in_one_frame() -> None:
    state = ObservableGameState.from_inputs(
        LANDLORD_HAND[1:],
        round_id="round-fast-opponent",
        landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.RIGHT,
        remaining_cards={
            PlayerSeat.SELF: 19,
            PlayerSeat.RIGHT: 17,
            PlayerSeat.LEFT: 17,
        },
        played_cards=("3",),
        last_play=("3",),
        last_player=PlayerSeat.SELF,
    )
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_state=state,
    )

    update = tracker.update(_scene(
        frame_id=1,
        visible_hand=LANDLORD_HAND[1:],
        right_signal=VisualSignal.PLAY,
        right_cards=("8",),
        right_remaining=16,
        self_turn=False,
    ))

    assert update.event is not None
    assert update.event.actor is PlayerSeat.RIGHT
    assert update.event.cards == CardSet.parse(("8",))
    assert update.state is not None
    assert update.state.current_actor is PlayerSeat.LEFT


def test_compressed_two_passes_preserve_following_play_evidence() -> None:
    hand = ("3", "5", "6")
    state = ObservableGameState.from_inputs(
        hand,
        round_id="round-compressed-cycle",
        landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.SELF,
        remaining_cards={
            PlayerSeat.SELF: len(hand),
            PlayerSeat.RIGHT: 17,
            PlayerSeat.LEFT: 17,
        },
        played_cards=("4", "4", "4", "4"),
        last_play=("4", "4", "4", "4"),
        last_player=PlayerSeat.LEFT,
    )
    tracker = VisualEventTracker(stability_frames=3, initial_state=state)
    compressed = _scene(
        frame_id=1,
        visible_hand=hand,
        right_remaining=17,
        left_signal=VisualSignal.PLAY,
        left_cards=("A",),
        left_remaining=16,
        self_turn=False,
    )

    self_pass = tracker.update(compressed)
    right_pass = tracker.update(replace(
        compressed,
        frame_id=2,
        timestamp=2.0,
        seats=tuple(
            replace(seat, signal=VisualSignal.NEUTRAL, cards=())
            for seat in compressed.seats
        ),
    ))
    left_play = tracker.update(replace(
        compressed,
        frame_id=3,
        timestamp=3.0,
        seats=tuple(
            replace(seat, signal=VisualSignal.NEUTRAL, cards=())
            for seat in compressed.seats
        ),
    ))

    assert self_pass.event is not None and self_pass.event.is_pass
    assert self_pass.event.actor is PlayerSeat.SELF
    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.event.actor is PlayerSeat.RIGHT
    assert left_play.event is not None
    assert left_play.event.actor is PlayerSeat.LEFT
    assert left_play.event.cards == CardSet.parse(("A",))


def test_opponent_final_play_uses_tracked_count_when_counter_disappears() -> None:
    hand = ("5", "6", "7")
    state = ObservableGameState.from_inputs(
        hand,
        round_id="round-terminal-counter-hidden",
        landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.RIGHT,
        remaining_cards={
            PlayerSeat.SELF: len(hand),
            PlayerSeat.RIGHT: 1,
            PlayerSeat.LEFT: 17,
        },
        played_cards=("3",),
        hidden_played_count=32,
        last_play=("3",),
        last_player=PlayerSeat.SELF,
    )
    tracker = VisualEventTracker(stability_frames=3, initial_state=state)
    tracker.update(_scene(
        frame_id=1,
        visible_hand=hand,
        right_remaining=1,
        left_remaining=17,
        self_turn=False,
    ))

    finished = tracker.update(_scene(
        frame_id=2,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("A",),
        right_remaining=None,
        right_remaining_verified=False,
        left_remaining=17,
        self_turn=False,
    ))

    assert finished.event is not None
    assert finished.event.actor is PlayerSeat.RIGHT
    assert finished.event.cards == CardSet.parse(("A",))
    assert finished.mode is VisualTrackerMode.FINISHED
    assert finished.state is not None
    assert finished.state.remaining_for(PlayerSeat.RIGHT) == 0


def test_self_hand_difference_corrects_one_unique_ordered_rank_error() -> None:
    hand = ("5", "6", "7", "7", "7", "10", "J")
    state = ObservableGameState.from_inputs(
        hand,
        round_id="round-hand-correction",
        landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.SELF,
        remaining_cards={
            PlayerSeat.SELF: len(hand),
            PlayerSeat.RIGHT: 17,
            PlayerSeat.LEFT: 16,
        },
        played_cards=("6",),
        last_play=("6",),
        last_player=PlayerSeat.LEFT,
    )
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_state=state,
    )
    # True remaining hand after playing 10 is J, 7, 7, 7, 6, 5.  The middle
    # seven is transiently classified as 2 while selected cards settle.
    misread = ("J", "7", "2", "7", "6", "5")
    scene = _scene(
        frame_id=1,
        visible_hand=misread,
        right_remaining=17,
        left_remaining=16,
        self_turn=False,
    )

    applied = tracker.update(scene)

    assert applied.event is not None
    assert applied.event.cards == CardSet.parse(("10",))
    assert applied.state is not None
    assert applied.state.self_hand == CardSet.parse(("5", "6", "7", "7", "7", "J"))


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

    right_pass = tracker.update(_scene(
        frame_id=5,
        self_signal=VisualSignal.NEUTRAL,
        right_signal=VisualSignal.PASS,
    ))
    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.state is not None
    assert right_pass.state.current_actor is PlayerSeat.LEFT

    left_pass = tracker.update(_scene(
        frame_id=7,
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

    self_play = tracker.update(_scene(
        frame_id=3,
        visible_hand=remaining_hand,
        self_turn=False,
    ))
    tracker.update(_scene(
        frame_id=4,
        visible_hand=remaining_hand,
        self_turn=False,
    ))

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


def test_visual_tracker_recovers_self_play_when_both_opponents_pass_before_capture() -> None:
    tracker = VisualEventTracker(stability_frames=2)
    scanned = tracker.scan_current_scene(_scene(
        frame_id=100,
        visible_hand=("2", "K", "K", "J", "9"),
        left_signal=VisualSignal.PLAY,
        left_cards=("7", "7"),
        left_remaining=12,
        right_remaining=11,
        self_turn=True,
    ))
    assert scanned.initialized is True

    returned_to_self = _scene(
        frame_id=101,
        visible_hand=("2", "J", "9"),
        right_signal=VisualSignal.PASS,
        right_remaining=11,
        left_signal=VisualSignal.PASS,
        left_remaining=12,
        self_turn=True,
    )
    stabilizing = tracker.update(returned_to_self)
    self_play = tracker.update(replace(
        returned_to_self,
        frame_id=102,
        timestamp=102.0,
    ))
    right_pass = tracker.update(replace(
        returned_to_self,
        frame_id=103,
        timestamp=103.0,
    ))
    left_pass = tracker.update(replace(
        returned_to_self,
        frame_id=104,
        timestamp=104.0,
    ))

    assert stabilizing.event is None
    assert self_play.event is not None
    assert self_play.event.actor is PlayerSeat.SELF
    assert self_play.event.cards.cards == ("K", "K")
    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.event.actor is PlayerSeat.RIGHT
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.event.actor is PlayerSeat.LEFT
    assert left_pass.state is not None
    assert left_pass.state.revision == 3
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert left_pass.state.self_hand == CardSet.parse(("2", "J", "9"))
    assert not left_pass.state.trick_target
    assert left_pass.state.decision_ready is True


def test_low_confidence_turn_hint_cannot_create_inferred_pass() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        initial_stability_frames=1,
        round_id_factory=lambda _: "round-low-turn-pass",
    )
    tracker.update(_scene(frame_id=1, self_turn=True))
    remaining_hand = LANDLORD_HAND[1:]
    self_play = tracker.update(_scene(
        frame_id=2,
        visible_hand=remaining_hand,
        self_turn=False,
    ))
    assert self_play.event is not None
    assert self_play.state is not None
    assert self_play.state.current_actor is PlayerSeat.RIGHT

    low_turn = replace(
        _scene(
            frame_id=3,
            visible_hand=remaining_hand,
            right_remaining=17,
            self_turn=True,
        ),
        self_turn_confidence=0.40,
    )
    waiting = tracker.update(low_turn)

    assert waiting.event is None
    assert waiting.mode is VisualTrackerMode.TRACKING
    assert waiting.state is not None
    assert waiting.state.phase.value == "playing"
    assert waiting.state.current_actor is PlayerSeat.RIGHT


def test_low_confidence_remaining_count_cannot_create_inferred_pass() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        initial_stability_frames=1,
        round_id_factory=lambda _: "round-low-count-pass",
    )
    tracker.update(_scene(frame_id=1, self_turn=True))
    remaining_hand = LANDLORD_HAND[1:]
    self_play = tracker.update(_scene(
        frame_id=2,
        visible_hand=remaining_hand,
        self_turn=False,
    ))
    assert self_play.event is not None

    low_count = _scene(
        frame_id=3,
        visible_hand=remaining_hand,
        right_remaining=17,
        self_turn=True,
    )
    low_count = replace(
        low_count,
        seats=tuple(
            replace(
                observation,
                remaining_confidence=0.40,
            )
            if observation.seat is PlayerSeat.RIGHT
            else observation
            for observation in low_count.seats
        ),
    )
    waiting = tracker.update(low_count)

    assert waiting.event is None
    assert waiting.mode is VisualTrackerMode.TRACKING
    assert waiting.state is not None
    assert waiting.state.phase.value == "playing"
    assert waiting.state.current_actor is PlayerSeat.RIGHT


def test_stable_disappeared_self_turn_controls_confirm_self_pass() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=1,
    )
    hand = ("5", "9", "J", "K", "2")
    initialized = tracker.scan_current_scene(_scene(
        frame_id=1,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=True,
    ))
    assert initialized.initialized is True
    assert initialized.state is not None
    assert initialized.state.current_actor is PlayerSeat.SELF
    assert initialized.state.consecutive_passes == 1

    updates = [
        tracker.update(_scene(
            frame_id=frame_id,
            visible_hand=hand,
            right_signal=VisualSignal.PLAY,
            right_cards=("BJ",),
            right_remaining=13,
            left_remaining=14,
            self_turn=False,
        ))
        for frame_id in range(2, 5)
    ]

    assert updates[0].event is None
    assert updates[1].event is None
    confirmed = updates[2]
    assert confirmed.event is not None
    assert confirmed.event.actor is PlayerSeat.SELF
    assert confirmed.event.is_pass
    assert confirmed.event.source == "live_turn_inferred_pass"
    assert confirmed.state is not None
    assert confirmed.state.revision == 1
    assert confirmed.state.current_actor is PlayerSeat.RIGHT
    assert not confirmed.state.trick_target


def test_transient_disappeared_self_turn_controls_cannot_create_pass() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=1,
    )
    hand = ("5", "9", "J", "K", "2")
    tracker.scan_current_scene(_scene(
        frame_id=1,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=True,
    ))

    transient = tracker.update(_scene(
        frame_id=2,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=False,
    ))
    recovered = tracker.update(_scene(
        frame_id=3,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=True,
    ))

    assert transient.event is None
    assert recovered.event is None
    assert recovered.state is not None
    assert recovered.state.revision == 0
    assert recovered.state.current_actor is PlayerSeat.SELF
    assert recovered.state.trick_target.cards == ("BJ",)


def test_visible_self_pass_button_cannot_submit_pass_before_turn_ends() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=1,
    )
    hand = ("5", "9", "J", "K", "2")
    tracker.scan_current_scene(_scene(
        frame_id=1,
        visible_hand=hand,
        self_signal=VisualSignal.PASS,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=True,
    ))

    updates = [
        tracker.update(_scene(
            frame_id=frame_id,
            visible_hand=hand,
            self_signal=VisualSignal.PASS,
            right_signal=VisualSignal.PLAY,
            right_cards=("BJ",),
            right_remaining=13,
            left_remaining=14,
            self_turn=True,
        ))
        for frame_id in range(2, 7)
    ]

    assert all(update.event is None for update in updates)
    assert tracker.state is not None
    assert tracker.state.revision == 0
    assert tracker.state.current_actor is PlayerSeat.SELF
    assert tracker.state.trick_target.cards == ("BJ",)


def test_visual_tracker_finishes_on_empty_hand_then_initializes_next_round() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
        round_id_factory=lambda scene: f"round-{scene.frame_id}",
    )
    scanned = tracker.scan_current_scene(_scene(
        frame_id=1,
        visible_hand=("3",),
        right_remaining=17,
        left_remaining=17,
        self_turn=True,
    ))
    assert scanned.initialized is True
    assert scanned.state is not None
    first_round_id = scanned.state.round_id

    terminal_scene = _scene(
        frame_id=2,
        visible_hand=(),
        right_remaining=17,
        left_remaining=17,
        self_turn=False,
    )
    finished = tracker.update(terminal_scene)
    cleared = tracker.update(replace(
        terminal_scene,
        frame_id=3,
        timestamp=3.0,
    ))

    assert finished.event is not None
    assert finished.event.actor is PlayerSeat.SELF
    assert finished.event.cards.cards == ("3",)
    assert finished.mode is VisualTrackerMode.FINISHED
    assert finished.state is not None
    assert finished.state.phase.value == "finished"
    assert finished.state.remaining_for(PlayerSeat.SELF) == 0
    assert finished.state.winner is PlayerSeat.SELF

    assert cleared.mode is VisualTrackerMode.WAITING_FOR_ROUND
    assert cleared.state is None
    assert tracker.state is None
    assert "胜利" in cleared.message
    assert "状态已清除" in cleared.message

    next_round_scene = _scene(
        frame_id=5,
        visible_hand=LANDLORD_HAND,
        self_turn=True,
    )
    waiting = tracker.update(next_round_scene)
    next_round = tracker.update(replace(
        next_round_scene,
        frame_id=6,
        timestamp=6.0,
    ))

    assert waiting.mode is VisualTrackerMode.WAITING_FOR_ROUND
    assert next_round.initialized is True
    assert next_round.mode is VisualTrackerMode.TRACKING
    assert next_round.state is not None
    assert next_round.state.round_id != first_round_id
    assert next_round.state.revision == 0
    assert next_round.state.remaining_for(PlayerSeat.SELF) == 20


def test_visual_tracker_finishes_when_self_plays_a_final_pair() -> None:
    tracker = VisualEventTracker(stability_frames=2)
    scanned = tracker.scan_current_scene(_scene(
        frame_id=200,
        visible_hand=("K", "K"),
        left_signal=VisualSignal.PLAY,
        left_cards=("7", "7"),
        left_remaining=12,
        right_remaining=11,
        self_turn=True,
    ))
    assert scanned.initialized is True

    empty_hand = _scene(
        frame_id=201,
        visible_hand=(),
        left_remaining=12,
        right_remaining=11,
        self_turn=False,
    )
    finished = tracker.update(empty_hand)
    tracker.update(replace(
        empty_hand,
        frame_id=202,
        timestamp=202.0,
    ))

    assert finished.event is not None
    assert finished.event.actor is PlayerSeat.SELF
    assert finished.event.cards.cards == ("K", "K")
    assert finished.mode is VisualTrackerMode.FINISHED
    assert finished.state is not None
    assert finished.state.phase.value == "finished"
    assert finished.state.remaining_for(PlayerSeat.SELF) == 0
    assert finished.state.winner is PlayerSeat.SELF


def test_empty_crop_cannot_finish_an_invalid_multi_card_hand() -> None:
    tracker = VisualEventTracker(stability_frames=2)
    scanned = tracker.scan_current_scene(_scene(
        frame_id=210,
        visible_hand=("K", "J"),
        left_remaining=12,
        right_remaining=11,
        self_turn=True,
    ))
    assert scanned.initialized is True

    empty_hand = _scene(
        frame_id=211,
        visible_hand=(),
        left_remaining=12,
        right_remaining=11,
        self_turn=False,
    )
    tracker.update(empty_hand)
    update = tracker.update(replace(
        empty_hand,
        frame_id=212,
        timestamp=212.0,
    ))

    assert update.event is None
    assert update.mode is VisualTrackerMode.TRACKING
    assert update.state is not None
    assert update.state.phase.value == "playing"
    assert update.state.self_hand == CardSet.parse(("K", "J"))


def test_visual_tracker_finishes_when_opponent_plays_last_card() -> None:
    hand = ("5", "6", "7", "8", "9")
    initial_state = ObservableGameState.from_inputs(
        hand,
        round_id="round-opponent-final",
        landlord=PlayerSeat.SELF,
        current_actor=PlayerSeat.RIGHT,
        remaining_cards={
            PlayerSeat.SELF: len(hand),
            PlayerSeat.RIGHT: 1,
            PlayerSeat.LEFT: 17,
        },
        played_cards=("3",),
        hidden_played_count=30,
        last_play=("3",),
        last_player=PlayerSeat.SELF,
    )
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_state=initial_state,
    )
    terminal_scene = _scene(
        frame_id=1,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("A",),
        right_remaining=0,
        left_remaining=17,
        self_turn=False,
    )

    finished = tracker.update(terminal_scene)
    cleared = tracker.update(replace(
        terminal_scene,
        frame_id=2,
        timestamp=2.0,
    ))

    assert finished.event is not None
    assert finished.event.actor is PlayerSeat.RIGHT
    assert finished.event.cards.cards == ("A",)
    assert finished.mode is VisualTrackerMode.FINISHED
    assert finished.state is not None
    assert finished.state.phase.value == "finished"
    assert finished.state.remaining_for(PlayerSeat.RIGHT) == 0
    assert finished.state.winner is PlayerSeat.RIGHT
    assert cleared.mode is VisualTrackerMode.WAITING_FOR_ROUND


def test_selected_card_animation_cannot_replace_pristine_opening_hand() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=2,
        round_id_factory=lambda _: "round-selection-animation",
    )
    tracker.update(_scene(
        frame_id=1,
        visible_hand=LIVE_SELECTION_HAND,
        self_turn=True,
    ))
    initialized = tracker.update(_scene(
        frame_id=2,
        visible_hand=LIVE_SELECTION_HAND,
        self_turn=True,
    ))
    assert initialized.initialized is True

    selecting_first = tracker.update(_scene(
        frame_id=3,
        visible_hand=LIVE_SELECTION_ANIMATION_MISREAD,
        self_turn=True,
    ))
    selecting_stable = tracker.update(_scene(
        frame_id=4,
        visible_hand=LIVE_SELECTION_ANIMATION_MISREAD,
        self_turn=True,
    ))

    assert selecting_first.initialized is False
    assert selecting_stable.initialized is False
    assert selecting_stable.state is not None
    assert selecting_stable.state.round_id == "round-selection-animation"
    assert selecting_stable.state.self_hand == CardSet.parse(
        LIVE_SELECTION_HAND
    )

    remaining = list(LIVE_SELECTION_HAND)
    for rank in ("4", "5", "6", "7", "8", "9", "10", "J"):
        remaining.remove(rank)
    played = tracker.update(_scene(
        frame_id=5,
        visible_hand=tuple(remaining),
        self_turn=False,
    ))
    tracker.update(_scene(
        frame_id=6,
        visible_hand=tuple(remaining),
        self_turn=False,
    ))

    assert played.event is not None
    assert played.event.cards.cards == (
        "4", "5", "6", "7", "8", "9", "10", "J",
    )
    assert played.state is not None
    assert played.state.revision == 1
    assert played.state.current_actor is PlayerSeat.RIGHT
    assert played.state.remaining_for(PlayerSeat.SELF) == 12


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

    followups = [
        tracker.update(_scene(frame_id=frame_id, **current))
        for frame_id in range(3, 9)
    ]
    self_play = next(
        update
        for update in followups
        if update.event is not None and update.event.actor is PlayerSeat.SELF
    )
    right_play = next(
        update
        for update in followups
        if update.event is not None and update.event.actor is PlayerSeat.RIGHT
    )
    left_pass = next(
        update
        for update in followups
        if update.event is not None and update.event.actor is PlayerSeat.LEFT
    )

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
    later_updates = [
        tracker.update(_scene(frame_id=frame_id, **later))
        for frame_id in range(10, 16)
    ]
    right_lead = next(
        update
        for update in later_updates
        if update.event is not None and update.event.actor is PlayerSeat.RIGHT
    )
    left_play = next(
        update
        for update in later_updates
        if update.event is not None and update.event.actor is PlayerSeat.LEFT
    )

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
        right_remaining=15,
    ))

    assert update.mode is VisualTrackerMode.UNCERTAIN
    assert update.state is not None
    assert update.state.phase.value == "uncertain"
    assert "remaining count mismatch" in update.message


def test_visual_tracker_ignores_false_play_when_verified_count_is_unchanged() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        round_id_factory=lambda _: "round-false-play",
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

    assert update.mode is VisualTrackerMode.TRACKING
    assert update.event is None
    assert update.state is not None
    assert update.state.current_actor is PlayerSeat.RIGHT
    assert "余牌数没有减少" in update.message


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


def test_tracker_accepts_near_threshold_play_with_verified_count_drop() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        initial_stability_frames=1,
        confidence_threshold=0.70,
        round_id_factory=lambda _: "round-correlated-low-confidence",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(
        frame_id=2,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))
    tracker.update(_scene(
        frame_id=3,
        right_signal=VisualSignal.PLAY,
        right_cards=("9",),
        right_remaining=16,
    ))
    low_confidence_scene = _scene(
        frame_id=4,
        right_signal=VisualSignal.PLAY,
        right_cards=("9",),
        right_remaining=16,
        left_signal=VisualSignal.UNKNOWN,
        left_cards=("10",),
        left_remaining=16,
        self_turn=True,
    )
    low_confidence_scene = replace(
        low_confidence_scene,
        seats=tuple(
            replace(
                observation,
                cards=(
                    (_card("10", 0.68),)
                    if observation.seat is PlayerSeat.LEFT
                    else observation.cards
                ),
                confidence=(
                    0.68
                    if observation.seat is PlayerSeat.LEFT
                    else observation.confidence
                ),
            )
            for observation in low_confidence_scene.seats
        ),
    )

    update = tracker.update(low_confidence_scene)

    assert update.event is not None
    assert update.event.actor is PlayerSeat.LEFT
    assert update.event.cards.cards == ("10",)
    assert update.event.source == "live_visual_remaining_correlated"
    assert update.state is not None
    assert update.state.revision == 3
    assert update.state.current_actor is PlayerSeat.SELF
    assert update.state.trick_target.cards == ("10",)
    assert update.state.remaining_for(PlayerSeat.LEFT) == 16


def test_visible_play_is_processed_before_turn_based_pass_inference() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        initial_stability_frames=1,
        round_id_factory=lambda _: "round-play-before-pass",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(
        frame_id=2,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))

    animated_counter = tracker.update(_scene(
        frame_id=3,
        right_signal=VisualSignal.PLAY,
        right_cards=("9",),
        right_remaining=17,
        self_turn=True,
    ))
    corrected_counter = tracker.update(_scene(
        frame_id=4,
        right_signal=VisualSignal.PLAY,
        right_cards=("9",),
        right_remaining=16,
        self_turn=True,
    ))

    assert animated_counter.event is None
    assert animated_counter.state is not None
    assert animated_counter.state.current_actor is PlayerSeat.RIGHT
    assert corrected_counter.event is not None
    assert corrected_counter.event.actor is PlayerSeat.RIGHT
    assert corrected_counter.event.cards.cards == ("9",)
    assert corrected_counter.state is not None
    assert corrected_counter.state.current_actor is PlayerSeat.LEFT


def test_live_cycle_survives_counter_animation_without_rescan() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=2,
        confidence_threshold=0.70,
    )
    starting_hand = (
        "BJ", "SJ", "2", "A", "A", "K", "K", "K", "J", "9", "5",
    )
    initialized = tracker.scan_current_scene(_scene(
        frame_id=60,
        visible_hand=starting_hand,
        right_remaining=16,
        left_signal=VisualSignal.PLAY,
        left_cards=("10",),
        left_remaining=16,
        self_turn=True,
    ))
    assert initialized.initialized is True
    assert initialized.state is not None
    original_round_id = initialized.state.round_id

    after_a_cards = list(starting_hand)
    after_a_cards.remove("A")
    after_a = tuple(after_a_cards)
    self_play = tracker.update(_scene(
        frame_id=61,
        visible_hand=after_a,
        right_remaining=16,
        left_signal=VisualSignal.PLAY,
        left_cards=("10",),
        left_remaining=16,
        self_turn=False,
    ))
    tracker.update(_scene(
        frame_id=62,
        visible_hand=after_a,
        right_remaining=16,
        left_signal=VisualSignal.PLAY,
        left_cards=("10",),
        left_remaining=16,
        self_turn=False,
    ))
    assert self_play.event is not None
    assert self_play.event.cards.cards == ("A",)
    assert self_play.state is not None
    assert self_play.state.revision == 1
    assert self_play.state.current_actor is PlayerSeat.RIGHT

    animated_counter_updates = [
        tracker.update(_scene(
            frame_id=frame_id,
            visible_hand=after_a,
            right_signal=VisualSignal.PLAY,
            right_cards=("2",),
            right_remaining=16,
            left_remaining=16,
            self_turn=True,
        ))
        for frame_id in range(63, 67)
    ]
    assert all(update.event is None for update in animated_counter_updates)
    assert tracker.state is not None
    assert tracker.state.revision == 1
    assert tracker.state.current_actor is PlayerSeat.RIGHT

    right_play_scene = _scene(
        frame_id=67,
        visible_hand=after_a,
        right_signal=VisualSignal.PLAY,
        right_cards=("2",),
        right_remaining=15,
        left_remaining=16,
        self_turn=True,
    )
    right_play = tracker.update(right_play_scene)
    left_pass = tracker.update(replace(
        right_play_scene,
        frame_id=68,
        timestamp=68.0,
    ))

    assert right_play.event is not None
    assert right_play.event.actor is PlayerSeat.RIGHT
    assert right_play.event.cards.cards == ("2",)
    assert right_play.state is not None
    assert right_play.state.revision == 2
    assert right_play.state.remaining_for(PlayerSeat.RIGHT) == 15
    assert left_pass.event is not None and left_pass.event.is_pass
    assert left_pass.event.actor is PlayerSeat.LEFT
    assert left_pass.state is not None
    assert left_pass.state.revision == 3
    assert left_pass.state.current_actor is PlayerSeat.SELF
    assert left_pass.state.trick_target.cards == ("2",)
    assert left_pass.state.consecutive_passes == 1

    stable_updates = [
        tracker.update(replace(
            right_play_scene,
            frame_id=frame_id,
            timestamp=float(frame_id),
        ))
        for frame_id in range(69, 79)
    ]
    assert all(not update.initialized for update in stable_updates)
    assert tracker.state is not None
    assert tracker.state.round_id == original_round_id
    assert tracker.state.revision == 3
    assert tracker.state.phase.value == "playing"
    assert tracker.state.current_actor is PlayerSeat.SELF
    assert tracker.state.trick_target.cards == ("2",)


def test_current_scan_prefers_verified_left_unknown_over_old_right_play() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        confidence_threshold=0.70,
    )
    scene = _scene(
        frame_id=50,
        visible_hand=REAL_LANDLORD_HAND[:16],
        right_signal=VisualSignal.PLAY,
        right_cards=("9",),
        right_remaining=16,
        left_signal=VisualSignal.UNKNOWN,
        left_cards=("10",),
        left_remaining=16,
        self_turn=True,
    )
    scene = replace(
        scene,
        seats=tuple(
            replace(
                observation,
                cards=(
                    (_card("10", 0.68),)
                    if observation.seat is PlayerSeat.LEFT
                    else observation.cards
                ),
                confidence=(
                    0.68
                    if observation.seat is PlayerSeat.LEFT
                    else observation.confidence
                ),
            )
            for observation in scene.seats
        ),
    )

    update = tracker.scan_current_scene(scene)

    assert update.initialized is True
    assert update.state is not None
    assert update.state.current_actor is PlayerSeat.SELF
    assert update.state.trick_leader is PlayerSeat.LEFT
    assert update.state.trick_target.cards == ("10",)
    assert update.state.consecutive_passes == 0
    assert (
        "current_scan_accepted_left_play_from_verified_remaining"
        in update.state.warnings
    )


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
    right_play = tracker.update(replace(
        right_play_scene,
        frame_id=43,
        timestamp=43.0,
    ))
    tracker.update(replace(
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
    self_play = tracker.update(self_play_scene)
    tracker.update(replace(
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


def test_tracker_catches_up_three_missed_actions_from_live_table() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
    )
    hand = REAL_LANDLORD_HAND[:15]
    old_table = _scene(
        frame_id=80,
        visible_hand=hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("Q", "Q"),
        left_remaining=4,
        right_remaining=5,
        self_turn=True,
    )
    tracker.update(old_table)
    initialized = tracker.update(replace(
        old_table,
        frame_id=81,
        timestamp=81.0,
    ))
    assert initialized.initialized is True
    assert initialized.state is not None
    assert initialized.state.current_actor is PlayerSeat.SELF

    current_table = _scene(
        frame_id=82,
        visible_hand=hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("BJ",),
        left_remaining=3,
        right_remaining=5,
        self_turn=True,
    )
    self_pass = tracker.update(current_table)
    right_pass = tracker.update(replace(
        current_table,
        frame_id=83,
        timestamp=83.0,
    ))
    left_play = tracker.update(replace(
        current_table,
        frame_id=84,
        timestamp=84.0,
    ))
    tracker.update(replace(
        current_table,
        frame_id=85,
        timestamp=85.0,
    ))

    assert self_pass.event is not None and self_pass.event.is_pass
    assert self_pass.event.actor is PlayerSeat.SELF
    assert right_pass.event is not None and right_pass.event.is_pass
    assert right_pass.event.actor is PlayerSeat.RIGHT
    assert left_play.event is not None
    assert left_play.event.actor is PlayerSeat.LEFT
    assert left_play.event.cards.cards == ("BJ",)
    assert left_play.state is not None
    assert left_play.state.revision == 3
    assert left_play.state.current_actor is PlayerSeat.SELF
    assert left_play.state.trick_target.cards == ("BJ",)
    assert left_play.state.remaining_for(PlayerSeat.LEFT) == 3


def test_tracker_remembers_non_expected_blank_before_rapid_opponent_play() -> None:
    tracker = VisualEventTracker(
        stability_frames=1,
        initial_stability_frames=1,
    )
    hand = ("9", "J", "K", "K", "2")
    initialized = tracker.scan_current_scene(_scene(
        frame_id=90,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_signal=VisualSignal.PASS,
        left_remaining=14,
        self_turn=True,
    ))
    assert initialized.initialized is True
    assert initialized.state is not None
    assert initialized.state.current_actor is PlayerSeat.SELF
    assert initialized.state.consecutive_passes == 1

    # The left seat becomes blank while self is still the expected actor. This
    # edge must remain armed until left's next visible action.
    tracker.update(_scene(
        frame_id=91,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=True,
    ))
    self_pass = tracker.update(_scene(
        frame_id=92,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("BJ",),
        right_remaining=13,
        left_remaining=14,
        self_turn=False,
    ))
    assert self_pass.event is not None and self_pass.event.is_pass

    rapid_opponents = _scene(
        frame_id=93,
        visible_hand=hand,
        right_signal=VisualSignal.PLAY,
        right_cards=("3", "3"),
        right_remaining=11,
        left_signal=VisualSignal.PLAY,
        left_cards=("7", "7"),
        # Simulate the real 12 -> unverified template fallback 16 failure.
        left_remaining=16,
        self_turn=True,
    )
    rapid_opponents = replace(
        rapid_opponents,
        seats=tuple(
            replace(
                observation,
                remaining_verified=False,
            )
            if observation.seat is PlayerSeat.LEFT
            else observation
            for observation in rapid_opponents.seats
        ),
    )
    right_play = tracker.update(rapid_opponents)
    left_play = tracker.update(replace(
        rapid_opponents,
        frame_id=94,
        timestamp=94.0,
    ))

    assert right_play.event is not None
    assert right_play.event.actor is PlayerSeat.RIGHT
    assert right_play.event.cards.cards == ("3", "3")
    assert left_play.event is not None
    assert left_play.event.actor is PlayerSeat.LEFT
    assert left_play.event.cards.cards == ("7", "7")
    assert left_play.state is not None
    assert left_play.state.revision == 3
    assert left_play.state.current_actor is PlayerSeat.SELF
    assert left_play.state.trick_target.cards == ("7", "7")
    assert left_play.state.remaining_for(PlayerSeat.LEFT) == 12


def test_tracker_ignores_stable_unknown_blob_when_remaining_is_unchanged() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=1,
        round_id_factory=lambda _: "round-unknown-blob",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(
        frame_id=2,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))
    tracker.update(_scene(
        frame_id=3,
        self_signal=VisualSignal.PLAY,
        self_cards=("3",),
    ))

    tracker.update(_scene(
        frame_id=4,
        right_signal=VisualSignal.UNKNOWN,
        right_remaining=17,
    ))
    update = tracker.update(_scene(
        frame_id=5,
        right_signal=VisualSignal.UNKNOWN,
        right_remaining=17,
    ))

    assert update.mode is VisualTrackerMode.TRACKING
    assert update.state is not None
    assert update.state.phase.value == "playing"
    assert update.state.current_actor is PlayerSeat.RIGHT
    assert "余牌未减少" in update.message


def test_tracker_recovers_after_transient_bad_self_hand_read() -> None:
    tracker = VisualEventTracker(
        stability_frames=3,
        initial_stability_frames=2,
        round_id_factory=lambda _: "round-transient-hand",
    )
    tracker.update(_scene(frame_id=1))
    tracker.update(_scene(frame_id=2))
    bad_hand = (*LANDLORD_HAND[:-2], "BJ")

    tracker.update(_scene(
        frame_id=3,
        visible_hand=bad_hand,
        self_turn=True,
    ))
    paused = tracker.update(_scene(
        frame_id=4,
        visible_hand=bad_hand,
        self_turn=True,
    ))
    recovered = tracker.update(_scene(
        frame_id=5,
        visible_hand=LANDLORD_HAND,
        self_turn=True,
    ))

    assert paused.mode is VisualTrackerMode.TRACKING
    assert paused.state is not None
    assert paused.state.phase.value == "uncertain"
    assert paused.state.decision_ready is False
    assert recovered.mode is VisualTrackerMode.TRACKING
    assert recovered.state is not None
    assert recovered.state.phase.value == "playing"
    assert recovered.state.round_id == "round-transient-hand"


def test_tracker_rebuilds_stably_drifted_state_on_self_turn() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
    )
    starting_hand = REAL_LANDLORD_HAND[:16]
    initialized = tracker.scan_current_scene(_scene(
        frame_id=100,
        visible_hand=starting_hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("4",),
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    ))
    assert initialized.initialized is True
    drifted_hand = starting_hand[1:]
    drifted = _scene(
        frame_id=101,
        visible_hand=drifted_hand,
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    )

    updates = []
    for frame_id in range(101, 105):
        updates.append(tracker.update(replace(
            drifted,
            frame_id=frame_id,
            timestamp=float(frame_id),
        )))

    assert updates[1].state is not None
    assert updates[1].state.phase.value == "uncertain"
    rebuilt = updates[-1]
    assert rebuilt.initialized is True
    assert rebuilt.mode is VisualTrackerMode.TRACKING
    assert rebuilt.state is not None
    assert rebuilt.state.round_id.startswith("auto-scan-")
    assert rebuilt.state.self_hand == CardSet.parse(drifted_hand)
    assert "active_state_drift_recovered" in rebuilt.warnings


def test_tracker_does_not_rebuild_from_stale_trick_without_count_change() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
    )
    hand = REAL_LANDLORD_HAND[:16]
    initialized = tracker.scan_current_scene(_scene(
        frame_id=120,
        visible_hand=hand,
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    ))
    assert initialized.initialized is True
    assert initialized.state is not None
    original_round_id = initialized.state.round_id

    stale_display = _scene(
        frame_id=121,
        visible_hand=hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("10",),
        left_remaining=7,
        right_remaining=12,
        self_turn=True,
    )
    updates = [
        tracker.update(replace(
            stale_display,
            frame_id=frame_id,
            timestamp=float(frame_id),
        ))
        for frame_id in range(121, 130)
    ]

    assert all(not update.initialized for update in updates)
    assert tracker.state is not None
    assert tracker.state.round_id == original_round_id
    assert not tracker.state.trick_target


def test_tracker_never_recovers_to_increased_remaining_counts() -> None:
    tracker = VisualEventTracker(
        stability_frames=2,
        initial_stability_frames=2,
    )
    hand = REAL_LANDLORD_HAND[:15]
    initialized = tracker.scan_current_scene(_scene(
        frame_id=110,
        visible_hand=hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("10",),
        left_remaining=4,
        right_remaining=5,
        self_turn=True,
    ))
    assert initialized.initialized is True
    impossible = _scene(
        frame_id=111,
        visible_hand=hand,
        left_signal=VisualSignal.PLAY,
        left_cards=("10",),
        left_remaining=16,
        right_remaining=5,
        self_turn=True,
    )

    updates = [
        tracker.update(replace(
            impossible,
            frame_id=frame_id,
            timestamp=float(frame_id),
        ))
        for frame_id in range(111, 119)
    ]

    assert all(not update.initialized for update in updates)
    assert tracker.state is not None
    assert tracker.state.remaining_for(PlayerSeat.LEFT) == 4


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
