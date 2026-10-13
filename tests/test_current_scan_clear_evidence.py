from __future__ import annotations

from dataclasses import replace

import pytest

from src.state.events import PlayerSeat
from src.tracking.visual_events import (
    VisualEventTracker,
    VisualTrackerMode,
    _current_scan_payload,
)
from src.vision.scene_types import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    VisualCard,
    VisualSignal,
)

# Structural reproductions of AI-reviewed opening transitions, not model fixtures or
# frame-number exceptions. The bottom cards have not joined the hand yet.
PRE_BOTTOM_HAND = tuple("BJ SJ 2 2 2 2 A A K 10 10 9 7 6 6 6 6".split())
FULL_HAND = (*PRE_BOTTOM_HAND, "9", "7", "4")
FARMER_HAND = tuple("SJ 2 A A K Q Q J J J 7 6 5 5 5 4 4".split())


def _card(rank: str, confidence: float = 0.99) -> VisualCard:
    return VisualCard(rank, confidence, (0, 0, 10, 20))


def _scene(
    *,
    hand: tuple[str, ...] = PRE_BOTTOM_HAND,
    landlord: PlayerSeat = PlayerSeat.SELF,
    right_remaining: int = 17,
) -> SceneObservation:
    return SceneObservation(
        frame_id=1,
        timestamp=1.0,
        window_pixel_box=(0, 0, 100, 100),
        self_hand=tuple(_card(rank) for rank in hand),
        seats=tuple(
            SeatObservation(
                seat=seat,
                signal=VisualSignal.NEUTRAL,
                remaining_count=(
                    len(hand) if seat is PlayerSeat.SELF
                    else right_remaining if seat is PlayerSeat.RIGHT else 17
                ),
                role=SeatRole.LANDLORD if seat is landlord else SeatRole.FARMER,
                confidence=0.99,
                remaining_confidence=0.99,
                role_confidence=0.99,
                remaining_verified=True,
            )
            for seat in PlayerSeat
        ),
        self_turn=True,
        self_turn_confidence=0.99,
        confidence=0.99,
    )


def _replace_seat(scene: SceneObservation, observation: SeatObservation) -> SceneObservation:
    return replace(scene, seats=tuple(
        observation if seat.seat is observation.seat else seat for seat in scene.seats
    ))


def _at(scene: SceneObservation, frame_id: int) -> SceneObservation:
    return replace(scene, frame_id=frame_id, timestamp=float(frame_id))


def _unknown(scene: SceneObservation, seat: PlayerSeat) -> SceneObservation:
    return _replace_seat(scene, replace(
        scene.seat(seat), signal=VisualSignal.UNKNOWN,
        cards=(_card("SJ", 0.0), _card("SJ", 0.0), _card("8", 0.91)),
        confidence=0.0,
    ))


def _with_right_target(scene: SceneObservation) -> SceneObservation:
    return _replace_seat(scene, replace(
        scene.seat(PlayerSeat.RIGHT), signal=VisualSignal.PLAY,
        cards=(_card("4"),), remaining_count=16,
    ))


@pytest.mark.parametrize("landlord", [PlayerSeat.SELF, PlayerSeat.RIGHT])
def test_opening_transition_waits_then_accepts_complete_initial_scene(
    landlord: PlayerSeat,
) -> None:
    # 002: self landlord before receiving bottom cards. 001: right landlord
    # before receiving bottom cards. Both have misleading active-turn evidence.
    hand = PRE_BOTTOM_HAND if landlord is PlayerSeat.SELF else FARMER_HAND
    transition = _unknown(_unknown(_scene(hand=hand, landlord=landlord), PlayerSeat.LEFT),
                          PlayerSeat.RIGHT)
    tracker = VisualEventTracker(stability_frames=3, initial_stability_frames=2)
    for frame_id in range(1, 5):
        update = tracker.update(_at(transition, frame_id))
        assert not update.initialized
        assert update.state is None
        assert update.mode is VisualTrackerMode.WAITING_FOR_ROUND
    manual = tracker.scan_current_scene(_at(transition, 5))
    assert not manual.initialized
    assert manual.state is None
    assert "场上牌尚未确认为空或不出" in manual.message

    initial = _scene(
        hand=FULL_HAND if landlord is PlayerSeat.SELF else FARMER_HAND,
        landlord=landlord,
        right_remaining=20 if landlord is PlayerSeat.RIGHT else 17,
    )
    initial = replace(initial, self_turn=landlord is PlayerSeat.SELF)
    waiting = tracker.update(_at(initial, 6))
    initialized = tracker.update(_at(initial, 7))
    assert waiting.state is None
    assert "1/2" in waiting.message
    assert initialized.initialized
    assert initialized.state is not None
    assert initialized.state.round_id == "live-7000"
    assert initialized.state.landlord is landlord
    assert initialized.state.hidden_played_count == 0
    assert initialized.state.revision == 0


@pytest.mark.parametrize("signal", [VisualSignal.NEUTRAL, VisualSignal.PASS])
def test_landlord_with_17_cards_can_scan_genuine_midgame_free_lead(signal: VisualSignal) -> None:
    scene = _scene()
    for seat in (PlayerSeat.LEFT, PlayerSeat.RIGHT):
        scene = _replace_seat(scene, replace(
            scene.seat(seat), signal=signal, pass_confidence=0.99,
        ))
    tracker = VisualEventTracker(stability_frames=3, initial_stability_frames=2)
    assert not tracker.update(scene).initialized
    update = tracker.update(_at(scene, 2))
    assert update.initialized
    assert update.state is not None
    assert update.state.hidden_played_count == 3
    assert update.state.current_actor is PlayerSeat.SELF
    assert update.state.trick_target.cards == ()
    assert update.state.round_id == "auto-scan-2000"


@pytest.mark.parametrize("signal", [VisualSignal.NEUTRAL, VisualSignal.PASS])
@pytest.mark.parametrize("right_target", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -0.01, 0.0, 0.69, 1.01])
def test_invalid_clear_confidence_cannot_support_scan(
    signal: VisualSignal, right_target: bool, invalid: float,
) -> None:
    scene = _with_right_target(_scene()) if right_target else _scene()
    left = scene.seat(PlayerSeat.LEFT)
    scene = _replace_seat(scene, replace(
        left, signal=signal,
        confidence=invalid if signal is VisualSignal.NEUTRAL else 0.99,
        pass_confidence=invalid if signal is VisualSignal.PASS else 0.99,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert payload is None
    assert "left 场上牌尚未确认为空或不出" in error
    manual = VisualEventTracker().scan_current_scene(scene)
    assert not manual.initialized
    assert manual.state is None


@pytest.mark.parametrize("signal", [VisualSignal.NEUTRAL, VisualSignal.PASS])
@pytest.mark.parametrize("confidence", [0.7, 1.0])
def test_clear_confidence_closed_bounds_do_not_raise_threshold(
    signal: VisualSignal, confidence: float,
) -> None:
    # No hidden-history discount here: test the exact existing threshold.
    scene = _scene(hand=FULL_HAND)
    scene = _replace_seat(scene, replace(
        scene.seat(PlayerSeat.LEFT), signal=signal,
        confidence=confidence, pass_confidence=confidence,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert error == ""
    assert payload is not None
    assert payload.confidence == pytest.approx(min(confidence, 0.99))


@pytest.mark.parametrize("right_target", [False, True])
@pytest.mark.parametrize("invalid_kind", ["unknown_empty", "unknown_cards", "neutral_cards", "pass_cards", "play_empty"])
def test_unresolved_left_is_not_clear(right_target: bool, invalid_kind: str) -> None:
    scene = _with_right_target(_scene()) if right_target else _scene()
    signal_name, card_kind = invalid_kind.split("_")
    scene = _replace_seat(scene, replace(
        scene.seat(PlayerSeat.LEFT), signal=VisualSignal(signal_name),
        cards=(_card("8", 0.0),) if card_kind == "cards" else (),
        confidence=0.0 if invalid_kind == "unknown_cards" else 0.99,
        pass_confidence=0.99,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert payload is None
    assert "left 场上牌尚未确认为空或不出" in error


def test_free_lead_requires_right_clear_as_well() -> None:
    scene = _unknown(_scene(), PlayerSeat.RIGHT)
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert payload is None
    assert "right 场上牌尚未确认为空或不出" in error


@pytest.mark.parametrize("signal", [VisualSignal.NEUTRAL, VisualSignal.PASS])
@pytest.mark.parametrize("right_target", [False, True])
def test_used_clear_confidence_contributes_to_final_confidence(
    signal: VisualSignal, right_target: bool,
) -> None:
    scene = _with_right_target(_scene()) if right_target else _scene()
    scene = _replace_seat(scene, replace(
        scene.seat(PlayerSeat.LEFT), signal=signal,
        confidence=0.8 if signal is VisualSignal.NEUTRAL else 0.01,
        pass_confidence=0.8 if signal is VisualSignal.PASS else 0.01,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert error == ""
    assert payload is not None
    assert payload.confidence == pytest.approx(0.8 * 0.95)
    assert payload.trick_target == (("4",) if right_target else ())
    assert payload.consecutive_passes == (1 if right_target else 0)


def test_right_clear_confidence_and_hidden_history_discount_are_not_discarded() -> None:
    scene = _scene()
    scene = _replace_seat(scene, replace(scene.seat(PlayerSeat.RIGHT), confidence=0.72))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert payload is None
    assert "综合置信度不足" in error


@pytest.mark.parametrize("signal", [VisualSignal.PLAY, VisualSignal.UNKNOWN])
def test_supported_left_target_takes_priority_over_unresolved_right(signal: VisualSignal) -> None:
    scene = _unknown(_scene(), PlayerSeat.RIGHT)
    confidence = 0.68 if signal is VisualSignal.UNKNOWN else 0.99
    scene = _replace_seat(scene, replace(
        scene.seat(PlayerSeat.LEFT), signal=signal,
        cards=(_card("4", confidence),), confidence=confidence, remaining_count=16,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert error == ""
    assert payload is not None
    assert payload.trick_target == ("4",)
    assert payload.trick_leader is PlayerSeat.LEFT
    assert payload.consecutive_passes == 0
    if signal is VisualSignal.UNKNOWN:
        assert "current_scan_accepted_left_play_from_verified_remaining" in payload.warnings


@pytest.mark.parametrize("seat", [PlayerSeat.LEFT, PlayerSeat.RIGHT])
def test_low_confidence_play_is_not_downgraded_to_empty_table(seat: PlayerSeat) -> None:
    scene = _replace_seat(_scene(), replace(
        _scene().seat(seat), signal=VisualSignal.PLAY,
        cards=(_card("4", 0.4),), remaining_count=16,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert payload is None
    assert "待压牌置信度不足" in error


@pytest.mark.parametrize("seat", [PlayerSeat.LEFT, PlayerSeat.RIGHT])
def test_unparseable_target_cannot_be_reinterpreted_as_free_lead(seat: PlayerSeat) -> None:
    scene = _replace_seat(_scene(), replace(
        _scene().seat(seat), signal=VisualSignal.UNKNOWN,
        cards=(_card("4", 0.68), _card("5", 0.68)),
        confidence=0.68, remaining_count=15,
    ))
    payload, error = _current_scan_payload(scene, confidence_threshold=0.7)
    assert payload is None
    assert "当前待压牌型无法确认" in error


def test_unresolved_frame_resets_automatic_scan_candidate() -> None:
    tracker = VisualEventTracker(stability_frames=3, initial_stability_frames=2)
    clean = _scene()
    assert "1/2" in tracker.update(clean).message
    rejected = tracker.update(_at(_unknown(clean, PlayerSeat.LEFT), 2))
    assert rejected.state is None
    assert not rejected.initialized
    restarted = tracker.update(_at(clean, 3))
    assert restarted.state is None
    assert "1/2" in restarted.message
    initialized = tracker.update(_at(clean, 4))
    assert initialized.initialized
    assert initialized.state is not None
    assert initialized.state.round_id == "auto-scan-4000"


def test_rejected_manual_rescan_preserves_existing_state() -> None:
    tracker = VisualEventTracker()
    original = tracker.scan_current_scene(_scene())
    assert original.initialized
    rejected = tracker.scan_current_scene(_at(_unknown(_scene(), PlayerSeat.LEFT), 2))
    assert not rejected.initialized
    assert rejected.state is original.state
    assert tracker.state is original.state
