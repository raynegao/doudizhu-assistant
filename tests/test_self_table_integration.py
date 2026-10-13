from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

import src.vision.scene_recognizer as vision
from src.config.live_layout import LiveLayoutConfig, NormalizedBox
from src.pipeline.live_runtime import _scene_log_signature
from src.state.cards import CardSet
from src.state.events import PlayerSeat, RoundPhase
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import VisualEventTracker
from src.vision.scene_types import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    SelfTableObservation,
    VisualCard,
    VisualSignal,
)


def state(hand="BJ", target="SJ"):
    return ObservableGameState.from_inputs(
        hand, round_id="terminal-integration", landlord=PlayerSeat.RIGHT,
        current_actor=PlayerSeat.SELF, last_play=target,
        last_player=PlayerSeat.LEFT if target else None,
        remaining_cards={PlayerSeat.SELF: len(CardSet.parse(hand)), PlayerSeat.LEFT: 8, PlayerSeat.RIGHT: 9},
        hidden_played_count=54 - len(CardSet.parse(hand)) - len(CardSet.parse(target)) - 17,
    )


def scene(frame, hand="BJ"):
    cards = tuple(VisualCard(rank, .93, (30 + i * 30, 40, 60 + i * 30, 100))
                  for i, rank in enumerate(hand.split()))
    seats = tuple(SeatObservation(seat, VisualSignal.NEUTRAL,
        remaining_count=None if seat is PlayerSeat.SELF else (8 if seat is PlayerSeat.LEFT else 9),
        role=SeatRole.LANDLORD if seat is PlayerSeat.RIGHT else SeatRole.FARMER,
        role_confidence=.99, confidence=.99, remaining_confidence=.99,
        remaining_verified=seat is not PlayerSeat.SELF) for seat in PlayerSeat)
    return SceneObservation(frame, float(frame), (0, 0, 1000, 800), (), seats, None, 0.0, .93,
        self_hand_empty_verified=True, self_table=SelfTableObservation(cards, True, "closed"))


@pytest.mark.parametrize("hand,target", [("BJ", "SJ"), ("SJ BJ", "2 2 2 2"), ("9 9", "8 8")])
@pytest.mark.parametrize("required", [3, 5])
def test_real_tracker_commits_exactly_after_full_gate_and_preserves_unknown(hand, target, required):
    tracker = VisualEventTracker(initial_state=state(hand, target), stability_frames=required)
    for frame in range(1, required):
        assert tracker.update(scene(frame, hand)).event is None
        assert tracker.state.revision == 0
    current = scene(required, hand)
    result = tracker.update(current)
    assert result.event is not None and result.event.source == "live_self_table_terminal"
    assert result.state.phase is RoundPhase.FINISHED
    assert result.state.winner is PlayerSeat.SELF
    assert result.event.confidence == .93
    assert current.self_turn is None
    assert current.seat(PlayerSeat.SELF).remaining_count is None
    assert current.seat(PlayerSeat.SELF).signal is VisualSignal.NEUTRAL


@pytest.mark.parametrize("interrupt", ["geometry", "scan", "bad_card", "old_frame", "role"])
def test_pauses_and_bad_frames_cannot_borrow_prior_stability(interrupt):
    tracker = VisualEventTracker(initial_state=state())
    assert tracker.update(scene(1)).event is None
    assert tracker.update(scene(2)).event is None
    broken = scene(3)
    if interrupt == "geometry":
        broken = replace(broken, warnings=("self_hand_geometry_unavailable: clipped",))
    elif interrupt == "bad_card":
        broken = replace(broken, self_table=SelfTableObservation((VisualCard("BJ", 0, (30, 40, 60, 100)),), True))
    elif interrupt == "role":
        broken = replace(broken, seats=tuple(replace(seat,
            role=SeatRole.LANDLORD if seat.seat is PlayerSeat.LEFT else SeatRole.FARMER)
            for seat in broken.seats))
    if interrupt == "scan":
        assert tracker.scan_current_scene(broken).event is None
    else:
        assert tracker.update(scene(1) if interrupt == "old_frame" else broken).event is None
    assert tracker.update(scene(4)).event is None
    assert tracker.update(scene(5)).event is None
    assert tracker.update(scene(6)).event is not None


@pytest.mark.parametrize("field", ["self_hand_empty_verified", "self_table"])
def test_missing_new_visual_evidence_preserves_legacy_unknown_behavior(field):
    tracker = VisualEventTracker(initial_state=state())
    for frame in range(1, 8):
        current = replace(scene(frame), **{field: False if field == "self_hand_empty_verified" else None})
        assert tracker.update(current).event is None
        assert tracker.state.revision == 0


def test_scene_log_signature_and_payload_include_new_real_evidence():
    current = scene(1)
    absent = replace(current, self_table=None)
    assert _scene_log_signature(current) != _scene_log_signature(absent)
    assert current.to_log_payload()["self_table"]["cards"][0]["rank"] == "BJ"
    assert current.to_log_payload()["self_hand_empty_verified"] is True


def test_existing_inactive_controls_path_remains_separate_from_new_n_frame_gate():
    tracker = VisualEventTracker(initial_state=state(), stability_frames=5)
    current = replace(scene(1), self_turn=False, self_turn_confidence=.99, self_table=None)
    result = tracker.update(current)
    assert result.event is not None and result.event.source == "live_hand_diff"
    assert result.state.phase is RoundPhase.FINISHED
    # Strict N-frame/raw-score claims apply to live_self_table_terminal only.


@pytest.mark.parametrize("geometry,allowed", [
    (vision.HandGeometry(0, (), False), False),
    (vision.HandGeometry(None, (), False, reason="ambiguous_gray_background"), False),
    (vision.HandGeometry(None, (), False), False),
])
def test_geometry_zero_alone_does_not_verify_a_black_hand_domain(monkeypatch, geometry, allowed):
    instance = object.__new__(vision.SceneRecognizer)
    instance.config = LiveLayoutConfig()
    instance.templates = SimpleNamespace(classify=lambda *_args: vision.TemplateMatch(None, 0.0))
    instance.remaining_reader = None
    instance._round_result_references = {}
    monkeypatch.setattr(vision, "_classify_round_result", lambda *_args, **_kw: None)
    monkeypatch.setattr(vision, "infer_hand_geometry", lambda *_args, **_kw: geometry)
    monkeypatch.setattr(instance, "_observe_hand", lambda *_args, **_kw: ())
    monkeypatch.setattr(instance, "_observe_seat", lambda _image, seat, **_kw: SeatObservation(seat, VisualSignal.NEUTRAL))
    calls = []
    def observe_table(_image):
        calls.append(True)
        return SelfTableObservation(reason="test")
    monkeypatch.setattr(instance, "_observe_self_table", observe_table)
    image = Image.new("RGB", (1000, 800), "black")
    result = instance.observe(SimpleNamespace(image=image, frame_id=1, timestamp=1.0, pixel_box=(0, 0, 1000, 800)))
    assert result.self_hand_empty_verified is allowed
    assert bool(calls) is allowed
    assert result.self_turn is None


def test_self_table_never_extends_into_configured_hand_roi():
    rois = dict(LiveLayoutConfig().rois)
    rois["self_play"] = NormalizedBox(.3, .3, .7, .8)
    instance = object.__new__(vision.SceneRecognizer)
    instance.config = LiveLayoutConfig(rois=rois)
    result = instance._observe_self_table(Image.new("RGB", (1000, 800)))
    assert not result.complete and result.reason == "table_roi_overlaps_hand"
