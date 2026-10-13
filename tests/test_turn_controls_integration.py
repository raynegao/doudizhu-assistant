from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as module
from src.config.live_layout import LiveLayoutConfig, NormalizedBox, live_layout_from_dict
from src.config.turn_controls import TurnControlsConfig
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState
from src.tracking.visual_events import VisualEventTracker
from src.vision.scene_types import (
    SceneObservation,
    SeatObservation,
    SeatRole,
    TemplateMatch,
    VisualCard,
    VisualSignal,
)


def _controls_image(kind):
    image = Image.new("RGB", (500, 240), (43, 73, 145))
    if kind == "button":
        ImageDraw.Draw(image).rounded_rectangle((80, 80, 420, 170), radius=45, fill=(242, 166, 31))
    elif kind == "unknown":
        ImageDraw.Draw(image).rectangle((80, 80, 420, 170), fill=(242, 166, 31))
    return image


def _observe(image, monkeypatch, config=None):
    controls = config or TurnControlsConfig()
    rois = dict(LiveLayoutConfig().rois)
    rois["self_turn"] = NormalizedBox(0, 0, 1, 1)
    instance = object.__new__(module.SceneRecognizer)
    instance.config = LiveLayoutConfig(rois=rois, turn_controls=controls)
    calls = []

    def template(kind, *_args, **_kwargs):
        calls.append(kind)
        # If observe still falls back, even an ambiguous image would look active.
        return TemplateMatch("active", 0.999) if kind == "turn" else TemplateMatch(None, 0.0)

    instance.templates = SimpleNamespace(classify=template)
    instance.remaining_reader = None
    instance._round_result_references = {}
    monkeypatch.setattr(module, "_classify_round_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "infer_hand_geometry", lambda *_args, **_kwargs: module.HandGeometry(0, (), True))
    monkeypatch.setattr(instance, "_observe_hand", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(instance, "_observe_seat", lambda _image, seat, **_kwargs: SeatObservation(seat, VisualSignal.NEUTRAL))
    frame = SimpleNamespace(image=image, frame_id=1, timestamp=1.0, pixel_box=(0, 0, *image.size))
    observation = instance.observe(frame)
    assert "turn" not in calls
    return observation


@pytest.mark.parametrize("kind,expected", [("button", True), ("empty", False), ("unknown", None)])
def test_actual_observer_preserves_structural_result_without_template_override(kind, expected, monkeypatch):
    scene = _observe(_controls_image(kind), monkeypatch)
    assert scene.self_turn is expected
    assert scene.self_turn_confidence == (0 if expected is None else 0.99)
    if expected is None:
        assert "self_turn_unavailable" in scene.warnings
        assert any(warning.startswith("self_turn_geometry:") for warning in scene.warnings)


def test_actual_observer_uses_custom_config_and_keeps_budget_failure_unknown(monkeypatch):
    scene = _observe(_controls_image("button"), monkeypatch,
                     replace(TurnControlsConfig(), maximum_input_pixels=10))
    assert scene.self_turn is None and scene.self_turn_confidence == 0
    assert "self_turn_geometry: turn_controls_input_budget_exceeded" in scene.warnings


def test_live_config_roundtrip_and_default_are_explicit():
    controls = replace(TurnControlsConfig(), maximum_collar_ratio=0.10, maximum_entities=2)
    config = LiveLayoutConfig(turn_controls=controls)
    assert live_layout_from_dict(json.loads(json.dumps(config.to_payload()))).turn_controls == controls
    assert live_layout_from_dict({}).turn_controls == TurnControlsConfig()
    with pytest.raises(ValueError, match="turn_controls"):
        LiveLayoutConfig(turn_controls={})


def test_structural_unknown_never_becomes_self_pass_in_tracking(monkeypatch):
    unknown = _observe(_controls_image("unknown"), monkeypatch)
    state = ObservableGameState.from_inputs(
        "A 2 SJ", landlord=PlayerSeat.RIGHT, current_actor=PlayerSeat.SELF,
        remaining_cards={PlayerSeat.SELF: 3, PlayerSeat.RIGHT: 5, PlayerSeat.LEFT: 6},
        played_cards="K", last_play="K", last_player=PlayerSeat.LEFT, hidden_played_count=39,
    )
    tracker = VisualEventTracker(initial_state=state)
    seats = tuple(SeatObservation(
        seat, VisualSignal.NEUTRAL, remaining_count=state.remaining_for(seat),
        role=SeatRole.LANDLORD if seat is PlayerSeat.RIGHT else SeatRole.FARMER,
        confidence=0.99, remaining_confidence=0.99, role_confidence=0.99,
        remaining_verified=True,
    ) for seat in PlayerSeat)
    scene = SceneObservation(1, 1.0, (0, 0, 500, 240),
                             tuple(VisualCard(rank, 0.99, (0, 0, 10, 20)) for rank in state.self_hand.cards),
                             seats, True, 0.99, 0.99)
    for frame in range(1, 3):
        assert tracker.update(replace(scene, frame_id=frame, timestamp=float(frame))).event is None
    for frame in range(3, 12):
        result = tracker.update(replace(scene, frame_id=frame, timestamp=float(frame),
                                        self_turn=unknown.self_turn, self_turn_confidence=unknown.self_turn_confidence))
        assert result.event is None
        assert tracker.state.round_id == state.round_id
        assert tracker.state.revision == state.revision
