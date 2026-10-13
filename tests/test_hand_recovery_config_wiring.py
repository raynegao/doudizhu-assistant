from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

import src.vision.scene_recognizer as module
from src.config.hand_geometry import ShortHandRecoveryConfig
from src.config.live_layout import LiveLayoutConfig, live_layout_from_dict


def test_short_hand_config_round_trip_including_legacy_defaults() -> None:
    recovery = ShortHandRecoveryConfig(minimum_difference=12.0)
    config = LiveLayoutConfig(short_hand_recovery=recovery)
    payload = json.loads(json.dumps(config.to_payload(), allow_nan=False))
    restored = live_layout_from_dict(payload)
    assert restored.short_hand_recovery == recovery
    assert live_layout_from_dict({}).short_hand_recovery == ShortHandRecoveryConfig()


@pytest.mark.parametrize("value", [None, {}, True, "default"])
def test_layout_requires_validated_short_hand_configuration(value) -> None:
    with pytest.raises(ValueError, match="short_hand_recovery"):
        LiveLayoutConfig(short_hand_recovery=value)


@pytest.mark.parametrize("route", ["observe", "seed", "hand"])
def test_all_scene_entrypoints_forward_exact_recovery_configuration(monkeypatch, route) -> None:
    recovery = ShortHandRecoveryConfig(minimum_difference=12.0)
    recognizer = object.__new__(module.SceneRecognizer)
    recognizer.config = LiveLayoutConfig(short_hand_recovery=recovery)
    recognizer._round_result_references = {}
    monkeypatch.setattr(module, "_classify_round_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_classify_role_badge", lambda _: (module.SeatRole.FARMER, 0.99))

    class ReachedGeometry(Exception):
        pass

    def geometry(_image, **kwargs):
        assert kwargs["recovery_config"] is recovery
        assert "expected_count" not in kwargs
        raise ReachedGeometry

    monkeypatch.setattr(module, "infer_hand_geometry", geometry)
    image = Image.new("RGB", (1000, 600), "navy")
    with pytest.raises(ReachedGeometry):
        if route == "observe":
            recognizer.observe(SimpleNamespace(
                image=image, frame_id=1, timestamp=1.0, pixel_box=(0, 0, 1000, 600),
            ))
        elif route == "seed":
            recognizer.seed_hand_references(image)
        else:
            recognizer._observe_hand(image, expected_count=17)
