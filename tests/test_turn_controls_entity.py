"""Pure turn-entity, configuration, import and bounded CPU regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import src.vision.turn_controls as turn_module
from src.config.turn_controls import TurnControlsConfig
from src.vision.turn_controls import TurnControlEvidence, observe_turn_controls


def _button_image(count=1, *, collar=6, shadow=3, scale=1):
    image = Image.new("RGB", (500, 240), (43, 73, 145))
    draw = ImageDraw.Draw(image)
    boxes = [(80, 80, 420, 170)] if count == 1 else [(30, 85, 215, 145), (285, 85, 470, 145)]
    for x1, y1, x2, y2 in boxes:
        if collar or shadow:
            draw.rounded_rectangle(
                (x1 - collar, y1 - collar, x2 + collar, y2 + collar + shadow),
                radius=(y2 - y1) // 2 + collar, fill=(35, 22, 12),
            )
        draw.rounded_rectangle(
            (x1, y1, x2, y2), radius=(y2 - y1) // 2, fill=(242, 166, 31),
        )
        draw.text((x1 + 55, y1 + 18), "AB", fill="white", font=ImageFont.load_default(size=22))
    return image.resize((round(500 * scale), round(240 * scale)), Image.Resampling.NEAREST)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("scale", [0.5, 1, 2])
@pytest.mark.parametrize("collar,shadow", [(0, 0), (3, 2), (6, 3)])
def test_complete_single_dual_entities(count, scale, collar, shadow):
    result = observe_turn_controls(_button_image(count, scale=scale, collar=collar, shadow=shadow))
    assert result.active is True and result.confidence >= 0.94
    assert len(result.entity_boxes) == count
    assert not result.template_fallback_allowed


@pytest.mark.parametrize("negative", [
    "dark_occlusion", "independent_white_text", "independent_dark_text",
    "cropped_entity", "thick_collar", "extra_warm_fragment", "bright_collar",
    "nonlocal_dark_extension", "independent_bbox_corner", "warm_rectangle",
    "attached_extra_warm_fragment",
])
def test_rejected_foreground_stays_unknown_not_false(negative):
    image = _button_image(collar=3, shadow=2)
    draw = ImageDraw.Draw(image)
    if negative == "dark_occlusion":
        draw.rectangle((190, 70, 340, 180), fill=(15, 15, 15))
    elif negative in {"independent_white_text", "independent_dark_text"}:
        draw.text((40, 15), "NO", fill="white" if negative.endswith("white_text") else "black",
                  font=ImageFont.load_default(size=30))
    elif negative == "cropped_entity":
        image = image.crop((0, 0, 400, 240))
    elif negative == "thick_collar":
        image = _button_image(collar=30, shadow=8)
    elif negative == "extra_warm_fragment":
        draw.rectangle((150, 30, 170, 45), fill=(242, 166, 31))
    elif negative == "bright_collar":
        draw.rounded_rectangle((74, 74, 426, 179), radius=50, fill="white")
        draw.rounded_rectangle((80, 80, 420, 170), radius=45, fill=(242, 166, 31))
    elif negative == "nonlocal_dark_extension":
        draw.rectangle((245, 168, 255, 220), fill="black")
    elif negative == "attached_extra_warm_fragment":
        draw.rectangle((245, 168, 250, 176), fill="black")
        draw.rectangle((240, 173, 254, 182), fill=(242, 166, 31))
    elif negative == "independent_bbox_corner":
        image = _button_image(collar=0, shadow=0)
        ImageDraw.Draw(image).rectangle((80, 80, 90, 91), fill="black")
    else:
        image = Image.new("RGB", (500, 240), (43, 73, 145))
        ImageDraw.Draw(image).rectangle((80, 80, 420, 170), fill=(242, 166, 31))
    result = observe_turn_controls(image)
    assert result.active is None and result.confidence == 0
    assert not result.entity_boxes and not result.template_fallback_allowed
    if negative == "attached_extra_warm_fragment":
        assert result.reason == "extra_warm_fragment_outside_seed"


@pytest.mark.parametrize("color", [(0, 0, 0), (128, 128, 128), (255, 255, 255)])
def test_empty_unverified_background_is_unknown(color):
    assert observe_turn_controls(Image.new("RGB", (500, 240), color)).active is None


def test_clean_supported_blue_background_is_inactive():
    result = observe_turn_controls(Image.new("RGB", (500, 240), (43, 73, 145)))
    assert result.active is False and result.confidence >= 0.95


@pytest.mark.parametrize("size", [(0, 0), (0, 20), (20, 0)])
def test_empty_input_is_unknown(size):
    assert observe_turn_controls(Image.new("RGB", size)).reason == "empty_input"


def test_old_tall_yellow_blocks_are_not_the_new_capsule_contract():
    image = Image.new("RGB", (500, 240), (43, 73, 145))
    for box in [(45, 55, 225, 205), (275, 55, 455, 205)]:
        ImageDraw.Draw(image).rounded_rectangle(box, radius=20, fill=(242, 166, 31))
    assert observe_turn_controls(image).active is None


def test_result_cannot_enable_template_fallback():
    evidence = observe_turn_controls(_button_image().crop((0, 0, 400, 240)))
    assert evidence.active is None and evidence.confidence == 0
    with pytest.raises(FrozenInstanceError):
        evidence.template_fallback_allowed = True
    with pytest.raises(TypeError):
        TurnControlEvidence(None, 0, "unknown", template_fallback_allowed=True)


def test_custom_geometry_color_and_entity_budget_are_consumed():
    image = _button_image()
    assert observe_turn_controls(image).active is True
    assert observe_turn_controls(image, config=replace(TurnControlsConfig(), maximum_collar_ratio=0.02)).reason == "connected_collar_too_thick"
    assert observe_turn_controls(image, config=replace(TurnControlsConfig(), warm_rgb_limits=(255, 80, 110))).active is None
    assert observe_turn_controls(_button_image(2), config=replace(TurnControlsConfig(), maximum_entities=1)).reason == "turn_controls_entity_budget_exceeded"


def test_one_rgb_conversion_per_observation(monkeypatch):
    image = _button_image(2)
    original = Image.Image.convert
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", spy)
    assert observe_turn_controls(image).active is True
    assert len(calls) == 1


def test_input_budget_is_checked_before_image_conversion(monkeypatch):
    image = _button_image()

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized ROI must not be converted")

    monkeypatch.setattr(Image.Image, "convert", forbidden)
    config = replace(TurnControlsConfig(), maximum_input_pixels=100)
    result = observe_turn_controls(image, config=config)
    assert result.active is None and result.reason == "turn_controls_input_budget_exceeded"


def test_checkerboard_stops_at_shared_run_budget():
    parity = np.indices((240, 500)).sum(axis=0) % 2
    pixels = np.full((240, 500, 3), (43, 73, 145), dtype=np.uint8)
    pixels[parity == 1] = (242, 166, 31)
    result = observe_turn_controls(Image.fromarray(pixels))
    assert result.active is None and result.reason == "turn_controls_run_budget_exceeded"


def test_component_membership_and_linear_overlap_sweep():
    mask = np.zeros((15, 15), dtype=bool)
    mask[2, 2:13] = True
    mask[2:13, 2] = True
    mask[9:11, 9:11] = True
    groups = turn_module._components(mask, [100])
    assert sorted(item.area for item in groups) == [4, 21]
    joined = np.array([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool)
    assert len(turn_module._components(joined, [100])) == 1


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_neighborhood_has_no_wraparound(radius):
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    expanded = turn_module._dilate(mask, radius)
    assert expanded.sum() == (radius + 1) ** 2 and not expanded[-1, -1]


def test_config_json_roundtrip_normalizes_lists_and_is_frozen():
    config = TurnControlsConfig.from_dict(json.loads(json.dumps(TurnControlsConfig().to_payload())))
    assert config == TurnControlsConfig()
    assert isinstance(config.warm_rgb_limits, tuple)
    with pytest.raises(FrozenInstanceError):
        config.maximum_entities = 2
    payload = config.to_payload()
    payload["warm_rgb_limits"][0] = 0
    assert config.warm_rgb_limits[0] == 150


@pytest.mark.parametrize("value", [None, [], "bad", True, {"unexpected": 1}, {1: 0}])
def test_invalid_config_objects_rejected(value):
    with pytest.raises(ValueError):
        TurnControlsConfig.from_dict(value)


@pytest.mark.parametrize("name", [item.name for item in fields(TurnControlsConfig)])
def test_every_config_field_rejects_boolean(name):
    default = getattr(TurnControlsConfig(), name)
    invalid = [True, *default[1:]] if isinstance(default, tuple) else True
    with pytest.raises(ValueError):
        TurnControlsConfig.from_dict({name: invalid})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "0.1", 10**1000])
@pytest.mark.parametrize("name", ["minimum_fill", "maximum_collar_channel_mean", "profile_end_fraction", "maximum_input_pixels"])
def test_invalid_numeric_config_values(value, name):
    with pytest.raises(ValueError):
        TurnControlsConfig.from_dict({name: value})


@pytest.mark.parametrize("payload", [
    {"button_aspect": [2]}, {"button_aspect": [6, 2]}, {"button_aspect": [0, 2]},
    {"button_aspect": [2, 2]}, {"button_height": [0.2, 1.01]},
    {"warm_rgb_limits": [151, 80]}, {"warm_rgb_limits": [151, 80, -1]},
    {"warm_rgb_limits": [151, 80, float("nan")]}, {"warm_differences": [65, 256]},
    {"blue_background_limits": [15, "15", 60]}, {"minimum_fill": 0},
    {"minimum_fill": 1.01}, {"maximum_collar_ratio": 0.5},
    {"profile_end_fraction": 0}, {"boundary_margin_ratio": -0.1},
    {"maximum_collar_channel_mean": 256}, {"maximum_entities": 0},
    {"maximum_entities": 1.5}, {"maximum_entities": 257},
    {"maximum_input_pixels": 16_000_001},
    {"minimum_component_pixels": 10, "maximum_input_pixels": 6},
])
def test_invalid_config_intervals_and_budgets(payload):
    with pytest.raises(ValueError):
        TurnControlsConfig.from_dict(payload)


def test_valid_nearly_half_profile_never_warns_on_empty_discrete_slice():
    config = replace(TurnControlsConfig(), profile_end_fraction=0.49999999999)
    result = observe_turn_controls(_button_image(), config=config)
    assert result.active in (True, None)


def test_wrong_runtime_config_type_is_rejected():
    with pytest.raises(ValueError):
        observe_turn_controls(_button_image(), config={})


def test_bounded_synthetic_cpu_and_lightweight_import_in_real_subprocess():
    code = """
import json, statistics, sys, time
import numpy as np
from PIL import Image, ImageDraw
from src.vision.turn_controls import observe_turn_controls
assert not any(name in sys.modules for name in ('torch', 'onnxruntime', 'src.pipeline.live_runtime'))
image = Image.new('RGB', (500, 240), (43,73,145))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((74,74,426,179),radius=50,fill=(35,22,12))
draw.rounded_rectangle((80,80,420,170),radius=45,fill=(242,166,31))
durations = []
for _ in range(12):
    started = time.perf_counter()
    assert observe_turn_controls(image).active is True
    durations.append(time.perf_counter()-started)
pixels = np.full((240,500,3),(43,73,145),dtype=np.uint8)
pixels[np.indices((240,500)).sum(axis=0)%2==1] = (242,166,31)
started = time.perf_counter()
assert observe_turn_controls(Image.fromarray(pixels)).reason == 'turn_controls_run_budget_exceeded'
print(json.dumps({'median_seconds':statistics.median(durations),'total_seconds':sum(durations),'noise_seconds':time.perf_counter()-started}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True, timeout=12,
    )
    timing = json.loads(result.stdout)
    # Deliberately broad CI bounds plus deterministic pixel/run budgets. This
    # is a runaway-work check, not a promise of real-game FPS on every host.
    assert timing["total_seconds"] < 8 and timing["noise_seconds"] < 3
