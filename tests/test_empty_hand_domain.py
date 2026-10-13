from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from src.config.hand_empty import HandEmptyConfig
from src.config.live_layout import (
    LiveLayoutConfig,
    NormalizedBox,
    live_layout_from_dict,
    load_live_layout,
)
from src.state.events import PlayerSeat
from src.vision.hand_empty import observe_empty_hand_domain
from src.vision.scene_recognizer import HandGeometry, SceneRecognizer, infer_hand_geometry
from src.vision.scene_types import SelfTableObservation

BLUE = (35, 75, 120)


def observe(image, config=None, geometry=None):
    config = config or HandEmptyConfig()
    geometry = geometry or infer_hand_geometry(image)
    return observe_empty_hand_domain(image, config=config, geometry_count=geometry.count,
        has_candidates=bool(geometry.boxes), geometry_reason=geometry.reason)


def test_real_blue_geometry_is_none_but_independent_empty_domain_is_verified():
    image = Image.new("RGB", (1000, 380), BLUE)
    geometry = infer_hand_geometry(image)
    assert geometry == HandGeometry(None, (), False)
    result = observe(image, geometry=geometry)
    assert result.verified and result.blue_fraction == 1
    assert result.effective_box == (0, 0, 1000, 380)
    assert geometry.count is None


@pytest.mark.parametrize("color", [(0, 0, 0), (235, 235, 235), (114, 114, 114), (15, 150, 30), (30, 40, 55)])
def test_unsupported_background_is_not_empty(color):
    assert not observe(Image.new("RGB", (1000, 380), color)).verified


@pytest.mark.parametrize("box", [(0, 0, 0, 0), (498, 0, 500, 90), (500, 120, 500, 200),
                                  (999, 0, 999, 379), (100, 188, 101, 190)])
@pytest.mark.parametrize("color", [(235, 235, 235), (114, 114, 114), (0, 0, 0)])
def test_local_slivers_fail_even_if_global_blue_fraction_passes(box, color):
    image = Image.new("RGB", (1000, 380), BLUE)
    ImageDraw.Draw(image).rectangle(box, fill=color)
    result = observe(image)
    assert not result.verified


@pytest.mark.parametrize("geometry", [HandGeometry(1, (), False),
    HandGeometry(None, ((5, 5, 10, 10),), False),
    HandGeometry(None, (), False, reason="ambiguous_gray_hand"),
    HandGeometry(1, ((5, 0, 10, 10),), False, (0,), "rank_top_outside_input_roi")])
def test_candidates_or_unknown_geometry_cannot_be_overridden(geometry):
    assert not observe(Image.new("RGB", (1000, 380), BLUE), geometry=geometry).verified


def test_fixed_ui_band_has_explicit_scope_and_boundary():
    config = HandEmptyConfig(bottom_ui_fraction=.03)
    image = Image.new("RGB", (1000, 380), BLUE)
    # This declared UI band is outside the calibrated hand domain. This test
    # documents the blind area; it does NOT claim to distinguish a card sliver
    # from a UI ornament at these exact excluded pixels.
    ImageDraw.Draw(image).rectangle((100, 369, 199, 379), fill="white")
    assert not observe(image).verified
    result = observe(image, config=config)
    assert result.verified and result.excluded_bottom_px == 11
    assert result.effective_box == (0, 0, 1000, 369)
    image.putpixel((100, 368), (235, 235, 235))
    assert not observe(image, config=config).verified


def test_global_guard_rejects_a_large_foreground_even_inside_excluded_band():
    image = Image.new("RGB", (1000, 380), BLUE)
    ImageDraw.Draw(image).rectangle((0, 369, 999, 379), fill="white")
    assert not observe(image, config=HandEmptyConfig(bottom_ui_fraction=.03)).verified


@pytest.mark.parametrize("size,mode", [((10, 10), "RGB"), ((1600, 1000), "RGB"), ((40, 40), "L")])
def test_unusable_input_is_unknown(size, mode):
    image = Image.new(mode, size)
    assert not observe(image, geometry=HandGeometry(None, (), False)).verified


@pytest.mark.parametrize("name,value", [("bottom_ui_fraction", True), ("bottom_ui_fraction", -.01),
    ("bottom_ui_fraction", .030001), ("bottom_ui_fraction", float("nan")),
    ("bottom_ui_fraction", float("inf")), ("minimum_blue_fraction", .9949),
    ("minimum_blue_fraction", True), ("blue_red_gap", 0), ("blue_green_gap", 256),
    ("blue_minimum", 0), ("maximum_input_pixels", 0), ("minimum_dimension", True),
    ("blue_red_gap", 1), ("blue_green_gap", 1), ("blue_minimum", 1), ("minimum_dimension", 1)])
def test_config_rejects_invalid_or_weakened_boundaries(name, value):
    with pytest.raises(ValueError):
        HandEmptyConfig.from_dict({name: value})


@pytest.mark.parametrize("value", [None, [], {"unknown": 1}])
def test_config_rejects_unknown_shape(value):
    with pytest.raises(ValueError):
        HandEmptyConfig.from_dict(value)


def test_known_legacy_layout_resolves_explicit_default_without_rewriting(tmp_path):
    path = tmp_path / "old-config.json"
    original = json.dumps({"rois": {"self_hand": [.025, .625, .975, .895]}}).encode()
    path.write_bytes(original)
    loaded = load_live_layout(path)
    assert loaded.hand_empty.bottom_ui_fraction == .03
    assert live_layout_from_dict(loaded.to_payload()) == loaded
    assert path.read_bytes() == original
    # Explicit zero wins; neither image content nor the absence of slots can
    # silently re-enable the classic profile's bottom exclusion.
    assert live_layout_from_dict({"hand_empty": {}}).hand_empty.bottom_ui_fraction == 0
    assert live_layout_from_dict({"app_name": "another skin"}).hand_empty.bottom_ui_fraction == 0
    assert live_layout_from_dict({"rois": {"self_hand": [.02, .625, .975, .895]}}).hand_empty.bottom_ui_fraction == 0


@pytest.mark.parametrize("color,verified", [(BLUE, True), ((0, 0, 0), False), ((114, 114, 114), False)])
def test_real_geometry_to_observe_without_injecting_zero(monkeypatch, color, verified):
    config = LiveLayoutConfig()
    recognizer = SceneRecognizer(config, predictor=lambda _images: ())
    # Unrelated table classification is intercepted only to prove whether the
    # real geometry + hand observer + empty detector actually reaches it.
    calls = []
    monkeypatch.setattr(recognizer, "_observe_self_table",
        lambda _image: calls.append(True) or SelfTableObservation(reason="test"))
    image = Image.new("RGB", (1000, 800), color)
    geometry = infer_hand_geometry(config.crop(image, "self_hand"))
    assert geometry.count is None
    result = recognizer.observe(SimpleNamespace(image=image, frame_id=1, timestamp=1,
                                                pixel_box=(0, 0, 1000, 800)))
    assert result.self_hand_empty_verified is verified
    assert bool(calls) is verified
    assert result.self_hand_empty_evidence.verified is verified
    assert result.seat(PlayerSeat.SELF).remaining_count is None


def test_layout_rejects_wrong_empty_config_type():
    with pytest.raises(ValueError, match="HandEmptyConfig"):
        replace(LiveLayoutConfig(), hand_empty={})


def test_custom_layout_does_not_implicitly_gain_a_ui_band():
    config = replace(LiveLayoutConfig(), rois={**LiveLayoutConfig().rois,
        "self_hand": NormalizedBox(.03, .6, .97, .9)})
    assert config.hand_empty.bottom_ui_fraction == 0


def test_near_gray_background_cannot_be_promoted_by_a_weakened_blue_predicate():
    image = Image.new("RGB", (1000, 380), (64, 64, 66))
    assert infer_hand_geometry(image) == HandGeometry(None, (), False)
    assert not observe(image).verified
    with pytest.raises(ValueError):
        HandEmptyConfig(blue_red_gap=1, blue_green_gap=1, blue_minimum=1)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_geometry_count_is_not_a_count(value):
    assert not observe_empty_hand_domain(Image.new("RGB", (1000, 380), BLUE),
        config=HandEmptyConfig(), geometry_count=value, has_candidates=False,
        geometry_reason="").verified
