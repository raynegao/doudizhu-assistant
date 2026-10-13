from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config.hand_photometry import HandPhotometryConfig
from src.config.live_layout import LiveLayoutConfig, live_layout_from_dict
from src.vision.hand_photometry import prepare_hand_rank_crop


def _printed_crop(paper: int = 124, ink: tuple[int, int, int] = (65, 8, 4)) -> Image.Image:
    image = Image.new("RGB", (126, 210), (paper,) * 3)
    draw = ImageDraw.Draw(image)
    draw.line((25, 20, 70, 20, 70, 85, 58, 98, 30, 98, 22, 87), fill=ink, width=7)
    return image


def _prepare(image: Image.Image, **kwargs):
    return prepare_hand_rank_crop(image, geometry_verified=True, config=HandPhotometryConfig(**kwargs))


@pytest.mark.parametrize("paper", [90, 110, 123, 124, 150, 180])
@pytest.mark.parametrize("ink", [(60, 8, 4), (10, 10, 10)])
def test_supported_gray_paper_uses_one_gain_without_mutating_original(paper, ink) -> None:
    original = _printed_crop(paper, ink)
    before = original.tobytes()
    result = _prepare(original)
    assert result.evidence.reason == "neutral_gray_paper_gain"
    assert result.evidence.usable
    assert result.evidence.background_level == paper
    assert result.evidence.gain == 250 / paper
    assert result.image.size == original.size
    assert result.image.mode == "RGB"
    assert result.image.getpixel((100, 100)) == (250, 250, 250)
    assert result.image.getpixel((40, 20)) == tuple(round(value * 250 / paper) for value in ink)
    assert original.tobytes() == before
    # Already corrected input is stable; repeated pipeline exposure cannot
    # continue multiplying it or silently change a model's input a second time.
    again = _prepare(result.image)
    assert again.image is result.image
    assert again.evidence.gain == 1


@pytest.mark.parametrize("paper", [190, 220, 246, 250, 255])
def test_white_card_is_byte_preserved_including_antialiasing_and_red_ink(paper) -> None:
    image = _printed_crop(paper, (130, 16, 8))
    ImageDraw.Draw(image).line((5, 5, 60, 5), fill=(125, 125, 125), width=2)
    before = image.tobytes()
    result = _prepare(image)
    assert result.image is image
    assert result.image.tobytes() == before
    assert result.evidence.usable
    assert result.evidence.gain == 1


@pytest.mark.parametrize("paper", [90, 124, 180])
def test_unprinted_gray_panel_does_not_become_a_trusted_white_card(paper) -> None:
    image = Image.new("RGB", (126, 210), (paper,) * 3)
    result = _prepare(image)
    assert result.image is image
    assert result.evidence.reason == "ink_contrast_unsupported"
    assert not result.evidence.usable


def test_weak_ink_contrast_is_not_treated_as_a_supported_dark_rank() -> None:
    image = _printed_crop(90, (65, 8, 4))
    result = _prepare(image)
    assert result.image is image
    assert result.evidence.reason == "ink_contrast_unsupported"
    assert not result.evidence.usable


def test_gray_white_mixed_crop_is_not_normalized_or_trusted() -> None:
    image = _printed_crop()
    ImageDraw.Draw(image).rectangle((75, 0, 125, 209), fill="white")
    result = _prepare(image)
    assert result.image is image
    assert result.evidence.reason == "bright_or_mixed_paper"
    assert not result.evidence.usable


def test_no_model_or_rank_dependent_choice_between_two_gray_paper_levels() -> None:
    image = _printed_crop()
    ImageDraw.Draw(image).rectangle((63, 0, 125, 209), fill=(165, 165, 165))
    result = _prepare(image)
    assert result.image is image
    assert result.evidence.reason == "paper_level_ambiguous"
    assert not result.evidence.usable


def test_original_paper_shading_survives_a_uniform_selection_overlay() -> None:
    paper = np.repeat(np.linspace(124, 117, 210).round().astype(np.uint8)[:, None, None], 126, axis=1)
    image = Image.fromarray(np.repeat(paper, 3, axis=2))
    ImageDraw.Draw(image).line((25, 20, 70, 20, 70, 85, 58, 98, 30, 98), fill=(60, 8, 4), width=7)
    before = image.tobytes()
    result = _prepare(image)
    assert result.evidence.reason == "neutral_gray_paper_gain" and result.evidence.usable
    assert result.image.getpixel((100, 0))[0] > result.image.getpixel((100, 209))[0]
    assert image.tobytes() == before


def test_broad_illumination_gradient_is_not_explained_as_selected_paper() -> None:
    paper = np.repeat(np.linspace(90, 180, 210).round().astype(np.uint8)[:, None, None], 126, axis=1)
    image = Image.fromarray(np.repeat(paper, 3, axis=2))
    ImageDraw.Draw(image).rectangle((20, 20, 35, 80), fill=(10, 10, 10))
    result = _prepare(image)
    assert result.image is image and not result.evidence.usable
    assert result.evidence.reason == "paper_level_ambiguous"


def test_gray_majority_alone_does_not_override_dark_occluding_quadrant() -> None:
    image = _printed_crop()
    ImageDraw.Draw(image).rectangle((63, 105, 125, 209), fill=(5, 5, 5))
    result = _prepare(image)
    assert result.image is image
    assert result.evidence.reason == "paper_spatial_support_missing"
    assert not result.evidence.usable


def test_no_exposure_estimate_from_neutral_ink_on_colored_background() -> None:
    image = Image.new("RGB", (126, 210), (90, 110, 140))
    ImageDraw.Draw(image).rectangle((20, 20, 65, 65), fill=(124, 124, 124))
    result = _prepare(image)
    assert result.image is image
    assert result.evidence.gain == 1


@pytest.mark.parametrize("reason", ["unknown", "clipped"])
def test_geometry_denial_never_restores_brightness_or_trust(reason) -> None:
    image = _printed_crop()
    result = prepare_hand_rank_crop(image, geometry_verified=False, config=HandPhotometryConfig())
    assert result.image is image
    assert result.evidence.reason == "geometry_unverified"
    assert not result.evidence.usable


@pytest.mark.parametrize("size", [(0, 0), (0, 100), (10, 100), (100, 10)])
def test_empty_or_tiny_crops_are_not_resized_or_invented(size) -> None:
    image = Image.new("RGB", size)
    result = _prepare(image)
    assert result.image is image and result.evidence.gain == 1
    assert not result.evidence.usable


@pytest.mark.parametrize("size,minimum", [((19, 40), 20), ((126, 210), 127)])
def test_unsupported_gray_size_is_not_a_confident_passthrough(size, minimum) -> None:
    image = _printed_crop().resize(size)
    result = _prepare(image, minimum_crop_side=minimum)
    assert result.image is image and not result.evidence.usable
    assert result.evidence.reason == "unsupported_crop"


def test_unsupported_size_can_preserve_explicitly_white_paper_without_a_gain() -> None:
    image = _printed_crop(250, (130, 16, 8))
    result = _prepare(image, minimum_crop_side=127)
    assert result.image is image and result.evidence.usable
    assert result.evidence.gain == 1


def test_one_pixel_wide_gray_crop_cannot_create_nan_spatial_support() -> None:
    image = Image.new("RGB", (1, 20), (125, 125, 125))
    image.putpixel((0, 4), (0, 0, 0))
    result = _prepare(image, minimum_crop_side=2)
    assert not result.evidence.usable and result.image is image
    with pytest.raises(ValueError, match="at least 2"):
        HandPhotometryConfig(minimum_crop_side=1)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_unsupported_color_modes_are_not_silently_reinterpreted(mode) -> None:
    image = _printed_crop().convert(mode)
    result = _prepare(image)
    assert result.image is image and not result.evidence.usable


def test_scalar_gain_preserves_ink_color_without_uint8_wraparound() -> None:
    image = _printed_crop(100, (50, 5, 2))
    image.putpixel((120, 200), (230, 10, 20))
    result = _prepare(image)
    assert result.image.getpixel((120, 200)) == (255, 25, 50)
    assert result.image.getpixel((40, 20)) == (125, 12, 5)


def test_background_neutrality_uses_signed_channels() -> None:
    image = _printed_crop()
    data = np.asarray(image).copy()
    data[:120, :, :] = (5, 124, 124)
    tinted = Image.fromarray(data)
    result = _prepare(tinted)
    assert result.image is tinted and result.evidence.gain == 1
    assert not result.evidence.usable


def test_gain_and_gates_come_from_supplied_config() -> None:
    image = _printed_crop()
    result = _prepare(image, target_paper=240)
    assert result.evidence.gain == 240 / 124
    assert result.image.getpixel((100, 100)) == (240, 240, 240)
    refused = _prepare(image, background_min=130)
    assert refused.image is image


def test_configured_gain_limit_prevents_extreme_correction() -> None:
    image = _printed_crop()
    refused = _prepare(image, maximum_gain=1.5)
    assert refused.image is image
    assert not refused.evidence.usable
    assert refused.evidence.reason == "gain_limit_exceeded"


@pytest.mark.parametrize("field", list(HandPhotometryConfig.__dataclass_fields__))
@pytest.mark.parametrize("value", [True, "0.8", None, float("nan"), float("inf"), -1, 10**999])
def test_configuration_rejects_invalid_scalar_values(field, value) -> None:
    with pytest.raises(ValueError, match="hand_photometry"):
        HandPhotometryConfig.from_dict({field: value})


@pytest.mark.parametrize("payload", [
    {"background_min": 180, "background_max": 100},
    {"background_max": 250, "target_paper": 250},
    {"paper_tolerance": 90}, {"minimum_paper_fraction": 0.5},
    {"minimum_gray_fraction": 0.7},
    {"minimum_ink_fraction": 0.4, "maximum_ink_fraction": 0.3},
    {"target_paper": 0}, {"target_paper": 256}, {"minimum_crop_side": 1}, {"unknown": 1},
    [], None,
])
def test_configuration_rejects_invalid_intervals_unknown_fields_and_roots(payload) -> None:
    with pytest.raises(ValueError, match="hand_photometry"):
        HandPhotometryConfig.from_dict(payload)


def test_live_layout_snapshot_roundtrip_and_explicit_defaults_do_not_mutate_old_payload() -> None:
    legacy = {"app_name": "fixture"}
    config = live_layout_from_dict(legacy)
    assert config.hand_photometry == HandPhotometryConfig()
    assert legacy == {"app_name": "fixture"}
    custom = replace(config, hand_photometry=HandPhotometryConfig(target_paper=245))
    payload = json.loads(json.dumps(custom.to_payload()))
    restored = live_layout_from_dict(payload)
    assert restored == custom
    with pytest.raises(ValueError, match="hand_photometry"):
        LiveLayoutConfig(hand_photometry={})
