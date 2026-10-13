from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.vision.scene_recognizer import (
    RoundResultOutcome,
    _classify_round_result,
    _load_builtin_round_result_references,
)


def modal(*, outcome=RoundResultOutcome.VICTORY, color=(250, 220, 60), button=True, dx=0):
    reference = _load_builtin_round_result_references()[outcome][0]
    mask = np.zeros((48, 96), dtype=np.uint8)
    for index in reference:
        mask[index // 96, index % 96] = 255
    image = Image.new("RGB", (960, 640), (20, 30, 55))
    glyph = Image.fromarray(mask).resize((250, 125), Image.Resampling.NEAREST)
    image.paste(Image.new("RGB", glyph.size, color), (355 + dx, 68), glyph)
    if button:
        ImageDraw.Draw(image).rounded_rectangle((360, 410, 600, 510), radius=30, fill=(220, 150, 35))
    return image


@pytest.mark.parametrize("color", [(250, 220, 60), (255, 240, 120), (255, 255, 255)])
@pytest.mark.parametrize("dx", [-5, 0, 5])
def test_explicit_victory_signature_matches_bright_gold_or_pale(color, dx):
    result = _classify_round_result(modal(color=color, dx=dx))
    assert result is not None and result.outcome is RoundResultOutcome.VICTORY


def test_gold_without_continue_button_is_not_result():
    assert _classify_round_result(modal(button=False)) is None


def test_gold_without_explicit_reference_never_means_not_defeat_equals_victory():
    assert _classify_round_result(modal(), references={}) is None
    refs = _load_builtin_round_result_references()
    assert _classify_round_result(modal(), references={RoundResultOutcome.DEFEAT: refs[RoundResultOutcome.DEFEAT]}) is None


def test_gold_defeat_shape_is_not_relabelled_victory():
    assert _classify_round_result(modal(outcome=RoundResultOutcome.DEFEAT)) is None


@pytest.mark.parametrize("color", [(145, 125, 75), (180, 180, 75), (110, 220, 250)])
def test_dim_background_gold_and_other_colors_are_not_bright_title_ink(color):
    assert _classify_round_result(modal(color=color)) is None


@pytest.mark.parametrize("shape", ["bars", "one_solid_panel", "letters"])
def test_ordinary_gold_ui_with_continue_shaped_button_still_requires_title_identity(shape):
    image = modal()
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 50, 660, 210), fill=(20, 30, 55))
    if shape == "bars":
        draw.rectangle((380, 85, 430, 180), fill=(250, 220, 60))
        draw.rectangle((510, 85, 560, 180), fill=(250, 220, 60))
    elif shape == "one_solid_panel":
        draw.rectangle((380, 85, 560, 180), fill=(250, 220, 60))
    else:
        draw.text((400, 90), "PLAY", fill=(250, 220, 60))
    assert _classify_round_result(image) is None
