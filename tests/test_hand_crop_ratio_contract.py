"""Pure-pixel contracts for primary versus recovered hand rank-crop ratios."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import src.vision.scene_recognizer as module
from src.config.hand_geometry import ShortHandRecoveryConfig

PRIMARY_RATIOS = (0.535, 0.60)
NONDEFAULT_RATIOS = [(0.4, 0.6), (0.7, 0.5)]


def _fan(*, starts=(50, 190), gray_slots=(), clipped=False, interior_ink=False):
    image = Image.new("RGB", (650, 400), (45, 75, 150))
    draw = ImageDraw.Draw(image)
    top = -25 if clipped else 30
    font = ImageFont.load_default(size=45)
    for index, left in enumerate(starts):
        face = (135, 135, 135) if index in gray_slots else (235, 235, 235)
        draw.rounded_rectangle((left, top, left + 199, top + 279), radius=8,
                               fill=face, outline=(70, 70, 70), width=2)
        draw.text((left + 15, max(2, top + 10)), "J", fill=(40, 5, 5), font=font)
    if clipped:
        # Straight interior strokes support the oversegmented primary seeds,
        # but lack the closing bottom corners required by actual recovery.
        for left in (120, 260):
            draw.line((left, 0, left, 247), fill="black", width=2)
    if interior_ink:
        draw.rectangle((251, 32, 295, 66), fill="black")
    return image


def _prepare(image, ratios):
    pixels = np.asarray(image)
    high, low = pixels.max(axis=2), pixels.min(axis=2)
    gray = (low >= 90) & (high < 175) & (high - low <= 18)
    mask = module._card_white_mask(image) | gray
    grayscale, difference = module._hand_edge_arrays(image)
    config = ShortHandRecoveryConfig(rank_crop_ratios=ratios)
    boxes = module._recover_short_hand_boxes(
        image, mask, maximum=20, config=config,
        grayscale=grayscale, horizontal_difference=difference,
    )
    assert len(boxes) == 2
    return mask, grayscale, difference, config, boxes


def _spy(monkeypatch, name):
    original = getattr(module, name)
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(module, name, record)
    return calls


@pytest.mark.parametrize("ratios", NONDEFAULT_RATIOS)
def test_left_roi_edge_uses_explicit_recovered_crop_contract(ratios):
    image = _fan(starts=(0, 100))
    mask, grayscale, difference, config, boxes = _prepare(image, ratios)
    assert boxes[0][0] == 2  # The first physical rounded corner is clipped horizontally.
    result = module._anchor_hand_slots(
        image, mask, boxes, 2, config=config,
        grayscale=grayscale, horizontal_difference=difference,
        input_crop_ratios=config.rank_crop_ratios,
    )
    assert result.usable and result.count == 2 and result.boxes == boxes
    assert result.clipped_slots == () and result.reason == ""
    if ratios == (0.4, 0.6):
        assert boxes == ((2, 32, 68, 142), (100, 32, 166, 142))


@pytest.mark.parametrize("gray_slots", [(), (0,), (0, 1)])
@pytest.mark.parametrize("ratios", NONDEFAULT_RATIOS)
def test_smaller_and_larger_rank_crops_preserve_physical_slots(ratios, gray_slots):
    image = _fan(gray_slots=gray_slots)
    _mask, _grayscale, _difference, _config, baseline = _prepare(image, PRIMARY_RATIOS)
    mask, grayscale, difference, config, boxes = _prepare(image, ratios)
    result = module._anchor_hand_slots(
        image, mask, boxes, 2, config=config,
        grayscale=grayscale, horizontal_difference=difference,
        input_crop_ratios=ratios,
    )
    assert result.usable and result.count == len(result.boxes) == 2
    assert result.boxes == boxes
    assert [(x, y) for x, y, _, _ in boxes] == [(x, y) for x, y, _, _ in baseline]
    height = boxes[0][3] - boxes[0][1]
    base_height = baseline[0][3] - baseline[0][1]
    assert (height < base_height) if ratios[0] < PRIMARY_RATIOS[0] else (height > base_height)
    assert boxes[0][2] - boxes[0][0] == round(height * ratios[1])


@pytest.mark.parametrize("gray_slots", [(), (0,)])
@pytest.mark.parametrize("ratios", NONDEFAULT_RATIOS)
def test_primary_crops_keep_primary_contract_even_with_custom_recovery_config(
    ratios, gray_slots, monkeypatch,
):
    image = _fan(gray_slots=gray_slots)
    baseline = module.infer_hand_geometry(image)
    assert baseline.usable and baseline.count == 2
    config = ShortHandRecoveryConfig(rank_crop_ratios=ratios)
    anchors = _spy(monkeypatch, "_anchor_hand_slots")
    checks = _spy(monkeypatch, "_verify_short_hand_origins")

    def no_fallback(*_args, **_kwargs):
        raise AssertionError("This physical fan must stay on the primary path")

    monkeypatch.setattr(module, "_recover_short_hand_boxes", no_fallback)
    assert module.infer_hand_geometry(image, recovery_config=config) == baseline
    assert len(anchors) == 1
    assert anchors[0][1]["config"] is config
    assert anchors[0][1]["input_crop_ratios"] == PRIMARY_RATIOS
    assert len(checks) == (0 if gray_slots else 1)
    for _args, kwargs, _result in checks:
        assert kwargs["input_crop_ratios"] == PRIMARY_RATIOS


@pytest.mark.parametrize("gray_slots", [(0,), (0, 1)])
@pytest.mark.parametrize("ratios", NONDEFAULT_RATIOS)
def test_actual_gray_fallback_passes_recovered_ratio_and_shared_arrays(
    ratios, gray_slots, monkeypatch,
):
    image = _fan(gray_slots=gray_slots)
    config = ShortHandRecoveryConfig(rank_crop_ratios=ratios)
    # Only force the primary detector to yield no candidate. Recovery and
    # anchoring still independently evaluate the actual gray/white pixels.
    monkeypatch.setattr(module, "_gray_hand_boxes", lambda *_args, **_kwargs: ())
    recoveries = _spy(monkeypatch, "_recover_short_hand_boxes")
    anchors = _spy(monkeypatch, "_anchor_hand_slots")
    result = module.infer_hand_geometry(image, maximum=3, recovery_config=config)
    assert result.usable and result.count == 2
    assert len(recoveries) == len(anchors) == 1
    recovery_kwargs, recovered = recoveries[0][1:]
    anchor_kwargs = anchors[0][1]
    assert recovered == result.boxes
    assert recovery_kwargs["maximum"] == 3
    assert anchor_kwargs["input_crop_ratios"] == ratios
    for key in ("config", "grayscale", "horizontal_difference"):
        assert anchor_kwargs[key] is recovery_kwargs[key]
    assert anchor_kwargs["config"] is config


@pytest.mark.parametrize("ratios", NONDEFAULT_RATIOS)
def test_actual_white_clipped_fallback_changes_contract_only_after_recovery(ratios, monkeypatch):
    image = _fan(clipped=True)
    config = ShortHandRecoveryConfig(rank_crop_ratios=ratios)
    primary_boxes = tuple((x, 0, x + 81, 135) for x in (52, 120, 190, 260))
    # The real first anchor must reject these supported but top-clipped
    # candidates; neither anchor nor recovery has a fabricated final result.
    monkeypatch.setattr(module, "_infer_white_hand_count", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr(module, "_infer_white_hand_boxes", lambda *_args, **_kwargs: primary_boxes)
    anchors = _spy(monkeypatch, "_anchor_hand_slots")
    recoveries = _spy(monkeypatch, "_recover_short_hand_boxes")
    checks = _spy(monkeypatch, "_verify_short_hand_origins")
    result = module.infer_hand_geometry(image, recovery_config=config)
    assert len(anchors) == 2 and len(recoveries) == len(checks) == 1
    first, second = anchors
    assert first[1]["input_crop_ratios"] == PRIMARY_RATIOS
    assert first[2].count == 4 and first[2].clipped_slots == (0, 1, 2, 3)
    assert second[1]["input_crop_ratios"] == ratios
    assert checks[0][1]["input_crop_ratios"] == ratios
    assert result.boxes == recoveries[0][2] == second[2].boxes
    assert result.count == 2 and not result.usable
    assert result.clipped_slots == (0, 1)
    assert result.reason == "rank_top_outside_input_roi"
    for key in ("config", "grayscale", "horizontal_difference"):
        assert first[1][key] is second[1][key] is recoveries[0][1][key] is checks[0][1][key]


@pytest.mark.parametrize("ratios", NONDEFAULT_RATIOS)
def test_horizontal_repair_reuses_input_ratio_not_its_recovery_output_ratio(ratios, monkeypatch):
    image = _fan(interior_ink=True)
    mask, grayscale, difference, input_config, boxes = _prepare(image, ratios)
    first, (left, top, right, bottom) = boxes
    shifted = (first, (left + 20, top, right + 20, bottom))
    candidate = module._anchor_hand_slots(
        image, mask, shifted, 2, config=input_config,
        grayscale=grayscale, horizontal_difference=difference, input_crop_ratios=ratios,
    )
    assert candidate.boxes == shifted
    config = ShortHandRecoveryConfig(minimum_edge_support=0.72)
    recoveries = _spy(monkeypatch, "_recover_short_hand_boxes")
    anchors = _spy(monkeypatch, "_anchor_hand_slots")
    result = module._verify_short_hand_origins(
        image, mask, candidate, maximum=3, config=config,
        grayscale=grayscale, horizontal_difference=difference, input_crop_ratios=ratios,
    )
    assert result.usable and result.count == 2 and result.boxes == boxes
    assert len(recoveries) == len(anchors) == 1
    assert recoveries[0][2] != boxes  # Recovery sizes use their own configuration.
    assert anchors[0][1]["input_crop_ratios"] == ratios
    for calls in (recoveries, anchors):
        kwargs = calls[0][1]
        assert kwargs["config"] is config
        assert kwargs["grayscale"] is grayscale
        assert kwargs["horizontal_difference"] is difference
