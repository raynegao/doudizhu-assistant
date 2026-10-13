"""Synthetic unknown-extent handoff to independently verified horizontal recovery."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import src.vision.scene_recognizer as module
from src.config.hand_geometry import ShortHandRecoveryConfig

UNKNOWN_EXTENT = "hand_slot_extent_unverified:candidate_count_unconfirmed"


def _fan(*, letters="J", clipped=False):
    image = Image.new("RGB", (650, 400), (45, 75, 150))
    draw = ImageDraw.Draw(image)
    top = -55 if clipped else 30
    font = ImageFont.load_default(size=45)
    for left in (50, 150):
        draw.rounded_rectangle((left, top, left + 199, top + 279), radius=8,
                               fill=(235, 235, 235), outline=(70, 70, 70), width=2)
        draw.text((left + 15, top + 10), letters, fill=(40, 5, 5), font=font)
    # Interior ink affects the incorrectly shifted sample, not the true rim.
    # This makes the upstream extent failure arise from pixels, not a stub.
    if not clipped:
        draw.rectangle((211, 32, 255, 66), fill="black")
    return image


def _prepare(*, letters="J", config=None):
    image = _fan(letters=letters)
    config = config or ShortHandRecoveryConfig()
    mask = module._card_white_mask(image)
    grayscale, difference = module._hand_edge_arrays(image)
    physical = module._recover_short_hand_boxes(
        image, mask, maximum=20, config=config,
        grayscale=grayscale, horizontal_difference=difference,
    )
    assert len(physical) == 2
    good = module._anchor_hand_slots(
        image, mask, physical, len(physical), config=config,
        grayscale=grayscale, horizontal_difference=difference,
    )
    assert good.usable and good.count == 2
    first, (left, top, right, bottom) = good.boxes
    shifted = (first, (left + 20, top, right + 20, bottom))
    unknown = module._anchor_hand_slots(
        image, mask, shifted, len(shifted), config=config,
        grayscale=grayscale, horizontal_difference=difference,
    )
    assert unknown.count is None and not unknown.usable and unknown.boxes == shifted
    assert unknown.reason == UNKNOWN_EXTENT
    return image, mask, grayscale, difference, config, good, unknown


def _verify(prepared, *, geometry=None, maximum=20):
    image, mask, grayscale, difference, config, _good, unknown = prepared
    return module._verify_short_hand_origins(
        image, mask, unknown if geometry is None else geometry, maximum=maximum,
        config=config, grayscale=grayscale, horizontal_difference=difference,
    )


def _assert_unknown(result, original):
    assert result.count is None and not result.usable
    assert result.boxes == original.boxes
    assert result.clipped_slots == ()
    assert "candidate_count_unconfirmed" in result.reason


@pytest.mark.parametrize("letters", ["A", "K", "Q", "J", "10"])
def test_real_extent_unknown_requires_physical_horizontal_recovery(letters):
    prepared = _prepare(letters=letters)
    good = prepared[-2]
    result = _verify(prepared)
    assert result == good
    assert result.count == 2 and result.usable


@pytest.mark.parametrize("returned_count", [0, 1, 3])
def test_recovery_failure_or_cardinality_mismatch_preserves_unknown(returned_count, monkeypatch):
    prepared = _prepare()
    good, unknown = prepared[-2:]
    recovered = (*good.boxes, good.boxes[-1])[:returned_count]
    monkeypatch.setattr(module, "_recover_short_hand_boxes", lambda *_args, **_kwargs: recovered)
    _assert_unknown(_verify(prepared), unknown)


def test_actual_maximum_is_a_cap_not_a_candidate_count(monkeypatch):
    prepared = _prepare()
    calls = []
    original = module._recover_short_hand_boxes

    def recover(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["maximum"] == 1
        assert "count" not in kwargs and "expected_count" not in kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_recover_short_hand_boxes", recover)
    _assert_unknown(_verify(prepared, maximum=1), prepared[-1])
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["unusable", "wrong_count", "vertical_rewrite"])
def test_second_anchor_must_confirm_all_slots_without_new_geometry(failure, monkeypatch):
    prepared = _prepare()
    good, unknown = prepared[-2:]
    if failure == "unusable":
        result = replace(good, usable=False)
    elif failure == "wrong_count":
        result = replace(good, count=1)
    else:
        result = replace(good, boxes=tuple((x, y + 10, right, bottom + 10)
                                          for x, y, right, bottom in good.boxes))
    monkeypatch.setattr(module, "_anchor_hand_slots", lambda *_args, **_kwargs: result)
    _assert_unknown(_verify(prepared), unknown)


def test_pure_vertical_unknown_cannot_be_accepted_by_horizontal_flow(monkeypatch):
    prepared = _prepare()
    good = prepared[-2]
    shifted = tuple((x, y + 20, right, bottom + 20) for x, y, right, bottom in good.boxes)
    unknown = module.HandGeometry(None, shifted, False, reason=UNKNOWN_EXTENT)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Supported x positions cannot justify horizontal recovery")

    monkeypatch.setattr(module, "_recover_short_hand_boxes", forbidden)
    _assert_unknown(_verify(prepared, geometry=unknown), unknown)


def test_combined_xy_error_cannot_hide_a_vertical_rewrite():
    prepared = _prepare()
    original = prepared[-1]
    shifted = tuple((x, y + 20, right, bottom + 20) for x, y, right, bottom in original.boxes)
    unknown = replace(original, boxes=shifted)
    _assert_unknown(_verify(prepared, geometry=unknown), unknown)


def test_missing_top_cannot_be_accepted_even_without_clipped_flag(monkeypatch):
    prepared = list(_prepare())
    prepared[0] = _fan(clipped=True)
    prepared[1] = module._card_white_mask(prepared[0])
    prepared[2], prepared[3] = module._hand_edge_arrays(prepared[0])
    original = prepared[-1]
    unknown = replace(original, boxes=tuple((x, 0, right, bottom - y)
                                            for x, y, right, bottom in original.boxes))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A missing physical top must stop before horizontal recovery")

    monkeypatch.setattr(module, "_recover_short_hand_boxes", forbidden)
    _assert_unknown(_verify(prepared, geometry=unknown), unknown)


@pytest.mark.parametrize("reason", ["ambiguous_gray_hand", "hand_slot_extent_mismatch:candidate_count_unconfirmed"])
def test_unrelated_unknown_reason_cannot_enter_horizontal_recovery(reason, monkeypatch):
    prepared = _prepare()
    unknown = replace(prepared[-1], reason=reason)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("This handoff only accepts the specific unverified-extent reason")

    monkeypatch.setattr(module, "_recover_short_hand_boxes", forbidden)
    assert _verify(prepared, geometry=unknown) == unknown


def test_actual_config_and_existing_arrays_reach_recovery_and_second_anchor(monkeypatch):
    config = ShortHandRecoveryConfig(minimum_edge_support=.72)
    prepared = _prepare(config=config)
    image, _mask, grayscale, difference, _config, good, _unknown = prepared
    original_recover = module._recover_short_hand_boxes
    original_anchor = module._anchor_hand_slots
    calls = []

    def record(stage, function, *args, **kwargs):
        assert kwargs["config"] is config
        assert kwargs["grayscale"] is grayscale
        assert kwargs["horizontal_difference"] is difference
        calls.append(stage)
        return function(*args, **kwargs)

    monkeypatch.setattr(module, "_recover_short_hand_boxes",
                        lambda *args, **kwargs: record("recover", original_recover, *args, **kwargs))
    monkeypatch.setattr(module, "_anchor_hand_slots",
                        lambda *args, **kwargs: record("anchor", original_anchor, *args, **kwargs))
    original_diff = np.diff
    original_convert = Image.Image.convert

    def no_full_difference(array, *args, **kwargs):
        assert array.ndim != 2, "The shared full-ROI difference must not be recomputed"
        return original_diff(array, *args, **kwargs)

    def no_grayscale_conversion(value, mode=None, *args, **kwargs):
        assert not (mode == "L" and value.size == image.size), "The shared grayscale must be reused"
        return original_convert(value, mode, *args, **kwargs)

    monkeypatch.setattr(np, "diff", no_full_difference)
    monkeypatch.setattr(Image.Image, "convert", no_grayscale_conversion)
    assert _verify(prepared, maximum=3) == good
    assert calls == ["recover", "anchor"]
