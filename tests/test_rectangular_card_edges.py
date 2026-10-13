from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config.hand_geometry import ShortHandRecoveryConfig
from src.vision.rectangular_card_edges import rectangular_slot_supported

CONFIG = ShortHandRecoveryConfig()


def _card(*, selected=False, border_width=2, raised=False):
    image = Image.new("L", (300, 380), 68)
    top = 20 if raised else 70
    face, border = (114, 60) if selected else (228, 125)
    draw = ImageDraw.Draw(image)
    draw.rectangle((74, top, 276, top + 280), fill=face, outline=border, width=border_width)
    draw.rectangle((86, top + 20, 98, top + 58), fill=20)
    return np.asarray(image), top + border_width, top + 281 - border_width


def _supported(gray, *, left=74, top=72, right=124, bottom=349, config=CONFIG):
    return rectangular_slot_supported(
        gray, left=left, top=top, right=right, bottom=bottom, config=config,
    )


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("raised", [False, True])
@pytest.mark.parametrize("border_width", [1, 2, 3])
@pytest.mark.parametrize("offsets", [(0, 0), (2, -1), (-2, 2)])
def test_square_card_accepts_visible_face_and_nearby_candidate_endpoints(
    selected, raised, border_width, offsets,
):
    gray, top, bottom = _card(selected=selected, border_width=border_width, raised=raised)
    assert _supported(gray, top=top + offsets[0], bottom=bottom + offsets[1])


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("raised", [(12, 13), tuple(range(13)), tuple(range(14))])
def test_existing_dense_primary_slots_have_independent_square_extents(selected, raised):
    image = Image.new("L", (1000, 380), 68)
    draw = ImageDraw.Draw(image)
    starts = [74 + index * 50 for index in range(14)]
    for index, left in enumerate(starts):
        top = 20 if index in raised else 70
        face, rim = (114, 60) if selected and index in raised else (228, 125)
        draw.rectangle((left, top, left + 202, top + 280), fill=face, outline=rim, width=2)
        draw.rectangle((left + 12, top + 20, left + 24, top + 58), fill=20)
    gray = np.asarray(image)
    for index, left in enumerate(starts):
        top = (20 if index in raised else 70) + 2
        right = starts[index + 1] if index + 1 < len(starts) else left + 90
        assert _supported(gray, left=left, top=top + 2, right=right, bottom=top + 276)


@pytest.mark.parametrize("kind", [
    "vertical_only", "top_only", "bottom_only", "different_x", "side_gap",
    "different_rim", "reverse_polarity", "crossed_lines_without_face",
])
def test_strokes_cannot_substitute_for_two_connected_corners(kind):
    image = Image.new("L", (300, 380), 228)
    draw = ImageDraw.Draw(image)
    draw.line((74, 70, 74, 350), fill=125, width=2)
    if kind != "bottom_only" and kind != "vertical_only":
        draw.line((74, 70, 115, 70), fill=125, width=2)
    if kind != "top_only" and kind != "vertical_only":
        draw.line((95 if kind == "different_x" else 74, 350, 115, 350), fill=125, width=2)
    if kind == "side_gap":
        draw.rectangle((73, 150, 78, 270), fill=228)
    elif kind == "different_rim":
        draw.line((74, 72, 74, 349), fill=5, width=2)
    elif kind == "reverse_polarity":
        image = Image.fromarray(255 - np.asarray(image))
    elif kind == "crossed_lines_without_face":
        # No face lies inside any endpoint allowed by alignment tolerance.
        draw.rectangle((76, 72, 115, 100), fill=125)
        draw.rectangle((76, 320, 115, 349), fill=125)
    assert not _supported(np.asarray(image), bottom=350)


@pytest.mark.parametrize("color", [0, 68, 114, 228])
def test_featureless_panels_are_not_closed_slots(color):
    assert not _supported(np.full((380, 300), color, dtype=np.uint8))


@pytest.mark.parametrize("clip", ["top", "bottom"])
def test_clipped_card_has_no_two_visible_corners(clip):
    image = Image.new("L", (300, 380), 68)
    draw = ImageDraw.Draw(image)
    top, bottom = (-10, 270) if clip == "top" else (110, 390)
    draw.rectangle((74, top, 276, bottom), fill=228, outline=125, width=2)
    assert not _supported(np.asarray(image), top=max(1, top + 2), bottom=min(379, bottom - 1))


@pytest.mark.parametrize("offsets", [(30, 30), (-30, -30), (0, -30), (30, 0)])
def test_real_corners_far_from_proposed_extent_do_not_authorize_it(offsets):
    gray, top, bottom = _card()
    assert not _supported(gray, top=top + offsets[0], bottom=bottom + offsets[1])


def test_top_and_bottom_on_different_sides_cannot_be_combined():
    image = Image.new("L", (300, 380), 228)
    draw = ImageDraw.Draw(image)
    draw.line((70, 70, 70, 350), fill=125, width=2)
    draw.line((70, 70, 120, 70), fill=125, width=2)
    draw.line((79, 70, 79, 350), fill=125, width=2)
    draw.line((79, 350, 120, 350), fill=125, width=2)
    assert not _supported(np.asarray(image), bottom=350)


def test_matching_t_decoration_is_an_explicit_local_evidence_limit():
    # This shares exactly the relevant pixels of an overlapped square rim.
    # The caller's primary-grid/cardinality gate, not this bool, owns the slot.
    image = Image.new("L", (300, 380), 228)
    draw = ImageDraw.Draw(image)
    draw.line((74, 70, 74, 350), fill=125, width=2)
    draw.line((20, 70, 120, 70), fill=125, width=2)
    draw.line((20, 350, 120, 350), fill=125, width=2)
    assert _supported(np.asarray(image), bottom=350) is True


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_existing_grayscale_is_not_mutated_or_interpreted_with_uint8_wrap(dtype):
    gray, top, bottom = _card()
    gray = gray.astype(dtype)
    before = gray.copy()
    assert _supported(gray, top=top, bottom=bottom)
    np.testing.assert_array_equal(gray, before)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 256.0])
def test_invalid_local_pixel_values_fail_closed(value):
    gray, top, bottom = _card()
    invalid = gray.astype(np.float64)
    invalid[:, 60:100] = value
    assert not _supported(invalid, top=top, bottom=bottom)


@pytest.mark.parametrize("gray", [
    np.ones((300,), dtype=np.uint8), np.ones((20, 20, 3), dtype=np.uint8),
    np.ones((380, 300), dtype=bool), np.ones((380, 300), dtype=complex),
    np.ones((380, 300), dtype=object), [[1, 2], [3, 4]],
])
def test_invalid_grayscale_shape_and_type_fail_closed(gray):
    assert not _supported(gray)


@pytest.mark.parametrize("bounds", [
    {"left": -1}, {"right": 301}, {"left": 124}, {"right": 75},
    {"top": 0}, {"bottom": 380}, {"top": 349}, {"bottom": 90},
    {"left": True}, {"top": 72.5},
])
def test_invalid_or_insufficient_extent_fails_closed(bounds):
    gray, _, _ = _card()
    assert not _supported(gray, **bounds)


def test_custom_support_is_respected_without_changing_configuration():
    gray, top, bottom = _card()
    strict = replace(CONFIG, minimum_corner_support=1.0)
    assert _supported(gray, top=top + 2, bottom=bottom - 1, config=strict)
    assert strict.minimum_corner_support == 1.0
    assert CONFIG.minimum_corner_support == 0.85


def test_invalid_configuration_is_an_explicit_programming_error():
    gray, _, _ = _card()
    with pytest.raises(ValueError, match="ShortHandRecoveryConfig"):
        _supported(gray, config=None)
