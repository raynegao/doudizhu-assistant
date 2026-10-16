"""Counter glyphs clipped by the ROI boundary are decorations or unknown, never a leading 1.

Independent acceptance session acceptance-20261014-006 (frozen ``04f83e90``):
the left remaining-count ROI cuts into the landlord avatar. While the avatar
animated, its skin-coloured edge grew into a 35 px tall, 10 px wide run at the
first ROI columns, passed the glyph size gates, matched no digit and fell back
to ``1``: a visible ``4`` was read as ``14`` and ``3`` as ``13``. A third
candidate made a visible ``12`` unknown. The avatar edge is clipped by the ROI
and shorter than the real digit, which a counter glyph never is.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.vision.scene_recognizer import (
    CvRemainingReader,
    _remaining_digit_masks,
    _remaining_glyph_signature,
    _remaining_signature_hole_count,
)


def _digit_mask(digit: str) -> np.ndarray:
    image = Image.new("L", (32, 48), 0)
    draw = ImageDraw.Draw(image)
    if digit == "1":
        draw.line((4, 9, 17, 1, 17, 44), fill=255, width=6)
        draw.line((4, 44, 29, 44), fill=255, width=5)
    elif digit == "2":
        draw.line((2, 8, 9, 2, 24, 2, 29, 8, 29, 16, 3, 44, 29, 44), fill=255, width=5)
    elif digit == "9":
        draw.rectangle((2, 1, 29, 22), outline=255, width=5)
        draw.rectangle((25, 19, 29, 43), fill=255)
        draw.rectangle((4, 42, 29, 46), fill=255)
    else:
        raise AssertionError(digit)
    mask = np.asarray(image, dtype=np.uint8) > 0
    rows, columns = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1].copy()


def _reader() -> CvRemainingReader:
    reader = object.__new__(CvRemainingReader)
    reader._digit_references = {
        digit: [_remaining_glyph_signature(_digit_mask(digit))] for digit in "129"
    }
    reader._digit_reference_holes = {
        digit: tuple((ref, _remaining_signature_hole_count(ref)) for ref in refs)
        for digit, refs in reader._digit_references.items()
    }
    return reader


def _counter(value: str) -> Image.Image:
    image = Image.new("RGB", (140, 90), (43, 73, 145))
    left = 30
    for digit in value:
        mask = _digit_mask(digit)
        glyph = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
        image.paste((245, 180, 50), (left, 20), glyph)
        left += glyph.width + 12
    return image


def _with_edge_run(image: Image.Image, *, height: int, side: str = "left",
                   width: int = 14, top: int = 28) -> Image.Image:
    # Avatar skin passes the yellow counter mask: r>145, g>70, b<165, r-b>45, g-b>5.
    skin = (236, 190, 150)
    draw = ImageDraw.Draw(image)
    left = 0 if side == "left" else image.width - width
    draw.rectangle((left, top, left + width - 1, top + height - 1), fill=skin)
    return image


def test_plain_counters_still_read() -> None:
    assert _reader().read(_counter("9")).count == 9
    assert _reader().read(_counter("12")).count == 12


@pytest.mark.parametrize("side", ["left", "right"])
def test_shorter_run_clipped_by_the_roi_edge_is_dropped(side: str) -> None:
    image = _with_edge_run(_counter("9"), height=35, side=side)
    assert len(_remaining_digit_masks(image)) == 1
    assert _reader().read(image).count == 9


def test_edge_decoration_cannot_turn_a_single_digit_into_a_teen() -> None:
    result = _reader().read(_with_edge_run(_counter("9"), height=35))
    assert result.count == 9


def test_edge_decoration_does_not_make_two_real_digits_unknown() -> None:
    assert _reader().read(_with_edge_run(_counter("12"), height=35)).count == 12


def test_clipped_run_with_digit_height_makes_the_read_unknown() -> None:
    image = _with_edge_run(_counter("9"), height=48, top=20)
    assert _remaining_digit_masks(image) == ()
    assert _reader().read(image).count is None


def test_only_clipped_candidates_are_unknown() -> None:
    image = Image.new("RGB", (140, 90), (43, 73, 145))
    assert _remaining_digit_masks(_with_edge_run(image, height=48, top=20)) == ()


def test_sub_glyph_edge_slivers_remain_ignored_as_before() -> None:
    image = _with_edge_run(_counter("9"), height=10, width=4)
    assert _reader().read(image).count == 9
