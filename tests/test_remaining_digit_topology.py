from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as scene_module
from src.vision.scene_recognizer import (
    CvRemainingReader,
    _glyph_similarity,
    _remaining_glyph_signature,
    _remaining_signature_hole_count,
    _remaining_signature_holes,
    _remaining_zero_hole_incompatible,
)


def _digit_mask(digit: str) -> np.ndarray:
    """Artificial glyphs: no recording, model, local template or work dependency."""
    image = Image.new("L", (32, 48), 0)
    draw = ImageDraw.Draw(image)
    if digit == "0":
        draw.rectangle((2, 1, 29, 46), outline=255, width=5)
    elif digit == "1":
        draw.line((4, 9, 17, 1, 17, 44), fill=255, width=6)
        draw.line((4, 44, 29, 44), fill=255, width=5)
    elif digit == "2":
        draw.line((2, 8, 9, 2, 24, 2, 29, 8, 29, 16, 3, 44, 29, 44), fill=255, width=5)
    elif digit == "8":
        draw.rectangle((2, 1, 29, 24), outline=255, width=5)
        draw.rectangle((2, 23, 29, 46), outline=255, width=5)
    elif digit in {"9", "6"}:
        draw.rectangle((2, 1, 29, 22), outline=255, width=5)
        draw.rectangle((25, 19, 29, 43), fill=255)
        draw.rectangle((4, 42, 29, 46), fill=255)
        if digit == "6":
            image = image.rotate(180)
    else:
        raise AssertionError(digit)
    mask = np.asarray(image, dtype=np.uint8) > 0
    rows, columns = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1].copy()


def _reader() -> CvRemainingReader:
    reader = object.__new__(CvRemainingReader)
    reader._digit_references = {
        digit: [_remaining_glyph_signature(_digit_mask(digit))] for digit in "012689"
    }
    reader._digit_reference_holes = {
        digit: tuple((ref, _remaining_signature_hole_count(ref)) for ref in refs)
        for digit, refs in reader._digit_references.items()
    }
    return reader


def _counter(value: str, *, scale: float = 1.0, shift: tuple[int, int] = (0, 0)) -> Image.Image:
    image = Image.new("RGB", (140, 90), (43, 73, 145))
    left, top = 20 + shift[0], 20 + shift[1]
    for digit in value:
        mask = _digit_mask(digit)
        glyph = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
        glyph = glyph.resize(
            (round(glyph.width * scale), round(glyph.height * scale)),
            resample=Image.Resampling.NEAREST,
        )
        image.paste((245, 180, 50), (left, top), glyph)
        left += glyph.width + 12
    return image


@pytest.mark.parametrize(
    "value,expected",
    [("19", 19), ("18", 18), ("10", 10), ("9", 9), ("0", None), ("20", 20), ("29", None)],
)
def test_counter_domain_is_checked_after_glyph_evidence(value: str, expected: int | None) -> None:
    assert _reader().read(_counter(value)).count == expected


@pytest.mark.parametrize("value", ["19", "18", "10", "9", "0", "20"])
@pytest.mark.parametrize("scale,shift", [(0.9, (-2, 3)), (1.0, (2, -3)), (1.1, (3, 2))])
def test_counter_topology_survives_small_scale_and_translation(
    value: str,
    scale: float,
    shift: tuple[int, int],
) -> None:
    expected = int(value) if value != "0" else None
    assert _reader().read(_counter(value, scale=scale, shift=shift)).count == expected


@pytest.mark.parametrize("digit", ["9", "6"])
def test_off_center_loop_vetoes_a_better_foreground_zero_score(
    monkeypatch: pytest.MonkeyPatch,
    digit: str,
) -> None:
    reader = _reader()
    observed = _remaining_glyph_signature(_digit_mask(digit))
    zero = reader._digit_references["0"][0]

    def biased_similarity(left: frozenset[int], right: frozenset[int]) -> float:
        if left == observed:
            if right == zero:
                return 0.71
            return 0.66 if right == observed else 0.30
        return _glyph_similarity(left, right)

    monkeypatch.setattr(scene_module, "_glyph_similarity", biased_similarity)
    assert biased_similarity(observed, zero) > biased_similarity(observed, observed)
    assert reader.read(_counter("1" + digit)).count == int("1" + digit)


def test_zero_veto_does_not_directly_assign_nine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scene_module, "_glyph_similarity", lambda *_: 0.10)
    digit, _ = _reader()._classify_digit(_digit_mask("9"), index=1, digit_count=2, prefix=["1"])
    assert digit is None


@pytest.mark.parametrize("last_hole_row,expected_veto", [(31, True), (32, False)])
@pytest.mark.parametrize("shift", [(-2, -3), (0, 0), (2, 3)])
def test_zero_veto_midpoint_boundary_is_strict_and_translation_invariant(
    last_hole_row: int,
    expected_veto: bool,
    shift: tuple[int, int],
) -> None:
    # Foreground y=6..57 has midpoint 31.5. A hole ending at 31 is a veto;
    # one reaching 32 is not. Do not move this boundary to fit a recording.
    mask = np.zeros((64, 64), dtype=bool)
    mask[6:58, 20:44] = True
    mask[12 : last_hole_row + 1, 26:38] = False
    mask = np.roll(mask, shift, axis=(0, 1))
    signature = frozenset(int(index) for index in np.flatnonzero(mask))
    holes = _remaining_signature_holes(signature)
    assert len(holes) == 1
    assert _remaining_signature_hole_count(signature) == len(holes)
    assert _remaining_zero_hole_incompatible(signature, holes) is expected_veto


@pytest.mark.parametrize("digit,hole_count", [("0", 1), ("1", 0), ("8", 2), ("9", 1)])
def test_hole_components_preserve_existing_count_semantics(digit: str, hole_count: int) -> None:
    signature = _remaining_glyph_signature(_digit_mask(digit))
    holes = _remaining_signature_holes(signature)
    assert len(holes) == _remaining_signature_hole_count(signature) == hole_count
    if digit in {"0", "1", "8"}:
        assert _remaining_zero_hole_incompatible(signature, holes) is False


def test_empty_open_and_tiny_holes_do_not_create_zero_veto() -> None:
    assert _remaining_signature_holes(frozenset()) == ()
    assert _remaining_zero_hole_incompatible(frozenset(), ()) is False
    mask = np.zeros((64, 64), dtype=bool)
    mask[6:58, 20:44] = True
    mask[12, 25:28] = False  # Three pixels retain the existing <4 noise rule.
    signature = frozenset(int(index) for index in np.flatnonzero(mask))
    assert _remaining_signature_holes(signature) == ()
    mask[12:25, 20:35] = False  # Open to the exterior, not an enclosed hole.
    signature = frozenset(int(index) for index in np.flatnonzero(mask))
    assert _remaining_signature_holes(signature) == ()
