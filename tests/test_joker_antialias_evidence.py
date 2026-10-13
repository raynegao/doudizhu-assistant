"""A neutral coverage fringe is not a second ink color or a rank prediction."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config.joker_color import JokerColorConfig
from src.vision.card_classifier import CardPrediction
from src.vision.rank_glyph import extract_rank_glyph, read_joker_color
from src.vision.scene_recognizer import _fuse_card_prediction

OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _card(*, shift=(0, 0), gray=115, thickness=5):
    image = Image.new("RGB", (96, 144), (245, 245, 245))
    dx, dy = shift
    draw = ImageDraw.Draw(image)
    draw.line([(17+dx, 14+dy), (49+dx, 14+dy)], fill=(20, 20, 20), width=thickness)
    draw.line([(43+dx, 14+dy), (43+dx, 58+dy), (37+dx, 66+dy),
               (24+dx, 66+dy), (18+dx, 58+dy)], fill=(20, 20, 20), width=thickness)
    glyph = extract_rank_glyph(image)
    core = {p for p in glyph.pixels if all((p[0]+x, p[1]+y) in glyph.pixels for x, y in OFFSETS)}
    edge = {p for p in glyph.pixels-core if any((p[0]+x, p[1]+y) in core for x, y in OFFSETS)}
    for point in edge:
        image.putpixel(point, (gray, gray, gray))
    assert extract_rank_glyph(image).pixels == glyph.pixels
    return image, core, edge


def _read(image, config=None):
    return read_joker_color(image, extract_rank_glyph(image), config or JokerColorConfig())


@pytest.mark.parametrize("shift", [(0, 0), (4, 7), (11, 2)])
@pytest.mark.parametrize("gray", [101, 115, 129])
@pytest.mark.parametrize("thickness", [4, 5, 7])
def test_black_interior_and_one_pixel_neutral_fringe_keep_measured_fraction(shift, gray, thickness):
    image, _, edge = _card(shift=shift, gray=gray, thickness=thickness)
    before = image.tobytes(), extract_rank_glyph(image)
    evidence = _read(image)
    assert evidence.color == "black"
    assert evidence.reason == "black_rank_ink_antialiased_edge"
    assert evidence.black_fraction == pytest.approx(1-len(edge)/len(before[1].pixels))
    assert evidence.black_fraction < JokerColorConfig().min_color_fraction
    assert (image.tobytes(), extract_rank_glyph(image)) == before


@pytest.mark.parametrize("kind", ["gray_interior", "red_interior", "red_fringe", "blue_fringe",
                                   "all_gray", "gray_bridge", "small_core", "gray_paper"])
def test_weak_interior_thick_gray_or_conflicting_edge_stays_unknown(kind):
    image, core, edge = _card()
    if kind in {"gray_interior", "red_interior"}:
        image.putpixel(sorted(core)[len(core)//2], (115, 115, 115) if kind == "gray_interior" else (160, 20, 20))
    elif kind in {"red_fringe", "blue_fringe"}:
        image.putpixel(sorted(edge)[0], (160, 20, 20) if kind == "red_fringe" else (40, 40, 120))
    elif kind == "all_gray":
        for point in extract_rank_glyph(image).pixels:
            image.putpixel(point, (115, 115, 115))
    elif kind == "gray_bridge":
        # A remote one-pixel gray branch touches the rank but no solid black
        # neighbor. It must not be explained by some distant black interior.
        for x in range(49, 58):
            image.putpixel((x, 14), (115, 115, 115))
    elif kind == "small_core":
        assert _read(image, replace(JokerColorConfig(), min_component_pixels=len(core)+1)).color == "unknown"
        return
    else:
        pixels = np.asarray(image).copy()
        pixels[(pixels == 245).all(axis=2)] = (155, 155, 155)
        image = Image.fromarray(pixels)
    assert _read(image).color == "unknown"


def test_clipped_antialiased_rank_keeps_original_refusal():
    image, _, _ = _card()
    glyph = extract_rank_glyph(image)
    assert glyph.box is not None
    image = image.crop((glyph.box[0], 0, image.width, image.height))
    evidence = _read(image)
    assert (evidence.color, evidence.reason) == ("unknown", "rank_component_clipped")


def test_thin_serif_can_touch_core_connected_black_edge_instead_of_core_directly():
    image, _, _ = _card()
    for point in ((49, 14), (50, 14)):
        image.putpixel(point, (20, 20, 20))
    image.putpixel((51, 14), (115, 115, 115))
    glyph = extract_rank_glyph(image)
    core = {p for p in glyph.pixels if all((p[0]+x, p[1]+y) in glyph.pixels for x, y in OFFSETS)}
    assert (51, 14) in glyph.pixels
    assert not any((51+x, 14+y) in core for x, y in OFFSETS)
    assert _read(image).color == "black"


def test_gray_fringe_cannot_rely_on_diagonally_disconnected_black_speck():
    image, _, _ = _card()
    image.putpixel((50, 11), (20, 20, 20))
    image.putpixel((51, 11), (115, 115, 115))
    # The unchanged glyph selection is eight-connected; black support must
    # still have an orthogonal path to the actual solid interior.
    assert {(50, 11), (51, 11)} <= extract_rank_glyph(image).pixels
    assert _read(image).color == "unknown"


def test_black_channel_configuration_still_gates_the_actual_interior():
    image, _, _ = _card()
    assert _read(image, replace(JokerColorConfig(), black_max_channel=19)).color == "unknown"
    assert _read(image, replace(JokerColorConfig(), black_max_channel=20)).color == "black"


def test_custom_red_gate_cannot_be_overridden_by_black_core():
    image, _, edge = _card()
    for point in edge:
        image.putpixel(point, (115, 90, 90))
    evidence = _read(image, replace(JokerColorConfig(), red_channel_margin=20))
    assert 0 < evidence.red_fraction < 0.9
    assert evidence.black_fraction < 0.9
    assert evidence.color == "unknown"


@pytest.mark.parametrize("rank,expected", [("SJ", 0.973), ("BJ", 0), ("J", 0.973), ("A", 0.973)])
def test_antialias_support_never_changes_identity_or_increases_model_score(rank, expected):
    image, _, _ = _card()
    prediction = CardPrediction(rank, 0.973, {rank: 0.973})
    found, confidence, evidence = _fuse_card_prediction(prediction, None, joker_color=_read(image))
    assert (found, confidence) == (rank, expected)
    assert evidence.raw_confidence == 0.973


def test_red_decoration_outside_selected_glyph_cannot_vote():
    image, _, _ = _card()
    ImageDraw.Draw(image).polygon(((75, 0), (95, 0), (95, 55)), fill=(160, 20, 20))
    evidence = _read(image)
    assert evidence.color == "black" and evidence.red_fraction == 0


def test_ordinary_card_cannot_become_joker_from_antialiased_reference():
    image, _, _ = _card()
    found, confidence, evidence = _fuse_card_prediction(
        CardPrediction("K", 0.973, {"K": 0.973}), ("SJ", 0.999), joker_color=_read(image),
    )
    assert (found, confidence) == ("K", 0)
    assert evidence.fusion_reason == "joker_reference_requires_raw_identity"
