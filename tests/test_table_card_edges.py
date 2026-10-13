from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as scene_module
from src.config.live_layout import LiveLayoutConfig
from src.state.events import PlayerSeat
from src.vision.scene_recognizer import (
    SceneRecognizer,
    SeatRole,
    _card_white_mask,
    _filter_table_card_edges,
    infer_hand_geometry,
    segment_card_boxes,
)


def _table(kind: str) -> Image.Image:
    image = Image.new("RGB", (780, 420), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    if kind == "empty":
        return image
    if kind == "text_only":
        draw.text((20, 110), "PASS", fill="white")
        return image
    starts = (14, 88)
    if kind == "separated":
        starts = (14, 245, 480)
    elif kind == "vertical_wrap":
        starts = tuple(14 + index * 74 for index in range(9))
    for left in starts:
        draw.rounded_rectangle(
            (left, 91, left + 168, 332), radius=8,
            fill=(230, 230, 230), outline=(125, 125, 125), width=2,
        )
        if kind in {"pair_with_stem", "vertical_word"}:
            draw.rectangle((left + 27, 104, left + 31, 160), fill=(0, 0, 0))
        if kind == "vertical_word":
            for y in (175, 215, 255):
                draw.rectangle((left + 27, y, left + 31, y + 22), fill=(0, 0, 0))
    if kind == "vertical_wrap":
        draw.rounded_rectangle(
            (starts[-1], 194, starts[-1] + 168, 414), radius=8,
            fill=(230, 230, 230), outline=(125, 125, 125), width=2,
        )
    return image


@pytest.mark.parametrize("kind,count", [
    ("empty", 0), ("text_only", 0), ("separated", 3), ("plain_pair", 2),
    ("pair_with_stem", 2), ("vertical_word", 2), ("vertical_wrap", 10),
])
def test_table_edge_filter_preserves_real_cards_and_drops_glyph_stems(kind, count) -> None:
    image = _table(kind)
    old_boxes = segment_card_boxes(image)
    boxes = segment_card_boxes(image, table_edges=True)

    assert len(boxes) == count
    if kind in {"pair_with_stem", "vertical_word"}:
        assert len(old_boxes) == 6
        assert boxes == (old_boxes[0], old_boxes[3])
    else:
        assert boxes == old_boxes
    if kind == "vertical_wrap":
        wrapped = [box for box in boxes if box[0] == 606]
        assert len(wrapped) == 2
        assert wrapped[1][1] == 194


@pytest.mark.parametrize("support_count", [0, 1])
def test_too_few_supported_borders_retains_original_candidate(support_count) -> None:
    image = Image.new("RGB", (300, 240), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    if support_count:
        draw.line((19, 10, 19, 224), fill="black", width=1)
    boxes = ((20, 10, 100, 156), (80, 10, 160, 156), (140, 10, 220, 156))

    assert _filter_table_card_edges(image, boxes) == boxes


def test_lower_body_support_is_required_not_only_long_upper_stem() -> None:
    image = Image.new("RGB", (320, 230), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    for left in (20, 160):
        draw.line((left - 1, 10, left - 1, 209), fill="black")
    # 60% continuous support alone would accept this upper glyph stroke;
    # only 20% of the lower half has support, so it is not a card border.
    draw.line((89, 10, 89, 129), fill="black")
    boxes = ((20, 10, 100, 146), (90, 10, 170, 146), (160, 10, 240, 146))

    assert _filter_table_card_edges(image, boxes) == (boxes[0], boxes[2])


def test_default_segmentation_and_hand_fallback_do_not_enable_table_filter(monkeypatch) -> None:
    image = _table("plain_pair")
    mask = _card_white_mask(image)
    monkeypatch.setattr(
        scene_module, "_filter_table_card_edges",
        lambda *_args: pytest.fail("table filter used in legacy/hand path"),
    )

    assert len(segment_card_boxes(image)) == 2
    assert infer_hand_geometry(image).count == 2
    assert np.array_equal(_card_white_mask(image), mask)


def test_separated_cards_skip_overlap_filter_even_when_enabled(monkeypatch) -> None:
    image = _table("separated")
    monkeypatch.setattr(
        scene_module, "_filter_table_card_edges",
        lambda *_args: pytest.fail("separated cards must not use overlap filter"),
    )

    assert len(segment_card_boxes(image, table_edges=True)) == 3


def test_overlapping_table_does_not_repeat_full_segmentation(monkeypatch) -> None:
    original = scene_module._segment_overlapping_card_boxes
    calls = []

    def spy(image, **kwargs):
        calls.append(image.size)
        return original(image, **kwargs)

    monkeypatch.setattr(scene_module, "_segment_overlapping_card_boxes", spy)

    assert len(segment_card_boxes(_table("pair_with_stem"), table_edges=True)) == 2
    assert len(calls) == 1


def test_opponent_observation_explicitly_enables_table_edges(tmp_path, monkeypatch) -> None:
    recognizer = SceneRecognizer(
        LiveLayoutConfig(templates_dir=tmp_path), predictor=lambda _crops: (),
        remaining_reader=lambda _: {},
    )
    calls = []

    def segment_spy(image, *, table_edges=False):
        calls.append(table_edges)
        return ()

    monkeypatch.setattr(scene_module, "segment_card_regions", segment_spy)
    recognizer._observe_seat(
        Image.new("RGB", (1000, 600), (45, 75, 145)), PlayerSeat.LEFT,
        role=(SeatRole.FARMER, 0.99), remaining=(15, 0.99, True),
    )

    assert calls == [True]
