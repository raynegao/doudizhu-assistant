from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as scene_module
from src.config.live_layout import LiveLayoutConfig
from src.vision.card_classifier import CardPrediction
from src.vision.scene_recognizer import (
    SceneRecognizer,
    _card_white_mask,
    _infer_white_hand_boxes,
    _infer_white_hand_count,
    infer_hand_geometry,
    infer_overlapping_hand_boxes,
    infer_visible_hand_count,
    segment_card_boxes,
)


def _hand(
    count: int = 14,
    *,
    selected: tuple[int, ...] = (),
    raised: tuple[int, ...] = (),
    top: int = 70,
    raised_top: int = 20,
) -> tuple[Image.Image, tuple[int, ...]]:
    image = Image.new("RGB", (1000, 380), (35, 75, 120))
    draw = ImageDraw.Draw(image)
    first = (image.width - (202 + (count - 1) * 50)) // 2
    starts = tuple(first + index * 50 for index in range(count))
    for index, left in enumerate(starts):
        y = raised_top if index in raised else top
        face = (114, 114, 114) if index in selected else (228, 228, 228)
        border = (60, 60, 60) if index in selected else (125, 125, 125)
        draw.rectangle((left, y, left + 202, y + 280), fill=face, outline=border, width=2)
        draw.rectangle((left + 12, y + 20, left + 24, y + 58), fill=(20, 20, 20))
    return image, starts


@pytest.mark.parametrize("count,selected", [
    (14, (6,)), (14, (12,)), (14, (13,)), (14, (12, 13)),
    (14, tuple(range(14))), (1, (0,)), (2, (0, 1)), (3, (0, 1, 2)),
])
def test_gray_selection_keeps_slots_and_full_rank_crops(count, selected) -> None:
    image, starts = _hand(count, selected=selected)
    geometry = infer_hand_geometry(image)

    assert geometry.usable
    assert geometry.count == count
    assert infer_visible_hand_count(image) == count
    assert infer_overlapping_hand_boxes(image, count) == geometry.boxes
    for box, left in zip(geometry.boxes, starts, strict=True):
        assert abs(box[0] - left) <= 3
        assert box[0] <= left + 12 and box[2] > left + 24
        assert box[1] <= 90 and box[3] > 128


@pytest.mark.parametrize("count", [1, 2, 3, 14, 17])
def test_normal_geometry_preserves_white_baseline_boxes(count) -> None:
    image, _ = _hand(count)
    expected_count = _infer_white_hand_count(image)
    expected_boxes = _infer_white_hand_boxes(image, expected_count)

    geometry = infer_hand_geometry(image)

    assert geometry.count == expected_count == count
    assert geometry.boxes == expected_boxes
    assert geometry.usable


@pytest.mark.parametrize("raised", [(12, 13), tuple(range(13)), tuple(range(14))])
def test_raised_slots_use_individual_tops_without_enlarging_crops(raised) -> None:
    plain, starts = _hand()
    image, _ = _hand(raised=raised)
    baseline = infer_hand_geometry(plain)
    geometry = infer_hand_geometry(image)

    assert geometry.usable
    assert geometry.count == 14
    for index, (box, old_box, left) in enumerate(zip(geometry.boxes, baseline.boxes, starts, strict=True)):
        assert box[3] - box[1] == old_box[3] - old_box[1]
        assert box[2] - box[0] == old_box[2] - old_box[0]
        rank_top = (20 if index in raised else 70) + 20
        assert box[1] <= rank_top and box[3] > rank_top + 38
        assert box[0] <= left + 12 < box[2]
    if len(raised) != 14:
        assert max(box[1] for box in geometry.boxes) - min(box[1] for box in geometry.boxes) >= 45


@pytest.mark.parametrize("raised_top", [-5, -20, -50])
def test_clipped_selection_keeps_count_but_marks_slots_unusable(raised_top) -> None:
    image, _ = _hand(raised=(12, 13), raised_top=raised_top)
    geometry = infer_hand_geometry(image)

    assert geometry.count == 14
    assert len(geometry.boxes) == 14
    assert not geometry.usable
    assert geometry.clipped_slots == (12, 13)
    assert geometry.reason == "rank_top_outside_input_roi"


@pytest.mark.parametrize("kind", ["full", "wide_panel", "portrait_without_rank"])
def test_ambiguous_gray_background_does_not_guess_a_count(kind) -> None:
    image = Image.new("RGB", (1000, 380), (35, 75, 120))
    draw = ImageDraw.Draw(image)
    if kind == "full":
        draw.rectangle((0, 0, 999, 379), fill=(114, 114, 114))
    elif kind == "wide_panel":
        draw.rectangle((25, 20, 975, 350), fill=(114, 114, 114))
    else:
        draw.rectangle((400, 70, 602, 350), fill=(114, 114, 114))

    geometry = infer_hand_geometry(image, count=17)

    assert geometry.count is None
    assert geometry.boxes == ()
    assert not geometry.usable
    assert geometry.reason.startswith("ambiguous_gray_")
    assert infer_visible_hand_count(image) is None
    assert infer_overlapping_hand_boxes(image, 17) == ()


def test_empty_blue_roi_is_not_reported_as_ambiguous_gray_hand() -> None:
    geometry = infer_hand_geometry(Image.new("RGB", (1000, 380), (35, 75, 120)))
    assert geometry.count is None
    assert geometry.boxes == ()
    assert geometry.reason == ""


def test_hand_gray_support_does_not_change_non_hand_white_helpers() -> None:
    image, _ = _hand(1, selected=(0,))
    before_mask = _card_white_mask(image)
    before_boxes = segment_card_boxes(image)

    assert infer_hand_geometry(image).count == 1

    assert not before_mask.any()
    assert before_boxes == ()
    assert np.array_equal(_card_white_mask(image), before_mask)
    assert segment_card_boxes(image) == before_boxes


def test_clipped_slots_get_zero_confidence_and_cannot_teach_references(tmp_path: Path, monkeypatch) -> None:
    image, _ = _hand(raised=(12, 13), raised_top=-5)
    predictions = []

    def predictor(crops):
        return tuple(CardPrediction(rank="7", confidence=0.99, probabilities={"7": 0.99}) for _ in crops)

    recognizer = SceneRecognizer(LiveLayoutConfig(templates_dir=tmp_path), predictor=predictor)
    monkeypatch.setattr(recognizer, "_match_rank_reference", lambda *_args, **_kwargs: None)
    before = {rank: list(values) for rank, values in recognizer._rank_references.items()}
    fingerprint = recognizer._reference_hand_fingerprint

    cards = recognizer._observe_hand(image, expected_count=17, raw_predictions_out=predictions)

    assert len(cards) == len(predictions) == 14
    assert [i for i, card in enumerate(cards) if card.confidence == 0] == [12, 13]
    recognizer._remember_rank_references(image, cards, raw_predictions=predictions, reset=True)
    assert recognizer._rank_references == before
    assert recognizer._reference_hand_fingerprint == fingerprint


def test_provided_geometry_is_reused_without_second_inference(tmp_path: Path, monkeypatch) -> None:
    image, _ = _hand(3)
    geometry = infer_hand_geometry(image)
    recognizer = SceneRecognizer(
        LiveLayoutConfig(templates_dir=tmp_path),
        predictor=lambda crops: tuple(CardPrediction(rank="7", confidence=0.99, probabilities={}) for _ in crops),
    )
    monkeypatch.setattr(scene_module, "infer_hand_geometry", lambda *_args, **_kwargs: pytest.fail("duplicate geometry"))

    cards = recognizer._observe_hand(image, expected_count=17, geometry=geometry)

    assert len(cards) == 3
    assert tuple(card.box for card in cards) == geometry.boxes


@pytest.mark.parametrize("gray_background", [False, True])
def test_scene_distinguishes_unknown_gray_from_empty_blue_roi(tmp_path, monkeypatch, gray_background) -> None:
    color = (114, 114, 114) if gray_background else (35, 75, 120)
    image = Image.new("RGB", (1000, 600), color)
    recognizer = SceneRecognizer(
        LiveLayoutConfig(templates_dir=tmp_path), predictor=lambda _crops: (),
        remaining_reader=lambda _: {},
    )
    monkeypatch.setattr(scene_module, "_classify_round_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scene_module, "_classify_turn_controls", lambda _: (False, 0.99))

    scene = recognizer.observe(SimpleNamespace(
        image=image, frame_id=1, timestamp=1.0, pixel_box=(0, 0, 1000, 600),
    ))

    assert scene.self_hand == ()
    warnings = [warning for warning in scene.warnings if warning.startswith("self_hand_geometry_unavailable:")]
    assert bool(warnings) is gray_background
    if gray_background:
        assert scene.self_turn is None
        assert scene.self_turn_confidence == scene.confidence == 0.0
    else:
        assert scene.self_turn is False
