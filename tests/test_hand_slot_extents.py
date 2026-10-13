"""Physical hand-slot extents and fail-closed observation, using synthetic RGB.

No recordings, work artifacts, archived source, model weights or real inference
are used. The predictor in the observation integration test is an explicit stub.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import src.vision.scene_recognizer as module
from src.config.hand_geometry import ShortHandRecoveryConfig
from src.config.live_layout import LiveLayoutConfig
from src.vision.card_classifier import CardPrediction


def _fan(
    count=6, *, letters="A", selected=(), raised=(), clipped=False, banner=None, rounded=True,
):
    body_height, step, top = 300, 110, 60
    starts = tuple(40 + step * index for index in range(count))
    image = Image.new("RGB", (starts[-1] + 260, 400), (40, 75, 135))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for index, left in enumerate(starts):
        y = (-20 if clipped else 15) if index in raised else top
        color = (135, 135, 135) if index in selected else (235, 235, 235)
        body = (left, y, left + 215, y + body_height)
        if rounded:
            draw.rounded_rectangle(body, radius=10, fill=color, outline=(70, 70, 70), width=2)
        else:
            draw.rectangle(body, fill=color, outline=(70, 70, 70), width=2)
        draw.text((left + 15, y + 20), letters, fill=(25, 8, 8), font=font)
    if banner:
        upper, lower, color = banner
        draw.rectangle((starts[1] - 10, upper, starts[-1] + 212, lower), fill=color)
    # The fixture supplies its drawn slots; no game-state count is available.
    first_top = -20 if clipped and raised else (15 if raised else top)
    crop_height = round((top + body_height - first_top) * .535)
    crop_width = round(crop_height * .60)
    boxes = tuple(
        (left, max(0, first_top), left + crop_width, max(0, first_top) + crop_height)
        for left in starts
    )
    return image, boxes


def _face_mask(image):
    pixels = np.asarray(image)
    high, low = pixels.max(axis=2), pixels.min(axis=2)
    gray = (low >= 90) & (high < 175) & (high - low <= 18)
    return module._card_white_mask(image) | gray


def _anchor(image, boxes, **kwargs):
    return module._anchor_hand_slots(image, _face_mask(image), boxes, len(boxes), **kwargs)


def _unverified_fixture():
    image, boxes = _fan(banner=(140, 210, (130, 130, 130)))
    # Hide one upper corner while leaving its closed bottom visible.
    ImageDraw.Draw(image).rectangle(
        (boxes[2][0] - 5, 45, boxes[2][0] + 36, 105), fill=(40, 75, 135),
    )
    return image, boxes


@pytest.mark.parametrize("count", [4, 6, 11, 15, 20])
@pytest.mark.parametrize("banner", [(140, 210, (130, 130, 130)), (120, 210, (45, 75, 135))])
def test_middle_occlusion_keeps_independently_closed_slots(count, banner):
    image, boxes = _fan(count, banner=banner)
    result = _anchor(image, boxes)
    assert result.count == count and result.usable
    assert result.boxes == boxes


@pytest.mark.parametrize("letters", ["A", "K", "Q", "J", "10", "8", "JOKER"])
def test_occlusion_repair_does_not_depend_on_rank(letters):
    image, boxes = _fan(letters=letters, banner=(140, 210, (130, 130, 130)))
    result = _anchor(image, boxes)
    assert result.count == len(boxes) and result.usable
    assert result.boxes == boxes


@pytest.mark.parametrize("rounded", [True, False], ids=["rounded", "square"])
@pytest.mark.parametrize("raised", [(4, 5), (0, 1), (2, 3)])
def test_real_raised_slots_use_their_own_supported_top(rounded, raised):
    image, boxes = _fan(raised=raised, selected=raised, rounded=rounded)
    result = _anchor(image, boxes)
    assert result.count == len(boxes) and result.usable
    for index, box in enumerate(result.boxes):
        expected_top = 15 if index in raised else 60
        assert abs(box[1] - expected_top) <= 3
        assert 156 <= box[3] - box[1] <= 164


def test_unknown_retains_all_candidates_instead_of_a_filtered_count():
    image, boxes = _unverified_fixture()
    result = _anchor(image, boxes)
    assert result.count is None and not result.usable
    assert result.boxes == boxes
    assert "candidate_count_unconfirmed" in result.reason


def test_internal_straight_bar_cannot_certify_a_slot():
    image, boxes = _fan(2)
    left = boxes[0][0] + 55
    ImageDraw.Draw(image).line((left, 60, left, 360), fill=(5, 5, 5), width=2)
    forged = ((left, 60, left + 96, 220),)
    difference = module._hand_edge_arrays(image)[1]
    assert module._certify_hand_slot_extents(
        _face_mask(image), forged, difference, ShortHandRecoveryConfig(),
    ) is None


@pytest.mark.parametrize("both_ends", [False, True])
def test_curved_artwork_cannot_increase_the_confirmed_count(both_ends):
    image, boxes = _fan(banner=(140, 210, (130, 130, 130)))
    x = boxes[2][0] + 50
    points = [(x, 60), (x, 350), (x + 1, 353), (x + 3, 356), (x + 6, 358), (x + 9, 360)]
    if both_ends:
        points[:1] = [(x + 9, 60), (x + 6, 62), (x + 3, 64), (x + 1, 67), (x, 70)]
    ImageDraw.Draw(image).line(points, fill="black", width=2)
    candidates = boxes[:3] + ((x, 60, x + 96, 220),) + boxes[3:]
    result = _anchor(image, candidates)
    assert result.count is None and not result.usable
    assert result.boxes == candidates


def test_hidden_bottom_is_unknown_even_without_a_legacy_rewrite():
    image, boxes = _fan(banner=(140, 210, (130, 130, 130)))
    ImageDraw.Draw(image).rectangle((0, 290, image.width, 390), fill=(40, 75, 135))
    result = _anchor(image, boxes)
    assert result.count is None and not result.usable
    assert result.boxes == boxes


def test_clipped_true_selection_does_not_invent_visible_top_corners():
    image, boxes = _fan(raised=(4, 5), selected=(4, 5), clipped=True)
    result = _anchor(image, boxes)
    assert result.count == len(boxes) and not result.usable
    assert result.clipped_slots == (4, 5)


@pytest.mark.parametrize("remove", [0, 3, 5])
@pytest.mark.parametrize("selected", [(), (4, 5)])
def test_clipped_proposal_cannot_delete_a_real_candidate(remove, selected, monkeypatch):
    image, boxes = _fan(raised=(4, 5), selected=selected, clipped=True)
    retained = tuple(box for index, box in enumerate(boxes) if index != remove)
    clipped = tuple(index for index, box in enumerate(retained) if box[0] in {boxes[4][0], boxes[5][0]})
    proposal = module.HandGeometry(len(retained), retained, False, clipped, "rank_top_outside_input_roi")
    monkeypatch.setattr(module, "_propose_hand_slot_anchors", lambda *_args, **_kwargs: proposal)
    result = _anchor(image, boxes)
    assert result.count is None and not result.usable
    assert result.boxes == boxes and "candidate_count_unconfirmed" in result.reason


@pytest.mark.parametrize("selected", [(2,), tuple(range(6))])
def test_level_gray_hand_keeps_supported_geometry(selected):
    image, boxes = _fan(selected=selected)
    result = _anchor(image, boxes)
    assert result.count == len(boxes) and result.boxes == boxes and result.usable


def test_full_inference_does_not_count_curved_internal_decoration():
    image, _ = _fan()
    plain = module.infer_hand_geometry(image)
    assert plain.count == 6
    x = 310
    ImageDraw.Draw(image).line(
        [(x + 9, 60), (x + 6, 62), (x + 3, 64), (x + 1, 67), (x, 70),
         (x, 350), (x + 1, 353), (x + 3, 356), (x + 6, 358), (x + 9, 360)],
        fill="black", width=2,
    )
    decorated = module.infer_hand_geometry(image)
    assert decorated.count in (None, 6)


def test_only_a_lower_edge_cannot_certify_a_slot():
    image = Image.new("RGB", (300, 400), (235, 235, 235))
    ImageDraw.Draw(image).line((90, 220, 90, 370), fill=(5, 5, 5), width=2)
    boxes = ((90, 60, 186, 220),)
    difference = module._hand_edge_arrays(image)[1]
    assert module._certify_hand_slot_extents(
        _face_mask(image), boxes, difference, ShortHandRecoveryConfig(),
    ) is None


def test_gray_dialog_and_empty_blue_never_gain_candidates():
    gray = module.infer_hand_geometry(Image.new("RGB", (650, 400), (135, 135, 135)))
    blue = module.infer_hand_geometry(Image.new("RGB", (650, 400), (40, 75, 135)))
    assert gray.count is None and not gray.usable and gray.boxes == ()
    assert blue.count is None and not blue.usable and blue.boxes == ()


def test_anchor_reuses_existing_difference_and_explicit_config(monkeypatch):
    config = replace(ShortHandRecoveryConfig(), minimum_corner_support=.95)
    seen = []
    original_certify = module._certify_hand_slot_extents

    def certify(mask, boxes, actual_difference, actual_config, **kwargs):
        seen.append((actual_difference, actual_config))
        return original_certify(mask, boxes, actual_difference, actual_config, **kwargs)

    monkeypatch.setattr(module, "_certify_hand_slot_extents", certify)
    image, boxes = _fan(banner=(140, 210, (130, 130, 130)))
    difference = module._hand_edge_arrays(image)[1]
    original_diff = np.diff

    def no_full_difference(array, *args, **kwargs):
        assert array.ndim != 2, "The supplied full-ROI difference must be reused"
        return original_diff(array, *args, **kwargs)

    monkeypatch.setattr(np, "diff", no_full_difference)
    _anchor(image, boxes, horizontal_difference=difference, config=config)
    assert seen
    assert all(actual_diff is difference and actual_config is config for actual_diff, actual_config in seen)
    assert config.maximum_slots == 3


def test_infer_geometry_forwards_its_actual_config_to_anchor(monkeypatch):
    config = replace(ShortHandRecoveryConfig(), minimum_corner_support=.95)
    seen = []
    original_anchor = module._anchor_hand_slots

    def anchor(*args, **kwargs):
        seen.append(kwargs.get("config"))
        return original_anchor(*args, **kwargs)

    monkeypatch.setattr(module, "_anchor_hand_slots", anchor)
    image, _ = _fan(banner=(140, 210, (130, 130, 130)))
    module.infer_hand_geometry(image, recovery_config=config)
    assert seen and all(value is config for value in seen)


def test_common_seed_offset_is_corrected_using_certified_real_tops():
    image, boxes = _fan()
    shifted = tuple((left, top + 70, right, bottom + 70) for left, top, right, bottom in boxes)
    difference = module._hand_edge_arrays(image)[1]
    extents = module._certify_hand_slot_extents(
        _face_mask(image), shifted, difference, ShortHandRecoveryConfig(),
    )
    assert extents is not None and len(extents) == len(boxes)
    assert all(abs(extent.top - 60) <= 3 for extent in extents)
    result = _anchor(image, shifted, horizontal_difference=difference)
    assert result.usable and result.count == len(boxes)
    assert result.clipped_slots == ()
    assert tuple(box[0] for box in result.boxes) == tuple(box[0] for box in boxes)
    assert all(abs(box[1] - 60) <= 3 for box in result.boxes)
    assert all(156 <= box[3] - box[1] <= 164 for box in result.boxes)
    assert result.boxes != shifted


def test_unknown_geometry_zeroes_real_observer_output_and_cannot_teach_references(
    tmp_path, monkeypatch,
):
    image, boxes = _unverified_fixture()
    geometry = _anchor(image, boxes)
    assert geometry.count is None and not geometry.usable and geometry.boxes == boxes
    ranks = ("3", "4", "5", "6", "7", "8")

    def predictor(crops):
        assert len(crops) == len(ranks)
        return tuple(CardPrediction(rank=rank, confidence=.99, probabilities={rank: .99}) for rank in ranks)

    recognizer = module.SceneRecognizer(LiveLayoutConfig(templates_dir=tmp_path), predictor=predictor)
    monkeypatch.setattr(recognizer, "_match_rank_reference", lambda *_args, **_kwargs: None)
    recognizer._rank_references.setdefault("3", []).append(frozenset({99991}))
    recognizer._reference_hand_fingerprint = ("9",)
    references_before = deepcopy(recognizer._rank_references)
    fingerprint_before = recognizer._reference_hand_fingerprint
    raw = []
    cards = recognizer._observe_hand(
        image, expected_count=19, geometry=geometry, raw_predictions_out=raw,
    )
    assert len(cards) == len(raw) == len(boxes)
    assert tuple(card.box for card in cards) == boxes
    assert tuple(card.rank for card in cards) == ranks
    assert all(prediction.confidence == .99 for prediction in raw)
    assert all(card.confidence == 0 for card in cards)
    assert all(
        card.evidence is not None and "geometry_unusable:" in card.evidence.fusion_reason
        for card in cards
    )

    def forbidden_learning(*_args, **_kwargs):
        raise AssertionError("Unknown geometry must stop before reference learning/reset")

    monkeypatch.setattr(recognizer, "_reset_rank_references", forbidden_learning)
    monkeypatch.setattr(module, "_rank_glyph_signature", forbidden_learning)
    recognizer._remember_rank_references(image, cards, raw_predictions=raw, reset=True)
    assert recognizer._rank_references == references_before
    assert recognizer._reference_hand_fingerprint == fingerprint_before
