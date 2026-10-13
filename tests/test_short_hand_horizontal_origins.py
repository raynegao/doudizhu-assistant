"""Physical short-hand origins and fail-closed integration; no model loading."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import src.vision.scene_recognizer as module
from src.config.hand_geometry import ShortHandRecoveryConfig
from src.config.live_layout import LiveLayoutConfig, NormalizedBox
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.scene_types import TemplateMatch


def _fan(count=2, *, letters="JOKER", gray=(), clipped=False, step=100):
    starts = tuple(50 + index * step for index in range(count))
    image = Image.new("RGB", (max(650, starts[-1] + 250), 400), (45, 75, 150))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=45)
    for index, x in enumerate(starts):
        top = -55 if clipped else 30
        color = (135, 135, 135) if index in gray else (235, 235, 235)
        draw.rounded_rectangle((x, top, x + 199, top + 279), radius=8,
                               fill=color, outline=(70, 70, 70), width=2)
        for row, letter in enumerate(letters):
            draw.text((x + 15, top + 10 + row * 43), letter, fill=(40, 5, 5), font=font)
    return image


def _physical(image):
    mask = module._card_white_mask(image)
    boxes = module._recover_short_hand_boxes(image, mask, maximum=20)
    assert len(boxes) == 2
    return module._anchor_hand_slots(image, mask, boxes, len(boxes))


def _shift(geometry, offset=20):
    first, (x, y, right, bottom) = geometry.boxes
    return replace(geometry, boxes=(first, (x + offset, y, right + offset, bottom)))


def _check(image, geometry, *, maximum=20, config=None):
    grayscale, difference = module._hand_edge_arrays(image)
    return module._verify_short_hand_origins(
        image, module._card_white_mask(image), geometry, maximum=maximum,
        config=config or ShortHandRecoveryConfig(),
        grayscale=grayscale, horizontal_difference=difference,
    )


@pytest.mark.parametrize("letters", ["JOKER", "A", "K", "Q", "J", "2345"])
def test_independent_dual_corners_relocate_only_horizontal_origins(letters):
    image = _fan(letters=letters)
    good = _physical(image)
    result = _check(image, _shift(good))
    assert asdict(result) == asdict(good)


@pytest.mark.parametrize("count", [1, 2, 3, 17, 20])
def test_supported_short_and_dense_geometry_is_unchanged(count, monkeypatch):
    image = _fan(count, letters="J", step=100 if count <= 3 else 20)
    old_count = module._infer_white_hand_count(image)
    boxes = module._infer_white_hand_boxes(image, old_count) if old_count else ()
    before = module._anchor_hand_slots(image, module._card_white_mask(image), boxes, old_count)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A supported origin must not invoke physical recovery")

    monkeypatch.setattr(module, "_recover_short_hand_boxes", forbidden)
    assert asdict(_check(image, before)) == asdict(before)


@pytest.mark.parametrize("both_ends", [False, True])
def test_curved_artwork_cannot_move_supported_origins(both_ends):
    image = _fan()
    before = _physical(image)
    points = [(100, 30), (100, 299), (101, 302), (103, 305), (106, 307), (109, 309)]
    if both_ends:
        points[:1] = [(109, 30), (106, 32), (103, 34), (101, 37), (100, 40)]
    ImageDraw.Draw(image).line(points, fill="black", width=2)
    assert _check(image, before) == before


def test_persistent_decoration_at_wrong_origin_remains_explicit_limitation():
    image = _fan()
    wrong = _shift(_physical(image))
    x = wrong.boxes[1][0]
    ImageDraw.Draw(image).line((x, 30, x, 309), fill="black", width=2)
    assert _check(image, wrong) == wrong


@pytest.mark.parametrize("gray", [(0,), (0, 1)])
def test_gray_branch_never_enters_new_white_contract(gray, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Gray geometry must keep its existing independent branch")

    monkeypatch.setattr(module, "_verify_short_hand_origins", forbidden)
    result = module.infer_hand_geometry(_fan(gray=gray))
    assert result.boxes


def test_missing_visible_top_cannot_authorize_horizontal_repair():
    image = _fan(clipped=True)
    mask = module._card_white_mask(image)
    boxes = module._recover_short_hand_boxes(image, mask, maximum=20)
    geometry = module._anchor_hand_slots(image, mask, boxes, len(boxes))
    assert not geometry.usable
    assert _check(image, geometry) == geometry
    incorrectly_usable = replace(geometry, usable=True, clipped_slots=(), reason="")
    assert _check(image, incorrectly_usable) == incorrectly_usable


@pytest.mark.parametrize("letters", ["A", "K", "Q", "J", "2345"])
def test_unresolved_visible_origin_preserves_candidates_but_not_trusted_count(letters, monkeypatch):
    image = _fan(letters=letters)
    wrong = _shift(_physical(image))
    monkeypatch.setattr(module, "_recover_short_hand_boxes", lambda *_args, **_kwargs: ())
    result = _check(image, wrong)
    assert result.boxes == wrong.boxes
    assert result.count is None and not result.usable and result.clipped_slots == ()
    assert result.reason == "horizontal_origin_unverified:candidate_count_unconfirmed"


@pytest.mark.parametrize("failure", ["cardinality", "corner", "supported_conflict", "half_width", "anchor"])
def test_incomplete_or_nonlocal_recovery_cannot_restore_old_usable(failure, monkeypatch):
    image = _fan()
    good = _physical(image)
    wrong = _shift(good)
    recovered = good.boxes
    if failure == "cardinality":
        recovered = recovered[:1]
    elif failure == "corner":
        monkeypatch.setattr(module, "_short_hand_corner", lambda *_args, **_kwargs: False)
    elif failure == "supported_conflict":
        first, second = recovered
        recovered = ((first[0] + 20, *first[1:]), second)
    elif failure == "half_width":
        width = wrong.boxes[1][2] - wrong.boxes[1][0]
        wrong = _shift(good, width // 2 + 1)
    else:
        monkeypatch.setattr(module, "_anchor_hand_slots", lambda *_args, **_kwargs:
                            replace(good, boxes=(good.boxes[0],)))
    monkeypatch.setattr(module, "_recover_short_hand_boxes", lambda *_args, **_kwargs: recovered)
    result = _check(image, wrong)
    assert result.count is None and not result.usable and result.boxes == wrong.boxes


def test_actual_configuration_and_maximum_are_forwarded_after_independent_recovery(monkeypatch):
    image = _fan()
    wrong = _shift(_physical(image))
    config = ShortHandRecoveryConfig(minimum_edge_support=0.72)
    calls = []
    recover = module._recover_short_hand_boxes

    def spy(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["config"] is config
        assert kwargs["maximum"] == 1
        assert "count" not in kwargs
        return recover(*args, **kwargs)

    monkeypatch.setattr(module, "_recover_short_hand_boxes", spy)
    result = _check(image, wrong, maximum=1, config=config)
    assert len(calls) == 1 and result.count is None and not result.usable


def test_custom_configuration_changes_evidence_gate_not_candidate_grid():
    image = _fan()
    wrong = _shift(_physical(image))
    result = _check(image, wrong, config=ShortHandRecoveryConfig(minimum_difference=255))
    assert result.count is None and not result.usable


def test_infer_reuses_anchor_arrays_for_horizontal_check_and_recovery(monkeypatch):
    image = _fan()
    good = _physical(image)
    wrong = _shift(good)
    monkeypatch.setattr(module, "_infer_white_hand_count", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(module, "_infer_white_hand_boxes", lambda *_args, **_kwargs: wrong.boxes)
    diff, convert, recover = np.diff, Image.Image.convert, module._recover_short_hand_boxes
    differences, conversions, recovery_arrays = [], [], []
    config = ShortHandRecoveryConfig(minimum_edge_support=0.72)

    def spy_diff(array, *args, **kwargs):
        if array.ndim == 2:
            differences.append(array.shape)
        return diff(array, *args, **kwargs)

    def spy_convert(image, mode=None, *args, **kwargs):
        if mode == "L":
            conversions.append(image.size)
        return convert(image, mode, *args, **kwargs)

    def spy_recover(*args, **kwargs):
        assert kwargs["maximum"] == 3 and kwargs["config"] is config
        recovery_arrays.append((kwargs["grayscale"], kwargs["horizontal_difference"]))
        return recover(*args, **kwargs)

    monkeypatch.setattr(np, "diff", spy_diff)
    monkeypatch.setattr(Image.Image, "convert", spy_convert)
    monkeypatch.setattr(module, "_recover_short_hand_boxes", spy_recover)
    result = module.infer_hand_geometry(image, maximum=3, recovery_config=config)
    assert result == good
    assert differences == [(image.height, image.width)]
    assert conversions == [image.size]
    assert len(recovery_arrays) == 1


@pytest.mark.parametrize("rank", ["A", "K", "Q", "J", "10", "3"])
def test_all_ordinary_rank_candidates_are_zero_when_origin_unverified(rank):
    image = _fan(letters=rank)
    wrong = _shift(_physical(image))
    geometry = replace(wrong, count=None, usable=False,
                       reason="horizontal_origin_unverified:candidate_count_unconfirmed")
    instance = object.__new__(module.SceneRecognizer)
    instance.config = LiveLayoutConfig()
    instance._rank_references = {}
    instance.predictor = lambda crops: tuple(CardPrediction(rank, 0.999, {rank: 0.999}) for _ in crops)
    cards = instance._observe_hand(image, expected_count=17, geometry=geometry)
    assert len(cards) == 2 and tuple(card.box for card in cards) == geometry.boxes
    assert all(card.confidence == 0.0 for card in cards)
    assert all("geometry_unusable:horizontal_origin_unverified" in card.evidence.fusion_reason for card in cards)


@pytest.mark.parametrize("ocr_count", [None, 7])
def test_real_observe_does_not_backfill_unconfirmed_count_from_candidate_slots(ocr_count, monkeypatch):
    image = _fan(letters="A")
    geometry = replace(_physical(image), count=None, usable=False,
                       reason="horizontal_origin_unverified:candidate_count_unconfirmed")
    rois = dict(LiveLayoutConfig().rois)
    rois["self_hand"] = NormalizedBox(0, 0, 1, 1)
    templates = SimpleNamespace(classify=lambda *_: TemplateMatch(None, 0.0),
                                template_images=lambda *_: ())
    instance = module.SceneRecognizer(
        LiveLayoutConfig(rois=rois),
        predictor=lambda crops: tuple(CardPrediction("A", 0.999, {"A": 0.999}) for _ in crops),
        template_matcher=templates,
        remaining_reader=lambda _: {} if ocr_count is None else {
            PlayerSeat.SELF: module.RemainingTextMatch(ocr_count, 0.995)},
    )
    monkeypatch.setattr(module, "infer_hand_geometry", lambda *_args, **_kwargs: geometry)
    monkeypatch.setattr(module, "_classify_round_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "segment_card_regions", lambda *_args, **_kwargs: ())
    scene = instance.observe(SimpleNamespace(image=image, frame_id=1, timestamp=1.0,
                                            pixel_box=(0, 0, image.width, image.height)))
    assert len(scene.self_hand) == 2 and all(card.confidence == 0.0 for card in scene.self_hand)
    own = scene.seat(PlayerSeat.SELF)
    assert own.remaining_count == ocr_count
    assert own.remaining_verified is (ocr_count is not None)
    assert any(warning.startswith("self_hand_geometry_unavailable: horizontal_origin_unverified")
               for warning in scene.warnings)
