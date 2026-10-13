from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as scene
from src.config.hand_geometry import ShortHandRecoveryConfig

BACKGROUND = (40, 75, 135)


def _rounded_slot(*, offset: int = 0, ink: bool = True):
    """An existing grid slot, with an early rim and nearby ordinary ink."""
    image = Image.new("RGB", (450, 450), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left = 50 + offset
    draw.rounded_rectangle(
        (left, 25, left + 240, 415), radius=10,
        fill=(240, 240, 240), outline=(120, 120, 120), width=2,
    )
    if ink:
        draw.rectangle((left + 18, 54, left + 21, 55), fill=(10, 10, 10))
        draw.rectangle((left + 8, 48, left + 11, 50), fill=(10, 10, 10))
    return image, ((left, 27, left + 126, 236),)


def _certify(image, boxes):
    grayscale, difference = scene._hand_edge_arrays(image)
    return scene._certify_hand_slot_extents(
        scene._card_white_mask(image), boxes, difference,
        ShortHandRecoveryConfig(), grayscale=grayscale,
    )


@pytest.mark.parametrize("offset", [0, 37, 91])
def test_early_rounded_rim_requires_same_x_paired_closures(offset) -> None:
    image, boxes = _rounded_slot(offset=offset)
    grayscale, difference = scene._hand_edge_arrays(image)
    mask = scene._card_white_mask(image)
    config = ShortHandRecoveryConfig()
    left = boxes[0][0]
    strip_right, rows = scene._hand_slot_face_rows(mask, boxes, 0)
    top, bottom = rows[0][0], rows[-1][1]
    assert top == 27
    # The original density-crossing endpoint fails, without weakening it.
    assert not scene._short_hand_corner(
        difference[::-1], (left, image.height - bottom, strip_right, image.height - top),
        left, config,
    )
    extents = _certify(image, boxes)
    assert extents is not None
    extent = extents[0]
    assert extent.left == left
    assert top - max(3, round((bottom - top) * config.alignment_ratio)) <= extent.top < top
    assert scene._short_hand_corner(
        difference, (left, extent.top, strip_right, extent.bottom), left, config,
    )
    assert scene._short_hand_corner(
        difference[::-1],
        (left, image.height - extent.bottom, strip_right, image.height - extent.top),
        left, config,
    )
    geometry = scene._anchor_hand_slots(
        image, mask, boxes, len(boxes), horizontal_difference=difference, grayscale=grayscale,
    )
    assert geometry.usable
    assert geometry.count == len(boxes)
    assert geometry.boxes == boxes  # No label/state-derived count or crop rebuild.


@pytest.mark.parametrize("damage", ["top", "bottom", "different_x", "roi_top", "roi_bottom"])
def test_missing_or_misaligned_closures_are_not_certified(damage) -> None:
    image, boxes = _rounded_slot()
    draw = ImageDraw.Draw(image)
    if damage == "top":
        draw.rectangle((40, 0, 300, 65), fill=BACKGROUND)
    elif damage == "bottom":
        draw.rectangle((40, 380, 300, 449), fill=BACKGROUND)
    elif damage == "different_x":
        draw.rectangle((40, 0, 62, 65), fill=BACKGROUND)
    elif damage == "roi_top":
        image = image.crop((0, 40, 450, 450))
    else:
        image = image.crop((0, 0, 450, 395))
    assert _certify(image, boxes) is None


@pytest.mark.parametrize("decoration", ["hook", "text", "interior_frame"])
def test_ordinary_ink_and_internal_decorations_cannot_certify_grid_slots(decoration) -> None:
    image = Image.new("RGB", (450, 450), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 350, 430), fill=(240, 240, 240))
    if decoration == "hook":
        draw.line([(90, 45), (90, 355), (80, 370), (55, 370)], fill=(15, 15, 15), width=5)
    elif decoration == "text":
        draw.text((50, 40), "J K Q 10", fill=(15, 15, 15))
    else:
        draw.rounded_rectangle((50, 40, 200, 150), radius=10, outline=(15, 15, 15), width=3)
    assert _certify(image, ((50, 27, 176, 236),)) is None


def test_failed_physical_evidence_preserves_candidates_as_unknown() -> None:
    image, boxes = _rounded_slot()
    ImageDraw.Draw(image).rectangle((40, 0, 300, 65), fill=BACKGROUND)
    grayscale, difference = scene._hand_edge_arrays(image)
    geometry = scene._anchor_hand_slots(
        image, scene._card_white_mask(image), boxes, len(boxes),
        horizontal_difference=difference, grayscale=grayscale,
    )
    assert not geometry.usable
    assert geometry.count is None
    assert geometry.boxes == boxes
    assert geometry.reason == "hand_slot_extent_unverified:candidate_count_unconfirmed"


@pytest.mark.parametrize("decoration_bottom", [150, 390])
def test_full_geometry_does_not_add_slots_for_internal_decorations(decoration_bottom) -> None:
    image = Image.new("RGB", (450, 450), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 350, 430), fill=(240, 240, 240))
    baseline = scene.infer_hand_geometry(image)
    draw.rounded_rectangle(
        (50, 40, 200, decoration_bottom), radius=10, outline=(15, 15, 15), width=3,
    )
    decorated = scene.infer_hand_geometry(image)
    assert baseline.usable and decorated.usable
    assert decorated.count == baseline.count
    assert decorated.boxes == baseline.boxes


def test_clean_seed_fast_path_does_not_enter_endpoint_search(monkeypatch) -> None:
    image, boxes = _rounded_slot(ink=False)
    grayscale, difference = scene._hand_edge_arrays(image)

    def unexpected(*args, **kwargs):
        pytest.fail("A clean unchanged seed must not enter endpoint certification")

    monkeypatch.setattr(scene, "_certify_hand_slot_extents", unexpected)
    geometry = scene._anchor_hand_slots(
        image, scene._card_white_mask(image), boxes, len(boxes),
        horizontal_difference=difference, grayscale=grayscale,
    )
    assert geometry.usable
    assert geometry.boxes == boxes


def test_extra_endpoint_search_is_local_and_reuses_difference(monkeypatch) -> None:
    image, boxes = _rounded_slot()
    grayscale, difference = scene._hand_edge_arrays(image)
    original = scene._short_hand_corner
    calls = []

    def record(array, bounds, left, config):
        assert np.shares_memory(array, difference)
        calls.append(bounds)
        return original(array, bounds, left, config)

    monkeypatch.setattr(scene, "_short_hand_corner", record)
    extents = scene._certify_hand_slot_extents(
        scene._card_white_mask(image), boxes, difference,
        ShortHandRecoveryConfig(), grayscale=grayscale,
    )
    assert extents is not None
    early_tops = [bounds[1] for bounds in calls if bounds[1] < 27]
    assert early_tops
    assert min(early_tops) >= 19  # 27 - round((414 - 27) * .02)
    assert len(calls) <= 50
