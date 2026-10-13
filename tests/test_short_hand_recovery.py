"""Pure, synthetic recovery/configuration tests; never run a classifier."""

from __future__ import annotations

import json
import warnings
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import src.vision.scene_recognizer as scene_module
from src.config.hand_geometry import ShortHandRecoveryConfig


def _existing_normal_geometry(image):
    mask = face_mask(image, vars(scene_module))
    pixels = np.asarray(image)
    high, low = pixels.max(axis=2), pixels.min(axis=2)
    gray = (low >= 90) & (high < 175) & (high - low <= 18)
    gray_bands = [
        run
        for run in scene_module._active_x_runs_from_mask(gray, min_column_ratio=0.45, max_gap=3)
        if run[1] - run[0] >= max(12, round(image.height * 0.04))
    ]
    if gray_bands:
        boxes = scene_module._gray_hand_boxes(image, mask, maximum=20)
        count = len(boxes) if boxes else None
    else:
        count = scene_module._infer_white_hand_count(image)
        boxes = scene_module._infer_white_hand_boxes(image, count) if count else ()
    if not boxes:
        return scene_module.HandGeometry(count, (), False)
    return scene_module._anchor_hand_slots(image, mask, boxes, count)


@pytest.fixture(scope="module")
def baseline():
    return {"infer_hand_geometry": _existing_normal_geometry}


@pytest.fixture(scope="module")
def integrated():
    return vars(scene_module)


def fan(
    count=2, *, gray=(), clipped=False, letters="JOKER", stagger=False, with_ink=True, step=100
):
    starts = tuple(50 + index * step for index in range(count))
    image = Image.new("RGB", (max(650, starts[-1] + 250), 400), (45, 75, 150))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=45)
    for index, x in enumerate(starts):
        top = -55 if clipped else 30
        if stagger and index == 0:
            top += 35
        fill = (135, 135, 135) if index in gray else (235, 235, 235)
        draw.rounded_rectangle(
            (x, top, x + 199, top + 279), radius=8, fill=fill, outline=(70, 70, 70), width=2
        )
        if with_ink:
            for row, letter in enumerate(letters):
                draw.text((x + 15, top + 10 + row * 43), letter, fill=(40, 5, 5), font=font)
    return image


def face_mask(image, namespace):
    pixels = np.asarray(image)
    high, low = pixels.max(axis=2), pixels.min(axis=2)
    return namespace["_card_white_mask"](image) | ((low >= 90) & (high < 175) & (high - low <= 18))


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("gray", [False, True])
def test_normal_short_old_geometry_and_coordinates_unchanged(baseline, integrated, count, gray):
    image = fan(count, gray=tuple(range(count)) if gray else (), letters="J")
    before = baseline["infer_hand_geometry"](image)
    after = integrated["infer_hand_geometry"](image)
    assert before.count == count
    assert asdict(before) == asdict(after)


@pytest.mark.parametrize("count", [17, 20])
def test_healthy_dense_hand_unchanged(baseline, integrated, count):
    image = fan(count, with_ink=False, step=50)
    draw = ImageDraw.Draw(image)
    for index in range(count):
        left = 50 + index * 50
        draw.rectangle((left + 12, 50, left + 24, 88), fill=(20, 20, 20))
    before = baseline["infer_hand_geometry"](image)
    after = integrated["infer_hand_geometry"](image)
    assert before.count == count and before.usable
    assert asdict(before) == asdict(after)


@pytest.mark.parametrize("gray", [(0,), (0, 1)])
def test_failed_gray_primary_recovers_pixels_without_expected_count(gray, monkeypatch):
    namespace = vars(scene_module)
    monkeypatch.setattr(scene_module, "_gray_hand_boxes", lambda *args, **kwargs: ())
    image = fan(gray=gray)
    result = namespace["infer_hand_geometry"](image)
    assert result.count == 2 and result.usable
    assert len(result.boxes) == 2


@pytest.mark.parametrize("clipped", [False, True])
def test_dual_cue_recovery_of_two_joker_bodies(integrated, clipped):
    image = fan(gray=(0,), clipped=clipped)
    mask = face_mask(image, integrated)
    boxes = integrated["_recover_short_hand_boxes"](image, mask, maximum=20)
    assert len(boxes) == 2
    result = integrated["_anchor_hand_slots"](image, mask, boxes, len(boxes))
    assert result.count == 2
    assert result.usable is not clipped
    assert result.clipped_slots == ((0, 1) if clipped else ())


@pytest.mark.parametrize("curved_both_ends", [False, True])
@pytest.mark.parametrize("clipped", [False, True])
def test_long_curved_decoration_cannot_create_trusted_third_slot(
    integrated, curved_both_ends, clipped
):
    image = fan(gray=(0,), clipped=clipped)
    clean = integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=20)
    assert len(clean) == 2
    draw = ImageDraw.Draw(image)
    top, bottom = (-55, 224) if clipped else (30, 309)
    points = [
        (100, top),
        (100, bottom - 10),
        (101, bottom - 7),
        (103, bottom - 4),
        (106, bottom - 2),
        (109, bottom),
    ]
    if curved_both_ends:
        points[:1] = [(109, top), (106, top + 2), (103, top + 4), (101, top + 7), (100, top + 10)]
    draw.line(points, fill="black", width=2)
    draw.rectangle((116, max(0, top) + 60, 124, max(0, top) + 80), fill="black")
    boxes = integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=20)
    # Two real slots or conservative refusal, never a third slot from artwork.
    assert boxes in ((), clean)


@pytest.mark.parametrize("clipped", [False, True])
def test_staggered_bodies_do_not_borrow_one_global_bottom(integrated, clipped):
    image = fan(gray=(0,), stagger=True, clipped=clipped)
    boxes = integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=20)
    assert boxes == ()


@pytest.mark.parametrize("gray", [(), (0,), (0, 1)])
def test_bottom_occlusion_is_not_fabricated_as_closed_body(integrated, gray):
    image = fan(gray=gray)
    ImageDraw.Draw(image).rectangle((0, 270, image.width, 340), fill=(45, 75, 150))
    assert (
        integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=20)
        == ()
    )


def test_full_roi_gray_panel_preserves_primary_background_rejection(monkeypatch):
    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("background must be rejected before recovery")

    monkeypatch.setattr(scene_module, "_recover_short_hand_boxes", unexpected_recovery)
    image = Image.new("RGB", (650, 400), (135, 135, 135))
    ImageDraw.Draw(image).text(
        (100, 80), "JOKER", fill="black", font=ImageFont.load_default(size=45)
    )
    result = scene_module.infer_hand_geometry(image)
    assert result.count is None and result.boxes == () and not result.usable
    assert result.reason == "ambiguous_gray_background"


def test_curved_gray_panels_without_ink_are_unknown(integrated):
    image = fan(gray=(0, 1), with_ink=False)
    assert (
        integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=20)
        == ()
    )


def test_explicit_wrong_expected_count_cannot_decide_gray_recovery(monkeypatch):
    namespace = vars(scene_module)
    monkeypatch.setattr(scene_module, "_gray_hand_boxes", lambda *args, **kwargs: ())
    image = fan(gray=(0,))
    assert namespace["infer_hand_geometry"](image, count=19).count == 2


def test_maximum_is_a_rejection_cap_not_a_requested_grid(integrated):
    image = fan(gray=(0,))
    assert (
        integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=1)
        == ()
    )


def test_recovery_parameters_are_immutable_and_do_not_hold_expected_count(integrated):
    config = ShortHandRecoveryConfig()
    assert "expected_count" not in asdict(config)
    with pytest.raises(FrozenInstanceError):
        config.maximum_slots = 2


def test_each_fallback_computes_full_roi_horizontal_difference_only_once(integrated, monkeypatch):
    original = np.diff
    full_image_calls = []

    def spy(array, *args, **kwargs):
        if array.ndim == 2:
            full_image_calls.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(np, "diff", spy)
    image = fan(gray=(0,))
    assert (
        len(
            integrated["_recover_short_hand_boxes"](image, face_mask(image, integrated), maximum=20)
        )
        == 2
    )
    assert full_image_calls == [(image.height, image.width)]


def test_clipped_top_is_never_claimed_as_a_visible_closed_corner(monkeypatch):
    namespace = vars(scene_module)
    original = namespace["_short_hand_corner"]
    inspected = []

    def spy(difference, bounds, x, config):
        inspected.append(bounds)
        return original(difference, bounds, x, config)

    monkeypatch.setattr(scene_module, "_short_hand_corner", spy)
    image = fan(gray=(0,), clipped=True)
    mask = face_mask(image, namespace)
    boxes = namespace["_recover_short_hand_boxes"](image, mask, maximum=20)
    assert len(boxes) == 2 and inspected
    assert all(bounds[1] == 0 for bounds in inspected)
    result = namespace["_anchor_hand_slots"](image, mask, boxes, 2)
    assert not result.usable and result.clipped_slots == (0, 1)


@pytest.mark.parametrize(
    "name",
    [
        "minimum_difference",
        "minimum_edge_support",
        "corner_depth_ratio",
        "minimum_corner_support",
        "corner_bend_ratio",
        "alignment_ratio",
    ],
)
@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -1.0, 0.0, "0.7", None])
def test_recovery_config_rejects_invalid_numbers(name, bad):
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig.from_dict({name: bad})


@pytest.mark.parametrize("bad", [True, False, 0, 4, 1.5, "2", None])
def test_recovery_config_rejects_invalid_slot_cap(bad):
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig.from_dict({"maximum_slots": bad})


@pytest.mark.parametrize(
    "name,bad",
    [
        ("fan_aspect", (1, 0.4)),
        ("fan_aspect", (0.4, 4)),
        ("final_aspect", (0.4, 1.1)),
        ("final_aspect", (0.4, 0.4)),
        ("rank_crop_ratios", (0, 0.6)),
        ("rank_crop_ratios", (0.5, 1.1)),
        ("lower_band", (0.96, 0.86)),
        ("lower_band", (0.86, 1.01)),
        ("lower_band", (float("nan"), 0.96)),
        ("lower_band", (True, 0.96)),
        ("lower_band", (0.86, float("inf"))),
        ("lower_band", "0.86,0.96"),
        ("lower_band", (0.86,)),
        ("lower_band", (0.86, 0.9, 0.96)),
    ],
)
def test_recovery_config_rejects_bad_intervals(name, bad):
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig.from_dict({name: bad})


@pytest.mark.parametrize("payload", [None, [], False, {"unexpected": 1}, {"expected_count": 2}])
def test_recovery_config_rejects_non_objects_and_unknown_keys(payload):
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig.from_dict(payload)


def test_recovery_config_round_trip_normalizes_json_lists_without_aliasing():
    original = ShortHandRecoveryConfig()
    payload = json.loads(json.dumps(original.to_payload()))
    restored = ShortHandRecoveryConfig.from_dict(payload)
    assert restored == original
    assert isinstance(restored.lower_band, tuple)
    payload["lower_band"][0] = 0.1
    assert restored.lower_band == (0.86, 0.96)
    assert ShortHandRecoveryConfig.from_dict({}) == original


def test_recovery_config_rejects_impossible_corner_and_oversized_difference():
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig(corner_bend_ratio=0.08)
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig(minimum_difference=256)
    with pytest.raises(ValueError):
        ShortHandRecoveryConfig(minimum_difference=10**400)


@pytest.mark.parametrize("bad", [False, {}, 2, "default"])
def test_geometry_rejects_wrong_recovery_config_type_even_on_normal_frame(bad):
    with pytest.raises(ValueError, match="recovery_config"):
        scene_module.infer_hand_geometry(fan(letters="J"), recovery_config=bad)


def test_explicit_default_config_preserves_geometry():
    image = fan(gray=(0,))
    assert scene_module.infer_hand_geometry(image) == scene_module.infer_hand_geometry(
        image, recovery_config=ShortHandRecoveryConfig()
    )


def test_valid_but_rounded_empty_band_is_rejected_without_runtime_warning():
    image = fan(gray=(0,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        boxes = scene_module._recover_short_hand_boxes(
            image,
            face_mask(image, vars(scene_module)),
            maximum=20,
            config=ShortHandRecoveryConfig(lower_band=(0.86, 0.860000001)),
        )
    assert boxes == ()


@pytest.mark.parametrize("size", [(0, 0), (0, 20), (20, 0), (1, 20), (20, 1), (7, 32)])
def test_new_recovery_helper_rejects_empty_or_narrow_images_without_warning(size):
    image = Image.new("RGB", size, "white")
    mask = np.ones((size[1], size[0]), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scene_module._recover_short_hand_boxes(image, mask, maximum=20) == ()


@pytest.mark.parametrize(
    "shape,bounds,x",
    [
        ((0, 0), (0, 0, 0, 0), 0),
        ((20, 0), (0, 0, 1, 18), 0),
        ((0, 20), (0, 0, 20, 0), 5),
        ((20, 7), (0, 0, 7, 18), 4),
    ],
)
def test_new_corner_helper_does_not_argmax_empty_axes(shape, bounds, x):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not scene_module._short_hand_corner(
            np.zeros(shape), bounds, x, ShortHandRecoveryConfig()
        )
