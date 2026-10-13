from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config.hand_geometry import ShortHandRecoveryConfig
from src.vision.clipped_hand_edges import clipped_trailing_edges_supported

CONFIG = ShortHandRecoveryConfig()


def _scene(*, count=14, raised_top=-50, selected=False, rounded=False):
    starts = tuple(74 + index * 50 for index in range(count))
    image = Image.new("RGB", (starts[-1] + 260, 380), (35, 75, 120))
    draw = ImageDraw.Draw(image)
    for index, left in enumerate(starts):
        clipped = index >= count - 2
        top = raised_top if clipped else 70
        face, rim = ((114, 114, 114), (60, 60, 60)) if selected and clipped else ((228, 228, 228), (125, 125, 125))
        bounds = (left, top, left + 202, top + 280)
        if rounded:
            draw.rounded_rectangle(bounds, radius=12, fill=face, outline=rim, width=2)
        else:
            draw.rectangle(bounds, fill=face, outline=rim, width=2)
        draw.rectangle((left + 12, top + 20, left + 24, top + 58), fill=(20, 20, 20))
    retained = tuple((left, 0, left + 112, 187) for left in starts)
    false_left = starts[-3] + 201
    candidates = (*retained, (false_left, 0, false_left + 112, 187))
    return image, candidates, retained, (count - 2, count - 1)


def _pixels(image):
    rgb = np.asarray(image)
    low, high = rgb.min(axis=2), rgb.max(axis=2)
    mask = (low >= 90) & (high - low <= 18)
    return np.asarray(image.convert("L")), mask


def _check(scene, *, callback=None, **changes):
    image, candidates, retained, clipped = scene
    gray, mask = _pixels(image)
    values = dict(candidate_boxes=candidates, retained_boxes=retained,
                  clipped_slots=clipped, config=CONFIG, corner_supported=callback)
    values.update(changes)
    return clipped_trailing_edges_supported(gray, mask, **values)


@pytest.mark.parametrize("count", [6, 14, 17])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("raised_top", [-35, -50, -80])
def test_only_exposed_right_suffix_has_independent_deletion_evidence(count, selected, raised_top):
    scene = _scene(count=count, selected=selected, raised_top=raised_top)
    assert _check(scene)
    assert len(scene[1]) == len(scene[2]) + 1


@pytest.mark.parametrize("kind", [
    "gray_real_left", "occluded_gray_real_left", "ink_bar_inside_face",
    "bottom_not_aligned", "no_occluder_bottom", "right_face_patch", "unknown_gray_background",
])
def test_missing_or_contradictory_pixel_evidence_never_deletes(kind):
    scene = _scene()
    image = scene[0]
    draw = ImageDraw.Draw(image)
    if kind in ("gray_real_left", "occluded_gray_real_left"):
        draw.rectangle((825, 231, 926, 350), fill=(114, 114, 114), outline=(60, 60, 60), width=2)
        if kind == "occluded_gray_real_left":
            draw.rectangle((820, 231, 930, 270), fill=(35, 75, 120))
    elif kind == "ink_bar_inside_face":
        draw.rectangle((827, 231, 900, 350), fill=(228, 228, 228))
    elif kind == "bottom_not_aligned":
        draw.rectangle((820, 300, 940, 370), fill=(35, 75, 120))
    elif kind == "no_occluder_bottom":
        draw.rectangle((727, 224, 751, 235), fill=(228, 228, 228))
    elif kind == "right_face_patch":
        draw.rectangle((830, 245, 853, 335), fill=(114, 114, 114))
    else:
        draw.rectangle((827, 231, 940, 370), fill=(114, 114, 114))
    assert not _check(scene)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("hidden_to", [245, 270, 300])
def test_naturally_occluded_real_left_edges_are_preserved(selected, hidden_to):
    scene = _scene()
    draw = ImageDraw.Draw(scene[0])
    face, rim = ((114, 114, 114), (60, 60, 60)) if selected else ((228, 228, 228), (125, 125, 125))
    draw.rectangle((825, 231, 926, 350), fill=face, outline=rim, width=2)
    draw.rectangle((820, 231, 930, hidden_to), fill=(35, 75, 120))
    assert not _check(scene)


def test_single_right_face_pixel_is_a_conservative_veto():
    scene = _scene()
    scene[0].putpixel((835, 270), (114, 114, 114))
    assert not _check(scene)


@pytest.mark.parametrize("kind", ["missing_bases", "missing_clipped_top", "missing_upper_crossing"])
def test_negative_tail_is_insufficient_without_both_source_bodies(kind):
    scene = _scene()
    draw = ImageDraw.Draw(scene[0])
    if kind == "missing_bases":
        draw.rectangle((0, 50, 672, 110), fill=(35, 75, 120))
    elif kind == "missing_clipped_top":
        draw.rectangle((670, 0, 760, 30), fill=(35, 75, 120))
    else:
        draw.rectangle((800, 195, 850, 227), fill=(35, 75, 120))
    assert not _check(scene)


def _rounded_callback(image):
    # Execute only the existing pure corner predicate on actual synthetic
    # pixels. No SceneRecognizer, model framework, predictions or constant mock.
    path = Path(__file__).resolve().parents[1] / "src/vision/scene_recognizer.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_short_hand_corner")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node,
    ], type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), "<pixel-corner-only>", "exec"), namespace)
    difference = np.abs(np.diff(np.asarray(image.convert("L"), dtype=np.int16), axis=1))
    calls = []

    def callback(left, top, right, bottom, require_top):
        calls.append(require_top)
        lower = namespace["_short_hand_corner"](difference, (left, top, right, bottom), left, CONFIG)
        upper = not require_top or namespace["_short_hand_corner"](
            difference[::-1], (left, image.height - bottom, right, image.height - top), left, CONFIG,
        )
        return bool(lower and upper)

    return callback, calls


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("raised_top", [-35, -50, -80])
def test_optional_rounded_callback_uses_real_closed_corner_pixels(selected, raised_top):
    scene = _scene(rounded=True, selected=selected, raised_top=raised_top)
    callback, calls = _rounded_callback(scene[0])
    assert _check(scene, callback=callback)
    assert True in calls and False in calls


def test_rounded_callback_cannot_certify_an_obscured_lower_corner():
    scene = _scene(rounded=True)
    ImageDraw.Draw(scene[0]).rectangle((722, 206, 749, 241), fill=(35, 75, 120))
    callback, _calls = _rounded_callback(scene[0])
    assert not _check(scene, callback=callback)


def test_rounded_side_without_its_flat_bottom_cannot_choose_a_callback_endpoint():
    scene = _scene(rounded=True)
    # Keep the curved left side but remove the independent horizontal end.
    ImageDraw.Draw(scene[0]).rectangle((748, 225, 834, 234), fill=(228, 228, 228))
    callback, _calls = _rounded_callback(scene[0])
    assert not _check(scene, callback=callback)


def test_rounded_tail_without_reverse_closure_is_not_a_proven_right_edge():
    scene = _scene(rounded=True)
    ImageDraw.Draw(scene[0]).rectangle((802, 333, 833, 356), fill=(35, 75, 120))
    callback, _calls = _rounded_callback(scene[0])
    assert not _check(scene, callback=callback)


@pytest.mark.parametrize("case", ["same_count", "internal_removal", "changed_prefix", "no_clipped", "all_clipped", "bad_index", "duplicate_index"])
def test_unproven_proposal_structure_cannot_authorize_deletion(case):
    scene = _scene()
    candidates, retained = scene[1:3]
    changes = {}
    if case == "same_count":
        changes["retained_boxes"] = candidates
    elif case == "internal_removal":
        changes["retained_boxes"] = candidates[:3] + candidates[4:]
    elif case == "changed_prefix":
        changes["retained_boxes"] = ((retained[0][0] + 1, *retained[0][1:]), *retained[1:])
    elif case == "no_clipped":
        changes["clipped_slots"] = ()
    elif case == "all_clipped":
        changes["clipped_slots"] = tuple(range(len(retained)))
    elif case == "bad_index":
        changes["clipped_slots"] = (99,)
    else:
        changes["clipped_slots"] = (12, 12)
    assert not _check(scene, **changes)


def test_every_proposed_tail_deletion_needs_its_own_proof():
    scene = _scene()
    assert _check(scene)  # The first trailing contour is physically proved.
    candidates = (*scene[1], (880, 0, scene[0].width, 187))
    assert all(0 <= x < right <= scene[0].width for x, _y, right, _bottom in candidates)
    assert not _check(scene, candidate_boxes=candidates)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 256])
def test_invalid_pixels_fail_closed(value):
    scene = _scene()
    gray, mask = _pixels(scene[0])
    gray = gray.astype(np.float64)
    gray[100, 100] = value
    assert not clipped_trailing_edges_supported(
        gray, mask, candidate_boxes=scene[1], retained_boxes=scene[2], clipped_slots=scene[3], config=CONFIG,
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_pixel_inputs_are_reused_without_mutation(dtype):
    scene = _scene()
    gray, mask = _pixels(scene[0])
    gray = gray.astype(dtype)
    before = gray.copy(), mask.copy()
    assert clipped_trailing_edges_supported(
        gray, mask, candidate_boxes=scene[1], retained_boxes=scene[2], clipped_slots=scene[3], config=CONFIG,
    )
    np.testing.assert_array_equal(gray, before[0])
    np.testing.assert_array_equal(mask, before[1])
