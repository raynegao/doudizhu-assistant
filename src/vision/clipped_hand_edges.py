from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from src.config.hand_geometry import ShortHandRecoveryConfig
from src.vision.rectangular_card_edges import rectangular_slot_supported

Box = tuple[int, int, int, int]
CornerSupport = Callable[[int, int, int, int, bool], bool]


def clipped_trailing_edges_supported(
    grayscale: np.ndarray,
    mask: np.ndarray,
    *,
    candidate_boxes: Sequence[Box],
    retained_boxes: Sequence[Box],
    clipped_slots: Sequence[int],
    config: ShortHandRecoveryConfig,
    corner_supported: CornerSupport | None = None,
) -> bool:
    """Prove only a proposed trailing deletion, never infer slots or card ranks.

    Retained boxes must be the exact primary-grid prefix. Every removed origin
    must instead be a right-facing contour exposed below a positively observed
    clipped card. A short/negative edge or missing face alone proves nothing.
    Unknown or occluded evidence returns False; callers must retain candidates.

    ``mask`` is the hand-only white/gray face mask. The optional pure pixel
    callback certifies existing rounded corners as
    (left, top, right, bottom, require_top); it must not consult state or models.
    """
    if not isinstance(config, ShortHandRecoveryConfig):
        raise ValueError("clipped edge evidence requires ShortHandRecoveryConfig")
    if corner_supported is not None and not callable(corner_supported):
        raise ValueError("corner_supported must be callable")
    if (
        not isinstance(grayscale, np.ndarray) or grayscale.ndim != 2
        or grayscale.dtype.kind not in "uif"
        or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
        or mask.shape != grayscale.shape or not grayscale.size
        or not np.isfinite(grayscale).all()
        or grayscale.min() < 0 or grayscale.max() > 255
    ):
        return False
    height, width = mask.shape
    candidates, retained = tuple(candidate_boxes), tuple(retained_boxes)
    if (
        not 1 < len(retained) < len(candidates)
        or candidates[:len(retained)] != retained
        or not _valid_boxes(candidates, width, height)
        or not clipped_slots
        or any(isinstance(index, bool) or not isinstance(index, int)
               or not 0 <= index < len(retained) for index in clipped_slots)
        or len(set(clipped_slots)) != len(clipped_slots)
    ):
        return False
    clipped = set(clipped_slots)
    bases = []
    for index, (left, _seed_top, right, _seed_bottom) in enumerate(retained):
        if index in clipped:
            continue
        exposed_right = retained[index + 1][0] if index + 1 < len(retained) else right
        margin = max(3, round((exposed_right - left) * 0.10))
        face = mask[:, left + margin:exposed_right - margin]
        if not face.size:
            continue
        rows = _runs(face.mean(axis=1) >= 0.50)
        rows = [run for run in rows if run[1] - run[0] >= max(6, round(height * 0.02))]
        if not rows or rows[0][0] <= 0 or rows[-1][1] >= height - 2:
            continue
        top, bottom = rows[0][0], rows[-1][1]
        if rectangular_slot_supported(
            grayscale, left=left, top=top, right=exposed_right, bottom=bottom, config=config,
        ) or (corner_supported is not None
              and corner_supported(left, top, exposed_right, bottom, True) is True):
            bases.append((top, bottom))
    if not bases:
        return False
    body_height = int(np.median([bottom - top for top, bottom in bases]))
    alignment = max(4, round(body_height * config.alignment_ratio))
    base_bottoms = [bottom for _top, bottom in bases]
    if max(base_bottoms) - min(base_bottoms) > alignment:
        return False
    radius = max(8, round(body_height * config.corner_depth_ratio))
    minimum_span = max(18, round(height * 0.25))
    clipped_bottoms: list[int] = []
    for index in sorted(clipped):
        left, _top, right, _bottom = retained[index]
        exposed_right = retained[index + 1][0] if index + 1 < len(retained) else right
        observed_bottoms = []
        for x in range(max(1, left - alignment), min(width - 1, left + alignment + 1)):
            columns: np.ndarray = grayscale[:, x - 1:x + 1].astype(np.float64, copy=False)
            positive = columns[:, 1] - columns[:, 0] >= config.minimum_difference
            runs = _runs(positive)
            if not runs or runs[0][0] != 0 or runs[0][1] < minimum_span:
                continue
            stop = runs[0][1]
            # A clipped face must really reach the input boundary beside this
            # same inner rim. A passed-in clipped index is not pixel evidence.
            upper_face = mask[:3, x + 2:min(exposed_right, x + 5)]
            if not upper_face.size or float(upper_face.mean()) < config.minimum_corner_support:
                continue
            level = float(np.median(columns[:stop, 0]))
            if float((np.abs(columns[:stop, 0] - level) <= config.minimum_difference).mean()) < config.minimum_corner_support:
                continue
            for bottom in range(stop, min(height - 2, stop + radius) + 1):
                if _lower_rim(
                    grayscale, x=x, bottom=bottom, left=x + 3,
                    right=min(exposed_right - 2, x + radius), level=level, config=config,
                ) or (corner_supported is not None
                      and _lower_rim(
                          grayscale, x=x, bottom=bottom, left=x + radius,
                          right=exposed_right - 2, level=level, config=config,
                      )
                      and corner_supported(left, 0, exposed_right, bottom, False) is True):
                    # Curved-side callbacks can accept several nearby ends.
                    # A visible inner horizontal rim independently fixes the
                    # actual endpoint; do not choose an arbitrary callback hit.
                    observed_bottoms.append(bottom)
        if not observed_bottoms or max(observed_bottoms) - min(observed_bottoms) > alignment:
            return False
        clipped_bottoms.append(min(observed_bottoms))
    for left, _top, _right, _bottom in candidates[len(retained):]:
        edges = []
        for x in range(max(1, left - alignment), min(width - 1, left + alignment + 1)):
            columns = grayscale[:, x - 1:x + 1].astype(np.float64, copy=False)
            delta = columns[:, 1] - columns[:, 0]
            # A genuine gray left rim includes a 60->114 positive inner edge,
            # even when its outer edge is negative or its upper part is hidden.
            if any(stop - start >= radius for start, stop in _runs(delta >= config.minimum_difference)):
                return False
            for start, stop in _runs(delta <= -config.minimum_difference):
                if (
                    stop - start >= minimum_span
                    and any(-alignment <= value - stop <= radius + alignment for value in base_bottoms)
                    and min(abs(start - value) for value in clipped_bottoms) <= alignment
                ):
                    edges.append((x, start, stop, float(np.median(columns[start:stop, 1]))))
        supported = False
        for x, start, stop, level in edges:
            arm_left, arm_right = x - radius, x + radius
            if arm_left < 0 or arm_right > width or radius - 3 < 6:
                continue
            left_face = mask[start:stop, arm_left:x - 3]
            right_face = mask[start:stop, x + 3:arm_right]
            above_face = mask[max(0, start - radius):start - 3, arm_left:arm_right]
            if (
                not above_face.size
                or float(left_face.mean()) < config.minimum_corner_support
                or right_face.any()  # Any actual white/gray face vetoes deletion.
                or float(above_face.mean()) < config.minimum_corner_support
            ):
                continue
            bottom_proposals = {
                bottom for base in base_bottoms
                for bottom in range(max(start + 3, base - alignment), min(height - 1, base + alignment) + 1)
            }
            if any(
                (abs(bottom - stop) <= alignment
                 and _lower_rim(grayscale, x=x, bottom=bottom, left=arm_left,
                                right=x - 3, level=level, config=config))
                or (_lower_rim(grayscale, x=x, bottom=bottom, left=x - 2 * radius,
                               right=x - radius, level=level, config=config)
                    and _right_turn_supported(grayscale, x, bottom, body_height, config))
                for bottom in bottom_proposals
            ):
                supported = True
                break
        if not supported:
            return False
    return True


def _right_turn_supported(
    grayscale: np.ndarray, x: int, bottom: int, body_height: int,
    config: ShortHandRecoveryConfig,
) -> bool:
    """Follow negative edges turning left, using the existing corner limits.

    This is the right-contour counterpart of a left rounded corner, evaluated
    on a small strip. The independent flat bottom rim is checked by the caller.
    """
    radius = max(8, round(body_height * config.corner_depth_ratio))
    if bottom >= grayscale.shape[0] or bottom - radius < 0 or x - radius < 0 or x + 4 >= grayscale.shape[1]:
        return False
    strip: np.ndarray = grayscale[bottom - radius:bottom, x - radius:x + 5].astype(np.float64, copy=False)
    negative = strip[:, :-1] - strip[:, 1:]
    offsets = negative.argmax(axis=1).astype(float) - radius + 1
    valid = negative.max(axis=1) >= config.minimum_difference
    indexes = np.flatnonzero(valid)
    if (
        len(offsets) < 10 or not len(indexes)
        or float(valid.mean()) < config.minimum_corner_support
        or len(offsets) - 1 - indexes[-1] > max(3, body_height * config.alignment_ratio)
    ):
        return False
    offsets = offsets[:indexes[-1] + 1]
    base = float(np.median(offsets[:max(3, len(offsets) // 3)]))
    minimum_bend = max(2.0, body_height * config.corner_bend_ratio)
    return bool(
        abs(base) <= 3
        and base - float(np.median(offsets[-3:])) >= minimum_bend
        and np.count_nonzero(offsets <= base - minimum_bend) >= 2
        and float(np.mean(np.abs(np.diff(offsets)) <= max(2, body_height * 0.01)))
        >= config.minimum_corner_support
    )


def _lower_rim(
    grayscale: np.ndarray, *, x: int, bottom: int, left: int, right: int,
    level: float, config: ShortHandRecoveryConfig,
) -> bool:
    """A same-color horizontal closure, on the requested side of one rim."""
    if right - left < 6 or left < 0 or right > grayscale.shape[1] or not 3 <= bottom < grayscale.shape[0]:
        return False
    arm: np.ndarray = grayscale[bottom - 3:bottom + 1, left:right].astype(np.float64, copy=False)
    joined = (
        (arm[-2] - arm[-1] >= config.minimum_difference)
        & (np.abs(arm[-1] - level) <= config.minimum_difference)
    )
    return bool(
        float(joined.mean()) >= config.minimum_corner_support
        and float((arm[:-1] - level >= config.minimum_difference).mean()) >= config.minimum_corner_support
        and (right < x or left > x)
    )


def _valid_boxes(boxes: tuple[Box, ...], width: int, height: int) -> bool:
    previous = -1
    for box in boxes:
        if len(box) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in box):
            return False
        left, top, right, bottom = box
        if not (previous < left < right <= width and 0 <= top < bottom <= height):
            return False
        previous = left
    return True


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(values)
    if not len(indexes):
        return []
    starts = np.concatenate(([0], np.flatnonzero(np.diff(indexes) > 1) + 1))
    stops = np.concatenate((starts[1:], [len(indexes)]))
    return [(int(indexes[start]), int(indexes[stop - 1]) + 1) for start, stop in zip(starts, stops, strict=True)]
