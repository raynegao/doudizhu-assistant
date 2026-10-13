from __future__ import annotations

from numbers import Integral

import numpy as np

from src.config.hand_geometry import ShortHandRecoveryConfig


def rectangular_slot_supported(
    grayscale: np.ndarray,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    config: ShortHandRecoveryConfig,
) -> bool:
    """Certify a proposed extent, never discover a slot or infer a count.

    The caller must first validate the existing primary grid and must retain
    its cardinality. A decorative T junction painted with exactly the same
    pixels as a card rim is locally indistinguishable from an overlapping
    square card; this predicate alone cannot establish physical ownership.

    Both rightward corners must join the *same* sustained, dark-to-light side
    at a fixed x. Endpoint searches only accommodate the configured alignment
    tolerance. They do not choose the longest side or relax rounded corners.
    ``right`` limits this slot's exposed strip, before the next slot begins.
    """

    if not isinstance(config, ShortHandRecoveryConfig):
        raise ValueError("rectangular slot evidence requires ShortHandRecoveryConfig")
    if (
        not isinstance(grayscale, np.ndarray)
        or grayscale.ndim != 2
        or grayscale.dtype.kind not in "uif"
        or any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in (left, top, right, bottom))
    ):
        return False
    height, width = grayscale.shape
    if not (0 <= left < right <= width and 0 < top < bottom < height):
        return False
    body_height = bottom - top
    if body_height < 24:
        return False
    alignment = max(3, round(body_height * config.alignment_ratio))
    radius = max(8, round(body_height * config.corner_depth_ratio))
    minimum_arm = max(6, round(body_height * config.alignment_ratio))
    search_top = max(1, top - alignment)
    search_bottom = min(height - 1, bottom + alignment)
    minimum_difference = config.minimum_difference
    minimum_support = config.minimum_corner_support

    for side_x in range(max(1, left - alignment), min(right - 1, left + alignment + 1)):
        arm_end = min(right - 2, side_x + radius)
        if arm_end - side_x - 2 < minimum_arm:
            continue
        # Cast only a narrow local strip, so uint8 subtraction cannot wrap and
        # callers can reuse their existing grayscale without full-ROI copies.
        strip: np.ndarray = grayscale[search_top - 1:search_bottom + 1, side_x - 1:arm_end].astype(
            np.float64, copy=False,
        )
        if not np.isfinite(strip).all() or np.any((strip < 0) | (strip > 255)):
            continue
        outside, inside, arm = strip[:, 0], strip[:, 1], strip[:, 3:]
        rim_level = float(np.median(outside[1:-1]))
        side_support = (
            (inside - outside >= minimum_difference)
            & (np.abs(outside - rim_level) <= minimum_difference)
        )
        prefix = np.concatenate(([0], np.cumsum(side_support, dtype=np.int64)))
        origin = search_top - 1
        top_proposals = []
        bottom_proposals = []
        for candidate in range(search_top, min(top + alignment, height - 3) + 1):
            row = candidate - origin
            upper_rim = arm[row - 1]
            connected = (
                (arm[row] - upper_rim >= minimum_difference)
                & (np.abs(upper_rim - rim_level) <= minimum_difference)
            )
            if (
                float(connected.mean()) >= minimum_support
                and float((arm[row:row + 3] - rim_level >= minimum_difference).mean())
                >= minimum_support
            ):
                top_proposals.append(candidate)
        for candidate in range(max(bottom - alignment, 3), search_bottom + 1):
            row = candidate - origin
            lower_rim = arm[row]
            connected = (
                (arm[row - 1] - lower_rim >= minimum_difference)
                & (np.abs(lower_rim - rim_level) <= minimum_difference)
            )
            if (
                float(connected.mean()) >= minimum_support
                and float((arm[row - 3:row] - rim_level >= minimum_difference).mean())
                >= minimum_support
            ):
                bottom_proposals.append(candidate)
        for actual_top in top_proposals:
            for actual_bottom in bottom_proposals:
                span = actual_bottom - actual_top
                if span < 24:
                    continue
                start, stop = actual_top - origin, actual_bottom - origin
                corner_depth = min(span, max(8, round(span * config.corner_depth_ratio)))
                supported = (
                    (prefix[stop] - prefix[start]) / span,
                    (prefix[start + corner_depth] - prefix[start]) / corner_depth,
                    (prefix[stop] - prefix[stop - corner_depth]) / corner_depth,
                )
                if min(supported) >= minimum_support:
                    return True
    return False
