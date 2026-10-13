"""Pure, bounded button-entity observations; never infer a pass or use templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.config.turn_controls import TurnControlsConfig

Box: TypeAlias = tuple[int, int, int, int]
Mask: TypeAlias = NDArray[np.bool_]
Run: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class TurnControlEvidence:
    active: bool | None
    confidence: float
    reason: str
    entity_boxes: tuple[Box, ...] = ()
    # UNKNOWN is a structural refusal, not permission for a background match.
    template_fallback_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class _Component:
    box: Box
    area: int
    runs: tuple[Run, ...]


class _RunBudgetExceeded(Exception):
    pass


def _components(mask: Mask, budget: list[int]) -> tuple[_Component, ...]:
    """Four-connected row runs, with a linear overlap sweep and shared budget."""
    parent: list[int] = []
    runs: list[Run] = []
    previous: list[int] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for y, row in enumerate(mask):
        delta = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(delta == 1).tolist()
        ends = np.flatnonzero(delta == -1).tolist()
        budget[0] -= len(starts)
        if budget[0] < 0:
            raise _RunBudgetExceeded
        current: list[int] = []
        cursor = 0
        for left, right in zip(starts, ends, strict=True):
            index = len(parent)
            parent.append(index)
            runs.append((y, left, right))
            current.append(index)
            while cursor < len(previous) and runs[previous[cursor]][2] <= left:
                cursor += 1
            overlap = cursor
            while overlap < len(previous) and runs[previous[overlap]][1] < right:
                parent[find(index)] = find(previous[overlap])
                overlap += 1
        previous = current
    groups: dict[int, list[Run]] = {}
    for index, run in enumerate(runs):
        groups.setdefault(find(index), []).append(run)
    return tuple(
        _Component(
            (min(row[1] for row in group), group[0][0],
             max(row[2] for row in group), group[-1][0] + 1),
            sum(right - left for _, left, right in group), tuple(group),
        )
        for group in groups.values()
    )


def _dilate(mask: Mask, radius: int) -> Mask:
    """Bounded square neighborhood, with no per-pixel flood or wraparound."""
    if radius <= 0:
        return mask.copy()
    horizontal = np.pad(mask.astype(np.int32), ((0, 0), (radius, radius)))
    sums = np.pad(horizontal.cumsum(axis=1), ((0, 0), (1, 0)))
    horizontal = sums[:, 2 * radius + 1:] - sums[:, :-2 * radius - 1] > 0
    vertical = np.pad(horizontal.astype(np.int32), ((radius, radius), (0, 0)))
    sums = np.pad(vertical.cumsum(axis=0), ((1, 0), (0, 0)))
    return sums[2 * radius + 1:] - sums[:-2 * radius - 1] > 0


def _row_spans(component: _Component) -> NDArray[np.float64]:
    left, top, right, bottom = component.box
    starts = np.full(bottom - top, right)
    ends = np.full(bottom - top, left)
    for y, x1, x2 in component.runs:
        starts[y - top] = min(starts[y - top], x1)
        ends[y - top] = max(ends[y - top], x2)
    return (ends - starts) / (right - left)


def _seed_is_button(
    seed: _Component, shape: tuple[int, int], config: TurnControlsConfig,
) -> bool:
    height, width = shape
    left, top, right, bottom = seed.box
    box_width, box_height = right - left, bottom - top
    spans = _row_spans(seed)
    end_fraction = config.profile_end_fraction
    central = spans[round(box_height * end_fraction):round(box_height * (1 - end_fraction))]
    end_count = max(1, round(box_height * end_fraction))
    return bool(
        central.size
        and config.button_aspect[0] <= box_width / box_height <= config.button_aspect[1]
        and config.button_height[0] <= box_height / height <= config.button_height[1]
        and box_width / width >= config.minimum_button_width
        and seed.area / (box_width * box_height) >= config.minimum_fill
        and float(central.mean()) >= config.minimum_center_span
        and float(spans[:end_count].mean()) < config.maximum_end_span
        and float(spans[-end_count:].mean()) < config.maximum_end_span
    )


def _entity_refusal(
    entity: Mask, component: _Component, warm: Mask, blue: Mask, dark: Mask,
    threshold: int, margin: int, config: TurnControlsConfig, budget: list[int],
) -> str | None:
    seeds = [item for item in _components(warm & entity, budget) if item.area >= threshold]
    if not seeds:
        return "independent_nonbutton_foreground"
    seed = max(seeds, key=lambda item: item.area)
    sx1, sy1, sx2, sy2 = seed.box
    if any(
        other.box[0] < sx1 or other.box[1] < sy1
        or other.box[2] > sx2 or other.box[3] > sy2 for other in seeds
    ):
        return "extra_warm_fragment_outside_seed"
    if not _seed_is_button(seed, warm.shape, config):
        return "warm_seed_not_button"
    left, top, right, bottom = component.box
    collar_limit = max(config.minimum_margin_pixels, round((sy2 - sy1) * config.maximum_collar_ratio))
    if max(sx1 - left, sy1 - top, right - sx2, bottom - sy2) > collar_limit:
        return "connected_collar_too_thick"
    # Preserve seed holes as internal content while retaining the actual outer
    # row envelope. Do not claim every pixel in the bounding box as a button.
    seed_pixels = warm & entity
    envelope = np.zeros_like(warm)
    for y in np.flatnonzero(seed_pixels.any(axis=1)):
        columns = np.flatnonzero(seed_pixels[y])
        envelope[y, columns[0]:columns[-1] + 1] = True
    collar = entity & ~envelope
    if np.any(collar & ~_dilate(envelope, collar_limit)):
        return "collar_has_nonlocal_extension"
    if collar.any() and float(dark[collar].mean()) < config.minimum_dark_collar_ratio:
        return "collar_not_dark_rim_or_shadow"
    if component.area / ((right - left) * (bottom - top)) < config.minimum_entity_fill:
        return "entity_body_not_filled"
    exterior = _dilate(entity, margin) & ~entity
    if not exterior.any() or float(blue[exterior].mean()) < config.minimum_blue_background:
        return "entity_exterior_not_closed_blue_background"
    return None


def observe_turn_controls(
    image: Image.Image, *, config: TurnControlsConfig | None = None,
) -> TurnControlEvidence:
    """Observe entity geometry, not button text, game phase, or a pass event.

    All structural failures are UNKNOWN. Whole-ROI active/inactive templates
    must never override this observation. The fixed confidence is an evidence
    marker, not an empirically calibrated probability of correct game state.
    """
    if config is not None and not isinstance(config, TurnControlsConfig):
        raise ValueError("config must be a TurnControlsConfig")
    config = config or TurnControlsConfig()

    def unknown(reason: str) -> TurnControlEvidence:
        return TurnControlEvidence(None, 0.0, reason)

    width, height = image.size
    if width <= 0 or height <= 0:
        return unknown("empty_input")
    if width * height > config.maximum_input_pixels:
        return unknown("turn_controls_input_budget_exceeded")
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    red_min, green_min, blue_max = config.warm_rgb_limits
    red_gap, green_gap = config.warm_differences
    warm = (r > red_min) & (g > green_min) & (b < blue_max) & (r - b > red_gap) & (g - b > green_gap)
    br_gap, bg_gap, blue_min = config.blue_background_limits
    blue = (b > r + br_gap) & (b > g + bg_gap) & (b > blue_min)
    threshold = max(config.minimum_component_pixels, round(warm.size * config.minimum_component_ratio))
    if np.count_nonzero(warm) < threshold:
        if float(blue.mean()) >= config.minimum_blue_background:
            return TurnControlEvidence(False, 0.99, "verified_empty_blue_controls")
        return unknown("nonempty_or_unverified_background")
    budget = [config.maximum_component_runs]
    margin = max(config.minimum_margin_pixels, round(min(width, height) * config.boundary_margin_ratio))
    explained = np.zeros_like(warm)
    dark = pixels.mean(axis=2) <= config.maximum_collar_channel_mean
    boxes: list[Box] = []
    try:
        for component in _components(~blue, budget):
            if component.area < threshold:
                continue
            if len(boxes) >= config.maximum_entities:
                return unknown("turn_controls_entity_budget_exceeded")
            left, top, right, bottom = component.box
            if left < margin or top < margin or right > width - margin or bottom > height - margin:
                return unknown("entity_touches_roi_boundary")
            entity = np.zeros_like(warm)
            for y, x1, x2 in component.runs:
                entity[y, x1:x2] = True
            reason = _entity_refusal(entity, component, warm, blue, dark, threshold, margin, config, budget)
            if reason:
                return unknown(reason)
            explained |= entity
            boxes.append(component.box)
    except _RunBudgetExceeded:
        return unknown("turn_controls_run_budget_exceeded")
    if not boxes or float((~blue & ~explained).mean()) > 1 - config.minimum_blue_background:
        return unknown("unexplained_foreground_outside_entity")
    centers = [(box[1] + box[3]) / 2 for box in boxes]
    if max(centers) - min(centers) > height * config.boundary_margin_ratio:
        return unknown("button_row_misaligned")
    return TurnControlEvidence(True, 0.99, "complete_connected_button_entities", tuple(boxes))
