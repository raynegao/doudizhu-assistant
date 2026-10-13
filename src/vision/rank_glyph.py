from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from src.config.joker_color import JokerColorConfig


@dataclass(frozen=True)
class RankGlyphComponent:
    pixels: frozenset[tuple[int, int]]
    signature: frozenset[int]
    box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class JokerColorEvidence:
    color: Literal["red", "black", "unknown"]
    reason: str
    pixel_count: int = 0
    red_fraction: float = 0.0
    black_fraction: float = 0.0
    background_fraction: float = 0.0


def extract_rank_glyph(image: Image.Image) -> RankGlyphComponent:
    """Preserve the legacy grayscale component selection and 64x64 signature."""
    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    assert pixels is not None
    foreground = {(x, y) for y in range(height) for x in range(width) if cast(int, pixels[x, y]) < 130}
    components: list[set[tuple[int, int]]] = []
    while foreground:
        start = foreground.pop()
        component = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    neighbor = (x + offset_x, y + offset_y)
                    if neighbor in foreground:
                        foreground.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        if len(component) >= 20:
            components.append(component)

    def bounds(component: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1

    candidates = []
    for component in components:
        left, top, right, bottom = bounds(component)
        if (
            top < height * 0.45
            and left < width * 0.40
            and bottom - top > height * 0.12
            and right - left > width * 0.12
            and bottom - top < height * 0.70
            and right - left < width * 0.85
        ):
            candidates.append((component, (left, top, right, bottom)))
    if not candidates:
        return RankGlyphComponent(frozenset(), frozenset())
    primary_component, primary_box = min(candidates, key=lambda item: (item[1][1], item[1][0]))
    primary_top, primary_bottom = primary_box[1], primary_box[3]
    selected = set(primary_component)
    for component, box in candidates:
        if component is primary_component:
            continue
        if min(primary_bottom, box[3]) - max(primary_top, box[1]) > 0:
            selected.update(component)
    left = min(point[0] for point in selected)
    top = min(point[1] for point in selected)
    right = max(point[0] for point in selected) + 1
    bottom = max(point[1] for point in selected) + 1
    mask = Image.new("L", (right - left, bottom - top))
    mask_pixels = mask.load()
    assert mask_pixels is not None
    for x, y in selected:
        mask_pixels[x - left, y - top] = 255
    fitted = ImageOps.contain(mask, (52, 52), method=Image.Resampling.NEAREST)
    normalized = Image.new("L", (64, 64))
    normalized.paste(
        fitted,
        ((normalized.width - fitted.width) // 2, (normalized.height - fitted.height) // 2),
    )
    signature = frozenset(
        y * normalized.width + x
        for y in range(normalized.height)
        for x in range(normalized.width)
        if cast(int, normalized.getpixel((x, y))) > 0
    )
    return RankGlyphComponent(frozenset(selected), signature, (left, top, right, bottom))


def read_joker_color(
    image: Image.Image,
    glyph: RankGlyphComponent,
    config: JokerColorConfig,
) -> JokerColorEvidence:
    """Read only the selected rank ink; red body/landlord decorations do not vote."""
    count = len(glyph.pixels)
    if not glyph.pixels or glyph.box is None:
        return JokerColorEvidence("unknown", "rank_component_unavailable")
    if count < config.min_component_pixels:
        return JokerColorEvidence("unknown", "rank_component_too_small", count)
    left, top, right, bottom = glyph.box
    if left <= 0 or top <= 0 or right >= image.width or bottom >= image.height:
        return JokerColorEvidence("unknown", "rank_component_clipped", count)
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    points = tuple(glyph.pixels)
    xs = np.fromiter((point[0] for point in points), dtype=np.intp)
    ys = np.fromiter((point[1] for point in points), dtype=np.intp)
    ink = rgb[ys, xs]
    red, green, blue = ink[:, 0], ink[:, 1], ink[:, 2]
    red_fraction = float(np.mean(
        (red >= config.red_min_channel)
        & (red - green >= config.red_channel_margin)
        & (red - blue >= config.red_channel_margin)
    ))
    maximum = ink.max(axis=1)
    minimum = ink.min(axis=1)
    black_pixels = (
        (maximum <= config.black_max_channel)
        & (maximum - minimum <= config.black_max_chroma)
    )
    black_fraction = float(np.mean(black_pixels))
    margin = config.context_margin
    context_left, context_top = max(0, left - margin), max(0, top - margin)
    context = rgb[context_top:min(image.height, bottom + margin),
                  context_left:min(image.width, right + margin)]
    background = np.ones(context.shape[:2], dtype=bool)
    background[ys - context_top, xs - context_left] = False
    paper = context[background]
    background_fraction = 0.0
    if paper.size:
        background_fraction = float(np.mean(
            (paper.min(axis=1) >= config.background_min_channel)
            & (paper.max(axis=1) - paper.min(axis=1) <= config.background_max_chroma)
        ))
    if background_fraction < config.min_background_fraction:
        return JokerColorEvidence(
            "unknown", "rank_background_ambiguous", count,
            red_fraction, black_fraction, background_fraction,
        )
    is_red = red_fraction >= config.min_color_fraction
    is_black = black_fraction >= config.min_color_fraction
    color: Literal["red", "black", "unknown"] = "unknown"
    reason = "rank_ink_ambiguous"
    if is_red and not is_black:
        color, reason = "red", "red_rank_ink"
    elif is_black and not is_red:
        color, reason = "black", "black_rank_ink"
    elif red_fraction == 0.0 and not is_black and _has_black_antialias_support(
        glyph.pixels, points, black_pixels,
        neutral=maximum - minimum <= config.black_max_chroma,
        minimum_core_pixels=config.min_component_pixels,
    ):
        # Keep the measured whole-glyph fraction and raw model confidence.
        # Gray coverage pixels are not darker ink; their spatial attachment
        # to an independently black interior supplies the extra evidence.
        color, reason = "black", "black_rank_ink_antialiased_edge"
    return JokerColorEvidence(color, reason, count, red_fraction, black_fraction, background_fraction)


def _has_black_antialias_support(
    pixels: frozenset[tuple[int, int]],
    points: tuple[tuple[int, int], ...],
    black_pixels: NDArray[np.bool_],
    *,
    neutral: NDArray[np.bool_],
    minimum_core_pixels: int,
) -> bool:
    """Prove a one-pixel neutral edge attached to a sufficiently black core.

    Never discard a contradictory interior pixel or vote with nearby card
    decorations. A four-neighbor erosion identifies the solid interior. All
    of it must satisfy the original RGB black gate, and every nonblack edge
    pixel must touch black ink connected to that interior. This rejects thin
    gray fragments, thick gray borders and mixed-color glyphs without changing
    the grayscale component/signature or either original color threshold.
    """
    if not np.all(neutral):
        return False
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    core = {
        (x, y) for x, y in pixels
        if all((x + dx, y + dy) in pixels for dx, dy in offsets)
    }
    black = {point for point, is_black in zip(points, black_pixels, strict=True) if is_black}
    if len(core) < minimum_core_pixels or not core <= black:
        return False
    # A short antialiased serif can touch black edge ink rather than the
    # eroded core directly. Require a real black path, not gray bridging.
    supported = set(core)
    pending = list(core)
    while pending:
        x, y = pending.pop()
        for dx, dy in offsets:
            neighbor = (x + dx, y + dy)
            if neighbor in black and neighbor not in supported:
                supported.add(neighbor)
                pending.append(neighbor)
    return all(
        any((x + dx, y + dy) in supported for dx, dy in offsets)
        for x, y in pixels - black
    )
