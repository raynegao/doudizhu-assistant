from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.config.hand_photometry import HandPhotometryConfig
from src.vision.scene_types import CardPhotometryEvidence


@dataclass(frozen=True)
class PreparedHandCrop:
    image: Image.Image
    evidence: CardPhotometryEvidence


def prepare_hand_rank_crop(
    image: Image.Image,
    *,
    geometry_verified: bool,
    config: HandPhotometryConfig,
) -> PreparedHandCrop:
    """Correct a gray selection overlay without rank, history or model input.

    Never infer geometry or restore clipped pixels. A single multiplier keeps
    RGB ratios intact; no per-channel balance, autocontrast, rank-dependent
    retry or model-score selection is allowed. Rejection preserves the input.
    """
    def unchanged(reason: str, *, usable: bool = True) -> PreparedHandCrop:
        return PreparedHandCrop(image, CardPhotometryEvidence(reason=reason, usable=usable))

    if not geometry_verified:
        return unchanged("geometry_unverified", usable=False)
    if image.mode != "RGB" or min(image.size) == 0:
        return unchanged("unsupported_crop", usable=False)
    rgb = np.asarray(image, dtype=np.int16)
    low, high = rgb.min(axis=2), rgb.max(axis=2)
    if min(image.size) < config.minimum_crop_side:
        # Small, plainly white inputs need no gain, but insufficient pixels
        # must never turn an unsupported gray estimate into a trusted no-op.
        white = (low > config.background_max) & (high - low <= config.neutral_chroma)
        return unchanged(
            "unsupported_crop", usable=float(white.mean()) >= config.minimum_paper_fraction,
        )
    neutral = (high - low <= config.neutral_chroma)
    gray = neutral & (low >= config.background_min) & (high <= config.background_max)
    gray_fraction = float(gray.mean())
    # Bright paper plus gray paper is not a uniform selection overlay. Do not
    # saturate a second card, an unselected stripe or an occlusion highlight.
    if float(np.mean(low > config.background_max)) > config.maximum_bright_fraction:
        return unchanged(
            "bright_or_mixed_paper", usable=gray_fraction < config.minimum_gray_fraction,
        )
    if gray_fraction < config.minimum_paper_fraction:
        return unchanged(
            "gray_paper_not_dominant", usable=gray_fraction < config.minimum_gray_fraction,
        )
    levels = np.rint(rgb.mean(axis=2)).astype(np.int16)
    histogram = np.bincount(levels[gray], minlength=256)
    peak = int(np.argmax(histogram))
    paper = gray & (np.abs(levels - peak) <= config.paper_tolerance)
    support = float(paper.mean())
    if support < config.minimum_paper_fraction:
        return unchanged("paper_level_ambiguous", usable=False)
    height, width = paper.shape
    tiles = (
        paper[top:bottom, left:right]
        for top, bottom in ((0, height // 2), (height // 2, height))
        for left, right in ((0, width // 2), (width // 2, width))
    )
    if any(float(tile.mean()) < config.minimum_tile_fraction for tile in tiles):
        return unchanged("paper_spatial_support_missing", usable=False)
    # A flat gray field is not a printed rank. Colored ink also needs dark
    # support, so decorative color cannot determine the background estimate.
    ink_fraction = float(np.mean(high <= peak * config.ink_contrast_ratio))
    if not config.minimum_ink_fraction <= ink_fraction <= config.maximum_ink_fraction:
        return unchanged("ink_contrast_unsupported", usable=False)
    gain = config.target_paper / peak
    if gain > config.maximum_gain:
        return unchanged("gain_limit_exceeded", usable=False)
    lookup = [min(255, round(value * gain)) for value in range(256)]
    corrected = image.point(lookup * 3)
    return PreparedHandCrop(corrected, CardPhotometryEvidence(
        reason="neutral_gray_paper_gain", gain=gain,
        background_level=peak, paper_fraction=support,
    ))
