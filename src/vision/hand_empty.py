"""Positive, fixed-layout empty-domain evidence; no state or model access."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from src.config.hand_empty import HandEmptyConfig
from src.vision.scene_types import HandEmptyEvidence


def observe_empty_hand_domain(
    image: Image.Image, *, config: HandEmptyConfig,
    geometry_count: int | None, has_candidates: bool, geometry_reason: str,
) -> HandEmptyEvidence:
    if (has_candidates or geometry_reason or geometry_count not in (None, 0)
            or isinstance(geometry_count, bool)):
        return HandEmptyEvidence(reason="geometry_not_empty_candidate")
    width, height = image.size
    if (image.mode != "RGB" or min(width, height) < config.minimum_dimension
            or width * height > config.maximum_input_pixels):
        return HandEmptyEvidence(reason="unsupported_input")
    excluded = math.floor(height * config.bottom_ui_fraction)
    effective_box = (0, 0, width, height - excluded)
    pixels = np.asarray(image, dtype=np.int16)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    background = ((blue > red + config.blue_red_gap)
                  & (blue > green + config.blue_green_gap)
                  & (blue > config.blue_minimum))
    fraction = float(background.mean())
    # A global percentage alone misses even tall, three-pixel card slivers.
    # Inspect every pixel in the calibrated domain; never grow the exclusion
    # to fit the current foreground. The excluded UI band is a declared blind
    # area, not a region in which this routine claims to rule out arbitrary cards.
    verified = bool(background[:height - excluded].all()) and fraction >= config.minimum_blue_fraction
    return HandEmptyEvidence(
        verified=verified,
        reason="verified_calibrated_blue_hand_domain" if verified else "foreground_or_unsupported_background",
        effective_box=effective_box, excluded_bottom_px=excluded, blue_fraction=fraction,
    )
