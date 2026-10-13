from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class HandPhotometryConfig:
    """Paper-based exposure correction for verified hand rank crops only.

    These are image support gates, not card probabilities. Normal white crops
    are byte-preserved; only a dominant, spatially supported neutral gray
    paper level can determine a single RGB gain.
    """

    minimum_crop_side: int = 20
    neutral_chroma: int = 12
    background_min: int = 90
    background_max: int = 180
    target_paper: int = 250
    maximum_gain: float = 3.0
    # The client's unselected paper itself shades from about 248 to 234;
    # a half-bright selection preserves that gradient (124 to 117). Support
    # must include the paper gradient, not demand a flat histogram spike.
    paper_tolerance: int = 8
    minimum_gray_fraction: float = 0.25
    minimum_paper_fraction: float = 0.60
    minimum_tile_fraction: float = 0.25
    maximum_bright_fraction: float = 0.025
    ink_contrast_ratio: float = 0.70
    minimum_ink_fraction: float = 0.01
    maximum_ink_fraction: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "minimum_crop_side", "neutral_chroma", "background_min",
            "background_max", "target_paper", "paper_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
                raise ValueError(f"hand_photometry.{name} must be an integer in [1, 255]")
        if not self.background_min < self.background_max < self.target_paper:
            raise ValueError("hand_photometry requires background_min < background_max < target_paper")
        if self.minimum_crop_side < 2:
            raise ValueError("hand_photometry.minimum_crop_side must be at least 2 for spatial tiles")
        if self.paper_tolerance >= self.background_min:
            raise ValueError("hand_photometry.paper_tolerance must be below background_min")
        if not _finite(self.maximum_gain) or not 1 < self.maximum_gain <= 4:
            raise ValueError("hand_photometry.maximum_gain must be finite and in (1, 4]")
        for name in (
            "minimum_gray_fraction", "minimum_paper_fraction", "minimum_tile_fraction", "maximum_bright_fraction",
            "ink_contrast_ratio", "minimum_ink_fraction", "maximum_ink_fraction",
        ):
            value = getattr(self, name)
            if (
                not _finite(value) or not 0 < value < 1
            ):
                raise ValueError(f"hand_photometry.{name} must be finite and in (0, 1)")
        if self.minimum_paper_fraction <= 0.5:
            raise ValueError("hand_photometry.minimum_paper_fraction must exceed one half")
        if self.minimum_gray_fraction >= self.minimum_paper_fraction:
            raise ValueError("hand_photometry gray suspicion must precede dominant paper support")
        if self.minimum_ink_fraction >= self.maximum_ink_fraction:
            raise ValueError("hand_photometry ink fraction interval must be ordered")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> HandPhotometryConfig:
        if not isinstance(value, Mapping) or set(value) - {item.name for item in fields(cls)}:
            raise ValueError("hand_photometry must be an object containing only known fields")
        arguments: dict[str, Any] = dict(value)
        return cls(**arguments)


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
