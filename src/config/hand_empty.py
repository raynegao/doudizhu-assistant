from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class HandEmptyConfig:
    """Calibrated active-hand domain, not an inference from missing slots.

    The standalone default excludes nothing. A bottom UI band is only valid
    when layout calibration puts every legal hand/animation above that band.
    """

    bottom_ui_fraction: float = 0.0
    blue_red_gap: int = 15
    blue_green_gap: int = 15
    blue_minimum: int = 60
    minimum_blue_fraction: float = 0.995
    minimum_dimension: int = 32
    maximum_input_pixels: int = 1_500_000

    def __post_init__(self) -> None:
        for name, low, high in (("bottom_ui_fraction", 0.0, 0.03),
                                ("minimum_blue_fraction", 0.995, 1.0)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"hand_empty.{name} must be finite and in [{low}, {high}]")
        for name in ("blue_red_gap", "blue_green_gap", "blue_minimum",
                     "minimum_dimension", "maximum_input_pixels"):
            value = getattr(self, name)
            upper = 10_000_000 if name == "maximum_input_pixels" else 255
            lower = {"blue_red_gap": 15, "blue_green_gap": 15, "blue_minimum": 60,
                     "minimum_dimension": 32, "maximum_input_pixels": 1}[name]
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"hand_empty.{name} must be a positive bounded integer")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> HandEmptyConfig:
        if not isinstance(value, Mapping) or set(value) - {item.name for item in fields(cls)}:
            raise ValueError("hand_empty must be an object with known fields")
        arguments: dict[str, Any] = dict(value)
        return cls(**arguments)
