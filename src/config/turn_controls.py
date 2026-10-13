"""Validated configuration for local turn-control entity observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class TurnControlsConfig:
    # Strict lower R/G bounds and upper B bound for the warm interior seed.
    warm_rgb_limits: tuple[float, float, float] = (150.0, 80.0, 110.0)
    warm_differences: tuple[float, float] = (65.0, 25.0)
    # Strict lower B-R, B-G and B bounds for the supported blue background.
    blue_background_limits: tuple[float, float, float] = (15.0, 15.0, 60.0)
    minimum_component_ratio: float = 0.001
    button_aspect: tuple[float, float] = (2.0, 6.0)
    button_height: tuple[float, float] = (0.20, 0.75)
    minimum_button_width: float = 0.25
    minimum_fill: float = 0.65
    minimum_center_span: float = 0.85
    maximum_end_span: float = 0.97
    profile_end_fraction: float = 0.20
    boundary_margin_ratio: float = 0.01
    minimum_blue_background: float = 0.995
    maximum_collar_ratio: float = 0.15
    minimum_entity_fill: float = 0.80
    minimum_dark_collar_ratio: float = 0.70
    maximum_collar_channel_mean: float = 180.0
    minimum_component_pixels: int = 6
    minimum_margin_pixels: int = 2
    maximum_input_pixels: int = 1_000_000
    maximum_component_runs: int = 10_000
    maximum_entities: int = 16

    def __post_init__(self) -> None:
        for name, length, upper, interval in (
            ("warm_rgb_limits", 3, 255.0, False),
            ("warm_differences", 2, 255.0, False),
            ("blue_background_limits", 3, 255.0, False),
            ("button_aspect", 2, 20.0, True),
            ("button_height", 2, 1.0, True),
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != length:
                raise ValueError(f"turn_controls.{name} must have {length} numbers")
            normalized = tuple(_finite_number(name, item) for item in value)
            if any(item < 0 or item > upper for item in normalized):
                raise ValueError(f"turn_controls.{name} is out of range")
            if interval and not 0 < normalized[0] < normalized[1]:
                raise ValueError(f"turn_controls.{name} must be a positive ordered interval")
            object.__setattr__(self, name, normalized)
        for name in (
            "minimum_component_ratio", "minimum_button_width", "minimum_fill",
            "minimum_center_span", "maximum_end_span", "minimum_blue_background",
            "minimum_entity_fill", "minimum_dark_collar_ratio",
        ):
            number = _finite_number(name, getattr(self, name))
            if not 0 < number <= 1:
                raise ValueError(f"turn_controls.{name} must be in (0, 1]")
            object.__setattr__(self, name, number)
        for name in ("profile_end_fraction", "boundary_margin_ratio", "maximum_collar_ratio"):
            number = _finite_number(name, getattr(self, name))
            if not 0 < number < 0.5:
                raise ValueError(f"turn_controls.{name} must be in (0, 0.5)")
            object.__setattr__(self, name, number)
        number = _finite_number("maximum_collar_channel_mean", self.maximum_collar_channel_mean)
        if not 0 < number <= 255:
            raise ValueError("turn_controls.maximum_collar_channel_mean must be in (0, 255]")
        object.__setattr__(self, "maximum_collar_channel_mean", number)
        for name, upper in (
            ("minimum_component_pixels", 1_000_000), ("minimum_margin_pixels", 1_000),
            ("maximum_input_pixels", 16_000_000), ("maximum_component_runs", 1_000_000),
            ("maximum_entities", 256),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"turn_controls.{name} must be an integer in [1, {upper}]")
        if self.minimum_component_pixels > self.maximum_input_pixels:
            raise ValueError("turn_controls.minimum_component_pixels exceeds input budget")

    def to_payload(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, value: object) -> TurnControlsConfig:
        if not isinstance(value, Mapping):
            raise ValueError("turn_controls must be an object")
        if set(value) - {item.name for item in fields(cls)}:
            raise ValueError("turn_controls contains unknown fields")
        arguments: dict[str, Any] = dict(value)
        return cls(**arguments)


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"turn_controls.{name} must contain finite numbers")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"turn_controls.{name} must contain finite numbers") from exc
    if not math.isfinite(number):
        raise ValueError(f"turn_controls.{name} must contain finite numbers")
    return number
