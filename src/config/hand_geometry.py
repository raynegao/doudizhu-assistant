from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ShortHandRecoveryConfig:
    """Validated hand-only recovery geometry, never an expected card count."""

    maximum_slots: int = 3
    fan_aspect: tuple[float, float] = (0.40, 1.60)
    final_aspect: tuple[float, float] = (0.40, 0.90)
    rank_crop_ratios: tuple[float, float] = (0.535, 0.60)
    lower_band: tuple[float, float] = (0.86, 0.96)
    minimum_difference: float = 8.0
    minimum_edge_support: float = 0.70
    corner_depth_ratio: float = 0.08
    minimum_corner_support: float = 0.85
    corner_bend_ratio: float = 0.006
    alignment_ratio: float = 0.02

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_slots, bool)
            or not isinstance(self.maximum_slots, int)
            or not 1 <= self.maximum_slots <= 3
        ):
            raise ValueError("short_hand_recovery.maximum_slots must be an integer in [1, 3]")
        for name, upper, ordered, allow_zero in (
            ("fan_aspect", 3.0, True, False),
            ("final_aspect", 1.0, True, False),
            ("rank_crop_ratios", 1.0, False, False),
            ("lower_band", 1.0, True, True),
        ):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(f"short_hand_recovery.{name} must be a two-item interval")
            pair = tuple(_finite_number(name, item) for item in value)
            if any(item < 0 if allow_zero else item <= 0 for item in pair):
                raise ValueError(f"short_hand_recovery.{name} has an invalid lower bound")
            if any(item > upper for item in pair) or (ordered and pair[0] >= pair[1]):
                raise ValueError(f"short_hand_recovery.{name} has an invalid interval")
            # JSON lists cannot remain mutable inside a frozen configuration.
            object.__setattr__(self, name, pair)
        for name in (
            "minimum_difference",
            "minimum_edge_support",
            "corner_depth_ratio",
            "minimum_corner_support",
            "corner_bend_ratio",
            "alignment_ratio",
        ):
            number = _finite_number(name, getattr(self, name))
            upper = 255.0 if name == "minimum_difference" else 1.0
            if not 0 < number <= upper:
                raise ValueError(f"short_hand_recovery.{name} must be in (0, {upper}]")
            object.__setattr__(self, name, number)
        if self.corner_bend_ratio >= self.corner_depth_ratio:
            raise ValueError(
                "short_hand_recovery.corner_bend_ratio must be below corner_depth_ratio"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, value: object) -> ShortHandRecoveryConfig:
        if not isinstance(value, Mapping):
            raise ValueError("short_hand_recovery must be an object")
        names = {item.name for item in fields(cls)}
        if set(value) - names:
            raise ValueError("short_hand_recovery contains unknown fields")
        # The dataclass validates every supplied value without bool/string
        # coercion; omission alone chooses a default.
        arguments: dict[str, Any] = dict(value)
        return cls(**arguments)


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"short_hand_recovery.{name} must contain finite numbers")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"short_hand_recovery.{name} must contain finite numbers") from exc
    if not math.isfinite(number):
        raise ValueError(f"short_hand_recovery.{name} must contain finite numbers")
    return number
