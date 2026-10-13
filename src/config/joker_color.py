from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class JokerColorConfig:
    """Conservative RGB support gates, not calibrated class probabilities."""

    min_component_pixels: int = 40
    min_color_fraction: float = 0.90
    red_min_channel: int = 90
    red_channel_margin: int = 40
    black_max_channel: int = 100
    black_max_chroma: int = 30
    background_min_channel: int = 185
    background_max_chroma: int = 40
    min_background_fraction: float = 0.70
    context_margin: int = 2

    def __post_init__(self) -> None:
        for name in (
            "min_component_pixels", "red_min_channel", "red_channel_margin",
            "black_max_channel", "black_max_chroma", "background_min_channel",
            "background_max_chroma", "context_margin",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"joker_color.{name} must be an integer")
        if not 20 <= self.min_component_pixels <= 4096:
            raise ValueError("joker_color.min_component_pixels must be between 20 and 4096")
        if not 1 <= self.context_margin <= 8:
            raise ValueError("joker_color.context_margin must be between 1 and 8")
        for name in (
            "red_min_channel", "red_channel_margin", "black_max_channel",
            "black_max_chroma", "background_min_channel", "background_max_chroma",
        ):
            if not 0 <= getattr(self, name) <= 255:
                raise ValueError(f"joker_color.{name} must be between 0 and 255")
        for name in ("min_color_fraction", "min_background_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.5 < value <= 1.0
            ):
                raise ValueError(f"joker_color.{name} must be finite and in (0.5, 1]")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> JokerColorConfig:
        if not isinstance(value, Mapping):
            raise ValueError("joker_color must be an object")
        names = {item.name for item in fields(cls)}
        if set(value) - names:
            raise ValueError("joker_color contains unknown fields")
        defaults = cls()

        def integer(name: str) -> int:
            raw = value.get(name, getattr(defaults, name))
            if isinstance(raw, bool):
                raise ValueError(f"joker_color.{name} must be an integer")
            return int(str(raw))

        def fraction(name: str) -> float:
            raw = value.get(name, getattr(defaults, name))
            if isinstance(raw, bool):
                raise ValueError(f"joker_color.{name} must be a fraction")
            return float(str(raw))

        return cls(
            min_component_pixels=integer("min_component_pixels"),
            min_color_fraction=fraction("min_color_fraction"),
            red_min_channel=integer("red_min_channel"),
            red_channel_margin=integer("red_channel_margin"),
            black_max_channel=integer("black_max_channel"),
            black_max_chroma=integer("black_max_chroma"),
            background_min_channel=integer("background_min_channel"),
            background_max_chroma=integer("background_max_chroma"),
            min_background_fraction=fraction("min_background_fraction"),
            context_margin=integer("context_margin"),
        )
