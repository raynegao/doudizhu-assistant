from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class SelfTableConfig:
    """Paper-panel evidence limits; never inferred from expected cards."""

    paper_min_channel: int = 180
    paper_max_chroma: int = 35
    minimum_column_fraction: float = 0.15
    minimum_row_fraction: float = 0.25
    minimum_height_fraction: float = 0.25
    minimum_paper_fraction: float = 0.60
    maximum_input_pixels: int = 1_500_000

    def __post_init__(self) -> None:
        for name in ("paper_min_channel", "paper_max_chroma", "maximum_input_pixels"):
            value = getattr(self, name)
            upper = 255 if name != "maximum_input_pixels" else 10_000_000
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= upper:
                raise ValueError(f"self_table.{name} must be a positive bounded integer")
        for name in ("minimum_column_fraction", "minimum_row_fraction",
                     "minimum_height_fraction", "minimum_paper_fraction"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"self_table.{name} must be finite and in (0, 1]")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> SelfTableConfig:
        if not isinstance(value, Mapping) or set(value) - {item.name for item in fields(cls)}:
            raise ValueError("self_table must be an object with known fields")
        arguments: dict[str, Any] = dict(value)
        return cls(**arguments)
