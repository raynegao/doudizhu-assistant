from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class TableRankCropConfig:
    """Classifier input geometry for full card bodies, never a rank decision."""

    body_height_ratio: float = 0.535
    width_height_ratio: float = 0.60
    visible_corner_height_ratio: float = 0.68
    minimum_height: int = 24
    minimum_width: int = 16

    def __post_init__(self) -> None:
        for name in ("body_height_ratio", "width_height_ratio", "visible_corner_height_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value <= 1
                or not math.isfinite(value)
            ):
                raise ValueError(f"table_rank_crop.{name} must be finite and in (0, 1]")
        for name in ("minimum_height", "minimum_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"table_rank_crop.{name} must be a positive integer")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> TableRankCropConfig:
        if not isinstance(value, Mapping):
            raise ValueError("table_rank_crop must be an object")
        if set(value) - {item.name for item in fields(cls)}:
            raise ValueError("table_rank_crop contains unknown fields")
        arguments: dict[str, Any] = dict(value)
        return cls(**arguments)
