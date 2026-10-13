from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PIL import Image

from src.config.table_crops import TableRankCropConfig


class CardRegionKind(str, Enum):
    FULL_CARD = "full_card"
    VISIBLE_CORNER = "visible_corner"
    RANK_CROP = "rank_crop"


@dataclass(frozen=True)
class CardRegion:
    """Pixel geometry from segmentation, without predicted cards or state.

    Full bodies and overlapping visible corners have different established
    preprocessing contracts. RANK_CROP alone denotes a final model input.
    """

    box: tuple[int, int, int, int]
    kind: CardRegionKind

    def __post_init__(self) -> None:
        if (
            not isinstance(self.box, tuple)
            or len(self.box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.box)
            or not isinstance(self.kind, CardRegionKind)
        ):
            raise ValueError("card region requires a four-integer box and explicit geometry kind")
        left, top, right, bottom = self.box
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("card region must have a positive in-image extent")

    def classifier_box(self, config: TableRankCropConfig) -> tuple[int, int, int, int]:
        if not isinstance(config, TableRankCropConfig):
            raise ValueError("classifier crop requires TableRankCropConfig")
        if self.kind is CardRegionKind.RANK_CROP:
            return self.box
        left, top, right, bottom = self.box
        if self.kind is CardRegionKind.VISIBLE_CORNER:
            # Preserve the existing overlap input distribution. A visible
            # corner is an observation/reference ROI, not a final model crop.
            height = max(1, round((bottom - top) * config.visible_corner_height_ratio))
            return left, top, right, top + height
        height = min(bottom - top, max(config.minimum_height,
                                      round((bottom - top) * config.body_height_ratio)))
        width = min(right - left, max(config.minimum_width,
                                     round(height * config.width_height_ratio)))
        return left, top, left + width, top + height

    def classifier_crop(self, image: Image.Image, config: TableRankCropConfig) -> Image.Image:
        if self.box[2] > image.width or self.box[3] > image.height:
            raise ValueError("card region lies outside source image")
        return image.crop(self.classifier_box(config))
