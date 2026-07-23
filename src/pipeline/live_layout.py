from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


REQUIRED_LIVE_ROIS: tuple[str, ...] = (
    "self_hand",
    "self_play",
    "left_play",
    "right_play",
    "self_pass",
    "left_pass",
    "right_pass",
    "self_remaining",
    "left_remaining",
    "right_remaining",
    "self_role",
    "left_role",
    "right_role",
    "self_turn",
)


@dataclass(frozen=True)
class NormalizedBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not all(0.0 <= value <= 1.0 for value in self.to_tuple()):
            raise ValueError("normalized box values must be between 0 and 1")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("normalized box must have positive width and height")

    @classmethod
    def from_value(cls, value: object) -> "NormalizedBox":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"normalized ROI must be a four-item list: {value!r}")
        return cls(*(float(item) for item in value))

    def to_tuple(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def to_pixel_box(self, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = image_size
        box = (
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )
        left, top, right, bottom = box
        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValueError(f"ROI {box} is outside image size {image_size}")
        return box


def _default_rois() -> dict[str, NormalizedBox]:
    # Defaults are measured from the current macOS classic-layout screenshots.
    return {
        "self_hand": NormalizedBox(0.025, 0.625, 0.975, 0.895),
        "self_play": NormalizedBox(0.300, 0.330, 0.700, 0.590),
        "left_play": NormalizedBox(0.145, 0.105, 0.475, 0.430),
        "right_play": NormalizedBox(0.525, 0.105, 0.855, 0.430),
        "self_pass": NormalizedBox(0.420, 0.350, 0.580, 0.470),
        "left_pass": NormalizedBox(0.190, 0.105, 0.330, 0.225),
        "right_pass": NormalizedBox(0.700, 0.105, 0.840, 0.225),
        "self_remaining": NormalizedBox(0.035, 0.520, 0.145, 0.610),
        "left_remaining": NormalizedBox(0.100, 0.090, 0.155, 0.210),
        "right_remaining": NormalizedBox(0.845, 0.090, 0.900, 0.210),
        "self_role": NormalizedBox(0.020, 0.515, 0.115, 0.590),
        "left_role": NormalizedBox(0.020, 0.250, 0.115, 0.325),
        "right_role": NormalizedBox(0.885, 0.250, 0.985, 0.325),
        "self_turn": NormalizedBox(0.395, 0.385, 0.605, 0.540),
    }


@dataclass(frozen=True)
class LiveLayoutConfig:
    app_name: str = "斗地主"
    rois: Mapping[str, NormalizedBox] = field(default_factory=_default_rois)
    model_path: Path = Path("models/card_cnn.pt")
    device_name: str = "auto"
    templates_dir: Path = Path("data/live_game/templates")
    log_file: Path = Path("logs/live_assistant.jsonl")
    error_frames_dir: Path = Path("data/live_game/errors")
    interval_seconds: float = 0.25
    stability_frames: int = 3
    confidence_threshold: float = 0.70
    pass_threshold: float = 0.82
    template_threshold: float = 0.78
    simulations: int = 512
    max_depth: int = 60
    time_budget_ms: int = 1500
    min_rollouts_per_action: int = 32
    top_k: int = 3
    max_candidates: int = 20

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED_LIVE_ROIS if name not in self.rois]
        if missing:
            raise ValueError(f"live layout is missing required ROIs: {', '.join(missing)}")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.stability_frames <= 0:
            raise ValueError("stability_frames must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0 and 1")
        if not 0.0 <= self.template_threshold <= 1.0:
            raise ValueError("template_threshold must be between 0 and 1")
        if self.top_k <= 0 or self.max_candidates < self.top_k:
            raise ValueError("max_candidates must be at least top_k")

    def roi(self, name: str) -> NormalizedBox:
        try:
            return self.rois[name]
        except KeyError as exc:
            raise KeyError(f"unknown live ROI: {name}") from exc

    def crop(self, image: Image.Image, name: str) -> Image.Image:
        return image.crop(self.roi(name).to_pixel_box(image.size))

    def to_payload(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "rois": {
                name: list(box.to_tuple())
                for name, box in sorted(self.rois.items())
            },
            "model_path": self.model_path.as_posix(),
            "device_name": self.device_name,
            "templates_dir": self.templates_dir.as_posix(),
            "log_file": self.log_file.as_posix(),
            "error_frames_dir": self.error_frames_dir.as_posix(),
            "interval_seconds": self.interval_seconds,
            "stability_frames": self.stability_frames,
            "confidence_threshold": self.confidence_threshold,
            "pass_threshold": self.pass_threshold,
            "template_threshold": self.template_threshold,
            "simulations": self.simulations,
            "max_depth": self.max_depth,
            "time_budget_ms": self.time_budget_ms,
            "min_rollouts_per_action": self.min_rollouts_per_action,
            "top_k": self.top_k,
            "max_candidates": self.max_candidates,
        }


def live_layout_from_dict(data: Mapping[str, object]) -> LiveLayoutConfig:
    raw_rois = data.get("rois", {})
    if not isinstance(raw_rois, Mapping):
        raise ValueError("live layout rois must be an object")
    rois = _default_rois()
    rois.update({
        str(name): NormalizedBox.from_value(value)
        for name, value in raw_rois.items()
    })
    defaults = LiveLayoutConfig()
    return LiveLayoutConfig(
        app_name=str(data.get("app_name", defaults.app_name)),
        rois=rois,
        model_path=Path(str(data.get("model_path", defaults.model_path))),
        device_name=str(data.get("device_name", defaults.device_name)),
        templates_dir=Path(str(data.get("templates_dir", defaults.templates_dir))),
        log_file=Path(str(data.get("log_file", defaults.log_file))),
        error_frames_dir=Path(str(data.get("error_frames_dir", defaults.error_frames_dir))),
        interval_seconds=float(data.get("interval_seconds", defaults.interval_seconds)),
        stability_frames=int(data.get("stability_frames", defaults.stability_frames)),
        confidence_threshold=float(data.get("confidence_threshold", defaults.confidence_threshold)),
        pass_threshold=float(data.get("pass_threshold", defaults.pass_threshold)),
        template_threshold=float(data.get("template_threshold", defaults.template_threshold)),
        simulations=int(data.get("simulations", defaults.simulations)),
        max_depth=int(data.get("max_depth", defaults.max_depth)),
        time_budget_ms=int(data.get("time_budget_ms", defaults.time_budget_ms)),
        min_rollouts_per_action=int(
            data.get("min_rollouts_per_action", defaults.min_rollouts_per_action)
        ),
        top_k=int(data.get("top_k", defaults.top_k)),
        max_candidates=int(data.get("max_candidates", defaults.max_candidates)),
    )


def load_live_layout(path: Path) -> LiveLayoutConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("live layout root must be an object")
    return live_layout_from_dict(data)


def save_live_layout(path: Path, config: LiveLayoutConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_layout_preview(image: Image.Image, config: LiveLayoutConfig) -> Image.Image:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    colors = ("#ff4d4f", "#40a9ff", "#73d13d", "#faad14", "#9254de")
    for index, (name, roi) in enumerate(sorted(config.rois.items())):
        box = roi.to_pixel_box(preview.size)
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=max(2, preview.width // 800))
        label_box = draw.textbbox((box[0], box[1]), name, font=font)
        draw.rectangle(label_box, fill="black")
        draw.text((box[0], box[1]), name, fill=color, font=font)
    return preview


def render_roi_contact_sheet(
    image: Image.Image,
    config: LiveLayoutConfig,
    *,
    cell_size: tuple[int, int] = (320, 180),
) -> Image.Image:
    names = sorted(config.rois)
    columns = 3
    rows = (len(names) + columns - 1) // columns
    label_height = 24
    sheet = Image.new(
        "RGB",
        (columns * cell_size[0], rows * (cell_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, name in enumerate(names):
        column = index % columns
        row = index // columns
        x = column * cell_size[0]
        y = row * (cell_size[1] + label_height)
        crop = config.crop(image, name)
        fitted = ImageOps.contain(crop, cell_size)
        sheet.paste(
            fitted,
            (
                x + (cell_size[0] - fitted.width) // 2,
                y + label_height + (cell_size[1] - fitted.height) // 2,
            ),
        )
        draw.text((x + 4, y + 4), name, fill="black")
    return sheet


__all__ = [
    "LiveLayoutConfig",
    "NormalizedBox",
    "REQUIRED_LIVE_ROIS",
    "live_layout_from_dict",
    "load_live_layout",
    "render_layout_preview",
    "render_roi_contact_sheet",
    "save_live_layout",
]
