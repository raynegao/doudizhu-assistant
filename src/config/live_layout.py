from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config.hand_empty import HandEmptyConfig
from src.config.hand_geometry import ShortHandRecoveryConfig
from src.config.hand_photometry import HandPhotometryConfig
from src.config.joker_color import JokerColorConfig
from src.config.self_table import SelfTableConfig
from src.config.table_crops import TableRankCropConfig
from src.config.turn_controls import TurnControlsConfig

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
    def from_value(cls, value: object) -> NormalizedBox:
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
    evidence_dir: Path = Path("data/live_game/evidence")
    fixed_issue_registry: Path = Path("configs/live_evidence_fixed_issues.json")
    evidence_enabled: bool = True
    evidence_max_total_bytes: int = 512 * 1024 * 1024
    evidence_max_sessions: int = 20
    evidence_retention_days: float = 30.0
    evidence_protect_recent_days: float = 7.0
    evidence_max_frames_per_session: int = 180
    error_frame_cooldown_seconds: float = 60.0
    max_error_frames_per_round: int = 4
    max_error_frames_per_session: int = 25
    interval_seconds: float = 0.10
    recording_queue_frames: int = 4
    recording_io_timeout_seconds: float = 10.0
    initial_stability_frames: int = 2
    stability_frames: int = 3
    confidence_threshold: float = 0.70
    pass_threshold: float = 0.82
    template_threshold: float = 0.78
    joker_color: JokerColorConfig = field(default_factory=JokerColorConfig)
    short_hand_recovery: ShortHandRecoveryConfig = field(default_factory=ShortHandRecoveryConfig)
    hand_photometry: HandPhotometryConfig = field(default_factory=HandPhotometryConfig)
    table_rank_crop: TableRankCropConfig = field(default_factory=TableRankCropConfig)
    self_table: SelfTableConfig = field(default_factory=SelfTableConfig)
    hand_empty: HandEmptyConfig = field(default_factory=HandEmptyConfig)
    turn_controls: TurnControlsConfig = field(default_factory=TurnControlsConfig)
    simulations: int = 512
    max_depth: int = 60
    time_budget_ms: int = 1500
    min_rollouts_per_action: int = 32
    top_k: int = 3
    max_candidates: int = 20
    # Measured 2026-10-18 (docs/DECISION_ARENA.md): successive halving on the
    # same 20 candidates raised the finalists' samples from 64 to ~100 at equal
    # cost and did not lose strength; the shaped rollout policy weakened the
    # landlord in simulation and stays optional.
    allocation: str = "successive_halving"
    rollout_policy: str = "heuristic"

    def __post_init__(self) -> None:
        if not isinstance(self.joker_color, JokerColorConfig):
            raise ValueError("joker_color must be a JokerColorConfig")
        if not isinstance(self.short_hand_recovery, ShortHandRecoveryConfig):
            raise ValueError("short_hand_recovery must be a ShortHandRecoveryConfig")
        if not isinstance(self.hand_photometry, HandPhotometryConfig):
            raise ValueError("hand_photometry must be a HandPhotometryConfig")
        if not isinstance(self.table_rank_crop, TableRankCropConfig):
            raise ValueError("table_rank_crop must be a TableRankCropConfig")
        if not isinstance(self.self_table, SelfTableConfig):
            raise ValueError("self_table must be a SelfTableConfig")
        if not isinstance(self.hand_empty, HandEmptyConfig):
            raise ValueError("hand_empty must be a HandEmptyConfig")
        if not isinstance(self.turn_controls, TurnControlsConfig):
            raise ValueError("turn_controls must be a TurnControlsConfig")
        missing = [name for name in REQUIRED_LIVE_ROIS if name not in self.rois]
        if missing:
            raise ValueError(f"live layout is missing required ROIs: {', '.join(missing)}")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if (
            isinstance(self.recording_queue_frames, bool)
            or not isinstance(self.recording_queue_frames, int)
            or self.recording_queue_frames <= 0
        ):
            raise ValueError("recording_queue_frames must be a positive integer")
        if (
            not math.isfinite(self.recording_io_timeout_seconds)
            or self.recording_io_timeout_seconds <= 0
        ):
            raise ValueError("recording_io_timeout_seconds must be finite and positive")
        if self.error_frame_cooldown_seconds < 0:
            raise ValueError("error_frame_cooldown_seconds cannot be negative")
        if self.max_error_frames_per_round <= 0:
            raise ValueError("max_error_frames_per_round must be positive")
        if self.max_error_frames_per_session <= 0:
            raise ValueError("max_error_frames_per_session must be positive")
        if self.max_error_frames_per_round > self.max_error_frames_per_session:
            raise ValueError(
                "max_error_frames_per_round cannot exceed "
                "max_error_frames_per_session"
            )
        if self.evidence_max_total_bytes <= 0:
            raise ValueError("evidence_max_total_bytes must be positive")
        if self.evidence_max_sessions <= 0:
            raise ValueError("evidence_max_sessions must be positive")
        if self.evidence_retention_days < 0:
            raise ValueError("evidence_retention_days cannot be negative")
        if self.evidence_protect_recent_days < 0:
            raise ValueError("evidence_protect_recent_days cannot be negative")
        if self.evidence_max_frames_per_session <= 0:
            raise ValueError("evidence_max_frames_per_session must be positive")
        if self.initial_stability_frames <= 0:
            raise ValueError("initial_stability_frames must be positive")
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
        if self.allocation not in ("uniform", "successive_halving"):
            raise ValueError("allocation must be 'uniform' or 'successive_halving'")
        if self.rollout_policy not in ("heuristic", "shaped"):
            raise ValueError("rollout_policy must be 'heuristic' or 'shaped'")

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
            "evidence_dir": self.evidence_dir.as_posix(),
            "fixed_issue_registry": self.fixed_issue_registry.as_posix(),
            "evidence_enabled": self.evidence_enabled,
            "evidence_max_total_bytes": self.evidence_max_total_bytes,
            "evidence_max_sessions": self.evidence_max_sessions,
            "evidence_retention_days": self.evidence_retention_days,
            "evidence_protect_recent_days": self.evidence_protect_recent_days,
            "evidence_max_frames_per_session": self.evidence_max_frames_per_session,
            "error_frame_cooldown_seconds": self.error_frame_cooldown_seconds,
            "max_error_frames_per_round": self.max_error_frames_per_round,
            "max_error_frames_per_session": self.max_error_frames_per_session,
            "interval_seconds": self.interval_seconds,
            "recording_queue_frames": self.recording_queue_frames,
            "recording_io_timeout_seconds": self.recording_io_timeout_seconds,
            "initial_stability_frames": self.initial_stability_frames,
            "stability_frames": self.stability_frames,
            "confidence_threshold": self.confidence_threshold,
            "pass_threshold": self.pass_threshold,
            "template_threshold": self.template_threshold,
            "joker_color": self.joker_color.to_payload(),
            "short_hand_recovery": self.short_hand_recovery.to_payload(),
            "hand_photometry": self.hand_photometry.to_payload(),
            "table_rank_crop": self.table_rank_crop.to_payload(),
            "self_table": self.self_table.to_payload(),
            "hand_empty": self.hand_empty.to_payload(),
            "turn_controls": self.turn_controls.to_payload(),
            "simulations": self.simulations,
            "max_depth": self.max_depth,
            "time_budget_ms": self.time_budget_ms,
            "min_rollouts_per_action": self.min_rollouts_per_action,
            "top_k": self.top_k,
            "max_candidates": self.max_candidates,
            "allocation": self.allocation,
            "rollout_policy": self.rollout_policy,
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
    # Older snapshots have no empty-domain setting. Only the exact measured
    # classic layout gets its known bottom UI band; custom layouts default to
    # zero exclusion. Resolution is in memory; never rewrite an old snapshot.
    empty_default = {"bottom_ui_fraction": 0.03} if (
        rois == _default_rois() and data.get("app_name", defaults.app_name) == defaults.app_name
    ) else {}
    return LiveLayoutConfig(
        app_name=str(data.get("app_name", defaults.app_name)),
        rois=rois,
        model_path=Path(str(data.get("model_path", defaults.model_path))),
        device_name=str(data.get("device_name", defaults.device_name)),
        templates_dir=Path(str(data.get("templates_dir", defaults.templates_dir))),
        log_file=Path(str(data.get("log_file", defaults.log_file))),
        error_frames_dir=Path(str(data.get("error_frames_dir", defaults.error_frames_dir))),
        evidence_dir=Path(str(data.get("evidence_dir", defaults.evidence_dir))),
        fixed_issue_registry=Path(
            str(data.get("fixed_issue_registry", defaults.fixed_issue_registry))
        ),
        evidence_enabled=_boolean_value(
            data,
            "evidence_enabled",
            defaults.evidence_enabled,
        ),
        evidence_max_total_bytes=int(str(
            data.get(
                "evidence_max_total_bytes",
                defaults.evidence_max_total_bytes,
            )
        )),
        evidence_max_sessions=int(str(
            data.get("evidence_max_sessions", defaults.evidence_max_sessions)
        )),
        evidence_retention_days=float(str(
            data.get("evidence_retention_days", defaults.evidence_retention_days)
        )),
        evidence_protect_recent_days=float(str(
            data.get(
                "evidence_protect_recent_days",
                defaults.evidence_protect_recent_days,
            )
        )),
        evidence_max_frames_per_session=int(str(
            data.get(
                "evidence_max_frames_per_session",
                defaults.evidence_max_frames_per_session,
            )
        )),
        error_frame_cooldown_seconds=float(str(
            data.get(
                "error_frame_cooldown_seconds",
                defaults.error_frame_cooldown_seconds,
            )
        )),
        max_error_frames_per_round=int(str(
            data.get(
                "max_error_frames_per_round",
                defaults.max_error_frames_per_round,
            )
        )),
        max_error_frames_per_session=int(str(
            data.get(
                "max_error_frames_per_session",
                defaults.max_error_frames_per_session,
            )
        )),
        interval_seconds=float(str(data.get("interval_seconds", defaults.interval_seconds))),
        recording_queue_frames=int(str(
            data.get("recording_queue_frames", defaults.recording_queue_frames)
        )),
        recording_io_timeout_seconds=float(str(
            data.get("recording_io_timeout_seconds", defaults.recording_io_timeout_seconds)
        )),
        initial_stability_frames=int(str(
            data.get(
                "initial_stability_frames",
                defaults.initial_stability_frames,
            )
        )),
        stability_frames=int(str(data.get("stability_frames", defaults.stability_frames))),
        confidence_threshold=float(str(
            data.get("confidence_threshold", defaults.confidence_threshold)
        )),
        pass_threshold=float(str(data.get("pass_threshold", defaults.pass_threshold))),
        template_threshold=float(str(
            data.get("template_threshold", defaults.template_threshold)
        )),
        joker_color=JokerColorConfig.from_dict(data.get("joker_color", {})),
        short_hand_recovery=ShortHandRecoveryConfig.from_dict(data.get("short_hand_recovery", {})),
        hand_photometry=HandPhotometryConfig.from_dict(data.get("hand_photometry", {})),
        table_rank_crop=TableRankCropConfig.from_dict(data.get("table_rank_crop", {})),
        self_table=SelfTableConfig.from_dict(data.get("self_table", {})),
        hand_empty=HandEmptyConfig.from_dict(data.get("hand_empty", empty_default)),
        turn_controls=TurnControlsConfig.from_dict(data.get("turn_controls", {})),
        simulations=int(str(data.get("simulations", defaults.simulations))),
        max_depth=int(str(data.get("max_depth", defaults.max_depth))),
        time_budget_ms=int(str(data.get("time_budget_ms", defaults.time_budget_ms))),
        min_rollouts_per_action=int(str(
            data.get("min_rollouts_per_action", defaults.min_rollouts_per_action)
        )),
        top_k=int(str(data.get("top_k", defaults.top_k))),
        max_candidates=int(str(data.get("max_candidates", defaults.max_candidates))),
        allocation=str(data.get("allocation", defaults.allocation)),
        rollout_policy=str(data.get("rollout_policy", defaults.rollout_policy)),
    )


def load_live_layout(path: Path) -> LiveLayoutConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("live layout root must be an object")
    return live_layout_from_dict(data)


def _boolean_value(
    data: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


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
