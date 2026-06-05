from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_APP_NAME = "斗地主"
DEFAULT_WINDOW_OFFSET = (33, 473)
DEFAULT_ROI_HEIGHT = 203
DEFAULT_CALIBRATED_COUNT = 17
DEFAULT_CALIBRATED_START_X = 0
DEFAULT_CALIBRATED_START_Y = 10
DEFAULT_CALIBRATED_STEP_X = 60
DEFAULT_CALIBRATED_CROP_SIZE = (63, 105)


@dataclass(frozen=True)
class WindowInfo:
    app_name: str
    window_name: str
    window_box: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.window_box[2] - self.window_box[0]

    @property
    def height(self) -> int:
        return self.window_box[3] - self.window_box[1]


class WindowLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class RoiCalibration:
    window_name: str
    window_box: tuple[int, int, int, int]
    roi_box: tuple[int, int, int, int]
    count: int
    start_x: int
    start_y: int
    step_x: int
    crop_size: tuple[int, int]
    created_at: float
    app_name: str = DEFAULT_APP_NAME

    def to_runtime_config(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "window_name": self.window_name,
            "window_box": list(self.window_box),
            "roi_box": list(self.roi_box),
            "count": self.count,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "step_x": self.step_x,
            "crop_size": list(self.crop_size),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    roi_box: tuple[int, int, int, int] | None = None
    count: int | None = None
    start_x: int | None = None
    start_y: int | None = None
    step_x: int | None = None
    crop_size: tuple[int, int] | None = None
    app_name: str | None = None
    window_name: str | None = None
    window_box: tuple[int, int, int, int] | None = None
    model_path: Path | None = None
    device_name: str | None = None
    confidence_threshold: float | None = None
    last_play: str | None = None
    log_file: Path | None = None
    stability_window: int | None = None


def find_window(app_name: str = DEFAULT_APP_NAME, runner=subprocess.run) -> WindowInfo:
    script = (
        f'tell application "System Events" to tell process "{app_name}" '
        'to get {position, size, name} of first window'
    )
    try:
        result = runner(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise WindowLookupError(f"Cannot find a visible macOS window for app '{app_name}'. {detail}") from exc
    return parse_window_info(result.stdout, app_name=app_name)


def parse_window_info(output: str, app_name: str = DEFAULT_APP_NAME) -> WindowInfo:
    parts = [part.strip() for part in output.replace("\n", "").split(",")]
    if len(parts) < 5:
        raise ValueError(f"Cannot parse window info: {output!r}")
    try:
        left = int(float(parts[0]))
        top = int(float(parts[1]))
        width = int(float(parts[2]))
        height = int(float(parts[3]))
    except ValueError as exc:
        raise ValueError(f"Cannot parse window geometry: {output!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Window size must be positive: {output!r}")
    window_name = ",".join(parts[4:]).strip() or app_name
    return WindowInfo(
        app_name=app_name,
        window_name=window_name,
        window_box=(left, top, left + width, top + height),
    )


def calibrate_roi(
    window: WindowInfo,
    offset: tuple[int, int] = DEFAULT_WINDOW_OFFSET,
    roi_height: int = DEFAULT_ROI_HEIGHT,
    count: int = DEFAULT_CALIBRATED_COUNT,
    start_x: int = DEFAULT_CALIBRATED_START_X,
    start_y: int = DEFAULT_CALIBRATED_START_Y,
    step_x: int = DEFAULT_CALIBRATED_STEP_X,
    crop_size: tuple[int, int] = DEFAULT_CALIBRATED_CROP_SIZE,
    created_at: float | None = None,
) -> RoiCalibration:
    left, top, right, bottom = window.window_box
    roi_left = left + offset[0]
    roi_top = top + offset[1]
    roi_right = right
    roi_bottom = roi_top + roi_height
    if roi_left >= roi_right or roi_top >= roi_bottom or roi_bottom > bottom:
        raise ValueError(
            f"Calibrated ROI is outside window bounds: roi={(roi_left, roi_top, roi_right, roi_bottom)}, "
            f"window={window.window_box}"
        )
    return RoiCalibration(
        app_name=window.app_name,
        window_name=window.window_name,
        window_box=window.window_box,
        roi_box=(roi_left, roi_top, roi_right, roi_bottom),
        count=count,
        start_x=start_x,
        start_y=start_y,
        step_x=step_x,
        crop_size=crop_size,
        created_at=time.time() if created_at is None else created_at,
    )


def save_runtime_config(path: Path, calibration: RoiCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration.to_runtime_config(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_runtime_config(path: Path) -> RuntimeConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config root must be an object: {path}")
    return runtime_config_from_dict(data)


def runtime_config_from_dict(data: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        roi_box=_optional_roi_box(data.get("roi_box")),
        count=_optional_int(data.get("count")),
        start_x=_optional_int(data.get("start_x")),
        start_y=_optional_int(data.get("start_y")),
        step_x=_optional_int(data.get("step_x")),
        crop_size=_optional_crop_size(data.get("crop_size")),
        app_name=_optional_str(data.get("app_name")),
        window_name=_optional_str(data.get("window_name")),
        window_box=_optional_roi_box(data.get("window_box")),
        model_path=_optional_path(data.get("model_path")),
        device_name=_optional_str(data.get("device_name")),
        confidence_threshold=_optional_float(data.get("confidence_threshold")),
        last_play=_optional_str(data.get("last_play")),
        log_file=_optional_path(data.get("log_file")),
        stability_window=_optional_int(data.get("stability_window")),
    )


def _optional_roi_box(value: object) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_box(value)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(item) for item in value)  # type: ignore[return-value]
    raise ValueError(f"ROI/window box must be a string or four-item list: {value!r}")


def _optional_crop_size(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_crop_size(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = (int(item) for item in value)
        return width, height
    raise ValueError(f"Crop size must be a string or two-item list: {value!r}")


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(str(value))


def _parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Box must be left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise ValueError("Box must have right > left and bottom > top")
    return left, top, right, bottom


def _parse_crop_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise ValueError("Crop size must be WIDTHxHEIGHT")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)
