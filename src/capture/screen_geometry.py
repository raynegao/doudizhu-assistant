from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from PIL import Image, ImageGrab

from src.pipeline.calibration import WindowInfo, find_window


class ScreenGeometryError(RuntimeError):
    """Raised when logical macOS coordinates cannot be mapped to screenshot pixels."""


class WindowCaptureStatus(str, Enum):
    AVAILABLE = "available"
    NOT_OPEN = "not_open"
    MINIMIZED = "minimized"
    CAPTURE_ERROR = "capture_error"


class WindowAvailabilityError(ScreenGeometryError):
    def __init__(self, status: WindowCaptureStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ScreenGeometry:
    logical_size: tuple[int, int]
    pixel_size: tuple[int, int]

    def __post_init__(self) -> None:
        logical_width, logical_height = self.logical_size
        pixel_width, pixel_height = self.pixel_size
        if min(logical_width, logical_height, pixel_width, pixel_height) <= 0:
            raise ValueError("screen dimensions must be positive")

    @property
    def scale_x(self) -> float:
        return self.pixel_size[0] / self.logical_size[0]

    @property
    def scale_y(self) -> float:
        return self.pixel_size[1] / self.logical_size[1]

    def logical_to_pixel_box(
        self,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            raise ValueError("logical box must have positive width and height")
        pixel_box = (
            round(left * self.scale_x),
            round(top * self.scale_y),
            round(right * self.scale_x),
            round(bottom * self.scale_y),
        )
        pixel_left, pixel_top, pixel_right, pixel_bottom = pixel_box
        if (
            pixel_left < 0
            or pixel_top < 0
            or pixel_right > self.pixel_size[0]
            or pixel_bottom > self.pixel_size[1]
        ):
            raise ScreenGeometryError(
                f"window pixel box {pixel_box} is outside main screen {self.pixel_size}"
            )
        return pixel_box


@dataclass(frozen=True)
class CapturedWindow:
    frame_id: int
    timestamp: float
    image: Image.Image
    window: WindowInfo
    pixel_box: tuple[int, int, int, int]
    geometry: ScreenGeometry


@dataclass(frozen=True)
class WindowServerInfo:
    window_id: int
    window: WindowInfo
    is_onscreen: bool = True


_WINDOW_LIST_SWIFT = r"""
import CoreGraphics
import Foundation

let targetOwner = ProcessInfo.processInfo.environment["LIVE_CAPTURE_APP_NAME"] ?? ""
let options: CGWindowListOption = [.optionAll, .excludeDesktopElements]
let windows = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
    as? [[String: Any]] ?? []
for window in windows {
    let owner = window[kCGWindowOwnerName as String] as? String ?? ""
    if owner != targetOwner { continue }
    let id = window[kCGWindowNumber as String] as? Int ?? 0
    let name = window[kCGWindowName as String] as? String ?? ""
    let layer = window[kCGWindowLayer as String] as? Int ?? 0
    let isOnscreen = window[kCGWindowIsOnscreen as String] as? Bool ?? false
    guard let bounds = window[kCGWindowBounds as String] as? [String: Any],
          let x = bounds["X"] as? Double,
          let y = bounds["Y"] as? Double,
          let width = bounds["Width"] as? Double,
          let height = bounds["Height"] as? Double else {
        continue
    }
    print("\(id)\t\(x)\t\(y)\t\(width)\t\(height)\t\(layer)\t\(isOnscreen ? 1 : 0)\t\(name)")
}
"""


def parse_window_server_candidates(
    output: str,
    *,
    app_name: str,
) -> list[tuple[WindowServerInfo, int]]:
    candidates: list[tuple[WindowServerInfo, int]] = []
    for line in output.splitlines():
        parts = line.split("\t", 7)
        if len(parts) < 7:
            continue
        try:
            window_id = int(parts[0])
            left, top, width, height = (
                int(round(float(part))) for part in parts[1:5]
            )
            layer = int(parts[5])
        except ValueError:
            continue
        if len(parts) >= 8:
            is_onscreen = parts[6] == "1"
            window_name = parts[7]
        else:
            is_onscreen = True
            window_name = parts[6]
        if window_id <= 0 or width <= 0 or height <= 0:
            continue
        candidates.append(
            (
                WindowServerInfo(
                    window_id=window_id,
                    window=WindowInfo(
                        app_name=app_name,
                        window_name=window_name.strip() or app_name,
                        window_box=(left, top, left + width, top + height),
                    ),
                    is_onscreen=is_onscreen,
                ),
                layer,
            )
        )
    return candidates


def find_window_server_window(
    app_name: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WindowServerInfo:
    environment = dict(os.environ)
    environment["LIVE_CAPTURE_APP_NAME"] = app_name
    try:
        result = runner(
            ["/usr/bin/swift", "-e", _WINDOW_LIST_SWIFT],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ScreenGeometryError(
            f"cannot query macOS WindowServer for app '{app_name}': "
            f"{str(detail).strip()}"
        ) from exc
    candidates = parse_window_server_candidates(result.stdout, app_name=app_name)
    if not candidates:
        raise WindowAvailabilityError(
            WindowCaptureStatus.NOT_OPEN,
            f"未检测到“{app_name}”窗口，请打开斗地主",
        )
    layer_zero = [info for info, layer in candidates if layer == 0]
    selectable = layer_zero or [info for info, _ in candidates]
    return max(
        selectable,
        key=lambda info: info.window.width * info.window.height,
    )


def grab_window_by_id(
    window_id: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="live-window-capture-") as temp_dir:
        output_path = Path(temp_dir) / "window.png"
        try:
            runner(
                [
                    "/usr/sbin/screencapture",
                    "-x",
                    "-o",
                    f"-l{window_id}",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with Image.open(output_path) as image:
                return image.convert("RGB")
        except (OSError, subprocess.CalledProcessError, Image.UnidentifiedImageError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise ScreenGeometryError(
                f"window-level capture failed for window id {window_id}: "
                f"{str(detail).strip()}"
            ) from exc


def parse_desktop_bounds(output: str) -> tuple[int, int]:
    parts = [part.strip() for part in output.replace("\n", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"cannot parse desktop bounds: {output!r}")
    left, top, right, bottom = (int(float(part)) for part in parts)
    width = right - left
    height = bottom - top
    if left != 0 or top != 0 or width <= 0 or height <= 0:
        raise ValueError(f"unsupported desktop bounds: {output!r}")
    return width, height


def query_desktop_logical_size(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[int, int]:
    try:
        result = runner(
            [
                "osascript",
                "-e",
                'tell application "Finder" to get bounds of window of desktop',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ScreenGeometryError(
            f"cannot query macOS desktop bounds: {str(detail).strip()}"
        ) from exc
    return parse_desktop_bounds(result.stdout)


def detect_screen_geometry(
    *,
    grabber: Callable[..., Image.Image] = ImageGrab.grab,
    logical_size_provider: Callable[[], tuple[int, int]] = query_desktop_logical_size,
) -> ScreenGeometry:
    full_screen = grabber()
    return ScreenGeometry(
        logical_size=logical_size_provider(),
        pixel_size=full_screen.size,
    )


class MacWindowCapture:
    """Capture a macOS window with Retina-aware coordinate conversion."""

    def __init__(
        self,
        app_name: str = "斗地主",
        *,
        window_finder: Callable[[str], WindowInfo] = find_window,
        geometry_provider: Callable[[], ScreenGeometry] = detect_screen_geometry,
        grabber: Callable[..., Image.Image] = ImageGrab.grab,
        window_server_finder: Callable[[str], WindowServerInfo] = (
            find_window_server_window
        ),
        window_grabber: Callable[[int], Image.Image] = grab_window_by_id,
        prefer_window_level: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.app_name = app_name
        self._window_finder = window_finder
        self._geometry_provider = geometry_provider
        self._grabber = grabber
        self._window_server_finder = window_server_finder
        self._window_grabber = window_grabber
        self._prefer_window_level = prefer_window_level
        self._clock = clock
        self._geometry: ScreenGeometry | None = None
        self._window_server_info: WindowServerInfo | None = None

    def capture(self, frame_id: int) -> CapturedWindow:
        if self._prefer_window_level:
            return self._capture_window_level(frame_id)
        return self._capture_screen_crop(frame_id)

    def _capture_window_level(self, frame_id: int) -> CapturedWindow:
        info = self._window_server_finder(self.app_name)
        self._window_server_info = info
        if not info.is_onscreen:
            raise WindowAvailabilityError(
                WindowCaptureStatus.MINIMIZED,
                "斗地主窗口已最小化，当前无法识别；请还原窗口",
            )
        if self._geometry is None:
            self._geometry = self._geometry_provider()
        geometry = self._geometry
        pixel_box = geometry.logical_to_pixel_box(info.window.window_box)
        try:
            image = self._window_grabber(info.window_id).convert("RGB")
        except ScreenGeometryError:
            self._window_server_info = self._window_server_finder(self.app_name)
            info = self._window_server_info
            if not info.is_onscreen:
                raise WindowAvailabilityError(
                    WindowCaptureStatus.MINIMIZED,
                    "斗地主窗口已最小化，当前无法识别；请还原窗口",
                )
            pixel_box = geometry.logical_to_pixel_box(info.window.window_box)
            image = self._window_grabber(info.window_id).convert("RGB")
        expected_size = (
            pixel_box[2] - pixel_box[0],
            pixel_box[3] - pixel_box[1],
        )
        if image.size != expected_size:
            refreshed = self._window_server_finder(self.app_name)
            self._window_server_info = refreshed
            info = refreshed
            if not info.is_onscreen:
                raise WindowAvailabilityError(
                    WindowCaptureStatus.MINIMIZED,
                    "斗地主窗口已最小化，当前无法识别；请还原窗口",
                )
            pixel_box = geometry.logical_to_pixel_box(info.window.window_box)
            expected_size = (
                pixel_box[2] - pixel_box[0],
                pixel_box[3] - pixel_box[1],
            )
            if image.size != expected_size:
                raise ScreenGeometryError(
                    f"window-level image size {image.size} does not match "
                    f"expected {expected_size}"
                )
        return CapturedWindow(
            frame_id=frame_id,
            timestamp=self._clock(),
            image=image,
            window=info.window,
            pixel_box=pixel_box,
            geometry=geometry,
        )

    def _capture_screen_crop(self, frame_id: int) -> CapturedWindow:
        window = self._window_finder(self.app_name)
        if self._geometry is None:
            self._geometry = self._geometry_provider()
        geometry = self._geometry
        pixel_box = geometry.logical_to_pixel_box(window.window_box)
        try:
            image = self._grabber(bbox=pixel_box).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ScreenGeometryError(
                "Mac window capture failed. Grant Screen Recording permission, keep "
                f"'{self.app_name}' visible, and do not minimize it. Original error: {exc}"
            ) from exc
        expected_size = (
            pixel_box[2] - pixel_box[0],
            pixel_box[3] - pixel_box[1],
        )
        if image.size != expected_size:
            raise ScreenGeometryError(
                f"captured window size {image.size} does not match expected {expected_size}"
            )
        return CapturedWindow(
            frame_id=frame_id,
            timestamp=self._clock(),
            image=image,
            window=window,
            pixel_box=pixel_box,
            geometry=geometry,
        )


__all__ = [
    "CapturedWindow",
    "MacWindowCapture",
    "ScreenGeometry",
    "ScreenGeometryError",
    "WindowAvailabilityError",
    "WindowCaptureStatus",
    "WindowServerInfo",
    "detect_screen_geometry",
    "find_window_server_window",
    "grab_window_by_id",
    "parse_desktop_bounds",
    "parse_window_server_candidates",
    "query_desktop_logical_size",
]
