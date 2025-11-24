"""
Screen capture helpers for collecting training data and live frames.

Usage:
1. 运行脚本后，按 `S` 或 `Enter` 截图（按 `Q` 退出）。
2. 截图会保存到 `data/raw_screenshots/`，文件名形如 `screenshot_YYYYMMDD_HHMMSS.png`。
3. 保存成功后终端会提示保存路径。
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mss
from mss.base import ScreenShot
from mss import tools

try:
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover
    msvcrt = None

OUTPUT_DIR = Path("data/raw_screenshots")
DEFAULT_REGION: Dict[str, int] = {
    "top": 42,
    "left": 152,
    "width": 1620,
    "height": 946,
}


def capture_screen_region(region: Optional[Dict[str, int]] = None) -> ScreenShot:
    """
    Capture a screenshot of the specified region and return the raw ScreenShot object.
    """

    with mss.mss() as sct:
        monitor = region or sct.monitors[1]
        return sct.grab(monitor)


def save_screenshot(output_path: Path, screenshot: ScreenShot) -> None:
    """
    Persist a screenshot to disk for later labeling.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tools.to_png(screenshot.rgb, screenshot.size, output=str(output_path))


def _generate_filename(timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now()
    return f"screenshot_{ts.strftime('%Y%m%d_%H%M%S')}.png"


def _wait_for_keypress() -> str:
    """
    Wait for a single keypress. Returns one of {'capture', 'quit'}.
    """

    if msvcrt is None:
        user_input = input("按 S/Enter 截图，按 Q 退出：").strip().lower()
        if user_input in {"q", "quit"}:
            return "quit"
        return "capture"

    while True:
        key = msvcrt.getwch()
        if key in {"s", "S", "\r", "\n"}:
            return "capture"
        if key in {"q", "Q"}:
            return "quit"


def main() -> None:
    """
    CLI entry point for capturing screenshots on keypress.
    """

    print("Dou Dizhu Screenshot Capture 已启动。")
    print("按 S 或 Enter 截图，按 Q 退出。")

    while True:
        action = _wait_for_keypress()
        if action == "quit":
            print("已退出截图工具。")
            break

        screenshot = capture_screen_region(DEFAULT_REGION)
        filename = _generate_filename()
        output_path = OUTPUT_DIR / filename
        save_screenshot(output_path, screenshot)
        print(f"已保存到 {output_path}")


if __name__ == "__main__":
    main()
