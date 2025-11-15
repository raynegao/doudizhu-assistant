"""
Entrypoint that glues capture, detection, parsing, and simulation together.
"""

from typing import Any, Dict, Optional


def capture_screen(region: Optional[Dict[str, int]] = None) -> Any:
    """
    Capture a screen region and return an image compatible with YOLO.
    """

    pass


def process_frame(frame: Any) -> None:
    """
    Run the full detection → parsing → simulation pipeline on a frame.
    """

    pass


def main_loop() -> None:
    """
    Continuously capture the screen and update the UI with win-rate estimates.
    """

    pass


if __name__ == "__main__":
    main_loop()
