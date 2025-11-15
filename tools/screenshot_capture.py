"""
Screen capture helpers for collecting training data and live frames.
"""

from pathlib import Path
from typing import Dict, Optional


def capture_screen_region(region: Optional[Dict[str, int]] = None) -> bytes:
    """
    Capture a screenshot of the specified region and return raw bytes.
    """

    pass


def save_screenshot(output_path: Path, image_bytes: bytes) -> None:
    """
    Persist a screenshot to disk for later labeling.
    """

    pass
