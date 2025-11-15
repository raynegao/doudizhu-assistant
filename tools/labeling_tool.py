"""
Minimal labeling tool for annotating card bounding boxes.
"""

from pathlib import Path
from typing import List, Tuple


def launch_labeling_tool(image_dir: Path, label_dir: Path) -> None:
    """
    Open the annotation UI for a batch of screenshots.
    """

    pass


def handle_mouse_event(event: int, x: int, y: int, flags: int, params: Tuple) -> None:
    """
    Process mouse events for drawing bounding boxes.
    """

    pass


def save_annotations(label_path: Path, boxes: List[Tuple[int, int, int, int]], classes: List[int]) -> None:
    """
    Write YOLO-formatted labels collected from the UI.
    """

    pass
