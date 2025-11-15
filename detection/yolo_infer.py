"""
Helpers for running inference with the trained YOLO card detector.
"""

from pathlib import Path
from typing import Any, List, Tuple


def load_detector(weights_path: Path) -> Any:
    """
    Load the YOLO model weights for inference.
    """

    pass


def run_inference(model: Any, image_path: Path) -> List[Tuple[float, float, float, float, int, float]]:
    """
    Perform inference on a screenshot and return bounding boxes with class ids.
    """

    pass


def visualize_detections(image_path: Path, detections: List[Tuple[float, float, float, float, int, float]]) -> None:
    """
    Draw bounding boxes on the screenshot for debugging purposes.
    """

    pass


def main() -> None:
    """
    CLI entry point for running a single inference.
    """

    pass


if __name__ == "__main__":
    main()
