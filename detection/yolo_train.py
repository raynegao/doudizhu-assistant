"""
Utilities for training the YOLO model that detects playing cards.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Container for YOLO training configuration.
    """

    data_dir: Path
    model_name: str = "yolov8n.pt"
    epochs: int = 10
    batch_size: int = 16


def load_training_data(data_dir: Path) -> None:
    """
    Load and validate the YOLO-formatted dataset from disk.
    """

    pass


def train_model(config: TrainingConfig) -> None:
    """
    Execute the ultralytics training loop with the provided configuration.
    """

    pass


def export_trained_model(output_path: Path) -> None:
    """
    Save the best-performing trained weights to the models directory.
    """

    pass


def main() -> None:
    """
    CLI entry point for launching a training job.
    """

    pass


if __name__ == "__main__":
    main()
