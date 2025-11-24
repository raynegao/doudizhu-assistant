"""
Automated pipeline: synth data -> pseudo-label -> merge -> train.

Steps:
1) Generate synthetic data to `data/raw_screenshots_synth` and `data/labels_yolo_synth`.
2) Pseudo-label real screenshots to `data/labels_yolo_pseudo`.
3) Merge real + synth + pseudo into `data/yolo_dataset_merged`.
4) Train YOLO on the merged set using detection/yolo_train.py.

Usage (example):
    python tools/auto_pipeline.py ^
        --weights runs/train/doudizhu/weights/best.pt ^
        --synth-num 500 ^
        --train-epochs 50 ^
        --train-batch 16 ^
        --conf 0.35
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate synth -> pseudo-label -> merge -> train pipeline.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights for pseudo-labeling (e.g., best.pt).")
    parser.add_argument("--synth-num", type=int, default=500, help="Number of synthetic images to generate.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for pseudo-labeling.")
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--train-batch", type=int, default=16)
    parser.add_argument("--train-imgsz", type=int, default=640)
    args = parser.parse_args()

    # 1) Synthetic generation
    run(
        [
            sys.executable,
            "tools/synthetic_generator.py",
            "--num-images",
            str(args.synth_num),
            "--width",
            "1920",
            "--height",
            "1080",
            "--min-cards",
            "8",
            "--max-cards",
            "17",
        ]
    )

    # 2) Pseudo-label real screenshots
    run(
        [
            sys.executable,
            "tools/pseudo_label.py",
            "--weights",
            str(args.weights),
            "--images",
            "data/raw_screenshots",
            "--out",
            "data/labels_yolo_pseudo",
            "--conf",
            str(args.conf),
        ]
    )

    # 3) Merge datasets (real + synth + pseudo)
    run(
        [
            sys.executable,
            "tools/prepare_training_set.py",
            "--image-dirs",
            "data/raw_screenshots,data/raw_screenshots_synth,data/raw_screenshots",
            "--label-dirs",
            "data/labels_yolo,data/labels_yolo_synth,data/labels_yolo_pseudo",
            "--out",
            "data/yolo_dataset_merged",
        ]
    )

    # 4) Train on merged dataset
    run(
        [
            sys.executable,
            "detection/yolo_train.py",
            "--images",
            "data/yolo_dataset_merged/images/train",
            "--labels",
            "data/yolo_dataset_merged/labels/train",
            "--epochs",
            str(args.train_epochs),
            "--batch",
            str(args.train_batch),
            "--imgsz",
            str(args.train_imgsz),
        ]
    )


if __name__ == "__main__":
    main()
