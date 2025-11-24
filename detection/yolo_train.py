"""
Utilities for training the YOLO model that detects playing cards.

This script stages a YOLO-compatible dataset from:
- images: `data/raw_screenshots`
- labels: `data/labels_yolo`
It builds a `data/yolo_dataset/images/{train,val}` and matching `labels/{train,val}` tree,
writes `data/data.yaml`, then launches ultralytics training.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_logic.cards import RANK_ORDER


@dataclass
class TrainingConfig:
    """
    Container for YOLO training configuration.
    """

    images_dir: Path
    labels_dir: Path
    model_name: str = "yolov8n.pt"
    epochs: int = 10
    batch_size: int = 16
    imgsz: int = 640
    val_ratio: float = 0.1
    data_yaml: Path = Path("data/data.yaml")


def collect_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find images that have corresponding labels.
    """

    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*.png")):
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path]], val_ratio: float
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Shuffle and split paired (image, label) tuples into train/val lists.
    """

    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * (1 - val_ratio)))
    train = pairs[:split_idx]
    val = pairs[split_idx:] or pairs[:1]
    return train, val


def stage_dataset(
    root: Path, train_pairs: Sequence[Tuple[Path, Path]], val_pairs: Sequence[Tuple[Path, Path]]
) -> Tuple[List[Path], List[Path]]:
    """
    Create YOLO-style folder structure with images/labels under the same root.
    Returns (train_images, val_images) paths.
    """

    train_img_dir = root / "images" / "train"
    val_img_dir = root / "images" / "val"
    train_lbl_dir = root / "labels" / "train"
    val_lbl_dir = root / "labels" / "val"

    if root.exists():
        shutil.rmtree(root)
    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _copy_pairs(pairs: Sequence[Tuple[Path, Path]], img_dir: Path, lbl_dir: Path) -> List[Path]:
        out_imgs: List[Path] = []
        for img, lbl in pairs:
            img_dest = img_dir / img.name
            lbl_dest = lbl_dir / f"{img.stem}.txt"
            shutil.copy2(img, img_dest)
            shutil.copy2(lbl, lbl_dest)
            out_imgs.append(img_dest)
        return out_imgs

    train_imgs = _copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    val_imgs = _copy_pairs(val_pairs, val_img_dir, val_lbl_dir)
    return train_imgs, val_imgs


def write_yaml(dataset_root: Path, train_imgs: List[Path], val_imgs: List[Path], yaml_path: Path) -> Path:
    """
    Emit a data.yaml compatible with ultralytics.
    """

    content = {
        "path": str(dataset_root.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(RANK_ORDER),
        "names": RANK_ORDER,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return yaml_path


def train_model(config: TrainingConfig) -> None:
    """
    Execute the ultralytics training loop with the provided configuration.
    """

    pairs = collect_image_label_pairs(config.images_dir, config.labels_dir)
    if not pairs:
        raise RuntimeError(f"在 {config.images_dir} 未找到对应标签的图片，请先完成标注。")

    train_pairs, val_pairs = split_pairs(pairs, config.val_ratio)
    dataset_root = Path("data/yolo_dataset")
    train_imgs, val_imgs = stage_dataset(dataset_root, train_pairs, val_pairs)
    data_yaml = write_yaml(dataset_root, train_imgs, val_imgs, config.data_yaml)

    model = YOLO(config.model_name)
    model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.imgsz,
        project="runs/train",
        name="doudizhu",
        exist_ok=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on Dou Dizhu card images.")
    parser.add_argument("--images", type=Path, default=Path("data/raw_screenshots"), help="Directory with PNG screenshots.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels_yolo"), help="Directory with YOLO txt labels.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model name or path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--data-yaml", type=Path, default=Path("data/data.yaml"))
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point for launching a training job.
    """

    args = parse_args()
    config = TrainingConfig(
        images_dir=args.images,
        labels_dir=args.labels,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        val_ratio=args.val_ratio,
        data_yaml=args.data_yaml,
    )
    train_model(config)


if __name__ == "__main__":
    main()
