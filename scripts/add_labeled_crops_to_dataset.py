from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from scripts.generate_card_cls_dataset import augment_image, parse_image_size
from src.state.cards import normalize_rank
from src.vision.card_classifier import CARD_CLASSES, DEFAULT_IMAGE_SIZE


def add_labeled_crops(
    crop_dir: Path,
    labels: list[str],
    output_dir: Path,
    per_crop: int,
    val_ratio: float,
    image_size: tuple[int, int],
    seed: int,
) -> int:
    paths = sorted(crop_dir.glob("card_*.png"))
    if len(paths) != len(labels):
        raise ValueError(f"Expected {len(paths)} labels, got {len(labels)}")

    rng = random.Random(seed)
    count = 0
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        rank = normalize_rank(label)
        if rank not in CARD_CLASSES:
            raise ValueError(f"Unsupported rank: {label}")
        with Image.open(path) as image:
            for sample_index in range(per_crop):
                split = "val" if rng.random() < val_ratio else "train"
                output = output_dir / split / rank / f"real_{path.stem}_{index:02d}_{sample_index:04d}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                augment_image(image, rng, image_size).save(output)
                count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add labeled real card crops into the CNN dataset.")
    parser.add_argument("--crop-dir", required=True, help="Directory containing card_*.png crops.")
    parser.add_argument("--labels", required=True, help="Space-separated ranks matching sorted card_*.png files.")
    parser.add_argument("--output-dir", default="data/cards_cls", help="Dataset root with train/val rank folders.")
    parser.add_argument("--per-crop", type=int, default=30, help="Augmented samples to create per real crop.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=parse_image_size, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels = args.labels.split()
    count = add_labeled_crops(
        crop_dir=Path(args.crop_dir),
        labels=labels,
        output_dir=Path(args.output_dir),
        per_crop=args.per_crop,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"added {count} augmented real-crop samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
