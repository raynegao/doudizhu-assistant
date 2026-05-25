from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


RANKS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ")


def load_seed_images(seed_dir: Path) -> dict[str, list[Path]]:
    seeds: dict[str, list[Path]] = {}
    for rank in RANKS:
        paths = sorted((seed_dir / rank).glob("*.png"))
        if paths:
            seeds[rank] = paths
    return seeds


def augment_image(image: Image.Image, rng: random.Random, output_size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    crop_ratio = rng.uniform(0.0, 0.06)
    dx = int(width * crop_ratio)
    dy = int(height * crop_ratio)
    if dx > 0 and dy > 0:
        image = image.crop((
            rng.randint(0, dx),
            rng.randint(0, dy),
            width - rng.randint(0, dx),
            height - rng.randint(0, dy),
        ))

    image = image.resize(output_size, Image.Resampling.BICUBIC)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))

    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.7)))

    if rng.random() < 0.6:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(55, 92))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    if rng.random() < 0.35:
        pixels = image.load()
        for _ in range(int(output_size[0] * output_size[1] * 0.01)):
            x = rng.randrange(output_size[0])
            y = rng.randrange(output_size[1])
            delta = rng.randint(-18, 18)
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                max(0, min(255, r + delta)),
                max(0, min(255, g + delta)),
                max(0, min(255, b + delta)),
            )

    return image


def generate_dataset(
    seed_dir: Path,
    output_dir: Path,
    per_seed: int,
    val_ratio: float,
    image_size: tuple[int, int],
    seed: int,
) -> None:
    rng = random.Random(seed)
    seed_images = load_seed_images(seed_dir)
    if not seed_images:
        raise FileNotFoundError(f"No seed templates found under {seed_dir}")

    for rank, paths in seed_images.items():
        generated: list[Image.Image] = []
        source_names: list[str] = []
        for path in paths:
            with Image.open(path) as image:
                for _ in range(per_seed):
                    generated.append(augment_image(image, rng, image_size))
                    source_names.append(path.stem)

        indices = list(range(len(generated)))
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
        val_indices = set(indices[:val_count])

        for index, image in enumerate(generated):
            split = "val" if index in val_indices else "train"
            destination = output_dir / split / rank / f"{rank}_{source_names[index]}_{index:05d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)


def parse_image_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Image size must be WIDTHxHEIGHT, e.g. 64x96")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an augmented CNN card-classification dataset.")
    parser.add_argument("--seed-dir", default="data/cards_cls_seed", help="Rank-folder seed template directory.")
    parser.add_argument("--output-dir", default="data/cards_cls", help="Output dataset directory.")
    parser.add_argument("--per-seed", type=int, default=80, help="Augmented samples to generate per seed image.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--image-size", type=parse_image_size, default=(64, 96), help="Output image size WIDTHxHEIGHT.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generate_dataset(
        seed_dir=Path(args.seed_dir),
        output_dir=Path(args.output_dir),
        per_seed=args.per_seed,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
