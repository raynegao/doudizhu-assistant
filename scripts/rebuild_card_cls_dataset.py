from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from scripts.generate_card_cls_dataset import RANKS, augment_image, load_seed_images, parse_image_size
from src.vision.card_classifier import DEFAULT_IMAGE_SIZE


TEMPLATE_SPLITS: tuple[str, ...] = ("train", "val")
REAL_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class CropSource:
    crop_dir: Path
    labels: tuple[str, ...]
    note: str = ""


DEFAULT_CROP_SOURCES: tuple[CropSource, ...] = (
    CropSource(
        crop_dir=Path("data/roi_samples/window_mode_hand_roi_tight_001"),
        labels=("A", "K", "Q", "J", "10", "10", "9", "8", "7", "7", "6", "5", "4", "3", "3"),
    ),
    CropSource(
        crop_dir=Path("data/roi_samples/window_mode_jokers_hand_roi_001"),
        labels=("BJ", "2", "2", "K", "Q", "J", "J", "9", "9", "9", "8", "8", "8", "6", "4", "4", "3"),
    ),
    CropSource(
        crop_dir=Path("data/roi_samples/window_mode_jokers_sj_gray"),
        labels=("SJ",),
        note="temporary_gray_sj_from_bj_crop",
    ),
)

EXCLUDED_CROP_DIRS: tuple[Path, ...] = (
    Path("data/roi_samples/hand_roi_001_step135"),
)


def rebuild_dataset(
    seed_dir: Path,
    output_dir: Path,
    crop_sources: tuple[CropSource, ...],
    template_per_seed: int,
    real_per_crop: int,
    image_size: tuple[int, int],
    seed: int,
    clean: bool,
) -> dict[str, object]:
    if clean:
        for split in (*TEMPLATE_SPLITS, "test"):
            shutil.rmtree(output_dir / split, ignore_errors=True)
        (output_dir / "manifest.jsonl").unlink(missing_ok=True)

    rng = random.Random(seed)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    excluded = [str(path) for path in EXCLUDED_CROP_DIRS]

    with manifest_path.open("w", encoding="utf-8") as manifest:
        written += _write_template_samples(
            seed_dir=seed_dir,
            output_dir=output_dir,
            template_per_seed=template_per_seed,
            image_size=image_size,
            rng=rng,
            manifest=manifest,
        )
        written += _write_real_crop_samples(
            crop_sources=crop_sources,
            output_dir=output_dir,
            real_per_crop=real_per_crop,
            image_size=image_size,
            rng=rng,
            manifest=manifest,
        )

    counts = count_split_labels(output_dir)
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "sample_count": written,
        "excluded_crop_dirs": excluded,
        "counts": counts,
    }


def _write_template_samples(
    seed_dir: Path,
    output_dir: Path,
    template_per_seed: int,
    image_size: tuple[int, int],
    rng: random.Random,
    manifest,
) -> int:
    seed_images = load_seed_images(seed_dir)
    if not seed_images:
        raise FileNotFoundError(f"No seed templates found under {seed_dir}")

    written = 0
    for rank in RANKS:
        for source_path in seed_images.get(rank, []):
            split_counts = _split_counts(template_per_seed, {"train": 0.8, "val": 0.2})
            written += _augment_source_to_splits(
                source_path=source_path,
                rank=rank,
                output_dir=output_dir,
                split_counts=split_counts,
                image_size=image_size,
                rng=rng,
                manifest=manifest,
                prefix="template",
                note="template_seed",
            )
    return written


def _write_real_crop_samples(
    crop_sources: tuple[CropSource, ...],
    output_dir: Path,
    real_per_crop: int,
    image_size: tuple[int, int],
    rng: random.Random,
    manifest,
) -> int:
    written = 0
    for source in crop_sources:
        paths = sorted(source.crop_dir.glob("card_*.png"))
        if len(paths) != len(source.labels):
            raise ValueError(f"{source.crop_dir} has {len(paths)} card crops but {len(source.labels)} labels")
        for index, (path, rank) in enumerate(zip(paths, source.labels, strict=True)):
            split_counts = _split_counts(real_per_crop, {"train": 0.7, "val": 0.2, "test": 0.1})
            written += _augment_source_to_splits(
                source_path=path,
                rank=rank,
                output_dir=output_dir,
                split_counts=split_counts,
                image_size=image_size,
                rng=rng,
                manifest=manifest,
                prefix=f"real_{source.crop_dir.name}_{index:02d}",
                note=source.note or "real_crop",
            )
    return written


def _augment_source_to_splits(
    source_path: Path,
    rank: str,
    output_dir: Path,
    split_counts: dict[str, int],
    image_size: tuple[int, int],
    rng: random.Random,
    manifest,
    prefix: str,
    note: str,
) -> int:
    written = 0
    with Image.open(source_path) as image:
        for split, count in split_counts.items():
            for sample_index in range(count):
                augmentation_seed = rng.randrange(0, 2**31)
                sample_rng = random.Random(augmentation_seed)
                output = output_dir / split / rank / f"{prefix}_{source_path.stem}_{sample_index:04d}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                augment_image(image, sample_rng, image_size).save(output)
                manifest.write(json.dumps({
                    "output_path": str(output),
                    "split": split,
                    "label": rank,
                    "source_dir": str(source_path.parent),
                    "source_file": source_path.name,
                    "augmentation_seed": augmentation_seed,
                    "note": note,
                }, ensure_ascii=False) + "\n")
                written += 1
    return written


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < len(ratios):
        raise ValueError(f"total={total} is too small for splits={list(ratios)}")
    splits = list(ratios)
    counts = {split: max(1, int(total * ratios[split])) for split in splits}
    while sum(counts.values()) > total:
        largest = max(counts, key=counts.__getitem__)
        if counts[largest] == 1:
            break
        counts[largest] -= 1
    while sum(counts.values()) < total:
        largest_ratio = max(ratios, key=ratios.__getitem__)
        counts[largest_ratio] += 1
    return counts


def count_split_labels(output_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_counts: dict[str, int] = {}
        for rank in RANKS:
            split_counts[rank] = len(list((output_dir / split / rank).glob("*.png")))
        counts[split] = split_counts
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the cleaned Phase 2 card-classification dataset.")
    parser.add_argument("--seed-dir", default="data/cards_cls_seed")
    parser.add_argument("--output-dir", default="data/cards_cls")
    parser.add_argument("--template-per-seed", type=int, default=20)
    parser.add_argument("--real-per-crop", type=int, default=30)
    parser.add_argument("--image-size", type=parse_image_size, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clean", action="store_true", help="Append instead of rebuilding split directories.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = rebuild_dataset(
        seed_dir=Path(args.seed_dir),
        output_dir=Path(args.output_dir),
        crop_sources=DEFAULT_CROP_SOURCES,
        template_per_seed=args.template_per_seed,
        real_per_crop=args.real_per_crop,
        image_size=args.image_size,
        seed=args.seed,
        clean=not args.no_clean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
