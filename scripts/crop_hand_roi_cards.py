from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


def crop_hand_roi_cards(
    roi_path: Path,
    output_dir: Path,
    count: int,
    start_x: int,
    start_y: int,
    step_x: int,
    crop_size: tuple[int, int],
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if step_x <= 0:
        raise ValueError("step-x must be positive")

    image = ImageOps.exif_transpose(Image.open(roi_path)).convert("RGB")
    crop_width, crop_height = crop_size
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, object]] = []
    for index in range(count):
        left = start_x + index * step_x
        upper = start_y
        right = left + crop_width
        lower = upper + crop_height
        if right > image.width or lower > image.height:
            raise ValueError(
                f"Crop {index} exceeds ROI bounds: "
                f"box={(left, upper, right, lower)}, roi_size={image.size}"
            )

        card = image.crop((left, upper, right, lower))
        filename = f"card_{index:02d}.png"
        card.save(output_dir / filename)
        metadata.append({
            "index": index,
            "filename": filename,
            "box": [left, upper, right, lower],
        })

    (output_dir / "metadata.json").write_text(
        json.dumps({
            "source": str(roi_path),
            "count": count,
            "crop_size": [crop_width, crop_height],
            "step_x": step_x,
            "cards": metadata,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_crop_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Crop size must be WIDTHxHEIGHT, e.g. 126x210")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop overlapped hand-card ROI into visible card patches.")
    parser.add_argument("--roi", required=True, help="Input hand ROI image.")
    parser.add_argument("--output-dir", required=True, help="Output directory for card crops.")
    parser.add_argument("--count", type=int, default=16, help="Number of cards in the hand ROI.")
    parser.add_argument("--start-x", type=int, default=0, help="First card crop x offset in ROI pixels.")
    parser.add_argument("--start-y", type=int, default=0, help="First card crop y offset in ROI pixels.")
    parser.add_argument("--step-x", type=int, default=128, help="Horizontal offset between adjacent cards.")
    parser.add_argument("--crop-size", type=parse_crop_size, default=(126, 210), help="Visible crop size WIDTHxHEIGHT.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    crop_hand_roi_cards(
        roi_path=Path(args.roi),
        output_dir=Path(args.output_dir),
        count=args.count,
        start_x=args.start_x,
        start_y=args.start_y,
        step_x=args.step_x,
        crop_size=args.crop_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
