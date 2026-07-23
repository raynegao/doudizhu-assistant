from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

from src.pipeline.live_layout import load_live_layout
from src.state.cards import normalize_rank, validate_card_counts
from src.vision.scene_recognizer import segment_card_boxes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment and label one recorded Phase 6 play-area frame."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument("--seat", choices=("self", "left", "right"), required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/live_game/labeled_play_cards"),
    )
    parser.add_argument("--source-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_live_layout(args.config)
    image = Image.open(args.image).convert("RGB")
    play_crop = config.crop(image, f"{args.seat}_play")
    boxes = segment_card_boxes(play_crop)
    raw_labels = [
        token
        for token in re.split(r"[\s,，;；]+", args.labels.strip())
        if token
    ]
    labels = tuple(normalize_rank(token) for token in raw_labels)
    validate_card_counts(labels)
    if len(boxes) != len(labels):
        raise SystemExit(
            f"segmented {len(boxes)} cards but received {len(labels)} labels; "
            "check the ROI preview or choose a cleaner frame"
        )
    source_id = args.source_id or (
        f"{args.image.stem}-{args.seat}-{int(time.time() * 1000)}"
    )
    output_dir = args.output_root / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (box, label) in enumerate(zip(boxes, labels)):
            card = play_crop.crop(box)
            rank_height = max(1, round(card.height * 0.68))
            card = card.crop((0, 0, card.width, rank_height))
            path = output_dir / f"card_{index:02d}.png"
            card.save(path)
            manifest.write(json.dumps({
                "event": "labeled_live_play_card",
                "source_id": source_id,
                "seat": args.seat,
                "index": index,
                "label": label,
                "image": path.name,
                "source_image": args.image.as_posix(),
                "box": list(box),
            }, ensure_ascii=False) + "\n")
    (output_dir / "labels.txt").write_text(" ".join(labels) + "\n", encoding="utf-8")
    print(f"labeled crops: {output_dir}")
    print(f"labels: {' '.join(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
