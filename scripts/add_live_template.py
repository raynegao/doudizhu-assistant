from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from PIL import Image

from src.capture.screen_geometry import MacWindowCapture
from src.pipeline.live_layout import load_live_layout


VALID_KINDS = {"pass", "remaining", "role", "turn"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and label one Phase 6 live-game ROI template."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument("--kind", choices=sorted(VALID_KINDS), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--image", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_live_layout(args.config)
    if args.roi not in config.rois:
        raise SystemExit(
            f"unknown ROI {args.roi!r}; choose from {', '.join(sorted(config.rois))}"
        )
    if args.kind == "remaining":
        try:
            value = int(args.label)
        except ValueError as exc:
            raise SystemExit("remaining template label must be 0..20") from exc
        if not 0 <= value <= 20:
            raise SystemExit("remaining template label must be 0..20")
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", args.label).strip("_")
    if not safe_label:
        raise SystemExit("template label cannot be empty")
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
    else:
        image = MacWindowCapture(config.app_name).capture(frame_id=1).image
    crop = config.crop(image, args.roi)
    output_dir = config.templates_dir / args.kind / safe_label
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{args.roi}-{int(time.time() * 1000)}.png"
    crop.save(path)
    print(f"saved template: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
