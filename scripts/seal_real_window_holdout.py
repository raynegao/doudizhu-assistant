from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_real_window_holdout import seal_holdout_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal human-labeled real-window holdout data before model predictions exist."
        )
    )
    parser.add_argument("--model", type=Path, default=Path("models/card_cnn.pt"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/real_window_holdout/manifest.jsonl"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/cards_cls/manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_window_holdout/holdout-seal.json"),
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("runs/real-window-holdout"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seal = seal_holdout_manifest(
            model_path=args.model,
            manifest_path=args.manifest,
            training_manifest_path=args.training_manifest,
            output_path=args.output,
            predictions_root=args.predictions_root,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(seal, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
