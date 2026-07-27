from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.state.cards import RANKS
from src.vision.scene_recognizer import (
    _encode_glyph_signature,
    _glyph_similarity,
    _rank_glyph_signature,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export anonymized 64x64 binary rank-glyph features from "
            "locally labelled card crops."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/cards_cls/test"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/vision/assets/rank_glyph_signatures.json"
        ),
    )
    parser.add_argument("--max-per-rank", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_per_rank <= 0:
        raise SystemExit("--max-per-rank must be positive")
    exported: dict[str, list[str]] = {}
    missing: list[str] = []
    for rank in RANKS:
        rank_dir = args.source / rank
        signatures: list[frozenset[int]] = []
        for path in sorted(rank_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                with Image.open(path) as source:
                    signature = _rank_glyph_signature(
                        source.convert("RGB")
                    )
            except OSError:
                continue
            if not signature or any(
                _glyph_similarity(signature, existing) >= 0.98
                for existing in signatures
            ):
                continue
            signatures.append(signature)
        signatures = signatures[-args.max_per_rank :]
        if not signatures:
            missing.append(rank)
            continue
        exported[rank] = [
            _encode_glyph_signature(signature)
            for signature in signatures
        ]
    if missing:
        raise SystemExit(
            "missing usable rank samples: " + ", ".join(missing)
        )
    payload = {
        "format": "rank-glyph-signatures-v1",
        "grid_size": 64,
        "ranks": exported,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"exported {sum(map(len, exported.values()))} signatures "
        f"for {len(exported)} ranks to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
