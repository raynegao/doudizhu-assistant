from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.vision.card_classifier import load_checkpoint, predict_image_paths


def predict_crop_dir(model_path: Path, crop_dir: Path, device_name: str = "auto") -> list[dict[str, object]]:
    device = _resolve_device(device_name)
    model, classes, image_size = load_checkpoint(model_path, device=device)
    paths = sorted(crop_dir.glob("card_*.png"))
    if not paths:
        raise FileNotFoundError(f"No card_*.png crops found under {crop_dir}")
    predictions = predict_image_paths(model, paths, classes=classes, image_size=image_size, device=device)
    return [
        {
            "index": index,
            "file": str(path),
            "rank": prediction.rank,
            "confidence": prediction.confidence,
        }
        for index, (path, prediction) in enumerate(zip(paths, predictions, strict=True))
    ]


def _resolve_device(device_name: str) -> torch.device:
    from src.vision.card_classifier import select_device

    return select_device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict card ranks from cropped hand-card images.")
    parser.add_argument("--model", default="models/card_cnn.pt", help="PyTorch checkpoint path.")
    parser.add_argument("--crop-dir", required=True, help="Directory containing card_*.png crops.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = predict_crop_dir(Path(args.model), Path(args.crop_dir), device_name=args.device)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"{result['index']:02d} {result['rank']:>2} confidence={result['confidence']:.3f}")
        print("hand:", " ".join(str(result["rank"]) for result in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
