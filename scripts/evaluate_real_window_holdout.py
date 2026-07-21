"""Evaluate a separately collected real-window card-crop holdout manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.vision.card_classifier import CARD_CLASSES, load_checkpoint, predict_image_paths, select_device


def load_holdout_manifest(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_images: set[Path] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("image"), str) or not isinstance(record.get("label"), str):
            raise ValueError(f"line {line_number} requires string image and label")
        image = (path.parent / record["image"]).resolve()
        if record["label"] not in CARD_CLASSES:
            raise ValueError(f"line {line_number} has unsupported label {record['label']!r}")
        if not image.is_file():
            raise ValueError(f"line {line_number} image does not exist: {record['image']}")
        if image in seen_images:
            raise ValueError(f"line {line_number} duplicates image: {record['image']}")
        seen_images.add(image)
        records.append({"image": image, "label": record["label"], "source_id": record.get("source_id", "unknown")})
    if not records:
        raise ValueError("holdout manifest contains no samples")
    return records


def evaluate_holdout(model_path: Path, manifest_path: Path, output_dir: Path, device_name: str) -> dict[str, object]:
    records = load_holdout_manifest(manifest_path)
    device = select_device(device_name)
    model, classes, image_size = load_checkpoint(model_path, device=device)
    predictions = predict_image_paths(model, [record["image"] for record in records], classes=classes, image_size=image_size, device=device)
    errors = []
    correct = 0
    for record, prediction in zip(records, predictions, strict=True):
        expected = str(record["label"])
        if prediction.rank == expected:
            correct += 1
        else:
            errors.append({"image": str(record["image"]), "expected": expected, "predicted": prediction.rank, "confidence": round(prediction.confidence, 6), "source_id": record["source_id"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "real-window-holdout-v1", "model": str(model_path), "manifest": str(manifest_path), "sample_count": len(records), "accuracy": correct / len(records), "error_count": len(errors), "class_counts": dict(sorted(Counter(str(record["label"]) for record in records).items())), "source_counts": dict(sorted(Counter(str(record["source_id"]) for record in records).items())), "limitations": ["This report is valid only when every sample comes from real game-window screenshots excluded from training.", "A holdout result must not be merged into the fixed-ROI synthetic/local split metric."], "error_report": "errors.jsonl"}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a real-window independent card classifier holdout.")
    parser.add_argument("--model", default="models/card_cnn.pt")
    parser.add_argument("--manifest", required=True, help="JSONL manifest; image paths are relative to it.")
    parser.add_argument("--output-dir", default="runs/real-window-holdout")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        report = evaluate_holdout(Path(args.model), Path(args.manifest), Path(args.output_dir), args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
