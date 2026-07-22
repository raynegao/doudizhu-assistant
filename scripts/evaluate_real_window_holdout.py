"""Evaluate a separately collected real-window card-crop holdout manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from src.vision.card_classifier import (
    CARD_CLASSES,
    CardPrediction,
    load_checkpoint,
    predict_image_paths,
    select_device,
)


FOCUS_CONFUSION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(("SJ", "BJ")),
    frozenset(("10", "J")),
    frozenset(("6", "9")),
    frozenset(("J", "Q", "K", "A")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_holdout_manifest(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_images: set[Path] = set()
    seen_hashes: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("image"), str)
            or not isinstance(record.get("label"), str)
        ):
            raise ValueError(f"line {line_number} requires string image and label")

        image_value = str(record["image"])
        image = (path.parent / image_value).resolve()
        label = str(record["label"])
        source_id = record.get("source_id")
        if label not in CARD_CLASSES:
            raise ValueError(f"line {line_number} has unsupported label {label!r}")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"line {line_number} requires a non-empty source_id")
        if not image.is_file():
            raise ValueError(f"line {line_number} image does not exist: {image_value}")

        actual_sha256 = sha256_file(image)
        recorded_sha256 = record.get("sha256")
        if recorded_sha256 is not None and recorded_sha256 != actual_sha256:
            raise ValueError(f"line {line_number} sha256 does not match image: {image_value}")
        if image in seen_images:
            raise ValueError(f"line {line_number} duplicates image path: {image_value}")
        if actual_sha256 in seen_hashes:
            raise ValueError(f"line {line_number} duplicates image content: {image_value}")
        seen_images.add(image)
        seen_hashes.add(actual_sha256)
        records.append({
            "image": image,
            "image_value": image_value,
            "label": label,
            "source_id": source_id.strip(),
            "sha256": actual_sha256,
            "roi_sha256": record.get("roi_sha256"),
        })
    if not records:
        raise ValueError("holdout manifest contains no samples")
    return records


def load_training_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid training manifest JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"training manifest line {line_number} must be an object")
        for candidate in _training_record_paths(record, path):
            if candidate.is_file():
                hashes.add(sha256_file(candidate))
    return hashes


def find_training_leakage(
    records: Iterable[Mapping[str, object]],
    training_hashes: set[str],
) -> list[dict[str, str]]:
    return [
        {
            "image": str(record["image_value"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
        if str(record["sha256"]) in training_hashes
    ]


def evaluate_holdout(
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    device_name: str,
    *,
    training_manifest_path: Path | None = Path("data/cards_cls/manifest.jsonl"),
    minimum_samples: int = 300,
    minimum_per_class: int = 10,
    minimum_sources: int = 3,
    confidence_threshold: float = 0.70,
) -> dict[str, object]:
    records = load_holdout_manifest(manifest_path)
    leakage_checked = training_manifest_path is not None and training_manifest_path.is_file()
    training_hashes = (
        load_training_hashes(training_manifest_path)
        if leakage_checked and training_manifest_path is not None
        else set()
    )
    leakage = find_training_leakage(records, training_hashes)
    if leakage:
        raise ValueError(
            f"holdout leakage detected: {len(leakage)} sample(s) also occur in training data"
        )

    device = select_device(device_name)
    model, classes, image_size = load_checkpoint(model_path, device=device)
    predictions = predict_image_paths(
        model,
        [Path(record["image"]) for record in records],
        classes=classes,
        image_size=image_size,
        device=device,
    )
    report, rows = summarize_predictions(
        records,
        predictions,
        classes,
        leakage_checked=leakage_checked,
        training_manifest_path=training_manifest_path,
        minimum_samples=minimum_samples,
        minimum_per_class=minimum_per_class,
        minimum_sources=minimum_sources,
        confidence_threshold=confidence_threshold,
    )
    report.update({
        "model": str(model_path),
        "manifest": str(manifest_path),
        "artifacts": {
            "predictions": "predictions.jsonl",
            "errors": "errors.jsonl",
            "confusion_matrix": "confusion_matrix.csv",
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    _write_jsonl(
        output_dir / "errors.jsonl",
        [row for row in rows if not row["correct"]],
    )
    _write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        classes,
        rows,
    )
    return report


def summarize_predictions(
    records: list[Mapping[str, object]],
    predictions: list[CardPrediction],
    classes: tuple[str, ...],
    *,
    leakage_checked: bool,
    training_manifest_path: Path | None,
    minimum_samples: int,
    minimum_per_class: int,
    minimum_sources: int,
    confidence_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(records) != len(predictions):
        raise ValueError("prediction count does not match manifest sample count")
    rows: list[dict[str, object]] = []
    for record, prediction in zip(records, predictions, strict=True):
        expected = str(record["label"])
        predicted = prediction.rank
        rows.append({
            "image": str(record["image_value"]),
            "source_id": str(record["source_id"]),
            "sha256": str(record["sha256"]),
            "expected": expected,
            "predicted": predicted,
            "confidence": round(prediction.confidence, 6),
            "correct": expected == predicted,
            "low_confidence": prediction.confidence < confidence_threshold,
            "focus_error": expected != predicted and _is_focus_confusion(expected, predicted),
        })

    class_counts = Counter(str(record["label"]) for record in records)
    source_counts = Counter(str(record["source_id"]) for record in records)
    per_class: dict[str, dict[str, object]] = {}
    for label in classes:
        label_rows = [row for row in rows if row["expected"] == label]
        correct = sum(bool(row["correct"]) for row in label_rows)
        per_class[label] = {
            "count": len(label_rows),
            "correct": correct,
            "accuracy": correct / len(label_rows) if label_rows else None,
        }

    readiness_checks = [
        {
            "name": "training_leakage_checked",
            "passed": leakage_checked,
            "evidence": str(training_manifest_path) if leakage_checked else "not_available",
        },
        {
            "name": "minimum_sample_count",
            "passed": len(records) >= minimum_samples,
            "evidence": f"{len(records)}/{minimum_samples}",
        },
        {
            "name": "all_classes_covered",
            "passed": all(class_counts[label] > 0 for label in classes),
            "evidence": f"{sum(class_counts[label] > 0 for label in classes)}/{len(classes)}",
        },
        {
            "name": "minimum_samples_per_class",
            "passed": all(class_counts[label] >= minimum_per_class for label in classes),
            "evidence": f"minimum={min((class_counts[label] for label in classes), default=0)}/{minimum_per_class}",
        },
        {
            "name": "multiple_independent_sources",
            "passed": len(source_counts) >= minimum_sources,
            "evidence": f"{len(source_counts)}/{minimum_sources}",
        },
    ]
    correct_count = sum(bool(row["correct"]) for row in rows)
    return ({
        "schema_version": "real-window-holdout-v2",
        "sample_count": len(records),
        "correct_count": correct_count,
        "accuracy": correct_count / len(records),
        "error_count": len(records) - correct_count,
        "low_confidence_count": sum(bool(row["low_confidence"]) for row in rows),
        "focus_error_count": sum(bool(row["focus_error"]) for row in rows),
        "confidence_threshold": confidence_threshold,
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "per_class": per_class,
        "leakage_check": {
            "checked": leakage_checked,
            "overlap_count": 0,
            "training_manifest": str(training_manifest_path) if leakage_checked else None,
        },
        "publication_ready": all(bool(check["passed"]) for check in readiness_checks),
        "readiness_checks": readiness_checks,
        "limitations": [
            "Every sample must come from real game-window screenshots excluded from training.",
            "This result must not be merged with the fixed-ROI synthetic/local split metric.",
            "Publication readiness checks dataset independence and coverage, not a target accuracy.",
        ],
    }, rows)


def _training_record_paths(record: Mapping[str, object], manifest_path: Path) -> set[Path]:
    raw_paths: set[Path] = set()
    for key in ("output_path", "source_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            raw_paths.add(Path(value))
    source_dir = record.get("source_dir")
    source_file = record.get("source_file")
    if isinstance(source_dir, str) and isinstance(source_file, str):
        raw_paths.add(Path(source_dir) / source_file)

    resolved: set[Path] = set()
    for raw_path in raw_paths:
        if raw_path.is_absolute():
            resolved.add(raw_path.resolve())
            continue
        for base in (Path.cwd(), *manifest_path.resolve().parents):
            candidate = (base / raw_path).resolve()
            if candidate.exists():
                resolved.add(candidate)
                break
    return resolved


def _is_focus_confusion(expected: str, predicted: str) -> bool:
    return any(expected in group and predicted in group for group in FOCUS_CONFUSION_GROUPS)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_confusion_matrix(
    path: Path,
    classes: tuple[str, ...],
    rows: Iterable[Mapping[str, object]],
) -> None:
    matrix = {expected: Counter() for expected in classes}
    for row in rows:
        matrix[str(row["expected"])][str(row["predicted"])] += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected\\predicted", *classes])
        for expected in classes:
            writer.writerow([expected, *(matrix[expected][predicted] for predicted in classes)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a real-window independent card classifier holdout."
    )
    parser.add_argument("--model", default="models/card_cnn.pt")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest; image paths are relative to it.",
    )
    parser.add_argument("--training-manifest", default="data/cards_cls/manifest.jsonl")
    parser.add_argument("--output-dir", default="runs/real-window-holdout")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-samples", type=int, default=300)
    parser.add_argument("--minimum-per-class", type=int, default=10)
    parser.add_argument("--minimum-sources", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_holdout(
            Path(args.model),
            Path(args.manifest),
            Path(args.output_dir),
            args.device,
            training_manifest_path=Path(args.training_manifest),
            minimum_samples=args.minimum_samples,
            minimum_per_class=args.minimum_per_class,
            minimum_sources=args.minimum_sources,
            confidence_threshold=args.confidence_threshold,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
