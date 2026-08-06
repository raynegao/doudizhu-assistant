"""Evaluate a separately collected real-window card-crop holdout manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.capture.recording_integrity import (
    sha256_file,
    sha256_json_payload,
    sha256_python_implementation,
    write_json_atomic,
)
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
HOLDOUT_SEAL_SCHEMA_VERSION = "real-window-holdout-blind-seal-v1"
HOLDOUT_SESSION_SCHEMA_VERSION = "real-window-holdout-session-v2"
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ValueError(f"line {line_number} has an unsafe source_id")
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
            "crop_index": record.get("crop_index"),
            "sha256": actual_sha256,
            "roi_sha256": record.get("roi_sha256"),
        })
    if not records:
        raise ValueError("holdout manifest contains no samples")
    return records


def seal_holdout_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    training_manifest_path: Path,
    output_path: Path,
    predictions_root: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise ValueError(f"holdout seal already exists: {output_path}")
    if predictions_root.exists() and any(predictions_root.iterdir()):
        raise ValueError(
            "holdout labels must be sealed before prediction output exists: "
            f"{predictions_root}"
        )
    if not model_path.is_file():
        raise ValueError(f"holdout model is missing: {model_path}")
    if not training_manifest_path.is_file():
        raise ValueError(
            f"holdout training manifest is missing: {training_manifest_path}"
        )
    records = load_holdout_manifest(manifest_path)
    leakage = find_training_leakage(
        records,
        load_training_hashes(training_manifest_path),
    )
    if leakage:
        raise ValueError(
            f"holdout leakage detected before sealing: {len(leakage)} sample(s)"
        )
    summary, sessions = _validate_holdout_structure(records, manifest_path)
    project_root = Path(__file__).resolve().parents[1]
    seal: dict[str, object] = {
        "schema_version": HOLDOUT_SEAL_SCHEMA_VERSION,
        "completed_at": time.time(),
        "annotation_mode": "blind_without_model_predictions",
        "prediction_inputs_used": False,
        "model": _fingerprint(model_path),
        "manifest": _fingerprint(manifest_path),
        "training_manifest": _fingerprint(training_manifest_path),
        "implementation": {
            "project_root": project_root.as_posix(),
            "sha256": sha256_python_implementation(project_root),
        },
        "sessions": sessions,
        "summary": summary,
    }
    seal["report_sha256"] = sha256_json_payload(seal)
    write_json_atomic(output_path, seal)
    return seal


def validate_holdout_seal(
    seal_path: Path,
    *,
    model_path: Path,
    manifest_path: Path,
    training_manifest_path: Path,
) -> dict[str, object]:
    seal = _read_sealed_object(
        seal_path,
        schema_version=HOLDOUT_SEAL_SCHEMA_VERSION,
        label="holdout seal",
    )
    if seal.get("annotation_mode") != "blind_without_model_predictions":
        raise ValueError("holdout seal was not created in blind mode")
    if seal.get("prediction_inputs_used") is not False:
        raise ValueError("holdout seal declares prediction-assisted labels")
    completed_at = seal.get("completed_at")
    if (
        isinstance(completed_at, bool)
        or not isinstance(completed_at, (int, float))
    ):
        raise ValueError("holdout seal completion time is invalid")
    for name, path in (
        ("model", model_path),
        ("manifest", manifest_path),
        ("training_manifest", training_manifest_path),
    ):
        _validate_fingerprint(seal.get(name), path, f"holdout {name}")
    project_root = Path(__file__).resolve().parents[1]
    implementation = _mapping(seal.get("implementation"))
    if implementation.get("project_root") != project_root.as_posix():
        raise ValueError("holdout implementation root does not match")
    if implementation.get("sha256") != sha256_python_implementation(project_root):
        raise ValueError("holdout implementation changed after label sealing")
    records = load_holdout_manifest(manifest_path)
    summary, sessions = _validate_holdout_structure(records, manifest_path)
    if seal.get("summary") != summary:
        raise ValueError("holdout seal summary no longer matches the manifest")
    if seal.get("sessions") != sessions:
        raise ValueError("holdout session fingerprints changed after sealing")
    return seal


def _validate_holdout_structure(
    records: list[dict[str, object]],
    manifest_path: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    class_counts = Counter(str(record["label"]) for record in records)
    source_counts = Counter(str(record["source_id"]) for record in records)
    if len(records) < 300:
        raise ValueError("blind holdout seal requires at least 300 samples")
    if any(class_counts[label] < 10 for label in CARD_CLASSES):
        raise ValueError("blind holdout seal requires at least 10 samples per class")
    if len(source_counts) < 3:
        raise ValueError("blind holdout seal requires at least 3 independent sources")

    session_fingerprints: list[dict[str, str]] = []
    for source_id in sorted(source_counts):
        source_records = [
            record for record in records if record["source_id"] == source_id
        ]
        indices = [record.get("crop_index") for record in source_records]
        if indices != list(range(len(source_records))):
            raise ValueError(f"holdout source {source_id} crop indices are invalid")
        session_path = manifest_path.parent / "sessions" / source_id / "session.json"
        session = _read_sealed_object(
            session_path,
            schema_version=HOLDOUT_SESSION_SCHEMA_VERSION,
            label=f"holdout source {source_id}",
        )
        if session.get("source_id") != source_id:
            raise ValueError(f"holdout source {source_id} metadata does not match")
        if session.get("crop_count") != len(source_records):
            raise ValueError(f"holdout source {source_id} crop count does not match")
        if session.get("labels") != [
            str(record["label"]) for record in source_records
        ]:
            raise ValueError(f"holdout source {source_id} labels changed")
        if session.get("manifest") != manifest_path.resolve().as_posix():
            raise ValueError(f"holdout source {source_id} manifest path does not match")
        roi_hashes = {str(record.get("roi_sha256") or "") for record in source_records}
        if len(roi_hashes) != 1 or session.get("roi_sha256") not in roi_hashes:
            raise ValueError(f"holdout source {source_id} ROI fingerprint does not match")
        _validate_fingerprint(
            session.get("roi"),
            Path(str(_mapping(session.get("roi")).get("path") or "")),
            f"holdout source {source_id} ROI",
        )
        _validate_fingerprint(
            session.get("contact_sheet"),
            session_path.parent / "contact_sheet.png",
            f"holdout source {source_id} contact sheet",
        )
        session_fingerprints.append(_fingerprint(session_path))
    return ({
        "sample_count": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }, session_fingerprints)


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
    minimum_accuracy: float = 0.95,
    minimum_per_class_accuracy: float = 0.90,
    holdout_seal_path: Path | None = None,
    require_holdout_seal: bool = False,
) -> dict[str, object]:
    holdout_seal: dict[str, object] | None = None
    if holdout_seal_path is not None and holdout_seal_path.is_file():
        if training_manifest_path is None:
            raise ValueError("blind holdout seal requires a training manifest")
        holdout_seal = validate_holdout_seal(
            holdout_seal_path,
            model_path=model_path,
            manifest_path=manifest_path,
            training_manifest_path=training_manifest_path,
        )
    elif require_holdout_seal:
        raise ValueError("formal holdout evaluation requires a blind holdout seal")
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
        minimum_accuracy=minimum_accuracy,
        minimum_per_class_accuracy=minimum_per_class_accuracy,
        blind_holdout_sealed=holdout_seal is not None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    confusion_matrix_path = output_dir / "confusion_matrix.csv"
    _write_jsonl(predictions_path, rows)
    _write_jsonl(
        errors_path,
        [row for row in rows if not row["correct"]],
    )
    _write_confusion_matrix(
        confusion_matrix_path,
        classes,
        rows,
    )
    report.update({
        "schema_version": "real-window-holdout-v3",
        "created_at": time.time(),
        "model": str(model_path),
        "manifest": str(manifest_path),
        "inputs": {
            "model": _fingerprint(model_path),
            "manifest": _fingerprint(manifest_path),
            "training_manifest": (
                _fingerprint(training_manifest_path)
                if training_manifest_path is not None
                and training_manifest_path.is_file()
                else None
            ),
        },
        "artifacts": {
            "predictions": _fingerprint(predictions_path),
            "errors": _fingerprint(errors_path),
            "confusion_matrix": _fingerprint(confusion_matrix_path),
        },
        "holdout_seal": (
            _fingerprint(holdout_seal_path)
            if holdout_seal is not None and holdout_seal_path is not None
            else None
        ),
    })
    report["report_sha256"] = sha256_json_payload(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
    minimum_accuracy: float = 0.95,
    minimum_per_class_accuracy: float = 0.90,
    blind_holdout_sealed: bool = False,
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
            "name": "blind_labels_sealed_before_prediction",
            "passed": blind_holdout_sealed,
            "evidence": "holdout-seal.json" if blind_holdout_sealed else "not_available",
        },
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
    accuracy = correct_count / len(records)
    class_accuracies = [
        float(per_class[label]["accuracy"])
        for label in classes
        if isinstance(per_class[label]["accuracy"], (int, float))
    ]
    readiness_checks.extend((
        {
            "name": "minimum_accuracy",
            "passed": accuracy >= minimum_accuracy,
            "evidence": f"{accuracy:.6f}/{minimum_accuracy:.6f}",
        },
        {
            "name": "minimum_per_class_accuracy",
            "passed": (
                len(class_accuracies) == len(classes)
                and min(class_accuracies) >= minimum_per_class_accuracy
            ),
            "evidence": (
                f"minimum={min(class_accuracies, default=0.0):.6f}/"
                f"{minimum_per_class_accuracy:.6f}"
            ),
        },
    ))
    return ({
        "schema_version": "real-window-holdout-v3",
        "sample_count": len(records),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "error_count": len(records) - correct_count,
        "low_confidence_count": sum(bool(row["low_confidence"]) for row in rows),
        "focus_error_count": sum(bool(row["focus_error"]) for row in rows),
        "confidence_threshold": confidence_threshold,
        "accuracy_thresholds": {
            "overall": minimum_accuracy,
            "per_class": minimum_per_class_accuracy,
        },
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
            "Publication readiness requires both dataset independence and explicit accuracy thresholds.",
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


def _fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
    }


def _read_sealed_object(
    path: Path,
    *,
    schema_version: str,
    label: str,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"{label} schema is missing or unsupported")
    sealed = dict(payload)
    report_sha256 = sealed.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed):
        raise ValueError(f"{label} checksum is invalid")
    return payload


def _validate_fingerprint(value: object, path: Path, label: str) -> None:
    fingerprint = _mapping(value)
    expected = path.resolve()
    if fingerprint.get("path") != expected.as_posix():
        raise ValueError(f"{label} path does not match")
    if not expected.is_file():
        raise ValueError(f"{label} file is missing: {expected}")
    if fingerprint.get("sha256") != sha256_file(expected):
        raise ValueError(f"{label} checksum changed after sealing")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


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
    parser.add_argument("--minimum-accuracy", type=float, default=0.95)
    parser.add_argument(
        "--minimum-per-class-accuracy",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--holdout-seal",
        default="data/real_window_holdout/holdout-seal.json",
    )
    parser.add_argument("--require-seal", action="store_true")
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
            minimum_accuracy=args.minimum_accuracy,
            minimum_per_class_accuracy=args.minimum_per_class_accuracy,
            holdout_seal_path=Path(args.holdout_seal),
            require_holdout_seal=args.require_seal,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
