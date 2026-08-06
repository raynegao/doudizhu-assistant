from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.evaluate_real_window_holdout import (
    evaluate_holdout,
    find_training_leakage,
    load_holdout_manifest,
    load_training_hashes,
    seal_holdout_manifest,
    summarize_predictions,
    validate_holdout_seal,
)
from scripts.prepare_real_window_holdout import prepare_holdout_session
from src.capture.recording_integrity import sha256_file
from src.vision.card_classifier import (
    CARD_CLASSES,
    CardClassifierCNN,
    CardPrediction,
    save_checkpoint,
)


def _roi(path: Path) -> None:
    image = Image.new("RGB", (40, 20), "white")
    for x, color in ((0, (255, 0, 0)), (20, (0, 0, 255))):
        for xx in range(x, x + 10):
            for yy in range(12):
                image.putpixel((xx, yy), color)
    image.save(path)


def test_prepare_holdout_session_crops_hashes_and_registers_manifest(tmp_path: Path) -> None:
    roi_path = tmp_path / "fresh_roi.png"
    _roi(roi_path)
    output_root = tmp_path / "holdout"

    summary = prepare_holdout_session(
        roi_path=roi_path,
        output_root=output_root,
        source_id="window-a-round-001",
        labels=("3", "A"),
        count=2,
        start_x=0,
        start_y=0,
        step_x=20,
        crop_size=(10, 12),
    )

    assert summary["crop_count"] == 2
    assert Path(summary["contact_sheet"]["path"]).is_file()
    manifest = output_root / "manifest.jsonl"
    records = load_holdout_manifest(manifest)
    assert [record["label"] for record in records] == ["3", "A"]
    assert len({record["sha256"] for record in records}) == 2
    assert all(record["roi_sha256"] == summary["roi_sha256"] for record in records)

    with pytest.raises(ValueError, match="source-id already exists"):
        prepare_holdout_session(
            roi_path=roi_path,
            output_root=output_root,
            source_id="window-a-round-001",
            labels=("3", "A"),
            count=2,
            start_x=0,
            start_y=0,
            step_x=20,
            crop_size=(10, 12),
        )


def test_blind_holdout_seal_binds_labels_model_and_sessions(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "holdout"
    all_labels = [CARD_CLASSES[index % len(CARD_CLASSES)] for index in range(300)]
    for source_index in range(3):
        roi_path = tmp_path / f"source-{source_index}.png"
        roi = Image.new("RGB", (100, 1))
        for x in range(100):
            unique = source_index * 100 + x
            roi.putpixel((x, 0), (unique % 256, unique // 256, 1))
        roi.save(roi_path)
        prepare_holdout_session(
            roi_path=roi_path,
            output_root=output_root,
            source_id=f"source-{source_index}",
            labels=all_labels[source_index * 100 : (source_index + 1) * 100],
            count=100,
            start_x=0,
            start_y=0,
            step_x=1,
            crop_size=(1, 1),
        )
    model = tmp_path / "model.pt"
    model.write_bytes(b"frozen-model")
    training_manifest = tmp_path / "training.jsonl"
    training_manifest.write_text("{}\n", encoding="utf-8")
    manifest = output_root / "manifest.jsonl"
    seal_path = output_root / "holdout-seal.json"

    seal = seal_holdout_manifest(
        model_path=model,
        manifest_path=manifest,
        training_manifest_path=training_manifest,
        output_path=seal_path,
        predictions_root=tmp_path / "predictions-not-created",
    )
    validated = validate_holdout_seal(
        seal_path,
        model_path=model,
        manifest_path=manifest,
        training_manifest_path=training_manifest,
    )

    assert seal["prediction_inputs_used"] is False
    assert validated["summary"]["sample_count"] == 300
    assert len(validated["summary"]["source_counts"]) == 3
    assert min(validated["summary"]["class_counts"].values()) == 20

    model.write_bytes(b"tuned-after-looking")
    with pytest.raises(ValueError, match="model checksum changed"):
        validate_holdout_seal(
            seal_path,
            model_path=model,
            manifest_path=manifest,
            training_manifest_path=training_manifest,
        )


def test_formal_holdout_evaluation_requires_blind_seal(tmp_path: Path) -> None:
    image = tmp_path / "card.png"
    Image.new("RGB", (10, 10), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "label": "3", "source_id": "source-a"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a blind holdout seal"):
        evaluate_holdout(
            tmp_path / "missing-model.pt",
            manifest,
            tmp_path / "report",
            "cpu",
            training_manifest_path=tmp_path / "training.jsonl",
            holdout_seal_path=tmp_path / "missing-seal.json",
            require_holdout_seal=True,
        )


def test_training_manifest_hashes_block_exact_holdout_leakage(tmp_path: Path) -> None:
    image = tmp_path / "shared.png"
    Image.new("RGB", (10, 10), "white").save(image)
    holdout_manifest = tmp_path / "holdout.jsonl"
    holdout_manifest.write_text(
        json.dumps({"image": "shared.png", "label": "3", "source_id": "window-a"}) + "\n",
        encoding="utf-8",
    )
    training_manifest = tmp_path / "training.jsonl"
    training_manifest.write_text(
        json.dumps({"source_path": str(image)}) + "\n",
        encoding="utf-8",
    )

    records = load_holdout_manifest(holdout_manifest)
    leakage = find_training_leakage(records, load_training_hashes(training_manifest))
    assert leakage == [{"image": "shared.png", "sha256": records[0]["sha256"]}]


def test_holdout_summary_reports_per_class_focus_errors_and_readiness(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (10, 10), "red").save(first)
    Image.new("RGB", (10, 10), "blue").save(second)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join((
            json.dumps({"image": first.name, "label": "3", "source_id": "window-a"}),
            json.dumps({"image": second.name, "label": "A", "source_id": "window-b"}),
        )) + "\n",
        encoding="utf-8",
    )
    records = load_holdout_manifest(manifest)
    predictions = [
        CardPrediction(rank="3", confidence=0.99, probabilities={}),
        CardPrediction(rank="K", confidence=0.55, probabilities={}),
    ]

    report, rows = summarize_predictions(
        records,
        predictions,
        CARD_CLASSES,
        leakage_checked=True,
        training_manifest_path=tmp_path / "training.jsonl",
        minimum_samples=2,
        minimum_per_class=1,
        minimum_sources=2,
        confidence_threshold=0.70,
    )

    assert report["accuracy"] == 0.5
    assert report["per_class"]["3"]["accuracy"] == 1.0
    assert report["per_class"]["A"]["accuracy"] == 0.0
    assert report["focus_error_count"] == 1
    assert report["low_confidence_count"] == 1
    assert report["publication_ready"] is False
    assert rows[1]["focus_error"] is True


def test_holdout_evaluation_writes_complete_artifact_set(tmp_path: Path) -> None:
    image = tmp_path / "card.png"
    Image.new("RGB", (64, 96), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "label": "3", "source_id": "window-a"}) + "\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pt"
    save_checkpoint(model_path, CardClassifierCNN())
    output_dir = tmp_path / "report"

    report = evaluate_holdout(
        model_path,
        manifest,
        output_dir,
        "cpu",
        training_manifest_path=None,
        minimum_samples=1,
        minimum_per_class=1,
        minimum_sources=1,
    )

    assert report["sample_count"] == 1
    assert report["leakage_check"]["checked"] is False
    assert report["publication_ready"] is False
    assert report["schema_version"] == "real-window-holdout-v3"
    assert report["inputs"]["model"]["sha256"] == sha256_file(model_path)
    assert report["inputs"]["manifest"]["sha256"] == sha256_file(manifest)
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "predictions.jsonl").is_file()
    assert (output_dir / "errors.jsonl").is_file()
    assert (output_dir / "confusion_matrix.csv").is_file()
    assert report["artifacts"]["predictions"]["sha256"] == sha256_file(
        output_dir / "predictions.jsonl"
    )
