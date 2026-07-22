from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.evaluate_real_window_holdout import (
    evaluate_holdout,
    find_training_leakage,
    load_holdout_manifest,
    load_training_hashes,
    summarize_predictions,
)
from scripts.prepare_real_window_holdout import prepare_holdout_session
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
    assert Path(summary["contact_sheet"]).is_file()
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
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "predictions.jsonl").is_file()
    assert (output_dir / "errors.jsonl").is_file()
    assert (output_dir / "confusion_matrix.csv").is_file()
