from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.train_card_cnn import CardImageDataset
from src.vision.card_classifier import CARD_CLASSES, load_checkpoint, select_device


FOCUS_PAIRS: tuple[set[str], ...] = (
    {"SJ", "BJ"},
    {"10", "J"},
    {"6", "9"},
    {"A", "K", "Q", "J"},
)


def evaluate_model(
    model_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    batch_size: int,
    device_name: str,
) -> dict[str, object]:
    device = select_device(device_name)
    model, classes, image_size = load_checkpoint(model_path, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metrics: dict[str, dict[str, object]] = {}
    aggregate_matrix = [[0 for _ in classes] for _ in classes]
    errors: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        split_root = dataset_dir / split
        if not split_root.exists():
            continue
        dataset = CardImageDataset(split_root, image_size=image_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        sample_offset = 0
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).detach().cpu()
            for row_index, (expected_index, predicted_index) in enumerate(zip(labels, predictions, strict=True)):
                expected = classes[int(expected_index.item())]
                predicted = classes[int(predicted_index.item())]
                aggregate_matrix[int(expected_index.item())][int(predicted_index.item())] += 1
                total += 1
                if expected == predicted:
                    correct += 1
                else:
                    source_path = dataset.samples[sample_offset + row_index][0]
                    errors.append({
                        "split": split,
                        "path": str(source_path),
                        "expected": expected,
                        "predicted": predicted,
                        "focus_error": _is_focus_error(expected, predicted),
                    })
            sample_offset += len(labels)
        split_metrics[split] = {
            "count": total,
            "accuracy": correct / max(total, 1),
        }

    _write_confusion_matrix(output_dir / "card_cnn.confusion_matrix.csv", classes, aggregate_matrix)
    _write_error_report(output_dir / "card_cnn.error_report.jsonl", errors)
    metrics = {
        "model": str(model_path),
        "dataset": str(dataset_dir),
        "splits": split_metrics,
        "confusion_matrix": str(output_dir / "card_cnn.confusion_matrix.csv"),
        "error_report": str(output_dir / "card_cnn.error_report.jsonl"),
        "error_count": len(errors),
        "focus_error_count": sum(1 for error in errors if error["focus_error"]),
    }
    (output_dir / "card_cnn.eval.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _is_focus_error(expected: str, predicted: str) -> bool:
    return any(expected in pair and predicted in pair for pair in FOCUS_PAIRS)


def _write_confusion_matrix(path: Path, classes: tuple[str, ...], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["expected\\predicted", *classes])
        for label, row in zip(classes, matrix, strict=True):
            writer.writerow([label, *row])


def _write_error_report(path: Path, errors: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 2 CNN card classifier.")
    parser.add_argument("--model", default="models/card_cnn.pt")
    parser.add_argument("--dataset", default="data/cards_cls")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = evaluate_model(
        model_path=Path(args.model),
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
