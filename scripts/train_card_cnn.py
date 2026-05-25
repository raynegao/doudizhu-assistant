from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import warnings

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.vision.card_classifier import (
    CARD_CLASSES,
    DEFAULT_IMAGE_SIZE,
    CLASS_TO_INDEX,
    CardClassifierCNN,
    preprocess_image,
    save_checkpoint,
    select_device,
)


class CardImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, image_size: tuple[int, int]) -> None:
        self.root = root
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for rank in CARD_CLASSES:
            rank_dir = root / rank
            for path in sorted(rank_dir.glob("*.png")):
                self.samples.append((path, CLASS_TO_INDEX[rank]))
        if not self.samples:
            raise FileNotFoundError(f"No card images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return preprocess_image(image, self.image_size), label


def train_model(
    dataset_dir: Path,
    output: Path,
    onnx_output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: tuple[int, int],
    device_name: str,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = select_device(device_name)

    train_dataset = CardImageDataset(dataset_dir / "train", image_size=image_size)
    val_dataset = CardImageDataset(dataset_dir / "val", image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CardClassifierCNN(num_classes=len(CARD_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_count += images.size(0)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_loss = total_loss / max(total_count, 1)
        print(
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            metrics = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            save_checkpoint(output, model, image_size=image_size, extra={"metrics": metrics})

    model, _, _ = _load_best_model(output, device)
    export_onnx(model, onnx_output, image_size=image_size)
    (output.with_suffix(".metrics.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@torch.inference_mode()
def evaluate(
    model: CardClassifierCNN,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def export_onnx(model: CardClassifierCNN, output: Path, image_size: tuple[int, int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, image_size[1], image_size[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model,
            dummy,
            output,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )


def _load_best_model(path: Path, device: torch.device) -> tuple[CardClassifierCNN, tuple[str, ...], tuple[int, int]]:
    from src.vision.card_classifier import load_checkpoint

    return load_checkpoint(path, device=device)


def parse_image_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Image size must be WIDTHxHEIGHT, e.g. 64x96")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Phase 2 CNN card classifier.")
    parser.add_argument("--dataset", default="data/cards_cls", help="Dataset root with train/val rank folders.")
    parser.add_argument("--output", default="models/card_cnn.pt", help="PyTorch checkpoint path.")
    parser.add_argument("--onnx-output", default="models/card_cnn.onnx", help="ONNX export path.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=parse_image_size, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_model(
        dataset_dir=Path(args.dataset),
        output=Path(args.output),
        onnx_output=Path(args.onnx_output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
