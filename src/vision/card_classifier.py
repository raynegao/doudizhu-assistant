from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
import torch
from torch import Tensor, nn
import torch.nn.functional as F


CARD_CLASSES: tuple[str, ...] = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ")
CLASS_TO_INDEX: dict[str, int] = {rank: index for index, rank in enumerate(CARD_CLASSES)}
DEFAULT_IMAGE_SIZE: tuple[int, int] = (64, 96)


@dataclass(frozen=True)
class CardPrediction:
    rank: str
    confidence: float
    probabilities: dict[str, float]


class CardClassifierCNN(nn.Module):
    def __init__(self, num_classes: int = len(CARD_CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))


def select_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_image(image: Image.Image, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB").resize(image_size, Image.Resampling.BICUBIC)
    raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = raw.reshape(image_size[1], image_size[0], 3).permute(2, 0, 1).float() / 255.0
    return tensor


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[CardClassifierCNN, tuple[str, ...], tuple[int, int]]:
    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    classes = tuple(checkpoint.get("classes", CARD_CLASSES))
    image_size = tuple(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    model = CardClassifierCNN(num_classes=len(classes))
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model, classes, image_size  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    model: CardClassifierCNN,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    classes: Iterable[str] = CARD_CLASSES,
    extra: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "classes": tuple(classes),
        "image_size": image_size,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


@torch.inference_mode()
def predict_tensors(
    model: CardClassifierCNN,
    images: Tensor,
    classes: tuple[str, ...] = CARD_CLASSES,
    device: torch.device | str = "cpu",
) -> list[CardPrediction]:
    logits = model(images.to(device))
    probabilities = F.softmax(logits, dim=1).detach().cpu()
    results: list[CardPrediction] = []
    for row in probabilities:
        confidence, index = torch.max(row, dim=0)
        results.append(CardPrediction(
            rank=classes[int(index.item())],
            confidence=float(confidence.item()),
            probabilities={classes[i]: float(row[i].item()) for i in range(len(classes))},
        ))
    return results


@torch.inference_mode()
def predict_image_paths(
    model: CardClassifierCNN,
    paths: Iterable[Path],
    classes: tuple[str, ...] = CARD_CLASSES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    device: torch.device | str = "cpu",
) -> list[CardPrediction]:
    tensors = [preprocess_image(Image.open(path), image_size=image_size) for path in paths]
    if not tensors:
        return []
    batch = torch.stack(tensors)
    return predict_tensors(model, batch, classes=classes, device=device)


__all__ = [
    "CARD_CLASSES",
    "CLASS_TO_INDEX",
    "DEFAULT_IMAGE_SIZE",
    "CardClassifierCNN",
    "CardPrediction",
    "load_checkpoint",
    "predict_image_paths",
    "predict_tensors",
    "preprocess_image",
    "save_checkpoint",
    "select_device",
]
