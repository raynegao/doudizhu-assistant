from __future__ import annotations

from PIL import Image
import torch

from src.vision.card_classifier import CARD_CLASSES, CardClassifierCNN, preprocess_image


def test_card_classes_match_rule_engine_order() -> None:
    assert CARD_CLASSES == ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ")


def test_card_classifier_forward_shape() -> None:
    model = CardClassifierCNN(num_classes=len(CARD_CLASSES))
    logits = model(torch.zeros(2, 3, 96, 64))
    assert logits.shape == (2, len(CARD_CLASSES))


def test_preprocess_image_returns_normalized_chw_tensor() -> None:
    image = Image.new("RGB", (126, 210), color=(255, 128, 0))
    tensor = preprocess_image(image, image_size=(64, 96))
    assert tensor.shape == (3, 96, 64)
    assert torch.isclose(tensor.max(), torch.tensor(1.0))
    assert tensor.min() >= 0
