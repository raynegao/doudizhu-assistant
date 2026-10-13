"""Vision package with lazy legacy classifier exports.

Importing immutable observations must not initialize PyTorch. The existing
``from src.vision import CardClassifierCNN`` API still loads it on demand.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .card_classifier import (
        CARD_CLASSES as CARD_CLASSES,
    )
    from .card_classifier import (
        CLASS_TO_INDEX as CLASS_TO_INDEX,
    )
    from .card_classifier import (
        DEFAULT_IMAGE_SIZE as DEFAULT_IMAGE_SIZE,
    )
    from .card_classifier import (
        CardClassifierCNN as CardClassifierCNN,
    )
    from .card_classifier import (
        CardPrediction as CardPrediction,
    )
    from .card_classifier import (
        load_checkpoint as load_checkpoint,
    )
    from .card_classifier import (
        predict_image_paths as predict_image_paths,
    )
    from .card_classifier import (
        predict_tensors as predict_tensors,
    )
    from .card_classifier import (
        preprocess_image as preprocess_image,
    )
    from .card_classifier import (
        save_checkpoint as save_checkpoint,
    )
    from .card_classifier import (
        select_device as select_device,
    )

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


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".card_classifier", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
