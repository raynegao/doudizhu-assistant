from __future__ import annotations

import subprocess
import sys


def test_observation_types_and_helpers_do_not_import_inference_frameworks() -> None:
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import src.vision.scene_types; "
            "import src.tracking.observation_evidence; "
            "assert 'torch' not in sys.modules; "
            "assert 'src.pipeline.live_runtime' not in sys.modules; "
            "assert 'src.vision.scene_recognizer' not in sys.modules"
        )],
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_old_observation_imports_remain_identical() -> None:
    from src.tracking import observation_evidence, visual_events
    from src.vision import scene_recognizer, scene_types

    for name in (
        "VisualSignal", "SeatRole", "RoundResultOutcome", "TemplateMatch", "VisualCard",
        "SeatObservation", "RoundResultObservation", "SceneObservation",
    ):
        assert getattr(scene_recognizer, name) is getattr(scene_types, name)
    assert visual_events._seat_fingerprint is observation_evidence._seat_fingerprint


def test_legacy_classifier_exports_still_load_on_demand() -> None:
    import src.vision as vision
    from src.vision.card_classifier import CardClassifierCNN

    assert vision.CardClassifierCNN is CardClassifierCNN
    assert "CardClassifierCNN" in dir(vision)
    try:
        _ = vision.unknown_export
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown exports must raise AttributeError")
