from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from src.pipeline.runtime import (
    MacScreenFrameSource,
    Phase3Runtime,
    RuntimeCaptureError,
    RuntimeSettings,
    ScreenFrame,
    compute_crop_boxes,
    format_runtime_event,
)
from src.vision.card_classifier import CardPrediction


class FakeFrameSource:
    def __init__(self, image_size: tuple[int, int] = (450, 240)) -> None:
        self.image = Image.new("RGB", image_size, color=(255, 255, 255))

    def capture(self, frame_id: int) -> ScreenFrame:
        return ScreenFrame(
            frame_id=frame_id,
            timestamp=123.0,
            image=self.image,
            roi_box=(0, 0, self.image.width, self.image.height),
            source="test",
        )


def _predictor(ranks: list[str], confidences: list[float] | None = None):
    values = confidences or [0.99 for _ in ranks]

    def predict(crops):
        assert len(crops) == len(ranks)
        return [
            CardPrediction(rank=rank, confidence=values[index], probabilities={rank: values[index]})
            for index, rank in enumerate(ranks)
        ]

    return predict


def test_phase3_run_once_connects_observations_to_decision_log(tmp_path: Path) -> None:
    log_file = tmp_path / "phase3.jsonl"
    settings = RuntimeSettings(
        count=5,
        step_x=50,
        crop_size=(40, 80),
        log_file=log_file,
        last_play="",
    )
    runtime = Phase3Runtime(
        settings,
        frame_source=FakeFrameSource(),
        predictor=_predictor(["3", "4", "5", "6", "7"]),
    )

    event = runtime.run_once(frame_id=7)

    assert event.frame_id == 7
    assert event.source == "test"
    assert event.recognized_cards == ("3", "4", "5", "6", "7")
    assert event.candidate_count > 0
    assert event.recommended_action
    assert event.error is None
    [payload] = log_file.read_text(encoding="utf-8").splitlines()
    data = json.loads(payload)
    assert data["event"] == "phase3_recommendation"
    assert data["recognized_cards"] == ["3", "4", "5", "6", "7"]
    assert data["observations"][0]["box"] == [0, 20, 40, 100]
    assert "latency_ms" in data


def test_phase3_low_confidence_predictions_become_warnings() -> None:
    settings = RuntimeSettings(
        count=2,
        step_x=50,
        crop_size=(40, 80),
        confidence_threshold=0.70,
        log_file=None,
    )
    runtime = Phase3Runtime(
        settings,
        frame_source=FakeFrameSource(),
        predictor=_predictor(["3", "4"], [0.99, 0.42]),
    )

    event = runtime.run_once()

    assert event.warnings == ("low-confidence card 01: 4=0.420",)
    assert "WARNING:" in format_runtime_event(event)


def test_phase3_invalid_recognition_records_error_instead_of_crashing() -> None:
    settings = RuntimeSettings(
        count=5,
        step_x=50,
        crop_size=(40, 80),
        log_file=None,
    )
    runtime = Phase3Runtime(
        settings,
        frame_source=FakeFrameSource(),
        predictor=_predictor(["3", "3", "3", "3", "3"]),
    )

    event = runtime.run_once()

    assert event.candidate_count == 0
    assert event.recommended_action == ()
    assert event.error
    assert event.warnings
    assert event.reason.startswith("无法生成推荐")


def test_compute_crop_boxes_validates_bounds() -> None:
    with pytest.raises(ValueError, match="exceeds ROI bounds"):
        compute_crop_boxes(
            count=3,
            start_x=0,
            start_y=20,
            step_x=50,
            crop_size=(40, 80),
            image_size=(100, 100),
        )


def test_mac_screen_capture_error_is_actionable(monkeypatch) -> None:
    from PIL import ImageGrab

    def fail_grab(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(ImageGrab, "grab", fail_grab)
    source = MacScreenFrameSource((0, 0, 100, 100))

    with pytest.raises(RuntimeCaptureError, match="Screen Recording permission"):
        source.capture(frame_id=1)
