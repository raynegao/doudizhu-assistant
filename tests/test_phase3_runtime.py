from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from src.pipeline.runtime import (
    CardObservation,
    MacScreenFrameSource,
    Phase3Runtime,
    RuntimeCaptureError,
    RuntimeSettings,
    ScreenFrame,
    compute_crop_boxes,
    format_runtime_event,
)
from src.pipeline.calibration import (
    WindowInfo,
    calibrate_roi,
    load_runtime_config,
    parse_window_info,
    parse_window_candidates,
    save_runtime_config,
)
from src.pipeline.stabilizer import ObservationStabilizer
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
    assert data["raw_recognized_cards"] == ["3", "4", "5", "6", "7"]
    assert data["recognized_cards"] == ["3", "4", "5", "6", "7"]
    assert data["observations"][0]["box"] == [0, 20, 40, 100]
    assert data["stabilized"] is False
    assert data["stability_window"] == 1
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


def test_parse_window_info_and_calibrate_roi() -> None:
    window = parse_window_info("147, 82, 1176, 767, 斗地主", app_name="斗地主")

    assert window == WindowInfo(
        app_name="斗地主",
        window_name="斗地主",
        window_box=(147, 82, 1323, 849),
    )
    calibration = calibrate_roi(window, created_at=1.0)
    assert calibration.roi_box == (180, 555, 1323, 758)
    assert calibration.count == 17
    assert calibration.step_x == 60
    assert calibration.crop_size == (63, 105)


def test_parse_window_candidates_keeps_standard_game_window() -> None:
    output = (
        "157\t86\t66\t20\tWindow\tAXDialog\n"
        "147\t82\t1176\t767\t斗地主\tAXStandardWindow\n"
    )

    candidates = parse_window_candidates(output, app_name="斗地主")

    assert candidates == [
        (
            WindowInfo(
                app_name="斗地主",
                window_name="Window",
                window_box=(157, 86, 223, 106),
            ),
            "AXDialog",
        ),
        (
            WindowInfo(
                app_name="斗地主",
                window_name="斗地主",
                window_box=(147, 82, 1323, 849),
            ),
            "AXStandardWindow",
        ),
    ]


def test_runtime_config_roundtrip(tmp_path: Path) -> None:
    calibration = calibrate_roi(
        WindowInfo(app_name="斗地主", window_name="斗地主", window_box=(147, 82, 1323, 849)),
        created_at=1.0,
    )
    path = tmp_path / "phase3_runtime.local.json"
    save_runtime_config(path, calibration)

    config = load_runtime_config(path)

    assert config.app_name == "斗地主"
    assert config.roi_box == (180, 555, 1323, 758)
    assert config.count == 17
    assert config.crop_size == (63, 105)


def test_observation_stabilizer_majority_and_confidence_tiebreak() -> None:
    stabilizer = ObservationStabilizer(window_size=3)
    first = (
        CardObservation(index=0, rank="3", confidence=0.90, box=(0, 0, 1, 1)),
        CardObservation(index=1, rank="4", confidence=0.20, box=(1, 0, 2, 1)),
    )
    second = (
        CardObservation(index=0, rank="4", confidence=0.60, box=(0, 0, 1, 1)),
        CardObservation(index=1, rank="5", confidence=0.80, box=(1, 0, 2, 1)),
    )
    third = (
        CardObservation(index=0, rank="4", confidence=0.70, box=(0, 0, 1, 1)),
        CardObservation(index=1, rank="6", confidence=0.50, box=(1, 0, 2, 1)),
    )

    assert stabilizer.update(first).stable == first
    assert stabilizer.update(second).stable[0].rank == "3"
    result = stabilizer.update(third)

    assert result.stable[0].rank == "4"
    assert result.stable[0].confidence == pytest.approx(0.65)
    assert result.stable[1].rank == "5"
    assert result.stable[1].confidence == pytest.approx(0.80)


def test_phase3_runtime_uses_stabilized_cards_for_decision() -> None:
    settings = RuntimeSettings(
        count=1,
        step_x=50,
        crop_size=(40, 80),
        log_file=None,
        stability_window=3,
    )
    ranks_by_call = iter([["3"], ["4"], ["4"]])

    def predictor(crops):
        [rank] = next(ranks_by_call)
        return [CardPrediction(rank=rank, confidence=0.90, probabilities={rank: 0.90})]

    runtime = Phase3Runtime(
        settings,
        frame_source=FakeFrameSource(),
        predictor=predictor,
        stabilizer=ObservationStabilizer(window_size=3),
    )

    runtime.run_once()
    second = runtime.run_once()
    third = runtime.run_once()

    assert second.raw_recognized_cards == ("4",)
    assert second.recognized_cards == ("3",)
    assert third.raw_recognized_cards == ("4",)
    assert third.recognized_cards == ("4",)
    assert third.stability_window == 3


def test_run_phase3_cli_config_allows_cli_overrides(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_phase3_runtime

    config_path = tmp_path / "phase3_runtime.local.json"
    config_path.write_text(
        json.dumps({
            "roi_box": [10, 20, 110, 220],
            "count": 17,
            "start_x": 0,
            "start_y": 10,
            "step_x": 60,
            "crop_size": [63, 105],
            "device_name": "cpu",
            "stability_window": 3,
        }),
        encoding="utf-8",
    )
    captured = {}

    class DummyRuntime:
        def __init__(self, settings, stabilizer=None):
            captured["settings"] = settings
            captured["stabilizer"] = stabilizer

        def run_loop(self, max_frames=None, interval=1.0):
            captured["max_frames"] = max_frames
            captured["interval"] = interval
            return iter(())

    monkeypatch.setattr(run_phase3_runtime, "Phase3Runtime", DummyRuntime)

    assert run_phase3_runtime.main([
        "--config",
        str(config_path),
        "--count",
        "15",
        "--max-frames",
        "1",
    ]) == 0

    settings = captured["settings"]
    assert settings.roi_box == (10, 20, 110, 220)
    assert settings.count == 15
    assert settings.step_x == 60
    assert settings.stability_window == 3
    assert captured["stabilizer"] is not None
    assert captured["max_frames"] == 1
