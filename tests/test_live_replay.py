from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.evaluate_live_replay import evaluate_live_replay
from src.capture.recorded_window import RecordedWindowFrameSource


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_recorded_window_source_reads_contiguous_manifest(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (120, 80), "navy").save(frames / "000001.png")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{
        "frame_id": 1,
        "timestamp": 12.5,
        "full_image": "frames/000001.png",
    }])

    source = RecordedWindowFrameSource(manifest)
    frame = source.capture(1)

    assert source.frame_count == 1
    assert frame.timestamp == 12.5
    assert frame.image.size == (120, 80)


def test_live_replay_evaluator_reports_events_remaining_and_invariant(
    tmp_path: Path,
) -> None:
    predicted = tmp_path / "predicted.jsonl"
    expected = tmp_path / "expected.jsonl"
    scenes = tmp_path / "scenes.jsonl"
    action = {
        "event": "play_observed",
        "sequence_no": 1,
        "actor": "self",
        "cards": ["3"],
    }
    _write_jsonl(predicted, [
        action,
        {
            "event": "scene_observation",
            "frame_id": 1,
            "seats": [
                {"seat": "self", "remaining_count": 19},
                {"seat": "right", "remaining_count": 17},
                {"seat": "left", "remaining_count": 17},
            ],
        },
        {
            "event": "state_update",
            "state": {
                "remaining_cards": {"self": 19, "right": 17, "left": 17},
                "played_cards": ["3"],
            },
        },
    ])
    _write_jsonl(expected, [action])
    _write_jsonl(scenes, [{
        "frame_id": 1,
        "remaining": {"self": 19, "right": 17, "left": 17},
    }])

    report = evaluate_live_replay(
        predicted,
        expected,
        expected_scenes=scenes,
    )

    assert report["event_f1"] == 1.0
    assert report["card_exact_accuracy"] == 1.0
    assert report["remaining_accuracy"] == 1.0
    assert report["deck_invariant_passed"] is True
    assert report["passed"] is True
