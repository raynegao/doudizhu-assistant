from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.annotate_live_session import (
    assert_blind_annotation_preconditions,
    prepare_annotation_workbook,
    seal_annotation_bundle,
    validate_annotation_bundle,
    validate_sealed_annotation,
)
from scripts.evaluate_live_replay import evaluate_live_replay
from scripts.record_live_game import _validate_session_name
from scripts.record_live_game import main as record_live_main
from scripts.replay_live_game import main as replay_live_main
from src.capture.recorded_window import RecordedWindowFrameSource
from src.capture.recording_integrity import (
    FRAME_SCHEMA_VERSION,
    LEGACY_FRAME_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    VIDEO_FRAME_SCHEMA_VERSION,
    finalize_recording_session,
    inspect_recording_session,
    runtime_versions,
    sha256_directory,
    sha256_file,
    sha256_json_payload,
    sha256_python_implementation,
)
from src.capture.screen_geometry import CapturedWindow, ScreenGeometry
from src.config.live_layout import LiveLayoutConfig, save_live_layout
from src.pipeline.calibration import WindowInfo
from src.reporting.live_acceptance import (
    Phase6AcceptanceThresholds,
    audit_phase6_acceptance,
)
from src.reporting.live_diagnostics import analyze_live_log


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


def test_recorded_window_source_rejects_checksum_mismatch(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "000001.png"
    Image.new("RGB", (120, 80), "navy").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{
        "frame_id": 1,
        "timestamp": 12.5,
        "full_image": "frames/000001.png",
        "full_image_sha256": hashlib.sha256(b"not-the-image").hexdigest(),
    }])

    source = RecordedWindowFrameSource(manifest)

    with pytest.raises(ValueError, match="checksum mismatch"):
        source.capture(1)


def test_recording_session_name_blocks_paths() -> None:
    assert _validate_session_name("game-001") == "game-001"
    with pytest.raises(ValueError, match="session"):
        _validate_session_name("../outside")


def test_implementation_fingerprint_covers_evaluation_toolchain(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    scripts = tmp_path / "scripts"
    src.mkdir()
    scripts.mkdir()
    (src / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    evaluator = scripts / "evaluate_live_replay.py"
    evaluator.write_text("THRESHOLD = 0.95\n", encoding="utf-8")
    native = tmp_path / "native"
    native.mkdir()
    capture_helper = native / "capture.swift"
    capture_helper.write_text("let fps = 12\n", encoding="utf-8")
    original = sha256_python_implementation(tmp_path)

    evaluator.write_text("THRESHOLD = 0.10\n", encoding="utf-8")

    evaluator_changed = sha256_python_implementation(tmp_path)
    assert evaluator_changed != original

    capture_helper.write_text("let fps = 1\n", encoding="utf-8")

    assert sha256_python_implementation(tmp_path) != evaluator_changed


class _FakeRecordingCapture:
    def __init__(
        self,
        *,
        interrupt_after: int | None = None,
        backend: str = "screen_capture_kit_stream",
    ) -> None:
        self.calls = 0
        self.interrupt_after = interrupt_after
        self.backend = backend

    def capture(self, frame_id: int) -> CapturedWindow:
        self.calls += 1
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise KeyboardInterrupt
        return CapturedWindow(
            frame_id=frame_id,
            timestamp=time.time(),
            image=Image.new(
                "RGB",
                (100, 100),
                (frame_id % 256, 20, 80),
            ),
            window=WindowInfo(
                app_name="斗地主",
                window_name="fixture",
                window_box=(0, 0, 100, 100),
            ),
            pixel_box=(0, 0, 100, 100),
            geometry=ScreenGeometry(
                logical_size=(100, 100),
                pixel_size=(100, 100),
            ),
            capture_backend=self.backend,
        )


def test_live_recorder_writes_complete_integrity_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig())
    recordings = tmp_path / "recordings"
    capture = _FakeRecordingCapture()
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: capture,
    )

    result = record_live_main([
        "--config",
        str(config_path),
        "--session",
        "game-001",
        "--frames",
        "1",
        "--mark-complete",
        "--output-root",
        str(recordings),
    ])
    inspection = inspect_recording_session(recordings / "game-001")

    assert result == 0
    assert inspection.valid is True
    assert inspection.metadata["complete_game"] is True
    assert inspection.metadata["recording_state"] == "complete"
    assert len(inspection.manifest_rows[0]["rois"]) == len(LiveLayoutConfig().rois)
    assert inspection.manifest_rows[0]["schema_version"] == FRAME_SCHEMA_VERSION
    assert (
        inspection.manifest_rows[0]["roi_storage"]
        == "derived_from_full_rgb"
    )
    assert not (recordings / "game-001" / "rois").exists()
    assert inspection.metadata["evidence_split"] == "development"


def test_acceptance_recorder_uses_exact_lossless_video_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig(interval_seconds=0.001))
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )
    monkeypatch.setattr("scripts.record_live_game.time.sleep", lambda _seconds: None)

    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--frames",
        "3",
        "--mark-complete",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 0
    session_dir = recordings / "acceptance-001"
    inspection = inspect_recording_session(session_dir)

    assert inspection.valid is True
    assert not (session_dir / "frames").exists()
    assert inspection.metadata["frame_storage"]["mode"] == (
        "lossless_video_segments"
    )
    assert all(
        row["schema_version"] == VIDEO_FRAME_SCHEMA_VERSION
        for row in inspection.manifest_rows
    )
    source = RecordedWindowFrameSource(session_dir / "manifest.jsonl")
    try:
        assert [
            source.capture(frame_id).image.getpixel((0, 0))
            for frame_id in (1, 2, 3)
        ] == [(1, 20, 80), (2, 20, 80), (3, 20, 80)]
    finally:
        source.close()


def test_acceptance_lossless_video_detects_container_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig(interval_seconds=0.001))
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )
    monkeypatch.setattr("scripts.record_live_game.time.sleep", lambda _seconds: None)
    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--frames",
        "2",
        "--mark-complete",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 0
    session_dir = recordings / "acceptance-001"
    video_path = next((session_dir / "video").glob("*.mkv"))
    with video_path.open("ab") as stream:
        stream.write(b"tampered")

    inspection = inspect_recording_session(session_dir)

    assert any(
        "video segment 1 checksum mismatch" in issue
        for issue in inspection.issues
    )


def test_live_recorder_persists_interrupted_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig(interval_seconds=0.001))
    recordings = tmp_path / "recordings"
    capture = _FakeRecordingCapture(interrupt_after=1)
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: capture,
    )
    monkeypatch.setattr("scripts.record_live_game.time.sleep", lambda _seconds: None)

    result = record_live_main([
        "--config",
        str(config_path),
        "--session",
        "game-001",
        "--until-interrupt",
        "--output-root",
        str(recordings),
    ])
    inspection = inspect_recording_session(recordings / "game-001")

    assert result == 130
    assert inspection.valid is True
    assert inspection.metadata["recorded_frames"] == 1
    assert inspection.metadata["recording_state"] == "interrupted"
    assert inspection.metadata["complete_game"] is False


def test_acceptance_recorder_resumes_into_a_new_lossless_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig(interval_seconds=0.001))
    recordings = tmp_path / "recordings"
    first_capture = _FakeRecordingCapture(interrupt_after=2)
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: first_capture,
    )
    monkeypatch.setattr("scripts.record_live_game.time.sleep", lambda _seconds: None)
    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--until-interrupt",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 130
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )

    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--frames",
        "2",
        "--resume",
        "--mark-complete",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 0
    session_dir = recordings / "acceptance-001"
    inspection = inspect_recording_session(session_dir)

    assert inspection.valid is True
    assert len(inspection.metadata["frame_storage"]["segments"]) == 2
    assert [
        row["video_frame_index"] for row in inspection.manifest_rows
    ] == [0, 1, 0, 1]


def test_acceptance_recorder_defaults_to_ten_fps_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig(interval_seconds=0.25))
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )

    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--frames",
        "3",
        "--mark-complete",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 0
    session_dir = recordings / "acceptance-001"
    inspection = inspect_recording_session(session_dir)

    assert inspection.valid is True
    assert inspection.metadata["interval_seconds"] == 0.1
    assert inspection.metadata["max_capture_gap_seconds"] == pytest.approx(0.3)
    assert inspection.metadata["capture_cadence"]["effective_fps"] >= 9.0


def test_acceptance_recorder_refuses_fallback_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig())
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(
            backend="window_server_screenshot"
        ),
    )

    with pytest.raises(SystemExit, match="persistent ScreenCaptureKit stream"):
        record_live_main([
            "--config",
            str(config_path),
            "--session",
            "acceptance-001",
            "--frames",
            "1",
            "--mark-complete",
            "--evidence-split",
            "acceptance",
            "--output-root",
            str(recordings),
        ])

    metadata = json.loads(
        (recordings / "acceptance-001" / "session.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["recording_state"] == "failed"
    assert metadata["complete_game"] is False


def test_acceptance_integrity_rejects_slow_capture_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig(interval_seconds=0.25))
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )
    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--frames",
        "3",
        "--mark-complete",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 0
    session_dir = recordings / "acceptance-001"
    manifest_path = session_dir / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row, timestamp in zip(rows, (1.0, 1.25, 1.5), strict=True):
        row["timestamp"] = timestamp
    _write_jsonl(manifest_path, rows)
    metadata_path = session_dir / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "created_at": 0.0,
        "completed_at": 2.0,
        "capture_cadence": {
            "target_interval_seconds": 0.1,
            "sample_count": 2,
            "mean_gap_seconds": 0.25,
            "median_gap_seconds": 0.25,
            "p95_gap_seconds": 0.25,
            "max_gap_seconds": 0.25,
            "effective_fps": 4.0,
        },
    })
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    inspection = inspect_recording_session(session_dir)

    assert "acceptance recording median capture gap exceeds 0.15s: 0.250s" in (
        inspection.issues
    )
    assert "acceptance recording p95 capture gap exceeds 0.20s: 0.250s" in (
        inspection.issues
    )


def test_live_replay_uses_snapshot_and_writes_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model-v1")
    templates_path = tmp_path / "templates"
    templates_path.mkdir()
    (templates_path / "role.png").write_bytes(b"template-v1")
    config_path = tmp_path / "live.json"
    save_live_layout(
        config_path,
        LiveLayoutConfig(
            model_path=model_path,
            templates_dir=templates_path,
        ),
    )
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )
    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "game-001",
        "--frames",
        "1",
        "--mark-complete",
        "--output-root",
        str(recordings),
    ]) == 0

    class _FakeReplayRuntime:
        def __init__(self, config: object, **_kwargs: object) -> None:
            self.config = config

        def run_loop(self, *, max_frames: int) -> object:
            Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.log_file).write_text("{}\n", encoding="utf-8")
            yield SimpleNamespace(
                tracker_update=SimpleNamespace(
                    mode=SimpleNamespace(value="finished")
                )
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "scripts.replay_live_game.LiveGameRuntime",
        _FakeReplayRuntime,
    )
    output_dir = tmp_path / "replay"
    manifest = recordings / "game-001" / "manifest.jsonl"

    result = replay_live_main([
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--quiet",
    ])
    provenance = json.loads(
        (output_dir / "replay.json").read_text(encoding="utf-8")
    )

    assert result == 0
    assert provenance["schema_version"] == REPLAY_SCHEMA_VERSION
    assert provenance["config"]["path"] == (
        recordings / "game-001" / "config.snapshot.json"
    ).resolve().as_posix()
    assert provenance["model"]["sha256"] == sha256_file(model_path)
    assert provenance["templates"]["sha256"] == sha256_directory(templates_path)
    with pytest.raises(SystemExit, match="output already exists"):
        replay_live_main([
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--quiet",
        ])


def test_acceptance_replay_requires_blind_annotation_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    save_live_layout(config_path, LiveLayoutConfig())
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(
        "scripts.record_live_game.MacWindowCapture",
        lambda _app_name: _FakeRecordingCapture(),
    )
    assert record_live_main([
        "--config",
        str(config_path),
        "--session",
        "acceptance-001",
        "--frames",
        "1",
        "--mark-complete",
        "--evidence-split",
        "acceptance",
        "--output-root",
        str(recordings),
    ]) == 0

    with pytest.raises(SystemExit, match="valid blind annotation"):
        replay_live_main([
            "--manifest",
            str(recordings / "acceptance-001" / "manifest.jsonl"),
            "--output-dir",
            str(tmp_path / "replay"),
            "--quiet",
        ])


def test_live_replay_evaluator_reports_events_remaining_and_invariant(
    tmp_path: Path,
) -> None:
    predicted = tmp_path / "predicted.jsonl"
    expected = tmp_path / "expected.jsonl"
    scenes = tmp_path / "scenes.jsonl"
    action = {
        "event": "play_observed",
        "round_id": "round-001",
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
                {"seat": "self", "remaining_count": 18},
                {"seat": "right", "remaining_count": 17},
                {"seat": "left", "remaining_count": 17},
            ],
        },
        {
            "event": "state_update",
            "state": {
                "remaining_cards": {"self": 18, "right": 17, "left": 17},
                "played_cards": ["3"],
                "hidden_played_count": 1,
            },
        },
        {
            "event": "round_result_detected",
            "round_id": "round-001",
            "winner": "self",
            "outcome": "victory",
        },
    ])
    _write_jsonl(expected, [
        {**action, "round_id": "human-session-alias"},
        {
            "event": "round_result_detected",
            "winner": "self",
            "outcome": "victory",
        },
    ])
    _write_jsonl(scenes, [{
        "frame_id": 1,
        "after_sequence_no": 1,
        "remaining": {"self": 18, "right": 17, "left": 17},
    }])

    report = evaluate_live_replay(
        predicted,
        expected,
        expected_scenes=scenes,
        require_complete_round=True,
    )

    assert report["event_f1"] == 1.0
    assert report["card_exact_accuracy"] == 1.0
    assert report["remaining_accuracy"] == 1.0
    assert report["deck_invariant_passed"] is True
    assert report["round_result_accuracy"] == 1.0
    assert report["session_success"] is True
    assert report["passed"] is True
    assert report["inputs"]["predicted_log"]["sha256"] == hashlib.sha256(
        predicted.read_bytes()
    ).hexdigest()


def test_live_replay_requires_remaining_labels_for_every_expected_action(
    tmp_path: Path,
) -> None:
    predicted = tmp_path / "predicted.jsonl"
    expected = tmp_path / "expected.jsonl"
    scenes = tmp_path / "scenes.jsonl"
    actions = [
        {
            "event": "play_observed",
            "round_id": "round-001",
            "sequence_no": sequence,
            "actor": actor,
            "cards": [card],
        }
        for sequence, actor, card in (
            (1, "self", "3"),
            (2, "right", "4"),
        )
    ]
    result = {
        "event": "round_result_detected",
        "round_id": "round-001",
        "winner": "self",
        "outcome": "victory",
    }
    _write_jsonl(predicted, [
        *actions,
        {
            "event": "scene_observation",
            "frame_id": 1,
            "seats": [
                {"seat": "self", "remaining_count": 17},
                {"seat": "right", "remaining_count": 16},
                {"seat": "left", "remaining_count": 17},
            ],
        },
        {
            "event": "state_update",
            "state": {
                "remaining_cards": {"self": 17, "right": 16, "left": 17},
                "played_cards": ["3", "4"],
                "hidden_played_count": 2,
            },
        },
        result,
    ])
    _write_jsonl(expected, [*actions, result])
    _write_jsonl(scenes, [{
        "frame_id": 1,
        "after_sequence_no": 1,
        "remaining": {"self": 17, "right": 16, "left": 17},
    }])

    report = evaluate_live_replay(
        predicted,
        expected,
        expected_scenes=scenes,
        require_complete_round=True,
    )

    assert report["remaining_counts"] == {"correct": 3, "total": 3, "required": 6}
    assert report["thresholds"]["remaining_annotation_coverage"] is False
    assert report["passed"] is False


def _write_acceptance_session(
    recordings: Path,
    replays: Path,
    session: str,
    *,
    color: tuple[int, int, int],
) -> None:
    session_number = int(session.rsplit("-", 1)[-1])
    frame_timestamp = float(session_number * 1000)
    session_dir = recordings / session
    frames = session_dir / "frames"
    frames.mkdir(parents=True)
    image_path = frames / "000001.png"
    Image.new("RGB", (20, 10), color).save(image_path)
    roi_dir = session_dir / "rois" / "self_hand"
    roi_dir.mkdir(parents=True)
    roi_path = roi_dir / "000001.png"
    Image.new("RGB", (10, 5), color).save(roi_path)
    config_snapshot = session_dir / "config.snapshot.json"
    config_snapshot.write_text(
        json.dumps({"rois": {"self_hand": [0.0, 0.0, 1.0, 1.0]}}),
        encoding="utf-8",
    )
    config_sha256 = sha256_file(config_snapshot)
    _write_jsonl(session_dir / "manifest.jsonl", [{
        "schema_version": LEGACY_FRAME_SCHEMA_VERSION,
        "event": "recorded_frame",
        "session": session,
        "frame_id": 1,
        "timestamp": frame_timestamp,
        "full_image": "frames/000001.png",
        "full_image_sha256": sha256_file(image_path),
        "config_sha256": config_sha256,
        "image_size": [20, 10],
        "rois": {"self_hand": "rois/self_hand/000001.png"},
        "roi_sha256": {"self_hand": sha256_file(roi_path)},
    }])
    (session_dir / "session.json").write_text(
        json.dumps({
            "schema_version": SESSION_SCHEMA_VERSION,
            "session": session,
            "created_at": frame_timestamp - 1,
            "completed_at": frame_timestamp + 1,
            "config_snapshot": "config.snapshot.json",
            "config_sha256": config_sha256,
            "recorded_frames": 1,
            "recording_state": "complete",
            "complete_game": True,
            "evidence_split": "acceptance",
        }),
        encoding="utf-8",
    )
    expected_events = session_dir / "expected-events.jsonl"
    expected_scenes = session_dir / "expected-scenes.jsonl"
    action = {
        "event": "play_observed",
        "round_id": session,
        "sequence_no": 1,
        "actor": "self",
        "cards": ["3"],
    }
    result = {
        "event": "round_result_detected",
        "winner": "self",
        "outcome": "victory",
    }
    _write_jsonl(expected_events, [action, result])
    _write_jsonl(expected_scenes, [{
        "frame_id": 1,
        "after_sequence_no": 1,
        "remaining": {"self": 0, "right": 17, "left": 17},
    }])
    prepare_annotation_workbook(
        session_dir,
        columns=1,
        rows=1,
        thumbnail_width=160,
    )
    annotation = seal_annotation_bundle(session_dir)
    replay_dir = replays / session
    replay_dir.mkdir(parents=True)
    predicted = replay_dir / "events.jsonl"
    predicted_action = {**action, "round_id": f"runtime-{session}"}
    predicted_result = {**result, "round_id": f"runtime-{session}"}
    _write_jsonl(predicted, [
        predicted_action,
        {
            "event": "scene_observation",
            "frame_id": 1,
            "seats": [
                {"seat": "self", "remaining_count": 0},
                {"seat": "right", "remaining_count": 17},
                {"seat": "left", "remaining_count": 17},
            ],
        },
        {
            "event": "state_update",
            "state": {
                "remaining_cards": {"self": 0, "right": 17, "left": 17},
                "played_cards": ["3"],
                "hidden_played_count": 19,
            },
        },
        predicted_result,
    ])
    evaluation = evaluate_live_replay(
        predicted,
        expected_events,
        expected_scenes=expected_scenes,
        require_complete_round=True,
    )
    (replay_dir / "evaluation.json").write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )
    model_path = replays / "model.bin"
    model_path.write_bytes(b"model-v1")
    templates_path = replays / "templates"
    templates_path.mkdir(exist_ok=True)
    (templates_path / "template.bin").write_bytes(b"template-v1")
    replay_report: dict[str, object] = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "session": session,
            "created_at": float(annotation["completed_at"]) + 1.0,
            "manifest": {
                "path": (session_dir / "manifest.jsonl").resolve().as_posix(),
                "sha256": sha256_file(session_dir / "manifest.jsonl"),
            },
            "config": {
                "path": config_snapshot.resolve().as_posix(),
                "sha256": config_sha256,
            },
            "model": {
                "path": model_path.resolve().as_posix(),
                "sha256": sha256_file(model_path),
            },
            "templates": {
                "path": templates_path.resolve().as_posix(),
                "sha256": sha256_directory(templates_path),
            },
            "events_log": {
                "path": predicted.resolve().as_posix(),
                "sha256": sha256_file(predicted),
            },
            "annotation": {
                "path": (session_dir / "annotation.json").resolve().as_posix(),
                "sha256": sha256_file(session_dir / "annotation.json"),
            },
            "implementation": {
                "project_root": Path(__file__).resolve().parents[1].as_posix(),
                "sha256": sha256_python_implementation(
                    Path(__file__).resolve().parents[1]
                ),
                "runtime_versions": runtime_versions(),
            },
            "replayed_frames": 1,
            "final_mode": "finished",
        }
    replay_report["report_sha256"] = sha256_json_payload(replay_report)
    (replay_dir / "replay.json").write_text(
        json.dumps(replay_report),
        encoding="utf-8",
    )
    metadata_path = session_dir / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence_seal"] = {
        "implementation_sha256": replay_report["implementation"]["sha256"],
        "model_sha256": replay_report["model"]["sha256"],
        "templates_sha256": replay_report["templates"]["sha256"],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_blind_annotation_is_sealed_before_replay(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    session_dir = recordings / "game-001"

    annotation = validate_sealed_annotation(session_dir)

    assert annotation["prediction_inputs_used"] is False
    assert annotation["summary"] == {
        "frame_count": 1,
        "action_count": 1,
        "scene_count": 1,
        "result_count": 1,
    }
    with pytest.raises(ValueError, match="before replay output exists"):
        assert_blind_annotation_preconditions(
            session_dir,
            replays_root=replays,
        )


def test_blind_annotation_rejects_inconsistent_final_remaining(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    session_dir = recordings / "game-001"
    _write_jsonl(session_dir / "expected-scenes.jsonl", [{
        "frame_id": 1,
        "after_sequence_no": 1,
        "remaining": {"self": 1, "right": 17, "left": 17},
    }])

    with pytest.raises(ValueError, match="winner must have zero"):
        validate_annotation_bundle(session_dir)


def _write_holdout_report(path: Path, replays: Path) -> None:
    manifest = path.parent / "holdout-manifest.jsonl"
    training_manifest = path.parent / "training-manifest.jsonl"
    predictions = path.parent / "predictions.jsonl"
    errors = path.parent / "errors.jsonl"
    confusion_matrix = path.parent / "confusion.csv"
    manifest.write_text("{}\n" * 300, encoding="utf-8")
    training_manifest.write_text("{}\n", encoding="utf-8")
    predictions.write_text("{}\n" * 300, encoding="utf-8")
    errors.write_text("", encoding="utf-8")
    confusion_matrix.write_text("expected\\predicted\n", encoding="utf-8")

    def fingerprint(file_path: Path) -> dict[str, str]:
        return {
            "path": file_path.resolve().as_posix(),
            "sha256": sha256_file(file_path),
        }

    model_fingerprint = fingerprint(replays / "model.bin")
    manifest_fingerprint = fingerprint(manifest)
    training_fingerprint = fingerprint(training_manifest)
    session_fingerprints: list[dict[str, str]] = []
    for source in ("source-a", "source-b", "source-c"):
        session_path = path.parent / f"{source}.json"
        session_path.write_text(
            json.dumps({"source_id": source}),
            encoding="utf-8",
        )
        session_fingerprints.append(fingerprint(session_path))
    classes = (
        "3", "4", "5", "6", "7", "8", "9", "10",
        "J", "Q", "K", "A", "2", "SJ", "BJ",
    )
    class_counts = {label: 20 for label in classes}
    source_counts = {"source-a": 100, "source-b": 100, "source-c": 100}
    holdout_seal: dict[str, object] = {
        "schema_version": "real-window-holdout-blind-seal-v1",
        "completed_at": 1.0,
        "annotation_mode": "blind_without_model_predictions",
        "prediction_inputs_used": False,
        "model": model_fingerprint,
        "manifest": manifest_fingerprint,
        "training_manifest": training_fingerprint,
        "implementation": {
            "project_root": Path(__file__).resolve().parents[1].as_posix(),
            "sha256": sha256_python_implementation(
                Path(__file__).resolve().parents[1]
            ),
        },
        "sessions": session_fingerprints,
        "summary": {
            "sample_count": 300,
            "class_counts": class_counts,
            "source_counts": source_counts,
        },
    }
    holdout_seal["report_sha256"] = sha256_json_payload(holdout_seal)
    holdout_seal_path = path.parent / "holdout-seal.json"
    holdout_seal_path.write_text(
        json.dumps(holdout_seal),
        encoding="utf-8",
    )

    report: dict[str, object] = {
        "schema_version": "real-window-holdout-v3",
        "created_at": 2.0,
        "publication_ready": True,
        "sample_count": 300,
        "accuracy": 1.0,
            "class_counts": class_counts,
            "per_class": {
                label: {"count": 20, "correct": 20, "accuracy": 1.0}
                for label in classes
            },
            "source_counts": source_counts,
            "leakage_check": {"checked": True, "overlap_count": 0},
            "readiness_checks": [{"name": "all", "passed": True}],
            "inputs": {
                "model": model_fingerprint,
                "manifest": manifest_fingerprint,
                "training_manifest": training_fingerprint,
            },
            "artifacts": {
                "predictions": fingerprint(predictions),
                "errors": fingerprint(errors),
                "confusion_matrix": fingerprint(confusion_matrix),
            },
            "holdout_seal": fingerprint(holdout_seal_path),
        }
    report["report_sha256"] = sha256_json_payload(report)
    path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )


def test_phase6_acceptance_requires_independent_complete_sessions(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    for index in range(5):
        _write_acceptance_session(
            recordings,
            replays,
            f"game-{index + 1:03d}",
            color=(index * 20, 10, 20),
        )
    holdout = tmp_path / "holdout.json"
    _write_holdout_report(holdout, replays)

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=holdout,
    )

    assert report["session_count"] == 5
    assert report["metrics"]["event_f1"] == 1.0
    assert report["metrics"]["round_success_rate"] == 1.0
    assert report["passed"] is True


def test_phase6_acceptance_rejects_holdout_labels_sealed_after_prediction(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    holdout = tmp_path / "holdout.json"
    _write_holdout_report(holdout, replays)
    seal_path = tmp_path / "holdout-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["completed_at"] = 3.0
    seal.pop("report_sha256")
    seal["report_sha256"] = sha256_json_payload(seal)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    holdout_report = json.loads(holdout.read_text(encoding="utf-8"))
    holdout_report["holdout_seal"]["sha256"] = sha256_file(seal_path)
    holdout_report.pop("report_sha256")
    holdout_report["report_sha256"] = sha256_json_payload(holdout_report)
    holdout.write_text(json.dumps(holdout_report), encoding="utf-8")

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=holdout,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert "holdout labels were sealed after predictions existed" in report[
        "card_holdout"
    ]["evidence_issues"]
    assert report["passed"] is False


def test_phase6_acceptance_excludes_development_recordings(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    metadata_path = recordings / "game-001" / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence_split"] = "development"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert report["session_count"] == 0
    assert report["excluded_non_acceptance_sessions"] == [{
        "session": "game-001",
        "evidence_split": "development",
    }]
    assert report["checks"]["minimum_independent_sessions"] is False


def test_phase6_acceptance_rejects_code_changed_after_recording(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    metadata_path = recordings / "game-001" / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence_seal"]["implementation_sha256"] = "before-tuning"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert (
        "implementation_sha256 changed after acceptance recording"
        in report["sessions"][0]["evidence_issues"]
    )
    assert report["checks"]["all_evaluation_inputs_verified"] is False


def test_phase6_acceptance_rejects_cross_session_frame_reuse(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    _write_acceptance_session(
        recordings,
        replays,
        "game-002",
        color=(10, 20, 30),
    )

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=2),
    )

    assert report["checks"]["no_cross_session_frame_leakage"] is False
    assert report["passed"] is False


def test_recording_finalize_repairs_interrupted_frame_count(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    session_dir = recordings / "game-001"
    metadata_path = session_dir / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "recorded_frames": 0,
        "recording_state": "interrupted",
        "complete_game": False,
    })
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    before = inspect_recording_session(session_dir)
    finalized = finalize_recording_session(session_dir)

    assert "session recorded_frames does not match manifest length" in before.issues
    assert finalized.valid is True
    assert finalized.metadata["recorded_frames"] == 1
    assert finalized.metadata["recording_state"] == "complete"
    assert finalized.metadata["complete_game"] is True


def test_phase6_acceptance_rejects_tampered_recording_roi(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    roi_path = recordings / "game-001" / "rois" / "self_hand" / "000001.png"
    Image.new("RGB", (10, 5), "red").save(roi_path)

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert report["checks"]["all_recording_files_and_checksums_valid"] is False
    assert "frame 1 ROI self_hand checksum mismatch" in report["sessions"][0][
        "recording_issues"
    ]
    assert report["passed"] is False


def test_phase6_acceptance_rejects_annotations_changed_after_evaluation(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    expected_events = recordings / "game-001" / "expected-events.jsonl"
    expected_events.write_text(
        expected_events.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert report["checks"]["all_evaluation_inputs_verified"] is False
    assert "annotation expected_events checksum changed after replay" in report[
        "sessions"
    ][0]["evidence_issues"]
    assert "expected_events checksum changed after evaluation" in report[
        "sessions"
    ][0]["evidence_issues"]
    assert report["passed"] is False


def test_phase6_acceptance_rejects_edited_evaluation_metrics(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    evaluation_path = replays / "game-001" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["event_f1"] = 0.123456
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert "evaluation report content changed after generation" in report["sessions"][
        0
    ]["evidence_issues"]
    assert report["passed"] is False


def test_phase6_acceptance_rejects_model_changed_after_replay(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    (replays / "model.bin").write_bytes(b"model-v2")

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert report["checks"]["all_evaluation_inputs_verified"] is False
    assert "model checksum changed after replay" in report["sessions"][0][
        "evidence_issues"
    ]
    assert report["passed"] is False


def test_phase6_acceptance_rejects_stale_replay_implementation(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    replays = tmp_path / "replays"
    _write_acceptance_session(
        recordings,
        replays,
        "game-001",
        color=(10, 20, 30),
    )
    replay_path = replays / "game-001" / "replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["implementation"]["sha256"] = "0" * 64
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    report = audit_phase6_acceptance(
        recordings,
        replays,
        card_holdout_report=None,
        require_card_holdout=False,
        thresholds=Phase6AcceptanceThresholds(min_sessions=1),
    )

    assert "replay implementation changed after replay" in report["sessions"][0][
        "evidence_issues"
    ]
    assert report["passed"] is False


def test_live_log_diagnostics_groups_repeated_runtime_failures(
    tmp_path: Path,
) -> None:
    log = tmp_path / "live.jsonl"
    _write_jsonl(log, [
        {
            "event": "scene_observation",
            "timestamp": 1.0,
            "warnings": ["right_remaining_unavailable"],
        },
        {
            "event": "manual_current_scan",
            "success": True,
            "warnings": ["historical_played_cards_unknown=12"],
        },
        {
            "event": "live_runtime_snapshot",
            "round_id": "round-1",
            "tracker_mode": "tracking",
            "capture_latency_ms": 10.0,
            "total_latency_ms": 20.0,
            "warnings": ["right_remaining_unavailable"],
        },
        {
            "event": "play_observed",
            "round_id": "round-1",
            "warnings": [],
        },
        {
            "event": "live_decision",
            "round_id": "round-1",
            "warnings": [],
        },
    ])

    report = analyze_live_log(log)

    assert report["rounds"] == {
        "observed": 1,
        "with_actions": 1,
        "with_decisions": 1,
        "completed": 0,
    }
    assert report["manual_scans"]["success_rate"] == 1.0
    assert report["warning_categories"]["remaining_count_unavailable"] == 2
