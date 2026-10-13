from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_live_log import _latest_evidence_log, main
from src.reporting.live_evidence import LiveEvidenceStore


def test_latest_cli_reads_archived_only_session_with_source_provenance(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    store.start_session("scan-archive", runtime_version="test", registry_version=0)
    session = store.session_path("scan-archive")
    archive = session / "events.jsonl.20260905-120000"
    archive.write_text('{"event":"play_observed","round_id":"one"}\n')
    store.close_session("scan-archive")
    (root / "latest-session.json").write_text(json.dumps({"session_id": "scan-archive"}))
    output = tmp_path / "report.json"
    assert main(["--latest-evidence", "--evidence-root", str(root), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["event_counts"] == {"play_observed": 1}
    assert report["sources"] == [{"path": str(archive), "line_count": 1}]


def test_latest_evidence_log_uses_managed_session_id_not_pointer_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    store.start_session(
        "scan-safe",
        runtime_version="test",
        registry_version=1,
    )
    events = store.session_path("scan-safe") / "events.jsonl"
    events.write_text('{"event":"scan_session_started"}\n', encoding="utf-8")
    store.close_session("scan-safe")
    (root / "latest-session.json").write_text(
        json.dumps(
            {
                "session_id": "scan-safe",
                "events_file": "/tmp/untrusted.jsonl",
            }
        ),
        encoding="utf-8",
    )

    assert _latest_evidence_log(root) == events


def test_latest_evidence_log_falls_back_when_pointer_is_absent(
    tmp_path: Path,
) -> None:
    assert _latest_evidence_log(tmp_path / "missing") is None


def test_latest_evidence_log_falls_back_when_cleaned_session_is_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "latest-session.json").write_text(
        json.dumps({"session_id": "already-cleaned"}),
        encoding="utf-8",
    )

    assert _latest_evidence_log(root) is None


def test_latest_evidence_log_rejects_symlink_pointer(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside-pointer.json"
    outside.write_text(json.dumps({"session_id": "scan-safe"}), encoding="utf-8")
    (root / "latest-session.json").symlink_to(outside)

    with pytest.raises(ValueError, match="pointer cannot be a symlink"):
        _latest_evidence_log(root)


def test_latest_evidence_log_rejects_symlink_session(tmp_path: Path) -> None:
    outside_root = tmp_path / "outside-evidence"
    outside_store = LiveEvidenceStore(outside_root)
    outside_store.start_session(
        "scan-escaped",
        runtime_version="test",
        registry_version=1,
    )
    outside_session = outside_store.session_path("scan-escaped")
    (outside_session / "events.jsonl").write_text("{}\n", encoding="utf-8")
    outside_store.close_session("scan-escaped")

    root = tmp_path / "evidence"
    root.mkdir()
    (root / "scan-escaped").symlink_to(outside_session, target_is_directory=True)
    (root / "latest-session.json").write_text(
        json.dumps({"session_id": "scan-escaped"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a plain directory"):
        _latest_evidence_log(root)


def test_latest_evidence_log_rejects_symlink_metadata(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    store.start_session(
        "scan-metadata",
        runtime_version="test",
        registry_version=1,
    )
    session = store.session_path("scan-metadata")
    metadata = session / "evidence-session.json"
    outside = tmp_path / "outside-metadata.json"
    outside.write_bytes(metadata.read_bytes())
    metadata.unlink()
    metadata.symlink_to(outside)
    (root / "latest-session.json").write_text(
        json.dumps({"session_id": "scan-metadata"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata is not a plain file"):
        _latest_evidence_log(root)


def test_latest_evidence_log_rejects_symlink_events(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    store.start_session(
        "scan-events",
        runtime_version="test",
        registry_version=1,
    )
    store.close_session("scan-events")
    outside = tmp_path / "outside-events.jsonl"
    outside.write_text('{"event":"outside"}\n', encoding="utf-8")
    (store.session_path("scan-events") / "events.jsonl").symlink_to(outside)
    (root / "latest-session.json").write_text(
        json.dumps({"session_id": "scan-events"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="log is not a plain file"):
        _latest_evidence_log(root)
