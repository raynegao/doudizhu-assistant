from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.live_diagnostics import analyze_live_log
from src.reporting.live_log_reader import iter_live_log_records, list_live_log_sources


def test_reader_orders_numeric_slices_and_diagnostics_include_archives(tmp_path: Path) -> None:
    base = tmp_path / "events.jsonl"
    for suffix in (".10", ".2", ""):
        (tmp_path / f"events.jsonl.20260905-120000{suffix}").write_text(
            json.dumps({"event": "play_observed", "round_id": f"round{suffix}"}) + "\n",
            encoding="utf-8",
        )
    base.write_text('{"event":"round_result_detected","round_id":"round.10"}\n')
    (tmp_path / "events.jsonl.notes").write_text("not JSON")
    (tmp_path / "events.jsonl.20269999-000000").write_text("not JSON")
    (tmp_path / "events.jsonl.20260905-120000.01").write_text("not JSON")

    sources = list_live_log_sources(base)
    assert [source.path.name for source in sources] == [
        "events.jsonl.20260905-120000", "events.jsonl.20260905-120000.2",
        "events.jsonl.20260905-120000.10", "events.jsonl",
    ]
    records = tuple(iter_live_log_records(sources))
    assert records[2].source == sources[2].path and records[2].line_number == 1
    report = analyze_live_log(base)
    assert report["line_count"] == 4
    assert report["rounds"] == {"observed": 3, "with_actions": 3, "with_decisions": 0, "completed": 1}
    assert report["history_scope"] == "retained_log_slices_only"
    assert len(report["sources"]) == 4


def test_reader_can_analyze_archives_between_rotation_and_next_append(tmp_path: Path) -> None:
    base = tmp_path / "events.jsonl"
    archive = tmp_path / "events.jsonl.20260905-120000"
    archive.write_text('{"event":"live_decision"}\n')
    assert analyze_live_log(base)["line_count"] == 1


@pytest.mark.parametrize("name", ["events.jsonl", "events.jsonl.20260905-120000"])
def test_reader_rejects_matching_symlinks(tmp_path: Path, name: str) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{}\n')
    (tmp_path / name).symlink_to(outside)
    with pytest.raises(ValueError, match="plain file"):
        list_live_log_sources(tmp_path / "events.jsonl")


def test_reader_rejects_traversal_and_changed_files(tmp_path: Path) -> None:
    base = tmp_path / "events.jsonl"
    base.write_text('{}\n')
    with pytest.raises(ValueError, match="unsafe live log path"):
        list_live_log_sources(tmp_path / "other" / ".." / "events.jsonl")
    sources = list_live_log_sources(base)
    base.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="changed before reading"):
        tuple(iter_live_log_records(sources))


def test_reader_reports_the_corrupt_archive_and_line(tmp_path: Path) -> None:
    archive = tmp_path / "events.jsonl.20260905-120000"
    archive.write_text('{}\nnot JSON\n')
    with pytest.raises(ValueError, match=r"20260905-120000:2: invalid JSON"):
        analyze_live_log(tmp_path / "events.jsonl")


def test_diagnostics_distinguishes_known_pruned_history(tmp_path: Path) -> None:
    base = tmp_path / "events.jsonl"
    base.write_text('{"event":"live_log_history_pruned","archive_count":1}\n')
    report = analyze_live_log(base)
    assert report["history_known_pruned"] is True
    assert report["line_count"] == 1
