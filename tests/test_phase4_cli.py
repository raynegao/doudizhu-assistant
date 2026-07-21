from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_phase4_cli_replays_events_and_writes_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "phase4.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_phase4_decision",
            "--events-file",
            "examples/phase4_round.jsonl",
            "--simulations",
            "3",
            "--max-depth",
            "15",
            "--time-budget-ms",
            "0",
            "--seed",
            "9",
            "--top-k",
            "2",
            "--max-candidates",
            "6",
            "--log-file",
            str(log_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Top-K 推荐" in result.stdout
    assert "estimated_win" in result.stdout
    assert "推荐动作" in result.stdout
    [line] = log_file.read_text(encoding="utf-8").splitlines()
    payload = json.loads(line)
    assert payload["event"] == "phase4_recommendation"
    assert payload["state"]["decision_ready"] is True
    assert payload["completed_simulations"] == 3
    assert payload["top_k"]
    assert payload["opponent_estimate"]["sample_count"] == 3


def test_phase4_cli_rejects_invalid_input_without_log(tmp_path: Path) -> None:
    log_file = tmp_path / "phase4.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_phase4_decision",
            "--hand",
            "BJ BJ",
            "--log-file",
            str(log_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Phase 4 输入错误" in result.stderr
    assert not log_file.exists()


def test_phase4_cli_fallback_reports_unavailable_win_rate(tmp_path: Path) -> None:
    log_file = tmp_path / "phase4.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_phase4_decision",
            "--hand",
            "3 4 5",
            "--log-file",
            str(log_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "estimated_win=n/a" in result.stdout
    [line] = log_file.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["top_k"][0]["estimated_win_rate"] is None


def test_phase4_cli_blocks_unresolved_low_confidence_event(tmp_path: Path) -> None:
    events_file = tmp_path / "uncertain.jsonl"
    lines = Path("examples/phase4_round.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps({
        "event": "play_observed",
        "event_id": "turn-004-low",
        "sequence_no": 4,
        "actor": "self",
        "cards": ["5"],
        "confidence": 0.4,
    }))
    events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_file = tmp_path / "phase4.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_phase4_decision",
            "--events-file",
            str(events_file),
            "--log-file",
            str(log_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "phase=playing" in result.stderr
    assert "got uncertain" in result.stderr
    assert not log_file.exists()


def test_phase4_cli_wraps_malformed_event_without_traceback(tmp_path: Path) -> None:
    events_file = tmp_path / "malformed.jsonl"
    lines = Path("examples/phase4_round.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps({
        "event": "play_observed",
        "event_id": "turn-004",
        "sequence_no": 4,
        "actor": "self",
        "cards": 3,
    }))
    events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_file = tmp_path / "phase4.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_phase4_decision",
            "--events-file",
            str(events_file),
            "--log-file",
            str(log_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid observed action payload" in result.stderr
    assert "Traceback" not in result.stderr
    assert not log_file.exists()
