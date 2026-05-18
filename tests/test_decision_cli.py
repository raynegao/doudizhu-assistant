from __future__ import annotations

import json
import subprocess
import sys

from src.logic.decision import recommend_action
from src.logic.rules import PlayType
from src.state.game_state import GameStateSnapshot


def test_recommendation_is_deterministic_and_legal() -> None:
    state = GameStateSnapshot.from_inputs("3 4 5 6 7", "")
    first = recommend_action(state)
    second = recommend_action(state)
    assert first.action == second.action
    assert first.action.type is not PlayType.INVALID
    assert first.reason


def test_recommendation_uses_rocket_against_bomb_when_needed() -> None:
    state = GameStateSnapshot.from_inputs("SJ BJ 3", "2 2 2 2")
    result = recommend_action(state)
    assert str(result.action) == "SJ BJ"
    assert "火箭" in result.reason


def test_cli_smoke_writes_jsonl_log(tmp_path) -> None:
    log_file = tmp_path / "phase1.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.ui.cli",
            "--hand",
            "3 3 4 4 5 5 6 6 7 8 9 SJ BJ",
            "--last-play",
            "5 5",
            "--log-file",
            str(log_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "候选动作数" in result.stdout
    assert "推荐动作" in result.stdout
    assert "推荐理由" in result.stdout

    [line] = log_file.read_text(encoding="utf-8").splitlines()
    payload = json.loads(line)
    assert payload["event"] == "recommendation"
    assert payload["input_cards"]
    assert payload["last_play"] == ["5", "5"]
    assert payload["candidate_count"] >= 1
    assert payload["recommended_action"]
    assert payload["reason"]
    assert payload["warnings"] == []


def test_cli_rejects_invalid_input(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.ui.cli",
            "--hand",
            "BJ BJ",
            "--log-file",
            str(tmp_path / "phase1.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "输入错误" in result.stderr
