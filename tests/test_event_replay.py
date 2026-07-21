from __future__ import annotations

from pathlib import Path

import pytest

from src.logic.action_validation import validate_observed_action
from src.state.events import PlayerSeat
from src.state.replay import load_event_replay


def test_event_replay_loads_decision_ready_state() -> None:
    replay = load_event_replay(
        Path("examples/phase5/farmer_response.jsonl"),
        validator=validate_observed_action,
    )

    assert replay.event_count == 3
    assert replay.state.revision == 3
    assert replay.state.landlord is PlayerSeat.RIGHT
    assert replay.state.decision_ready
    assert replay.warnings == ()


def test_event_replay_reports_json_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"event":"game_started"}\n{"event":broken}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_event_replay(path, validator=validate_observed_action)

