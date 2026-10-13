from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import annotate_live_session
from scripts.evaluate_live_replay import _evaluate_results, _same_action, evaluate_live_replay
from src.state.round_result import infer_opening_landlord, parse_round_result


@pytest.mark.parametrize(("landlord", "winner"), product(("self", "right", "left"), repeat=2))
def test_team_result_all_seat_combinations(landlord: str, winner: str) -> None:
    outcome = "victory" if (landlord == winner) == (landlord == "self") else "defeat"
    row = {"landlord": landlord, "winner": winner, "outcome": outcome}
    facts = parse_round_result(row)
    assert facts.team_verified
    assert facts.winning_side == ("landlord" if winner == landlord else "farmers")
    with pytest.raises(ValueError):
        parse_round_result({**row, "outcome": "defeat" if outcome == "victory" else "victory"})


@pytest.mark.parametrize("invalid", [
    {}, {"winner": None}, {"outcome": "victory"},
    {"winner": "unknown", "outcome": "victory"},
    {"landlord": [], "winner": "left", "outcome": "victory"},
    {"winner": "left", "outcome": "victory", "winning_side": {}},
    {"landlord": "right", "winner": "left", "outcome": "unknown", "winning_side": "landlord"},
])
def test_result_malformed_fields_never_match(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_round_result(invalid)


def test_legacy_opening_role_only_uses_own_evidence() -> None:
    actions = [{"sequence_no": 1, "event": "play_observed", "actor": "right", "cards": ["3"]}]
    scenes = [{"after_sequence_no": 1, "remaining": {"self": 17, "right": 19, "left": 17}}]
    assert infer_opening_landlord(actions, scenes) == "right"
    row = {"winner": "left", "outcome": "victory"}
    assert parse_round_result(row, own_landlord_context="right").team_verified
    assert not parse_round_result(row).team_verified
    with pytest.raises(ValueError, match="own role"):
        parse_round_result({**row, "landlord": "left"}, own_landlord_context="right")
    assert infer_opening_landlord(actions, []) is None
    assert infer_opening_landlord(actions, [{
        "after_sequence_no": 1, "remaining": {"self": 5, "right": 4, "left": 7},
    }]) is None


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _evaluate(tmp_path: Path, result: dict[str, object], extra: list[dict[str, object]] | None = None):
    action = {"event": "play_observed", "sequence_no": 1, "actor": "left", "cards": ["3"]}
    expected_result = {
        "event": "round_result_detected", "landlord": "right", "winner": "left",
        "winning_side": "farmers", "outcome": "victory",
    }
    _write(tmp_path / "expected.jsonl", [action, expected_result])
    predicted_result = {"event": "round_result_detected", "round_id": "runtime", **result}
    rows = [
        {**action, "round_id": "runtime"},
        {"event": "state_update", "frame_id": 1, "state": {
            "round_id": "runtime", "landlord": "right",
            "remaining_cards": {"self": 17, "right": 17, "left": 0},
            "played_cards": ["3"], "hidden_played_count": 19,
        }},
        *(extra or []), predicted_result,
    ]
    _write(tmp_path / "predicted.jsonl", rows)
    return evaluate_live_replay(
        tmp_path / "predicted.jsonl", tmp_path / "expected.jsonl", require_complete_round=True,
    )


@pytest.mark.parametrize(("winner", "team", "exact", "unknown"), [
    (None, 1.0, 0.0, 1), ("self", 1.0, 0.0, 0), ("left", 1.0, 1.0, 0),
])
def test_team_and_exact_winner_are_separate(tmp_path, winner, team, exact, unknown):
    report = _evaluate(tmp_path, {"winner": winner, "outcome": "victory"})
    assert report["round_team_accuracy"] == team
    assert report["round_result_accuracy"] == exact
    assert report["round_result_counts"]["unknown_winner"] == unknown
    assert report["thresholds"]["complete_round"] is bool(exact)
    assert report["session_success"] is bool(exact)


@pytest.mark.parametrize("extra", [
    [{"event": "round_result_detected", "round_id": "runtime", "landlord": "right",
      "winner": "left", "outcome": "victory"}],
    [{"event": "state_update", "state": {"round_id": "new-auto-scan", "landlord": "right"}}],
    [{"event": "state_update", "state": {"round_id": "runtime", "landlord": "left"}}],
])
def test_duplicate_split_or_conflicting_rounds_cannot_pass(tmp_path, extra):
    report = _evaluate(tmp_path, {"winner": "left", "outcome": "victory"}, extra)
    assert report["round_result_accuracy"] == 0.0
    assert report["thresholds"]["complete_round"] is False
    assert report["session_success"] is False


def test_missing_or_contradictory_prediction_is_reported_not_assumed(tmp_path):
    report = _evaluate(tmp_path, {"outcome": "victory"})
    assert report["round_result_counts"]["invalid"] == 1
    assert report["round_result_accuracy"] == 0.0
    report = _evaluate(tmp_path, {"winner": "left", "outcome": "victory", "landlord": "left"})
    assert report["round_result_counts"]["invalid"] == 1


@pytest.mark.parametrize("round_value", [None, "", "  ", 123, [], {}])
def test_missing_or_non_string_result_round_cannot_alias_none_literal(round_value):
    result = {"landlord": "right", "winner": "left", "outcome": "victory"}
    predicted = {**result, "round_id": round_value}
    counts = _evaluate_results(
        [predicted], [predicted], [result], expected_landlord=None, round_alias="None",
    )
    assert counts["invalid"] == 1
    assert counts["exact"] == 0


def test_action_card_order_ignores_only_order_not_multiplicity():
    expected = {"event": "play_observed", "actor": "self", "cards": ["8", "8", "8", "3"]}
    assert _same_action(expected, {**expected, "cards": ["3", "8", "8", "8"]})
    # Both are legal triple-with-single plays and share the same set of ranks.
    assert not _same_action(expected, {**expected, "cards": ["8", "3", "3", "3"]})
    for cards in (["8", "8", "3"], "3888", {}, [None], ["3", []], [], ["8"] * 5):
        assert not _same_action(expected, {**expected, "cards": cards})
    for event, cards in (("play_observed", []), ("pass_observed", ["3"]),
                         ("play_observed", ["SJ", "SJ"]), ("play_observed", ["3", "4"])):
        invalid = {"event": event, "actor": "self", "cards": cards}
        assert not _same_action(invalid, invalid)
    passed = {"event": "pass_observed", "actor": "self", "cards": []}
    assert _same_action(passed, passed)


def test_literal_none_round_cannot_adopt_result_without_round(tmp_path):
    _evaluate(tmp_path, {"winner": "left", "outcome": "victory", "landlord": "right"})
    path = tmp_path / "predicted.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if "round_id" in row:
            row["round_id"] = "None"
        if row.get("event") == "state_update":
            row["state"]["round_id"] = "None"
    rows[-1].pop("round_id")
    _write(path, rows)
    report = evaluate_live_replay(path, tmp_path / "expected.jsonl", require_complete_round=True)
    assert report["event_f1"] == 1.0
    assert report["round_result_counts"]["invalid"] == 1
    assert report["session_success"] is False
    assert report["passed"] is False


@pytest.mark.parametrize(("landlord", "winner"), product(("self", "right", "left"), repeat=2))
def test_annotation_seals_team_result_without_human_rewrite(tmp_path, monkeypatch, landlord, winner):
    monkeypatch.setattr(annotate_live_session, "inspect_recording_session", lambda _: SimpleNamespace(
        valid=True, manifest_rows=[{"frame_id": 1}],
    ))
    outcome = "victory" if (landlord == winner) == (landlord == "self") else "defeat"
    events = [
        {"event": "play_observed", "sequence_no": 1, "actor": winner, "cards": ["3"]},
        {"event": "round_result_detected", "landlord": landlord, "winner": winner, "outcome": outcome},
    ]
    remaining = {seat: (0 if seat == winner else 5) for seat in ("self", "right", "left")}
    scenes = [{"frame_id": 1, "after_sequence_no": 1, "remaining": remaining}]
    _write(tmp_path / "expected-events.jsonl", events)
    _write(tmp_path / "expected-scenes.jsonl", scenes)
    assert annotate_live_session.validate_annotation_bundle(tmp_path)["result_count"] == 1
    remaining[next(seat for seat in remaining if seat != winner)] = 0
    _write(tmp_path / "expected-scenes.jsonl", scenes)
    with pytest.raises(ValueError, match="exactly one"):
        annotate_live_session.validate_annotation_bundle(tmp_path)
