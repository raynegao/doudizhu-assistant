from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


ACTION_EVENTS = {"play_observed", "pass_observed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 6 replay events and optional scene labels."
    )
    parser.add_argument("--predicted-log", type=Path, required=True)
    parser.add_argument("--expected-events", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/live-replay/evaluation.json"),
    )
    parser.add_argument("--require-thresholds", action="store_true")
    return parser


def evaluate_live_replay(
    predicted_log: Path,
    expected_events: Path,
    *,
    expected_scenes: Path | None = None,
) -> dict[str, object]:
    predicted_rows = _read_jsonl(predicted_log)
    expected_rows = _read_jsonl(expected_events)
    predicted_actions = [
        row for row in predicted_rows if row.get("event") in ACTION_EVENTS
    ]
    expected_actions = [
        row for row in expected_rows if row.get("event") in ACTION_EVENTS
    ]
    predicted_by_key = {_action_key(row): row for row in predicted_actions}
    expected_by_key = {_action_key(row): row for row in expected_actions}
    exact_matches = sum(
        _same_action(expected, predicted_by_key.get(key))
        for key, expected in expected_by_key.items()
    )
    precision = exact_matches / len(predicted_actions) if predicted_actions else 0.0
    recall = exact_matches / len(expected_actions) if expected_actions else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    expected_plays = [
        row for row in expected_actions if row.get("event") == "play_observed"
    ]
    exact_plays = sum(
        _same_action(row, predicted_by_key.get(_action_key(row)))
        for row in expected_plays
    )
    card_exact_accuracy = (
        exact_plays / len(expected_plays) if expected_plays else 0.0
    )

    invariant_checks = []
    for row in predicted_rows:
        if row.get("event") != "state_update":
            continue
        state = row.get("state")
        if not isinstance(state, dict):
            continue
        remaining = state.get("remaining_cards", {})
        played = state.get("played_cards", [])
        if isinstance(remaining, dict) and isinstance(played, list):
            invariant_checks.append(
                sum(int(value) for value in remaining.values()) + len(played) == 54
            )

    remaining_correct = 0
    remaining_total = 0
    if expected_scenes is not None:
        expected_scene_rows = {
            int(row["frame_id"]): row
            for row in _read_jsonl(expected_scenes)
        }
        predicted_scenes = {
            int(row["frame_id"]): row
            for row in predicted_rows
            if row.get("event") == "scene_observation"
        }
        for frame_id, expected in expected_scene_rows.items():
            predicted = predicted_scenes.get(frame_id)
            if predicted is None:
                expected_remaining = expected.get("remaining", {})
                if isinstance(expected_remaining, dict):
                    remaining_total += len(expected_remaining)
                continue
            observed_remaining = _scene_remaining(predicted)
            expected_remaining = expected.get("remaining", {})
            if not isinstance(expected_remaining, dict):
                continue
            for seat, value in expected_remaining.items():
                remaining_total += 1
                remaining_correct += observed_remaining.get(str(seat)) == int(value)
    remaining_accuracy = (
        remaining_correct / remaining_total if remaining_total else None
    )

    thresholds = {
        "event_f1": f1 >= 0.95,
        "card_exact_accuracy": card_exact_accuracy >= 0.95,
        "remaining_accuracy": (
            remaining_accuracy is not None and remaining_accuracy >= 0.98
        ),
        "deck_invariant": bool(invariant_checks) and all(invariant_checks),
    }
    return {
        "event_counts": {
            "expected": len(expected_actions),
            "predicted": len(predicted_actions),
            "exact": exact_matches,
        },
        "event_precision": round(precision, 6),
        "event_recall": round(recall, 6),
        "event_f1": round(f1, 6),
        "card_exact_accuracy": round(card_exact_accuracy, 6),
        "remaining_accuracy": (
            round(remaining_accuracy, 6)
            if remaining_accuracy is not None
            else None
        ),
        "deck_invariant_checks": len(invariant_checks),
        "deck_invariant_passed": bool(invariant_checks) and all(invariant_checks),
        "thresholds": thresholds,
        "passed": all(thresholds.values()),
        "limitations": (
            []
            if expected_scenes is not None
            else ["remaining_accuracy requires --expected-scenes annotations"]
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(payload)
    return rows


def _action_key(row: dict[str, object]) -> tuple[int, str]:
    return int(row["sequence_no"]), str(row["actor"])


def _same_action(
    expected: dict[str, object],
    predicted: dict[str, object] | None,
) -> bool:
    if predicted is None:
        return False
    return (
        expected.get("event") == predicted.get("event")
        and str(expected.get("actor")) == str(predicted.get("actor"))
        and list(expected.get("cards", [])) == list(predicted.get("cards", []))
    )


def _scene_remaining(row: dict[str, object]) -> dict[str, int]:
    values: dict[str, int] = {}
    seats = row.get("seats", [])
    if not isinstance(seats, list):
        return values
    for seat in seats:
        if not isinstance(seat, dict) or seat.get("remaining_count") is None:
            continue
        values[str(seat.get("seat"))] = int(seat["remaining_count"])
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_live_replay(
        args.predicted_log,
        args.expected_events,
        expected_scenes=args.expected_scenes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if args.require_thresholds and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
