from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from src.capture.recording_integrity import sha256_file, sha256_json_payload

ACTION_EVENTS = {"play_observed", "pass_observed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 6 replay events and optional scene labels."
    )
    parser.add_argument("--predicted-log", type=Path, required=True)
    parser.add_argument("--expected-events", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=Path)
    parser.add_argument(
        "--require-complete-round",
        action="store_true",
        help="Require an annotated and exactly matched round result.",
    )
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
    require_complete_round: bool = False,
) -> dict[str, object]:
    predicted_rows = _read_jsonl(predicted_log)
    expected_rows = _read_jsonl(expected_events)
    predicted_actions = [
        row for row in predicted_rows if row.get("event") in ACTION_EVENTS
    ]
    expected_actions = [
        row for row in expected_rows if row.get("event") in ACTION_EVENTS
    ]
    expected_sequences = [int(row["sequence_no"]) for row in expected_actions]
    if expected_sequences != list(range(1, len(expected_actions) + 1)):
        raise ValueError("expected action sequence_no must be contiguous and start at 1")
    predicted_round_ids = {
        str(row["round_id"])
        for row in predicted_actions
        if row.get("round_id") is not None
    }
    expected_default_round = (
        next(iter(predicted_round_ids)) if len(predicted_round_ids) == 1 else ""
    )
    expected_round_ids = {
        str(row["round_id"])
        for row in expected_actions
        if row.get("round_id") is not None
    }
    expected_round_alias = (
        expected_default_round
        if expected_default_round and len(expected_round_ids) <= 1
        else None
    )
    predicted_by_key, predicted_duplicate_keys = _index_actions(predicted_actions)
    expected_by_key, expected_duplicate_keys = _index_actions(
        expected_actions,
        default_round_id=expected_default_round,
        override_round_id=expected_round_alias,
    )
    if expected_duplicate_keys:
        raise ValueError(
            "expected events contain duplicate action keys: "
            + ", ".join(sorted(expected_duplicate_keys))
        )
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
        _same_action(
            row,
            predicted_by_key.get(
                _action_key(
                    row,
                    default_round_id=expected_default_round,
                    override_round_id=expected_round_alias,
                )
            ),
        )
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
            hidden_played_count = int(state.get("hidden_played_count", 0))
            invariant_checks.append(
                sum(int(value) for value in remaining.values())
                + len(played)
                + hidden_played_count
                == 54
            )

    remaining_correct = 0
    remaining_total = 0
    annotated_action_sequences: set[int] = set()
    if expected_scenes is not None:
        expected_scene_rows: dict[int, dict[str, object]] = {}
        for row in _read_jsonl(expected_scenes):
            frame_id = int(row["frame_id"])
            after_sequence_no = int(row["after_sequence_no"])
            if frame_id in expected_scene_rows:
                raise ValueError(f"duplicate expected scene frame_id: {frame_id}")
            if after_sequence_no in annotated_action_sequences:
                raise ValueError(
                    f"duplicate remaining annotation for action {after_sequence_no}"
                )
            if after_sequence_no not in expected_sequences:
                raise ValueError(
                    f"remaining annotation references unknown action {after_sequence_no}"
                )
            expected_remaining = row.get("remaining")
            if not isinstance(expected_remaining, dict) or set(expected_remaining) != {
                "self",
                "right",
                "left",
            }:
                raise ValueError(
                    "each expected scene must contain self/right/left remaining counts"
                )
            expected_scene_rows[frame_id] = row
            annotated_action_sequences.add(after_sequence_no)
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
    required_remaining_annotations = len(expected_actions) * 3
    remaining_annotation_coverage = bool(
        expected_actions
        and annotated_action_sequences == set(expected_sequences)
        and remaining_total >= required_remaining_annotations
    )

    predicted_results = [
        row for row in predicted_rows if row.get("event") == "round_result_detected"
    ]
    expected_results = [
        row for row in expected_rows if row.get("event") == "round_result_detected"
    ]
    predicted_results_by_round = {
        str(row.get("round_id")): row for row in predicted_results
    }
    result_round_alias = (
        next(iter(predicted_results_by_round))
        if len(predicted_results_by_round) == 1
        else None
    )
    exact_results = sum(
        _same_round_result(
            expected,
            predicted_results_by_round.get(
                result_round_alias or str(expected.get("round_id"))
            ),
            override_round_id=result_round_alias,
        )
        for expected in expected_results
    )
    round_result_accuracy = (
        exact_results / len(expected_results) if expected_results else None
    )
    complete_round_passed = bool(
        expected_results
        and len(predicted_results) == len(expected_results)
        and exact_results == len(expected_results)
    )

    thresholds = {
        "event_f1": f1 >= 0.95,
        "card_exact_accuracy": card_exact_accuracy >= 0.95,
        "remaining_accuracy": (
            remaining_accuracy is not None and remaining_accuracy >= 0.98
        ),
        "remaining_annotation_coverage": remaining_annotation_coverage,
        "deck_invariant": bool(invariant_checks) and all(invariant_checks),
    }
    if require_complete_round:
        thresholds["complete_round"] = complete_round_passed
    exact_event_sequence = bool(
        expected_actions
        and not predicted_duplicate_keys
        and len(predicted_actions) == len(expected_actions)
        and exact_matches == len(expected_actions)
    )
    session_success = (
        exact_event_sequence
        and complete_round_passed
        and bool(invariant_checks)
        and all(invariant_checks)
    ) if expected_results else None
    limitations: list[str] = []
    if expected_scenes is None:
        limitations.append("remaining_accuracy requires --expected-scenes annotations")
    if not expected_results:
        limitations.append(
            "session_success requires an expected round_result_detected annotation"
        )
    report: dict[str, object] = {
        "schema_version": "phase6-live-replay-evaluation-v2",
        "inputs": {
            "predicted_log": _input_fingerprint(predicted_log),
            "expected_events": _input_fingerprint(expected_events),
            "expected_scenes": (
                _input_fingerprint(expected_scenes)
                if expected_scenes is not None
                else None
            ),
        },
        "event_counts": {
            "expected": len(expected_actions),
            "predicted": len(predicted_actions),
            "exact": exact_matches,
            "expected_plays": len(expected_plays),
            "exact_plays": exact_plays,
            "predicted_duplicate_keys": len(predicted_duplicate_keys),
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
        "remaining_counts": {
            "correct": remaining_correct,
            "total": remaining_total,
            "required": required_remaining_annotations,
        },
        "deck_invariant_checks": len(invariant_checks),
        "deck_invariant_failures": sum(not value for value in invariant_checks),
        "deck_invariant_passed": bool(invariant_checks) and all(invariant_checks),
        "round_result_counts": {
            "expected": len(expected_results),
            "predicted": len(predicted_results),
            "exact": exact_results,
        },
        "round_result_accuracy": (
            round(round_result_accuracy, 6)
            if round_result_accuracy is not None
            else None
        ),
        "exact_event_sequence": exact_event_sequence,
        "session_success": session_success,
        "thresholds": thresholds,
        "passed": all(thresholds.values()),
        "limitations": limitations,
    }
    report["report_sha256"] = sha256_json_payload(report)
    return report


def _input_fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
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


def _action_key(
    row: dict[str, object],
    *,
    default_round_id: str = "",
    override_round_id: str | None = None,
) -> tuple[str, ...]:
    return (
        "sequence_actor",
        str(override_round_id or row.get("round_id") or default_round_id),
        str(int(row["sequence_no"])),
        str(row["actor"]),
    )


def _index_actions(
    rows: Iterable[dict[str, object]],
    *,
    default_round_id: str = "",
    override_round_id: str | None = None,
) -> tuple[dict[tuple[str, ...], dict[str, object]], set[str]]:
    indexed: dict[tuple[str, ...], dict[str, object]] = {}
    duplicates: set[str] = set()
    for row in rows:
        key = _action_key(
            row,
            default_round_id=default_round_id,
            override_round_id=override_round_id,
        )
        if key in indexed:
            duplicates.add("/".join(key))
        else:
            indexed[key] = row
    return indexed, duplicates


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


def _same_round_result(
    expected: dict[str, object],
    predicted: dict[str, object] | None,
    *,
    override_round_id: str | None = None,
) -> bool:
    if predicted is None:
        return False
    return (
        str(override_round_id or expected.get("round_id"))
        == str(predicted.get("round_id"))
        and str(expected.get("winner")) == str(predicted.get("winner"))
        and str(expected.get("outcome")) == str(predicted.get("outcome"))
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
        require_complete_round=args.require_complete_round,
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
