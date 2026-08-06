from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def analyze_live_log(path: Path) -> dict[str, object]:
    event_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    warning_categories: Counter[str] = Counter()
    tracker_modes: Counter[str] = Counter()
    window_statuses: Counter[str] = Counter()
    round_ids: set[str] = set()
    action_round_ids: set[str] = set()
    decision_round_ids: set[str] = set()
    completed_round_ids: set[str] = set()
    capture_latencies: list[float] = []
    total_latencies: list[float] = []
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    manual_scan_attempts = 0
    manual_scan_successes = 0
    line_count = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            line_count += 1
            event = str(payload.get("event", "<missing>"))
            event_counts[event] += 1
            round_id = payload.get("round_id")
            if round_id is not None:
                round_ids.add(str(round_id))
                if event in {"play_observed", "pass_observed"}:
                    action_round_ids.add(str(round_id))
                elif event == "live_decision":
                    decision_round_ids.add(str(round_id))
                elif event == "round_result_detected":
                    completed_round_ids.add(str(round_id))
            if event == "manual_current_scan":
                manual_scan_attempts += 1
                manual_scan_successes += int(payload.get("success") is True)
            mode = payload.get("tracker_mode") or payload.get("mode")
            if mode is not None:
                tracker_modes[str(mode)] += 1
            window_status = payload.get("window_status") or payload.get("status")
            if window_status is not None and event in {
                "live_runtime_snapshot",
                "live_window_status",
            }:
                window_statuses[str(window_status)] += 1
            _append_number(capture_latencies, payload.get("capture_latency_ms"))
            _append_number(total_latencies, payload.get("total_latency_ms"))
            timestamp = payload.get("timestamp")
            if isinstance(timestamp, (int, float)) and math.isfinite(timestamp):
                first_timestamp = (
                    float(timestamp)
                    if first_timestamp is None
                    else min(first_timestamp, float(timestamp))
                )
                last_timestamp = (
                    float(timestamp)
                    if last_timestamp is None
                    else max(last_timestamp, float(timestamp))
                )
            warnings = payload.get("warnings", [])
            if isinstance(warnings, list):
                for warning in warnings:
                    value = str(warning)
                    warning_counts[value] += 1
                    warning_categories[categorize_live_warning(value)] += 1

    return {
        "schema_version": "phase6-live-log-diagnostics-v1",
        "source": path.as_posix(),
        "line_count": line_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "event_counts": dict(event_counts.most_common()),
        "tracker_modes": dict(tracker_modes.most_common()),
        "window_statuses": dict(window_statuses.most_common()),
        "rounds": {
            "observed": len(round_ids),
            "with_actions": len(action_round_ids),
            "with_decisions": len(decision_round_ids),
            "completed": len(completed_round_ids),
        },
        "manual_scans": {
            "attempts": manual_scan_attempts,
            "successes": manual_scan_successes,
            "success_rate": _ratio(manual_scan_successes, manual_scan_attempts),
        },
        "latency_ms": {
            "capture": _distribution(capture_latencies),
            "total": _distribution(total_latencies),
        },
        "warning_categories": dict(warning_categories.most_common()),
        "top_warnings": [
            {"warning": warning, "count": count}
            for warning, count in warning_counts.most_common(30)
        ],
        "safety_events": {
            "decision_rejected": event_counts["live_decision_rejected"],
            "decision_errors": event_counts["live_decision_error"],
            "error_frames_saved": event_counts["error_frame_saved"],
        },
    }


def categorize_live_warning(warning: str) -> str:
    value = warning.lower()
    if "remaining_unavailable" in value or "余牌数尚未稳定" in warning:
        return "remaining_count_unavailable"
    if "self_hand_unavailable" in value or "self hand change" in value:
        return "self_hand_unavailable_or_inconsistent"
    if "role_unavailable" in value:
        return "role_unavailable"
    if "turn_unavailable" in value:
        return "turn_unavailable"
    if "未检测到“斗地主”窗口" in warning or "窗口当前无法识别" in warning:
        return "window_unavailable"
    if "最小化" in warning:
        return "window_minimized"
    if "automatic_current_game_scan" in value or "automatic_resync" in value:
        return "automatic_rescan"
    if "manual_current_game_scan" in value or "manual_scan" in value:
        return "manual_rescan"
    if "historical_played_cards_unknown" in value:
        return "unknown_history"
    if "low-confidence" in value or "confidence_outlier" in value:
        return "low_confidence"
    if "uniform_opponent_model" in value:
        return "uniform_opponent_model"
    if "rule_subset_only" in value:
        return "rule_subset_only"
    category = warning.split(":", 1)[0].strip().lower()
    category = re.sub(r"\d+(?:\.\d+)?", "#", category)
    return re.sub(r"\s+", "_", category) or "uncategorized"


def _append_number(values: list[float], value: object) -> None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        values.append(float(value))


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": len(ordered),
        "median": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(values: list[float], quantile: float) -> float:
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    fraction = index - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


__all__ = ["analyze_live_log", "categorize_live_warning"]
