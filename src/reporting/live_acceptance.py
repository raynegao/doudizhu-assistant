from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.capture.recording_integrity import (
    REPLAY_SCHEMA_VERSION,
    inspect_recording_session,
    runtime_versions,
    sha256_directory,
    sha256_file,
    sha256_json_payload,
    sha256_python_implementation,
)

_CARD_CLASSES = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ")


@dataclass(frozen=True)
class Phase6AcceptanceThresholds:
    min_sessions: int = 5
    event_f1: float = 0.95
    card_exact_accuracy: float = 0.95
    remaining_accuracy: float = 0.98
    round_success_rate: float = 0.80

    def __post_init__(self) -> None:
        if self.min_sessions <= 0:
            raise ValueError("min_sessions must be positive")
        for name, value in (
            ("event_f1", self.event_f1),
            ("card_exact_accuracy", self.card_exact_accuracy),
            ("remaining_accuracy", self.remaining_accuracy),
            ("round_success_rate", self.round_success_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def audit_phase6_acceptance(
    recordings_root: Path,
    replays_root: Path,
    *,
    card_holdout_report: Path | None,
    require_card_holdout: bool = True,
    thresholds: Phase6AcceptanceThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or Phase6AcceptanceThresholds()
    candidate_session_dirs = sorted({
        path.parent
        for pattern in ("*/session.json", "*/manifest.jsonl")
        for path in recordings_root.glob(pattern)
        if path.is_file()
    })
    session_dirs: list[Path] = []
    excluded_sessions: list[dict[str, str]] = []
    for session_dir in candidate_session_dirs:
        metadata_path = session_dir / "session.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        split = (
            str(metadata.get("evidence_split") or "unclassified")
            if isinstance(metadata, Mapping)
            else "unclassified"
        )
        if split != "acceptance":
            excluded_sessions.append({
                "session": session_dir.name,
                "evidence_split": split,
            })
            continue
        session_dirs.append(session_dir)
    hash_owners: dict[str, str] = {}
    cross_session_duplicates: list[dict[str, str]] = []
    session_time_ranges: list[tuple[str, float, float]] = []
    cross_session_time_overlaps: list[dict[str, object]] = []
    sessions: list[dict[str, object]] = []
    replay_model_hashes: set[str] = set()
    aggregate = {
        "expected_events": 0,
        "predicted_events": 0,
        "exact_events": 0,
        "expected_plays": 0,
        "exact_plays": 0,
        "remaining_correct": 0,
        "remaining_total": 0,
        "remaining_required": 0,
        "invariant_checks": 0,
        "invariant_failures": 0,
        "successful_rounds": 0,
    }

    for session_dir in session_dirs:
        session_name = session_dir.name
        inspection = inspect_recording_session(session_dir)
        manifest_path = session_dir / "manifest.jsonl"
        expected_events_path = session_dir / "expected-events.jsonl"
        expected_scenes_path = session_dir / "expected-scenes.jsonl"
        annotation_path = session_dir / "annotation.json"
        replay_dir = replays_root / session_name
        predicted_log_path = replay_dir / "events.jsonl"
        replay_provenance_path = replay_dir / "replay.json"
        evaluation_path = replay_dir / "evaluation.json"
        missing: list[str] = []

        metadata = inspection.metadata
        manifest_rows = inspection.manifest_rows
        if not expected_events_path.exists():
            missing.append("expected-events.jsonl")
        if not expected_scenes_path.exists():
            missing.append("expected-scenes.jsonl")
        annotation = _read_json_object(
            annotation_path,
            missing,
            "annotation.json",
        )
        replay_provenance = _read_json_object(
            replay_provenance_path,
            missing,
            "replay.json",
        )
        evaluation = _read_json_object(
            evaluation_path,
            missing,
            "evaluation.json",
        )
        evidence_issues = _replay_provenance_issues(
            replay_provenance,
            session_name=session_name,
            manifest=manifest_path,
            config_snapshot=session_dir / str(metadata.get("config_snapshot") or ""),
            predicted_log=predicted_log_path,
            expected_frames=len(manifest_rows),
            recording_seal=_mapping(metadata.get("evidence_seal")),
            annotation=annotation_path,
        )
        evidence_issues.extend(_annotation_evidence_issues(
            annotation,
            session_name=session_name,
            manifest=manifest_path,
            expected_events=expected_events_path,
            expected_scenes=expected_scenes_path,
            replay=replay_provenance,
        ))
        evidence_issues.extend(_evaluation_evidence_issues(
            evaluation,
            predicted_log=predicted_log_path,
            expected_events=expected_events_path,
            expected_scenes=expected_scenes_path,
        ))
        replay_model_sha256 = _mapping(
            replay_provenance.get("model")
        ).get("sha256")
        if isinstance(replay_model_sha256, str) and replay_model_sha256:
            replay_model_hashes.add(replay_model_sha256)

        for digest in inspection.full_frame_hashes:
            owner = hash_owners.setdefault(digest, session_name)
            if owner != session_name:
                cross_session_duplicates.append({
                    "sha256": digest,
                    "first_session": owner,
                    "duplicate_session": session_name,
                })
        if (
            inspection.first_frame_timestamp is not None
            and inspection.last_frame_timestamp is not None
        ):
            current_start = inspection.first_frame_timestamp
            current_end = inspection.last_frame_timestamp
            for previous_name, previous_start, previous_end in session_time_ranges:
                if current_start <= previous_end and previous_start <= current_end:
                    cross_session_time_overlaps.append({
                        "first_session": previous_name,
                        "second_session": session_name,
                        "first_range": [previous_start, previous_end],
                        "second_range": [current_start, current_end],
                    })
            session_time_ranges.append((session_name, current_start, current_end))

        metadata_valid = inspection.valid
        manifest_valid = inspection.valid
        complete_game = bool(metadata.get("complete_game")) if metadata else False
        evaluation_passed = bool(
            evaluation
            and not evidence_issues
            and evaluation.get("passed") is True
        )
        session_success = (
            evaluation.get("session_success")
            if evaluation and not evidence_issues
            else None
        )
        if evaluation and not evidence_issues:
            event_counts = _mapping(evaluation.get("event_counts"))
            remaining_counts = _mapping(evaluation.get("remaining_counts"))
            aggregate["expected_events"] += _as_int(event_counts.get("expected", 0))
            aggregate["predicted_events"] += _as_int(event_counts.get("predicted", 0))
            aggregate["exact_events"] += _as_int(event_counts.get("exact", 0))
            aggregate["expected_plays"] += _as_int(
                event_counts.get("expected_plays", 0)
            )
            aggregate["exact_plays"] += _as_int(event_counts.get("exact_plays", 0))
            aggregate["remaining_correct"] += _as_int(
                remaining_counts.get("correct", 0)
            )
            aggregate["remaining_total"] += _as_int(
                remaining_counts.get("total", 0)
            )
            aggregate["remaining_required"] += _as_int(
                remaining_counts.get("required", 0)
            )
            aggregate["invariant_checks"] += _as_int(
                evaluation.get("deck_invariant_checks", 0)
            )
            aggregate["invariant_failures"] += _as_int(
                evaluation.get("deck_invariant_failures", 0)
            )
            aggregate["successful_rounds"] += int(session_success is True)

        sessions.append({
            "session": session_name,
            "frame_count": len(manifest_rows),
            "complete_game": complete_game,
            "metadata_valid": metadata_valid,
            "manifest_valid": manifest_valid,
            "recording_issues": list(inspection.issues),
            "missing": missing,
            "evidence_issues": evidence_issues,
            "evaluation_path": evaluation_path.as_posix(),
            "evaluation_passed": evaluation_passed,
            "session_success": session_success,
        })

    event_precision = _ratio(
        aggregate["exact_events"],
        aggregate["predicted_events"],
    )
    event_recall = _ratio(
        aggregate["exact_events"],
        aggregate["expected_events"],
    )
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision is not None
        and event_recall is not None
        and event_precision + event_recall
        else None
    )
    card_exact_accuracy = _ratio(
        aggregate["exact_plays"],
        aggregate["expected_plays"],
    )
    remaining_accuracy = _ratio(
        aggregate["remaining_correct"],
        aggregate["remaining_total"],
    )
    round_success_rate = _ratio(
        aggregate["successful_rounds"],
        len(sessions),
    )

    holdout_payload: dict[str, object] | None = None
    if card_holdout_report is not None and card_holdout_report.exists():
        loaded = json.loads(card_holdout_report.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("card holdout report must be a JSON object")
        holdout_payload = loaded
    holdout_issues = _holdout_evidence_issues(
        holdout_payload or {},
        replay_model_hashes=replay_model_hashes,
    )
    holdout_ready = bool(
        holdout_payload
        and holdout_payload.get("publication_ready") is True
        and not holdout_issues
    )
    all_session_files_present = bool(sessions) and all(
        not session["missing"] for session in sessions
    )
    all_recording_integrity_valid = bool(sessions) and all(
        session["metadata_valid"] is True and session["manifest_valid"] is True
        for session in sessions
    )
    all_evaluation_inputs_verified = bool(sessions) and all(
        not session["evidence_issues"] for session in sessions
    )
    all_sessions_complete = bool(sessions) and all(
        session["complete_game"] is True for session in sessions
    )
    all_session_reports_passed = bool(sessions) and all(
        session["evaluation_passed"] is True for session in sessions
    )

    checks = {
        "minimum_independent_sessions": len(sessions) >= thresholds.min_sessions,
        "all_sessions_marked_complete": all_sessions_complete,
        "all_annotations_and_reports_present": all_session_files_present,
        "all_recording_files_and_checksums_valid": all_recording_integrity_valid,
        "all_evaluation_inputs_verified": all_evaluation_inputs_verified,
        "no_cross_session_frame_leakage": not cross_session_duplicates,
        "no_cross_session_time_overlap": not cross_session_time_overlaps,
        "event_f1": event_f1 is not None and event_f1 >= thresholds.event_f1,
        "card_exact_accuracy": (
            card_exact_accuracy is not None
            and card_exact_accuracy >= thresholds.card_exact_accuracy
        ),
        "remaining_accuracy": (
            remaining_accuracy is not None
            and remaining_accuracy >= thresholds.remaining_accuracy
        ),
        "remaining_annotation_coverage": (
            aggregate["remaining_required"] > 0
            and aggregate["remaining_total"] >= aggregate["remaining_required"]
        ),
        "deck_invariant": (
            aggregate["invariant_checks"] > 0
            and aggregate["invariant_failures"] == 0
        ),
        "round_success_rate": (
            round_success_rate is not None
            and round_success_rate >= thresholds.round_success_rate
        ),
        "all_session_reports_passed": all_session_reports_passed,
    }
    if require_card_holdout:
        checks["independent_card_holdout"] = holdout_ready

    remaining_work = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "phase6-acceptance-audit-v2",
        "recordings_root": recordings_root.as_posix(),
        "replays_root": replays_root.as_posix(),
        "threshold_values": {
            "min_sessions": thresholds.min_sessions,
            "event_f1": thresholds.event_f1,
            "card_exact_accuracy": thresholds.card_exact_accuracy,
            "remaining_accuracy": thresholds.remaining_accuracy,
            "round_success_rate": thresholds.round_success_rate,
        },
        "session_count": len(sessions),
        "excluded_non_acceptance_sessions": excluded_sessions,
        "sessions": sessions,
        "cross_session_duplicate_frames": cross_session_duplicates,
        "cross_session_time_overlaps": cross_session_time_overlaps,
        "aggregate_counts": aggregate,
        "metrics": {
            "event_precision": _rounded(event_precision),
            "event_recall": _rounded(event_recall),
            "event_f1": _rounded(event_f1),
            "card_exact_accuracy": _rounded(card_exact_accuracy),
            "remaining_accuracy": _rounded(remaining_accuracy),
            "round_success_rate": _rounded(round_success_rate),
        },
        "card_holdout": {
            "required": require_card_holdout,
            "report": card_holdout_report.as_posix()
            if card_holdout_report is not None
            else None,
            "publication_ready": holdout_ready,
            "evidence_issues": holdout_issues,
        },
        "checks": checks,
        "remaining_work": remaining_work,
        "passed": all(checks.values()),
    }


def _read_json_object(
    path: Path,
    missing: list[str],
    label: str,
) -> dict[str, object]:
    if not path.exists():
        missing.append(label)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _evaluation_evidence_issues(
    evaluation: Mapping[str, object],
    *,
    predicted_log: Path,
    expected_events: Path,
    expected_scenes: Path,
) -> list[str]:
    if not evaluation:
        return ["evaluation report is missing"]
    issues: list[str] = []
    if evaluation.get("schema_version") != "phase6-live-replay-evaluation-v2":
        issues.append("unsupported evaluation schema")
    sealed_payload = dict(evaluation)
    report_sha256 = sealed_payload.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed_payload):
        issues.append("evaluation report content changed after generation")
    inputs = _mapping(evaluation.get("inputs"))
    for name, expected_path in (
        ("predicted_log", predicted_log),
        ("expected_events", expected_events),
        ("expected_scenes", expected_scenes),
    ):
        fingerprint = _mapping(inputs.get(name))
        if not fingerprint:
            issues.append(f"missing {name} fingerprint")
            continue
        resolved = expected_path.resolve()
        if str(fingerprint.get("path")) != resolved.as_posix():
            issues.append(f"{name} path does not match this session")
            continue
        if not resolved.exists():
            issues.append(f"{name} file is missing")
            continue
        digest = sha256_file(resolved)
        if str(fingerprint.get("sha256")) != digest:
            issues.append(f"{name} checksum changed after evaluation")
    return issues


def _replay_provenance_issues(
    replay: Mapping[str, object],
    *,
    session_name: str,
    manifest: Path,
    config_snapshot: Path,
    predicted_log: Path,
    expected_frames: int,
    recording_seal: Mapping[str, object],
    annotation: Path,
) -> list[str]:
    if not replay:
        return ["replay provenance is missing"]
    issues: list[str] = []
    if replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
        issues.append("unsupported replay provenance schema")
    sealed_payload = dict(replay)
    report_sha256 = sealed_payload.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed_payload):
        issues.append("replay provenance content changed after generation")
    if replay.get("session") != session_name:
        issues.append("replay provenance session does not match")
    if replay.get("replayed_frames") != expected_frames:
        issues.append("replay provenance frame count does not match recording")
    for name, expected_path in (
        ("manifest", manifest),
        ("events_log", predicted_log),
        ("annotation", annotation),
    ):
        issues.extend(_fingerprint_issues(replay.get(name), name, expected_path))
    config = _mapping(replay.get("config"))
    issues.extend(_absolute_fingerprint_issues(config, "replay config"))
    if config_snapshot.is_file() and config.get("sha256") != sha256_file(config_snapshot):
        issues.append("replay config does not match the recording snapshot")

    model = _mapping(replay.get("model"))
    model_path = Path(str(model.get("path") or ""))
    if not model_path.is_absolute():
        issues.append("replay model path must be absolute")
    else:
        issues.extend(_fingerprint_issues(model, "model", model_path))

    templates = _mapping(replay.get("templates"))
    templates_path = Path(str(templates.get("path") or ""))
    if not templates_path.is_absolute():
        issues.append("replay templates path must be absolute")
    else:
        try:
            digest = sha256_directory(templates_path)
        except ValueError as exc:
            issues.append(str(exc))
        else:
            if templates.get("sha256") != digest:
                issues.append("templates checksum changed after replay")
    implementation = _mapping(replay.get("implementation"))
    project_root = Path(__file__).resolve().parents[2]
    if implementation.get("sha256") != sha256_python_implementation(project_root):
        issues.append("replay implementation changed after replay")
    if implementation.get("runtime_versions") != runtime_versions():
        issues.append("replay runtime dependencies changed after replay")
    if not recording_seal:
        issues.append("acceptance recording is missing its evidence seal")
    else:
        for sealed_name, replay_value in (
            ("implementation_sha256", implementation.get("sha256")),
            ("model_sha256", model.get("sha256")),
            ("templates_sha256", templates.get("sha256")),
        ):
            sealed_value = recording_seal.get(sealed_name)
            if not isinstance(sealed_value, str) or not sealed_value:
                issues.append(f"recording evidence seal is missing {sealed_name}")
            elif sealed_value != replay_value:
                issues.append(
                    f"{sealed_name} changed after acceptance recording"
                )
    return issues


def _annotation_evidence_issues(
    annotation: Mapping[str, object],
    *,
    session_name: str,
    manifest: Path,
    expected_events: Path,
    expected_scenes: Path,
    replay: Mapping[str, object],
) -> list[str]:
    if not annotation:
        return ["blind annotation seal is missing"]
    issues: list[str] = []
    if annotation.get("schema_version") != "phase6-blind-annotation-v1":
        issues.append("unsupported blind annotation schema")
    sealed_payload = dict(annotation)
    report_sha256 = sealed_payload.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed_payload):
        issues.append("blind annotation content changed after sealing")
    if annotation.get("session") != session_name:
        issues.append("blind annotation session does not match")
    if annotation.get("annotation_mode") != "blind_without_replay_predictions":
        issues.append("annotations were not sealed in blind mode")
    if annotation.get("prediction_inputs_used") is not False:
        issues.append("annotations declare prediction-assisted labels")
    for name, expected_path in (
        ("manifest", manifest),
        ("expected_events", expected_events),
        ("expected_scenes", expected_scenes),
    ):
        issues.extend(_fingerprint_issues(
            annotation.get(name),
            f"annotation {name}",
            expected_path,
        ))
    workbook = _mapping(annotation.get("workbook"))
    workbook_path = manifest.parent / "annotation-workbook.json"
    issues.extend(_fingerprint_issues(
        workbook,
        "annotation workbook",
        workbook_path,
    ))
    completed_at = annotation.get("completed_at")
    replay_created_at = replay.get("created_at")
    if (
        isinstance(completed_at, bool)
        or not isinstance(completed_at, (int, float))
        or isinstance(replay_created_at, bool)
        or not isinstance(replay_created_at, (int, float))
    ):
        issues.append("annotation/replay timestamps are missing or invalid")
    elif float(completed_at) > float(replay_created_at):
        issues.append("annotations were sealed after replay predictions existed")
    return issues


def _fingerprint_issues(
    value: object,
    name: str,
    expected_path: Path,
) -> list[str]:
    fingerprint = _mapping(value)
    if not fingerprint:
        return [f"missing {name} fingerprint"]
    resolved = expected_path.resolve()
    if str(fingerprint.get("path")) != resolved.as_posix():
        return [f"{name} path does not match this session"]
    if not resolved.is_file():
        return [f"{name} file is missing"]
    if fingerprint.get("sha256") != sha256_file(resolved):
        return [f"{name} checksum changed after replay"]
    return []


def _holdout_evidence_issues(
    report: Mapping[str, object],
    *,
    replay_model_hashes: set[str],
) -> list[str]:
    if not report:
        return ["card holdout report is missing"]
    issues: list[str] = []
    if report.get("schema_version") != "real-window-holdout-v3":
        issues.append("unsupported card holdout schema")
    sealed_payload = dict(report)
    report_sha256 = sealed_payload.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed_payload):
        issues.append("card holdout report content changed after generation")
    inputs = _mapping(report.get("inputs"))
    for name in ("model", "manifest", "training_manifest"):
        value = inputs.get(name)
        if name == "training_manifest" and value is None:
            issues.append("card holdout training manifest fingerprint is missing")
            continue
        issues.extend(_absolute_fingerprint_issues(value, f"holdout {name}"))
    model_sha256 = _mapping(inputs.get("model")).get("sha256")
    if len(replay_model_hashes) != 1:
        issues.append("replay sessions must use exactly one model checksum")
    elif model_sha256 not in replay_model_hashes:
        issues.append("card holdout model does not match replay model")
    artifacts = _mapping(report.get("artifacts"))
    for name in ("predictions", "errors", "confusion_matrix"):
        issues.extend(_absolute_fingerprint_issues(
            artifacts.get(name),
            f"holdout artifact {name}",
        ))
    sample_count = _safe_int(report.get("sample_count", 0))
    class_counts = _mapping(report.get("class_counts"))
    source_counts = _mapping(report.get("source_counts"))
    if sample_count < 300:
        issues.append("card holdout requires at least 300 samples")
    if any(_safe_int(class_counts.get(label, 0)) < 10 for label in _CARD_CLASSES):
        issues.append("card holdout requires at least 10 samples for every class")
    if len(source_counts) < 3:
        issues.append("card holdout requires at least 3 independent sources")
    accuracy = report.get("accuracy")
    if not isinstance(accuracy, (int, float)) or float(accuracy) < 0.95:
        issues.append("card holdout overall accuracy must be at least 0.95")
    per_class = _mapping(report.get("per_class"))
    for label in _CARD_CLASSES:
        class_accuracy = _mapping(per_class.get(label)).get("accuracy")
        if (
            not isinstance(class_accuracy, (int, float))
            or float(class_accuracy) < 0.90
        ):
            issues.append("card holdout per-class accuracy must be at least 0.90")
            break
    leakage = _mapping(report.get("leakage_check"))
    if leakage.get("checked") is not True or leakage.get("overlap_count") != 0:
        issues.append("card holdout training leakage check did not pass")
    readiness = report.get("readiness_checks")
    if not isinstance(readiness, list) or not readiness or any(
        not isinstance(check, Mapping) or check.get("passed") is not True
        for check in readiness
    ):
        issues.append("card holdout readiness checks did not all pass")
    predictions = _mapping(artifacts.get("predictions"))
    predictions_path = Path(str(predictions.get("path") or ""))
    if predictions_path.is_file() and _jsonl_row_count(predictions_path) != sample_count:
        issues.append("card holdout prediction count does not match sample_count")
    issues.extend(_blind_holdout_seal_issues(report))
    return issues


def _blind_holdout_seal_issues(
    report: Mapping[str, object],
) -> list[str]:
    issues: list[str] = []
    seal_fingerprint = _mapping(report.get("holdout_seal"))
    seal_path = Path(str(seal_fingerprint.get("path") or ""))
    if not seal_path.is_absolute():
        return ["blind holdout seal path must be absolute"]
    issues.extend(_fingerprint_issues(
        seal_fingerprint,
        "blind holdout seal",
        seal_path,
    ))
    if not seal_path.is_file():
        return issues
    try:
        loaded = json.loads(seal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [*issues, "blind holdout seal is invalid JSON"]
    if not isinstance(loaded, dict):
        return [*issues, "blind holdout seal must contain an object"]
    seal = loaded
    if seal.get("schema_version") != "real-window-holdout-blind-seal-v1":
        issues.append("unsupported blind holdout seal schema")
    sealed_payload = dict(seal)
    report_sha256 = sealed_payload.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed_payload):
        issues.append("blind holdout seal content changed after sealing")
    if seal.get("annotation_mode") != "blind_without_model_predictions":
        issues.append("holdout labels were not sealed in blind mode")
    if seal.get("prediction_inputs_used") is not False:
        issues.append("holdout labels declare prediction-assisted annotation")
    completed_at = seal.get("completed_at")
    evaluated_at = report.get("created_at")
    if (
        isinstance(completed_at, bool)
        or not isinstance(completed_at, (int, float))
        or isinstance(evaluated_at, bool)
        or not isinstance(evaluated_at, (int, float))
    ):
        issues.append("holdout seal/evaluation timestamps are missing or invalid")
    elif float(completed_at) > float(evaluated_at):
        issues.append("holdout labels were sealed after predictions existed")
    inputs = _mapping(report.get("inputs"))
    for name in ("model", "manifest", "training_manifest"):
        if seal.get(name) != inputs.get(name):
            issues.append(f"blind holdout seal {name} does not match evaluation")
    implementation = _mapping(seal.get("implementation"))
    project_root = Path(__file__).resolve().parents[2]
    if implementation.get("project_root") != project_root.as_posix():
        issues.append("blind holdout implementation root does not match")
    if implementation.get("sha256") != sha256_python_implementation(project_root):
        issues.append("blind holdout implementation changed after label sealing")
    sessions = seal.get("sessions")
    if not isinstance(sessions, list) or len(sessions) < 3:
        issues.append("blind holdout seal requires at least 3 session fingerprints")
    else:
        for index, fingerprint in enumerate(sessions, start=1):
            issues.extend(_absolute_fingerprint_issues(
                fingerprint,
                f"blind holdout session {index}",
            ))
    summary = _mapping(seal.get("summary"))
    if summary.get("sample_count") != report.get("sample_count"):
        issues.append("blind holdout sample count does not match evaluation")
    if summary.get("class_counts") != report.get("class_counts"):
        issues.append("blind holdout class counts do not match evaluation")
    if summary.get("source_counts") != report.get("source_counts"):
        issues.append("blind holdout source counts do not match evaluation")
    return issues


def _absolute_fingerprint_issues(value: object, name: str) -> list[str]:
    fingerprint = _mapping(value)
    if not fingerprint:
        return [f"missing {name} fingerprint"]
    path = Path(str(fingerprint.get("path") or ""))
    if not path.is_absolute():
        return [f"{name} path must be absolute"]
    if not path.is_file():
        return [f"{name} file is missing"]
    if fingerprint.get("sha256") != sha256_file(path):
        return [f"{name} checksum changed after evaluation"]
    return []


def _jsonl_row_count(path: Path) -> int:
    return sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"expected an integer-compatible value, got {value!r}")


def _safe_int(value: object) -> int:
    try:
        return _as_int(value)
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


__all__ = ["Phase6AcceptanceThresholds", "audit_phase6_acceptance"]
