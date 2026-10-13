from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.live_evidence import (
    EVIDENCE_SESSION_METADATA,
    EvidenceRetentionPolicy,
    EvidenceSessionStatus,
    FixedIssue,
    FixedIssueRegistry,
    LiveEvidenceError,
    LiveEvidenceStore,
    load_fixed_issue_registry,
    save_fixed_issue_registry,
    stable_issue_signature,
)
from src.reporting.live_log_reader import list_live_log_sources

DAY = 86400.0


@pytest.mark.parametrize("legacy", [False, True])
def test_fixed_cleanup_never_assumes_missing_unresolved_references_are_complete(
    tmp_path: Path, legacy: bool,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    fixed = stable_issue_signature("fixed")
    unresolved = stable_issue_signature("not fixed")
    store.start_session("missing-slice", runtime_version="test", registry_version=0, started_at=1.0)
    for signature in (fixed, unresolved):
        store.record_issue("missing-slice", "r1", signature=signature, summary=signature, observed_at=2.0)
    frame = _write_error_frame(store, "missing-slice", unresolved, name="shared.png", size=17)
    events = store.session_path("missing-slice") / "events.jsonl"
    lost = events.with_name(events.name + ".20260905-120000")
    events.rename(lost)
    events.write_text(json.dumps({"event": "error_frame_saved", "path": str(frame), "issue_signature": fixed}) + "\n")
    if legacy:
        store.close_session("missing-slice", closed_at=3.0)
    else:
        _seal_and_close_session(store, "missing-slice", closed_at=3.0)
    # Simulate a slice already lost to an older writer or manual relocation.
    # There is deliberately no pruning marker in the retained current file.
    lost.unlink()
    report = store.cleanup(
        EvidenceRetentionPolicy(),
        fixed_registry=FixedIssueRegistry.from_mapping(1, {fixed: 1}),
        now=4.0, dry_run=True,
    )
    assert not report.actions
    assert report.errors
    assert frame.read_bytes() == b"p" * 17


@pytest.mark.parametrize("conflicting_reference", [False, True])
def test_fixed_cleanup_reads_archives_and_keeps_shared_unresolved_frames(
    tmp_path: Path, conflicting_reference: bool,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    fixed = stable_issue_signature("fixed problem")
    unresolved = stable_issue_signature("unresolved problem")
    store.start_session("scan-rotated", runtime_version="test", registry_version=0, started_at=1.0)
    store.record_issue("scan-rotated", "r1", signature=fixed, summary="fixed", observed_at=2.0)
    store.record_issue("scan-rotated", "r2", signature=unresolved, summary="new", observed_at=3.0)
    frame = _write_error_frame(store, "scan-rotated", fixed, name="shared.png", size=17)
    events = store.session_path("scan-rotated") / "events.jsonl"
    archive = events.with_name(events.name + ".20260905-120000.10")
    events.rename(archive)
    events.write_text(
        json.dumps({"event": "error_frame_saved", "path": str(frame), "issue_signature": unresolved}) + "\n"
        if conflicting_reference else '{"event":"scan_session_stopped"}\n'
    )
    _seal_and_close_session(store, "scan-rotated", closed_at=4.0)
    report = store.cleanup(
        EvidenceRetentionPolicy(),
        fixed_registry=FixedIssueRegistry.from_mapping(version=1, fixed_versions={fixed: 1}),
        now=5.0, dry_run=False,
    )
    assert frame.exists() is conflicting_reference
    assert archive.exists() and events.exists()
    assert report.errors == ()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "corrupt", "ordinary_reference", "pruned"])
def test_fixed_cleanup_keeps_frames_when_archive_inspection_is_incomplete_or_shared(
    tmp_path: Path, unsafe_kind: str,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    fixed = stable_issue_signature("fixed problem")
    _closed_session(store, "scan-safe", at=1.0, registry_version=0, signature=fixed, error_frame_bytes=17)
    session = store.session_path("scan-safe")
    frame = session / "frames" / "error.png"
    archive = session / "events.jsonl.20260905-120000"
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside.jsonl"
        outside.write_text("{}\n")
        archive.symlink_to(outside)
    elif unsafe_kind == "corrupt":
        archive.write_text("not JSON\n")
    elif unsafe_kind == "pruned":
        archive.write_text('{"event":"live_log_history_pruned"}\n')
    else:
        archive.write_text(json.dumps({
            "event": "diagnostic_frame_promoted", "kind": "result-defeat", "path": str(frame)
        }) + "\n")
    report = store.cleanup(
        EvidenceRetentionPolicy(),
        fixed_registry=FixedIssueRegistry.from_mapping(version=1, fixed_versions={fixed: 1}),
        now=2.0, dry_run=False,
    )
    assert frame.read_bytes() == b"p" * 17
    assert not report.actions
    if unsafe_kind != "ordinary_reference":
        assert report.errors


def _closed_session(
    store: LiveEvidenceStore,
    session_id: str,
    *,
    at: float,
    registry_version: int = 1,
    signature: str | None = None,
    artifact_bytes: int = 0,
    error_frame_bytes: int = 0,
) -> None:
    store.start_session(
        session_id,
        runtime_version="test-runtime",
        registry_version=registry_version,
        started_at=at,
    )
    if signature is not None:
        store.record_issue(
            session_id,
            "round-001",
            signature=signature,
            summary="fixture issue",
            observed_at=at,
        )
    if artifact_bytes:
        (store.session_path(session_id) / "frame.bin").write_bytes(b"x" * artifact_bytes)
    if error_frame_bytes:
        assert signature is not None
        _write_error_frame(
            store,
            session_id,
            signature,
            name="error.png",
            size=error_frame_bytes,
        )
    _seal_and_close_session(store, session_id, closed_at=at)


def _seal_and_close_session(store: LiveEvidenceStore, session_id: str, *, closed_at: float) -> None:
    """Produce the same start/sequence/close proof as the real writer."""

    events = store.session_path(session_id) / "events.jsonl"
    sources = list_live_log_sources(events)
    slices = [(source.path, [json.loads(line) for line in source.path.read_text().splitlines()]) for source in sources]
    if not slices or slices[-1][0] != events:
        slices.append((events, []))
    slices[0][1].insert(0, {"event": "scan_session_started"})
    slices[-1][1].append({"event": "scan_session_stopped"})
    sequence = 0
    for path, rows in slices:
        for row in rows:
            sequence += 1
            row["scan_event_index"] = sequence
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    store.close_session(session_id, closed_at=closed_at, sealed_event_count=sequence)


def _write_error_frame(
    store: LiveEvidenceStore,
    session_id: str,
    signature: str,
    *,
    name: str,
    size: int,
    round_id: str = "round-001",
) -> Path:
    session_path = store.session_path(session_id)
    frames_path = session_path / "frames"
    frames_path.mkdir(exist_ok=True)
    frame_path = frames_path / name
    frame_path.write_bytes(b"p" * size)
    events_path = session_path / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "error_frame_saved",
                    "round_id": round_id,
                    "issue_signature": signature,
                    "path": frame_path.as_posix(),
                }
            )
            + "\n"
        )
    return frame_path


def test_stable_issue_signature_normalizes_volatile_counts_and_ids() -> None:
    first = stable_issue_signature(
        "right remaining count mismatch: screen=4, expected=5; frame=463"
    )
    second = stable_issue_signature(
        "right remaining count mismatch: screen=14, expected=15; frame=999"
    )

    assert first == second
    assert first.startswith("live:right-remaining-count-mismatch")


def test_store_records_session_round_and_repeated_issue_metadata(
    tmp_path: Path,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    signature = stable_issue_signature("vertical stacked play card missed")

    store.start_session(
        "scan-001",
        runtime_version="abc123",
        registry_version=3,
        started_at=10.0,
    )
    store.record_issue(
        "scan-001",
        "round-live-001",
        signature=signature,
        summary="missed second five",
        observed_at=11.0,
    )
    store.record_issue(
        "scan-001",
        "round-live-001",
        signature=signature,
        summary="missed second five again",
        observed_at=12.0,
    )
    closed = store.close_session("scan-001", closed_at=13.0)
    loaded = store.load_session("scan-001")

    assert closed.status is EvidenceSessionStatus.CLOSED
    assert loaded == closed
    assert loaded.rounds[0].round_id == "round-live-001"
    assert loaded.rounds[0].issues[0].occurrences == 2
    assert loaded.rounds[0].issues[0].last_seen_at == 12.0
    with pytest.raises(LiveEvidenceError, match="closed"):
        store.touch_round("scan-001", "round-live-002", observed_at=14.0)


def test_store_recovers_only_abandoned_or_stale_active_sessions(
    tmp_path: Path,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    store.start_session(
        "dead-owner",
        runtime_version="test",
        registry_version=1,
        started_at=10.0,
        owner_pid=111,
    )
    store.start_session(
        "live-owner",
        runtime_version="test",
        registry_version=1,
        started_at=10.0,
        owner_pid=222,
    )
    store.start_session(
        "old-ownerless",
        runtime_version="legacy",
        registry_version=0,
        started_at=1.0,
    )
    store.start_session(
        "recent-ownerless",
        runtime_version="legacy",
        registry_version=0,
        started_at=99.0,
    )

    recovered = store.recover_abandoned_sessions(
        owner_alive=lambda pid: pid == 222,
        now=100.0,
        ownerless_stale_seconds=10.0,
    )

    assert recovered == ("dead-owner", "old-ownerless")
    assert store.load_session("dead-owner").status is EvidenceSessionStatus.CLOSED
    assert store.load_session("old-ownerless").status is EvidenceSessionStatus.CLOSED
    assert store.load_session("live-owner").status is EvidenceSessionStatus.ACTIVE
    assert store.load_session("recent-ownerless").status is EvidenceSessionStatus.ACTIVE


def test_fixed_registry_round_trip_and_version_boundary(tmp_path: Path) -> None:
    signature = stable_issue_signature("known issue")
    registry = FixedIssueRegistry(
        version=4,
        issues=(
            FixedIssue(
                signature=signature,
                fixed_in_registry_version=4,
                fixed_in_runtime_version="fix-sha",
            ),
        ),
    )
    path = tmp_path / "fixed-issues.json"

    save_fixed_issue_registry(path, registry)
    loaded = load_fixed_issue_registry(path)

    assert loaded == registry
    assert loaded.resolves(signature, 3) is True
    assert loaded.resolves(signature, 4) is False


def test_cleanup_deletes_only_resolved_error_frames_and_keeps_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    fixed_signature = stable_issue_signature("remaining mismatch")
    unresolved_signature = stable_issue_signature("new segmentation regression")
    store.start_session(
        "pre-fix",
        runtime_version="old-runtime",
        registry_version=1,
        started_at=10.0,
    )
    store.record_issue(
        "pre-fix",
        "round-001",
        signature=fixed_signature,
        summary="fixed mismatch",
        observed_at=11.0,
    )
    store.record_issue(
        "pre-fix",
        "round-002",
        signature=unresolved_signature,
        summary="must be retained",
        observed_at=12.0,
    )
    fixed_frame = _write_error_frame(
        store,
        "pre-fix",
        fixed_signature,
        name="fixed.png",
        size=20,
        round_id="round-001",
    )
    unresolved_frame = _write_error_frame(
        store,
        "pre-fix",
        unresolved_signature,
        name="unresolved.png",
        size=30,
        round_id="round-002",
    )
    ordinary_frame = store.session_path("pre-fix") / "frames" / "ordinary.png"
    ordinary_frame.write_bytes(b"ordinary frame")
    with (store.session_path("pre-fix") / "events.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "event": "diagnostic_frame_saved",
                    "round_id": "round-002",
                    "path": ordinary_frame.as_posix(),
                }
            )
            + "\n"
        )
    _seal_and_close_session(store, "pre-fix", closed_at=13.0)
    _closed_session(
        store,
        "post-fix-regression",
        at=99.0,
        registry_version=2,
        signature=fixed_signature,
        error_frame_bytes=21,
    )
    store.start_session(
        "active-pre-fix",
        runtime_version="old-runtime",
        registry_version=1,
        started_at=5.0,
    )
    store.record_issue(
        "active-pre-fix",
        "round-active",
        signature=fixed_signature,
        summary="still recording",
        observed_at=5.0,
    )
    unmanaged = root / "existing-real-screenshots"
    unmanaged.mkdir()
    existing = unmanaged / "real.png"
    existing.write_bytes(b"never delete")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked-evidence").symlink_to(outside, target_is_directory=True)
    registry = FixedIssueRegistry.from_mapping(2, {fixed_signature: 2})
    policy = EvidenceRetentionPolicy(
        max_total_bytes=10**9,
        max_sessions=10,
        retention_days=365,
        protect_recent_unresolved_days=7,
    )

    preview = store.cleanup(
        policy,
        fixed_registry=registry,
        now=100.0,
        dry_run=True,
    )

    assert [action.session_id for action in preview.actions] == ["pre-fix"]
    assert preview.actions[0].performed is False
    assert preview.actions[0].target == "error_frame"
    assert preview.actions[0].relative_path == "frames/fixed.png"
    assert preview.actions[0].bytes == 20
    assert preview.actions[0].reasons == (
        f"issue_fixed:{fixed_signature}@registry_v2",
    )
    assert preview.fixed_registry_version == 2
    assert preview.policy == policy
    assert preview.projected_after_session_count == preview.before_session_count
    assert preview.projected_after_total_bytes == preview.before_total_bytes - 20
    assert preview.actual_after_session_count == preview.before_session_count
    assert preview.actual_after_total_bytes == preview.before_total_bytes
    assert preview.planned_reclaimed_bytes == 20
    assert preview.reclaimed_bytes == 0
    json.dumps(preview.to_payload())
    assert fixed_frame.exists()
    protected = {item.session_id: item.reasons for item in preview.protected_sessions}
    assert "active_session" in protected["active-pre-fix"]
    assert "recent_unresolved" in protected["post-fix-regression"]
    assert "recent_unresolved" in protected["pre-fix"]
    assert set(preview.ignored_entries) == {
        "existing-real-screenshots",
        "linked-evidence",
    }

    report = store.cleanup(
        policy,
        fixed_registry=registry,
        now=100.0,
        dry_run=False,
    )

    assert report.actions[0].performed is True
    assert report.actions[0].target == "error_frame"
    assert (root / "pre-fix").exists()
    assert not fixed_frame.exists()
    assert unresolved_frame.exists()
    assert ordinary_frame.read_bytes() == b"ordinary frame"
    assert (root / "pre-fix" / "events.jsonl").exists()
    assert (root / "pre-fix" / EVIDENCE_SESSION_METADATA).exists()
    assert report.actual_after_session_count == report.before_session_count
    assert report.actual_after_total_bytes == report.before_total_bytes - 20
    assert report.reclaimed_bytes == 20
    assert (root / "post-fix-regression").exists()
    assert (root / "active-pre-fix").exists()
    assert existing.read_bytes() == b"never delete"
    assert outside.exists()


def test_cleanup_never_treats_untrusted_registry_observation_as_pre_fix(
    tmp_path: Path,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    signature = stable_issue_signature("known issue")
    store.start_session(
        "registry-unavailable",
        runtime_version="current-runtime",
        registry_version=0,
        registry_trusted=False,
        started_at=10.0,
    )
    store.record_issue(
        "registry-unavailable",
        "round-1",
        signature=signature,
        summary="observed while registry was unavailable",
        observed_at=11.0,
    )
    store.close_session("registry-unavailable", closed_at=12.0)
    registry = FixedIssueRegistry(
        version=1,
        issues=(
            FixedIssue(
                signature=signature,
                fixed_in_registry_version=1,
                fixed_in_runtime_version="phase6-live-v1",
            ),
        ),
    )

    report = store.cleanup(
        EvidenceRetentionPolicy(
            max_total_bytes=10**9,
            max_sessions=10,
            retention_days=365,
            protect_recent_unresolved_days=7,
        ),
        fixed_registry=registry,
        supported_runtime_versions=frozenset({"phase6-live-v1"}),
        now=13.0,
        dry_run=False,
    )

    assert report.actions == ()
    assert (store.root / "registry-unavailable").exists()
    assert any(
        reason.startswith("latest_unresolved:") for reason in report.protected_sessions[0].reasons
    )


def test_cleanup_requires_registry_fix_to_match_supported_runtime(
    tmp_path: Path,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    signature = stable_issue_signature("known issue")
    _closed_session(
        store,
        "pre-fix",
        at=10.0,
        registry_version=0,
        signature=signature,
        error_frame_bytes=17,
    )
    frame_path = store.session_path("pre-fix") / "frames" / "error.png"
    registry = FixedIssueRegistry(
        version=1,
        issues=(
            FixedIssue(
                signature=signature,
                fixed_in_registry_version=1,
                fixed_in_runtime_version="phase6-live-v2",
            ),
        ),
    )
    policy = EvidenceRetentionPolicy(
        max_total_bytes=10**9,
        max_sessions=10,
        retention_days=365,
        protect_recent_unresolved_days=7,
    )

    incompatible = store.cleanup(
        policy,
        fixed_registry=registry,
        supported_runtime_versions=frozenset({"phase6-live-v1"}),
        now=11.0,
        dry_run=False,
    )

    assert incompatible.actions == ()
    assert frame_path.exists()

    compatible = store.cleanup(
        policy,
        fixed_registry=registry,
        supported_runtime_versions=frozenset({"phase6-live-v2"}),
        now=11.0,
        dry_run=False,
    )

    assert compatible.actions[0].performed is True
    assert compatible.actions[0].target == "error_frame"
    assert not frame_path.exists()
    assert (store.root / "pre-fix").exists()
    assert (store.root / "pre-fix" / "events.jsonl").exists()


def test_cleanup_capacity_can_override_latest_unresolved_soft_protection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    signature = stable_issue_signature("unresolved segmentation issue")
    _closed_session(
        store,
        "old-unresolved-copy",
        at=1 * DAY,
        signature=signature,
        artifact_bytes=80,
    )
    _closed_session(
        store,
        "latest-unresolved",
        at=2 * DAY,
        signature=signature,
        artifact_bytes=80,
    )
    _closed_session(
        store,
        "ordinary-old",
        at=1.5 * DAY,
        artifact_bytes=80,
    )
    policy = EvidenceRetentionPolicy(
        max_total_bytes=1,
        max_sessions=0,
        retention_days=1,
        protect_recent_unresolved_days=0,
    )

    report = store.cleanup(policy, now=10 * DAY, dry_run=False)

    assert {action.session_id for action in report.actions} == {
        "old-unresolved-copy",
        "latest-unresolved",
        "ordinary-old",
    }
    assert not (root / "latest-unresolved").exists()
    latest_action = next(
        action for action in report.actions if action.session_id == "latest-unresolved"
    )
    assert latest_action.reasons[0] == "capacity_limit"
    assert any(
        reason.startswith("soft_protection_overridden:latest_unresolved:")
        for reason in latest_action.reasons
    )
    assert report.protected_sessions == ()
    assert report.projected_limits_satisfied is True
    assert report.limits_satisfied is True


def test_cleanup_multiple_unique_unresolved_sessions_still_obey_capacity(
    tmp_path: Path,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    issue_names = ("alpha", "bravo", "charlie", "delta", "echo")
    for index, issue_name in enumerate(issue_names):
        _closed_session(
            store,
            f"unique-{index}",
            at=10.0 + index,
            signature=stable_issue_signature(f"unique unresolved issue {issue_name}"),
            artifact_bytes=20,
        )

    preview = store.cleanup(
        EvidenceRetentionPolicy(
            max_total_bytes=10**9,
            max_sessions=2,
            retention_days=365,
            protect_recent_unresolved_days=7,
        ),
        now=20.0,
        dry_run=True,
    )

    assert [action.session_id for action in preview.actions] == [
        "unique-0",
        "unique-1",
        "unique-2",
    ]
    assert all(action.reasons[0] == "capacity_limit" for action in preview.actions)
    assert all(
        any(
            reason.startswith("soft_protection_overridden:latest_unresolved:")
            for reason in action.reasons
        )
        for action in preview.actions
    )
    assert {item.session_id for item in preview.protected_sessions} == {
        "unique-3",
        "unique-4",
    }
    assert preview.projected_after_session_count == 2
    assert preview.actual_after_session_count == 5
    assert preview.projected_limits_satisfied is True

    report = store.cleanup(
        preview.policy,
        now=20.0,
        dry_run=False,
    )

    remaining = sorted(path.name for path in store.root.iterdir() if path.is_dir())
    assert remaining == ["unique-3", "unique-4"]
    assert report.actual_after_session_count == 2
    assert report.limits_satisfied is True


def test_cleanup_capacity_keeps_active_and_pinned_sessions_hard_protected(
    tmp_path: Path,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    store.start_session(
        "active",
        runtime_version="test",
        registry_version=0,
        started_at=10.0,
    )
    store.start_session(
        "pinned",
        runtime_version="test",
        registry_version=0,
        started_at=11.0,
        pinned=True,
    )
    store.close_session("pinned", closed_at=12.0)

    report = store.cleanup(
        EvidenceRetentionPolicy(max_total_bytes=0, max_sessions=0),
        now=20.0,
        dry_run=False,
    )

    assert report.actions == ()
    protected = {item.session_id: item.reasons for item in report.protected_sessions}
    assert protected == {
        "active": ("active_session",),
        "pinned": ("pinned",),
    }
    assert report.projected_after_session_count == 2
    assert report.actual_after_session_count == 2
    assert report.projected_limits_satisfied is False
    assert report.limits_satisfied is False


def test_cleanup_reports_invalid_metadata_without_deleting_it(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    invalid = root / "broken"
    invalid.mkdir(parents=True)
    metadata = invalid / EVIDENCE_SESSION_METADATA
    metadata.write_text("{not-json}\n", encoding="utf-8")

    report = LiveEvidenceStore(root).cleanup(
        EvidenceRetentionPolicy(max_total_bytes=0, max_sessions=0),
        now=100.0,
        dry_run=False,
    )

    assert report.invalid_entries
    assert metadata.exists()
    assert report.actions == ()


def test_cleanup_rejects_non_boolean_registry_trust_without_deleting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    store = LiveEvidenceStore(root)
    store.start_session(
        "tampered-trust",
        runtime_version="test",
        registry_version=0,
        started_at=10.0,
    )
    store.close_session("tampered-trust", closed_at=11.0)
    metadata = root / "tampered-trust" / EVIDENCE_SESSION_METADATA
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["registry_trusted"] = "false"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    report = store.cleanup(
        EvidenceRetentionPolicy(max_total_bytes=0, max_sessions=0),
        now=100.0,
        dry_run=False,
    )

    assert any("registry_trusted must be a boolean" in item for item in report.invalid_entries)
    assert metadata.exists()
    assert report.actions == ()


@pytest.mark.parametrize(
    ("max_total_bytes", "max_sessions", "expected_remaining"),
    (
        (10**9, 1, 1),
        (0, 10, 0),
    ),
)
def test_cleanup_enforces_count_and_byte_limits(
    tmp_path: Path,
    max_total_bytes: int,
    max_sessions: int,
    expected_remaining: int,
) -> None:
    store = LiveEvidenceStore(tmp_path / "evidence")
    _closed_session(store, "older", at=10.0, artifact_bytes=20)
    _closed_session(store, "newer", at=20.0, artifact_bytes=20)

    report = store.cleanup(
        EvidenceRetentionPolicy(
            max_total_bytes=max_total_bytes,
            max_sessions=max_sessions,
            retention_days=365,
            protect_recent_unresolved_days=0,
        ),
        now=30.0,
        dry_run=False,
    )

    remaining = [path for path in store.root.iterdir() if path.is_dir() and not path.is_symlink()]
    assert len(remaining) == expected_remaining
    assert all(action.reasons == ("capacity_limit",) for action in report.actions)
    assert report.limits_satisfied is True
