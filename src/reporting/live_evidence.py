from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from src.reporting.live_log_reader import (
    LiveLogSource,
    iter_live_log_records,
    list_live_log_sources,
)

EVIDENCE_SESSION_SCHEMA_VERSION = "phase6-live-evidence-session-v1"
FIXED_ISSUE_REGISTRY_SCHEMA_VERSION = "phase6-live-fixed-issues-v1"
EVIDENCE_SESSION_METADATA = "evidence-session.json"

_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_SIGNATURE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_PATTERN = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
_PATH_PATTERN = re.compile(r"(?:^|\s)(?:/[^\s:]+)+")


class LiveEvidenceError(ValueError):
    """Raised when managed live-evidence metadata is unsafe or malformed."""


class EvidenceSessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


def stable_issue_signature(issue_key: str, *, namespace: str = "live") -> str:
    """Return a stable, non-reversible signature for a recurring live issue.

    Callers should prefer a short semantic key such as
    ``remaining_count_mismatch``.  When a raw diagnostic message is supplied,
    volatile UUIDs, long hashes, paths and numbers are normalized before
    hashing so frame ids and observed counts do not create endless new issues.
    """

    canonical = unicodedata.normalize("NFKC", issue_key).strip().lower()
    if not canonical:
        raise LiveEvidenceError("issue_key cannot be empty")
    canonical = _UUID_PATTERN.sub("<uuid>", canonical)
    canonical = _HEX_PATTERN.sub("<hash>", canonical)
    canonical = _PATH_PATTERN.sub(" <path>", canonical)
    canonical = _NUMBER_PATTERN.sub("<n>", canonical)
    canonical = " ".join(canonical.split())
    readable = "-".join(re.findall(r"[a-z0-9]+", canonical))[:48]
    if not readable:
        readable = "issue"
    safe_namespace = "-".join(re.findall(r"[a-z0-9]+", namespace.lower()))
    if not safe_namespace:
        safe_namespace = "live"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{safe_namespace}:{readable}:{digest}"


@dataclass(frozen=True)
class EvidenceIssue:
    signature: str
    summary: str
    first_seen_at: float
    last_seen_at: float
    occurrences: int
    registry_version: int

    def __post_init__(self) -> None:
        _validate_signature(self.signature)
        _validate_timestamp(self.first_seen_at, "first_seen_at")
        _validate_timestamp(self.last_seen_at, "last_seen_at")
        if self.last_seen_at < self.first_seen_at:
            raise LiveEvidenceError("issue last_seen_at cannot precede first_seen_at")
        if self.occurrences <= 0:
            raise LiveEvidenceError("issue occurrences must be positive")
        if self.registry_version < 0:
            raise LiveEvidenceError("issue registry_version cannot be negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "summary": self.summary,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "occurrences": self.occurrences,
            "registry_version": self.registry_version,
        }

    @classmethod
    def from_payload(cls, payload: object) -> EvidenceIssue:
        value = _object_mapping(payload, "issue")
        return cls(
            signature=str(value["signature"]),
            summary=str(value.get("summary", "")),
            first_seen_at=float(str(value["first_seen_at"])),
            last_seen_at=float(str(value["last_seen_at"])),
            occurrences=int(str(value["occurrences"])),
            registry_version=int(str(value.get("registry_version", 0))),
        )


@dataclass(frozen=True)
class EvidenceRound:
    round_id: str
    started_at: float
    updated_at: float
    issues: tuple[EvidenceIssue, ...] = ()

    def __post_init__(self) -> None:
        if not self.round_id.strip():
            raise LiveEvidenceError("round_id cannot be empty")
        _validate_timestamp(self.started_at, "round started_at")
        _validate_timestamp(self.updated_at, "round updated_at")
        if self.updated_at < self.started_at:
            raise LiveEvidenceError("round updated_at cannot precede started_at")
        signatures = [issue.signature for issue in self.issues]
        if len(signatures) != len(set(signatures)):
            raise LiveEvidenceError("round issue signatures must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "round_id": self.round_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "issues": [issue.to_payload() for issue in self.issues],
        }

    @classmethod
    def from_payload(cls, payload: object) -> EvidenceRound:
        value = _object_mapping(payload, "round")
        raw_issues = value.get("issues", [])
        if not isinstance(raw_issues, list):
            raise LiveEvidenceError("round issues must be a list")
        return cls(
            round_id=str(value["round_id"]),
            started_at=float(str(value["started_at"])),
            updated_at=float(str(value["updated_at"])),
            issues=tuple(EvidenceIssue.from_payload(item) for item in raw_issues),
        )


@dataclass(frozen=True)
class EvidenceSession:
    session_id: str
    runtime_version: str
    registry_version: int
    status: EvidenceSessionStatus
    started_at: float
    updated_at: float
    closed_at: float | None = None
    pinned: bool = False
    owner_pid: int | None = None
    registry_trusted: bool = True
    rounds: tuple[EvidenceRound, ...] = ()
    sealed_event_count: int = 0

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)
        if not self.runtime_version.strip():
            raise LiveEvidenceError("runtime_version cannot be empty")
        if self.registry_version < 0:
            raise LiveEvidenceError("registry_version cannot be negative")
        if self.owner_pid is not None and self.owner_pid <= 0:
            raise LiveEvidenceError("owner_pid must be positive")
        if type(self.sealed_event_count) is not int or self.sealed_event_count < 0:
            raise LiveEvidenceError("sealed_event_count must be a nonnegative integer")
        _validate_timestamp(self.started_at, "session started_at")
        _validate_timestamp(self.updated_at, "session updated_at")
        if self.updated_at < self.started_at:
            raise LiveEvidenceError("session updated_at cannot precede started_at")
        if self.status is EvidenceSessionStatus.ACTIVE and self.closed_at is not None:
            raise LiveEvidenceError("active session cannot have closed_at")
        if self.status is EvidenceSessionStatus.CLOSED:
            if self.closed_at is None:
                raise LiveEvidenceError("closed session requires closed_at")
            _validate_timestamp(self.closed_at, "session closed_at")
            if self.closed_at < self.started_at:
                raise LiveEvidenceError("session closed_at cannot precede started_at")
        round_ids = [round_.round_id for round_ in self.rounds]
        if len(round_ids) != len(set(round_ids)):
            raise LiveEvidenceError("session round_ids must be unique")
        if any(round_.started_at < self.started_at for round_ in self.rounds):
            raise LiveEvidenceError("round cannot precede its evidence session")
        if any(round_.updated_at > self.updated_at for round_ in self.rounds):
            raise LiveEvidenceError("round cannot be newer than its evidence session")
        if self.closed_at is not None and any(
            round_.updated_at > self.closed_at for round_ in self.rounds
        ):
            raise LiveEvidenceError("session cannot close before its latest round")

    @property
    def issue_versions(self) -> dict[str, int]:
        versions: dict[str, int] = {}
        for round_ in self.rounds:
            for issue in round_.issues:
                versions[issue.signature] = max(
                    issue.registry_version,
                    versions.get(issue.signature, -1),
                )
        return versions

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": EVIDENCE_SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "runtime_version": self.runtime_version,
            "registry_version": self.registry_version,
            "status": self.status.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "pinned": self.pinned,
            "owner_pid": self.owner_pid,
            "registry_trusted": self.registry_trusted,
            "sealed_event_count": self.sealed_event_count,
            "rounds": [round_.to_payload() for round_ in self.rounds],
        }

    @classmethod
    def from_payload(cls, payload: object) -> EvidenceSession:
        value = _object_mapping(payload, "evidence session")
        if value.get("schema_version") != EVIDENCE_SESSION_SCHEMA_VERSION:
            raise LiveEvidenceError("unsupported evidence session schema_version")
        raw_rounds = value.get("rounds", [])
        if not isinstance(raw_rounds, list):
            raise LiveEvidenceError("session rounds must be a list")
        raw_closed_at = value.get("closed_at")
        return cls(
            session_id=str(value["session_id"]),
            runtime_version=str(value["runtime_version"]),
            registry_version=int(str(value.get("registry_version", 0))),
            status=EvidenceSessionStatus(str(value["status"])),
            started_at=float(str(value["started_at"])),
            updated_at=float(str(value["updated_at"])),
            closed_at=(float(str(raw_closed_at)) if raw_closed_at is not None else None),
            pinned=_mapping_bool(value, "pinned", False),
            owner_pid=(
                int(str(value["owner_pid"])) if value.get("owner_pid") is not None else None
            ),
            registry_trusted=_mapping_bool(value, "registry_trusted", True),
            rounds=tuple(EvidenceRound.from_payload(item) for item in raw_rounds),
            sealed_event_count=int(str(value.get("sealed_event_count", 0))),
        )


@dataclass(frozen=True)
class FixedIssue:
    signature: str
    fixed_in_registry_version: int
    fixed_in_runtime_version: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        _validate_signature(self.signature)
        if self.fixed_in_registry_version <= 0:
            raise LiveEvidenceError("fixed_in_registry_version must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "fixed_in_registry_version": self.fixed_in_registry_version,
            "fixed_in_runtime_version": self.fixed_in_runtime_version,
            "note": self.note,
        }

    @classmethod
    def from_payload(cls, payload: object) -> FixedIssue:
        value = _object_mapping(payload, "fixed issue")
        return cls(
            signature=str(value["signature"]),
            fixed_in_registry_version=int(str(value["fixed_in_registry_version"])),
            fixed_in_runtime_version=str(value.get("fixed_in_runtime_version", "")),
            note=str(value.get("note", "")),
        )


@dataclass(frozen=True)
class FixedIssueRegistry:
    """Versioned declarations of issue signatures fixed by the current code.

    A fix registered at version N resolves only occurrences captured with a
    registry version below N.  Seeing the same signature at N or later is a
    regression and therefore remains unresolved and protected.
    """

    version: int
    issues: tuple[FixedIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.version < 0:
            raise LiveEvidenceError("fixed issue registry version cannot be negative")
        signatures = [issue.signature for issue in self.issues]
        if len(signatures) != len(set(signatures)):
            raise LiveEvidenceError("fixed issue signatures must be unique")
        if any(issue.fixed_in_registry_version > self.version for issue in self.issues):
            raise LiveEvidenceError("fixed issue cannot refer to a future registry version")

    @classmethod
    def from_mapping(
        cls,
        version: int,
        fixed_versions: Mapping[str, int],
    ) -> FixedIssueRegistry:
        return cls(
            version=version,
            issues=tuple(
                FixedIssue(signature, fixed_version)
                for signature, fixed_version in sorted(fixed_versions.items())
            ),
        )

    def resolves(
        self,
        signature: str,
        observed_registry_version: int,
        *,
        supported_runtime_versions: frozenset[str] | None = None,
    ) -> bool:
        return (
            self.resolving_issue(
                signature,
                observed_registry_version,
                supported_runtime_versions=supported_runtime_versions,
            )
            is not None
        )

    def resolving_issue(
        self,
        signature: str,
        observed_registry_version: int,
        *,
        supported_runtime_versions: frozenset[str] | None = None,
    ) -> FixedIssue | None:
        for issue in self.issues:
            if (
                issue.signature == signature
                and observed_registry_version < issue.fixed_in_registry_version
                and (
                    supported_runtime_versions is None
                    or issue.fixed_in_runtime_version in supported_runtime_versions
                )
            ):
                return issue
        return None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": FIXED_ISSUE_REGISTRY_SCHEMA_VERSION,
            "version": self.version,
            "issues": [issue.to_payload() for issue in self.issues],
        }

    @classmethod
    def from_payload(cls, payload: object) -> FixedIssueRegistry:
        value = _object_mapping(payload, "fixed issue registry")
        if value.get("schema_version") != FIXED_ISSUE_REGISTRY_SCHEMA_VERSION:
            raise LiveEvidenceError("unsupported fixed issue registry schema_version")
        raw_issues = value.get("issues", [])
        if not isinstance(raw_issues, list):
            raise LiveEvidenceError("fixed issue registry issues must be a list")
        return cls(
            version=int(str(value["version"])),
            issues=tuple(FixedIssue.from_payload(item) for item in raw_issues),
        )


@dataclass(frozen=True)
class EvidenceRetentionPolicy:
    """Bounds for managed evidence.

    Active and pinned sessions are hard protected.  Recent and latest
    unresolved sessions are protected from routine retention cleanup, but are
    soft protected for capacity purposes so they cannot make the configured
    upper bounds permanently ineffective.
    """

    max_total_bytes: int = 512 * 1024 * 1024
    max_sessions: int = 100
    retention_days: float = 30.0
    protect_recent_unresolved_days: float = 7.0
    delete_fixed_evidence: bool = True

    def __post_init__(self) -> None:
        if self.max_total_bytes < 0:
            raise LiveEvidenceError("max_total_bytes cannot be negative")
        if self.max_sessions < 0:
            raise LiveEvidenceError("max_sessions cannot be negative")
        if self.retention_days < 0:
            raise LiveEvidenceError("retention_days cannot be negative")
        if self.protect_recent_unresolved_days < 0:
            raise LiveEvidenceError("protect_recent_unresolved_days cannot be negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "max_total_bytes": self.max_total_bytes,
            "max_sessions": self.max_sessions,
            "retention_days": self.retention_days,
            "protect_recent_unresolved_days": self.protect_recent_unresolved_days,
            "delete_fixed_evidence": self.delete_fixed_evidence,
        }


@dataclass(frozen=True)
class EvidenceCleanupAction:
    session_id: str
    bytes: int
    reasons: tuple[str, ...]
    performed: bool
    target: str = "session"
    relative_path: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "bytes": self.bytes,
            "reasons": list(self.reasons),
            "performed": self.performed,
            "target": self.target,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class ProtectedEvidenceSession:
    session_id: str
    bytes: int
    reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "bytes": self.bytes,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvidenceCleanupReport:
    evaluated_at: float
    fixed_registry_version: int
    policy: EvidenceRetentionPolicy
    dry_run: bool
    before_session_count: int
    before_total_bytes: int
    projected_after_session_count: int
    projected_after_total_bytes: int
    actual_after_session_count: int
    actual_after_total_bytes: int
    planned_reclaimed_bytes: int
    reclaimed_bytes: int
    projected_limits_satisfied: bool
    limits_satisfied: bool
    actions: tuple[EvidenceCleanupAction, ...]
    protected_sessions: tuple[ProtectedEvidenceSession, ...]
    invalid_entries: tuple[str, ...]
    ignored_entries: tuple[str, ...]
    errors: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "phase6-live-evidence-cleanup-v1",
            "evaluated_at": self.evaluated_at,
            "fixed_registry_version": self.fixed_registry_version,
            "policy": self.policy.to_payload(),
            "dry_run": self.dry_run,
            "before": {
                "session_count": self.before_session_count,
                "total_bytes": self.before_total_bytes,
            },
            "projected_after": {
                "session_count": self.projected_after_session_count,
                "total_bytes": self.projected_after_total_bytes,
                "limits_satisfied": self.projected_limits_satisfied,
            },
            "actual_after": {
                "session_count": self.actual_after_session_count,
                "total_bytes": self.actual_after_total_bytes,
                "limits_satisfied": self.limits_satisfied,
            },
            "planned_reclaimed_bytes": self.planned_reclaimed_bytes,
            "reclaimed_bytes": self.reclaimed_bytes,
            "actions": [action.to_payload() for action in self.actions],
            "protected_sessions": [session.to_payload() for session in self.protected_sessions],
            "invalid_entries": list(self.invalid_entries),
            "ignored_entries": list(self.ignored_entries),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class _ManagedSession:
    path: Path
    metadata: EvidenceSession
    bytes: int


@dataclass(frozen=True)
class _ManagedFixedFrame:
    session_id: str
    path: Path
    relative_path: str
    bytes: int
    fixed_issues: tuple[tuple[str, int], ...]
    stat_identity: tuple[int, int, int, int]
    event_sources: tuple[LiveLogSource, ...]


class LiveEvidenceStore:
    """Own metadata and cleanup for an isolated live-diagnostic evidence root.

    Cleanup only considers direct child directories containing a valid
    ``evidence-session.json`` whose session id matches the directory name.
    Existing logs, screenshots, unknown directories and symlinks are never
    adopted or removed implicitly.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def session_path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self.root / session_id

    def start_session(
        self,
        session_id: str,
        *,
        runtime_version: str,
        registry_version: int,
        started_at: float | None = None,
        pinned: bool = False,
        owner_pid: int | None = None,
        registry_trusted: bool = True,
    ) -> EvidenceSession:
        value = time.time() if started_at is None else started_at
        metadata = EvidenceSession(
            session_id=session_id,
            runtime_version=runtime_version,
            registry_version=registry_version,
            status=EvidenceSessionStatus.ACTIVE,
            started_at=value,
            updated_at=value,
            pinned=pinned,
            owner_pid=owner_pid,
            registry_trusted=registry_trusted,
        )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        directory = self.session_path(session_id)
        directory.mkdir(mode=0o700)
        try:
            self._write(metadata)
        except BaseException:
            directory.rmdir()
            raise
        return metadata

    def load_session(self, session_id: str) -> EvidenceSession:
        path = self.session_path(session_id) / EVIDENCE_SESSION_METADATA
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LiveEvidenceError(f"cannot read evidence metadata: {path}") from exc
        metadata = EvidenceSession.from_payload(payload)
        if metadata.session_id != session_id:
            raise LiveEvidenceError("metadata session_id does not match directory")
        return metadata

    def touch_round(
        self,
        session_id: str,
        round_id: str,
        *,
        observed_at: float | None = None,
    ) -> EvidenceSession:
        value = time.time() if observed_at is None else observed_at
        metadata = self.load_session(session_id)
        _require_active(metadata)
        rounds = list(metadata.rounds)
        for index, round_ in enumerate(rounds):
            if round_.round_id == round_id:
                rounds[index] = replace(round_, updated_at=max(round_.updated_at, value))
                break
        else:
            rounds.append(EvidenceRound(round_id, value, value))
        updated = replace(
            metadata, updated_at=max(metadata.updated_at, value), rounds=tuple(rounds)
        )
        self._write(updated)
        return updated

    def record_issue(
        self,
        session_id: str,
        round_id: str,
        *,
        signature: str,
        summary: str,
        observed_at: float | None = None,
        registry_version: int | None = None,
    ) -> EvidenceSession:
        _validate_signature(signature)
        value = time.time() if observed_at is None else observed_at
        metadata = self.load_session(session_id)
        _require_active(metadata)
        observed_registry_version = (
            metadata.registry_version if registry_version is None else registry_version
        )
        if observed_registry_version < 0:
            raise LiveEvidenceError("registry_version cannot be negative")
        rounds = list(metadata.rounds)
        for round_index, round_ in enumerate(rounds):
            if round_.round_id != round_id:
                continue
            issues = list(round_.issues)
            for issue_index, issue in enumerate(issues):
                if issue.signature == signature:
                    issues[issue_index] = replace(
                        issue,
                        summary=summary or issue.summary,
                        last_seen_at=max(issue.last_seen_at, value),
                        occurrences=issue.occurrences + 1,
                        registry_version=max(
                            issue.registry_version,
                            observed_registry_version,
                        ),
                    )
                    break
            else:
                issues.append(
                    EvidenceIssue(
                        signature=signature,
                        summary=summary,
                        first_seen_at=value,
                        last_seen_at=value,
                        occurrences=1,
                        registry_version=observed_registry_version,
                    )
                )
            rounds[round_index] = replace(
                round_,
                updated_at=max(round_.updated_at, value),
                issues=tuple(issues),
            )
            break
        else:
            rounds.append(
                EvidenceRound(
                    round_id=round_id,
                    started_at=value,
                    updated_at=value,
                    issues=(
                        EvidenceIssue(
                            signature=signature,
                            summary=summary,
                            first_seen_at=value,
                            last_seen_at=value,
                            occurrences=1,
                            registry_version=observed_registry_version,
                        ),
                    ),
                )
            )
        updated = replace(
            metadata, updated_at=max(metadata.updated_at, value), rounds=tuple(rounds)
        )
        self._write(updated)
        return updated

    def close_session(
        self,
        session_id: str,
        *,
        closed_at: float | None = None,
        sealed_event_count: int = 0,
    ) -> EvidenceSession:
        value = time.time() if closed_at is None else closed_at
        metadata = self.load_session(session_id)
        if metadata.status is EvidenceSessionStatus.CLOSED:
            return metadata
        updated = replace(
            metadata,
            status=EvidenceSessionStatus.CLOSED,
            updated_at=max(metadata.updated_at, value),
            closed_at=value,
            sealed_event_count=sealed_event_count,
        )
        self._write(updated)
        return updated

    def set_pinned(self, session_id: str, pinned: bool) -> EvidenceSession:
        metadata = self.load_session(session_id)
        updated = replace(metadata, pinned=pinned)
        self._write(updated)
        return updated

    def recover_abandoned_sessions(
        self,
        *,
        owner_alive: Callable[[int], bool] | None = None,
        now: float | None = None,
        ownerless_stale_seconds: float = 24 * 60 * 60,
    ) -> tuple[str, ...]:
        """Close active sessions whose runtime process can no longer exist.

        Older metadata has no owner PID and is recovered only after a
        conservative age threshold. Metadata is re-read before updating it so
        a concurrently changed session is never overwritten.
        """

        value = time.time() if now is None else now
        _validate_timestamp(value, "recovery now")
        if ownerless_stale_seconds < 0:
            raise LiveEvidenceError("ownerless_stale_seconds cannot be negative")
        is_alive = _owner_process_is_alive if owner_alive is None else owner_alive
        managed, _invalid, _ignored = self._inventory()
        recovered: list[str] = []
        for item in managed:
            metadata = item.metadata
            if metadata.status is not EvidenceSessionStatus.ACTIVE:
                continue
            abandoned = (
                not is_alive(metadata.owner_pid)
                if metadata.owner_pid is not None
                else value - metadata.updated_at >= ownerless_stale_seconds
            )
            if not abandoned:
                continue
            current = self.load_session(metadata.session_id)
            if current != metadata or current.status is not EvidenceSessionStatus.ACTIVE:
                continue
            self.close_session(
                metadata.session_id,
                closed_at=max(value, metadata.updated_at),
            )
            recovered.append(metadata.session_id)
        return tuple(recovered)

    def cleanup(
        self,
        policy: EvidenceRetentionPolicy,
        *,
        fixed_registry: FixedIssueRegistry | None = None,
        supported_runtime_versions: frozenset[str] | None = None,
        now: float | None = None,
        dry_run: bool = True,
    ) -> EvidenceCleanupReport:
        """Plan or perform bounded cleanup and return a complete audit report."""

        value = time.time() if now is None else now
        _validate_timestamp(value, "cleanup now")
        registry = fixed_registry or FixedIssueRegistry(version=0)
        managed, invalid, ignored = self._inventory()
        before_count = len(managed)
        before_bytes = sum(item.bytes for item in managed)
        unresolved = {
            item.metadata.session_id: _unresolved_signatures(
                item.metadata,
                registry,
                supported_runtime_versions=supported_runtime_versions,
            )
            for item in managed
        }
        newest_unresolved: dict[str, str] = {}
        for item in sorted(
            managed,
            key=lambda candidate: (
                candidate.metadata.updated_at,
                candidate.metadata.session_id,
            ),
        ):
            for signature in unresolved[item.metadata.session_id]:
                newest_unresolved[signature] = item.metadata.session_id

        hard_protected: dict[str, list[str]] = {}
        soft_protected: dict[str, list[str]] = {}
        grace_seconds = policy.protect_recent_unresolved_days * 86400.0
        for item in managed:
            metadata = item.metadata
            hard_reasons: list[str] = []
            soft_reasons: list[str] = []
            if metadata.status is EvidenceSessionStatus.ACTIVE:
                hard_reasons.append("active_session")
            if metadata.pinned:
                hard_reasons.append("pinned")
            outstanding = unresolved[metadata.session_id]
            if outstanding and value - metadata.updated_at <= grace_seconds:
                soft_reasons.append("recent_unresolved")
            newest = sorted(
                signature
                for signature in outstanding
                if newest_unresolved.get(signature) == metadata.session_id
            )
            if newest:
                soft_reasons.extend(f"latest_unresolved:{signature}" for signature in newest)
            if hard_reasons:
                hard_protected[metadata.session_id] = hard_reasons
            if soft_reasons:
                soft_protected[metadata.session_id] = soft_reasons

        selected_sessions: dict[str, list[str]] = {}
        by_id = {item.metadata.session_id: item for item in managed}
        closed_nonhard = [
            item
            for item in managed
            if item.metadata.session_id not in hard_protected
            and item.metadata.status is EvidenceSessionStatus.CLOSED
        ]
        fixed_frames: list[_ManagedFixedFrame] = []
        errors: list[str] = []
        if policy.delete_fixed_evidence:
            for item in closed_nonhard:
                discovered, discovery_errors = _fixed_frame_candidates(
                    item,
                    registry,
                    supported_runtime_versions=supported_runtime_versions,
                )
                fixed_frames.extend(discovered)
                errors.extend(discovery_errors)

        retention_seconds = policy.retention_days * 86400.0
        for item in closed_nonhard:
            session_id = item.metadata.session_id
            if session_id in soft_protected:
                continue
            if value - item.metadata.updated_at >= retention_seconds:
                selected_sessions[session_id] = ["retention_expired"]

        fixed_bytes_by_session: dict[str, int] = {}
        for frame in fixed_frames:
            fixed_bytes_by_session[frame.session_id] = (
                fixed_bytes_by_session.get(frame.session_id, 0) + frame.bytes
            )
        projected_count = before_count - len(selected_sessions)
        projected_bytes = before_bytes - sum(
            by_id[key].bytes for key in selected_sessions
        )
        projected_bytes -= sum(
            frame.bytes
            for frame in fixed_frames
            if frame.session_id not in selected_sessions
        )

        unprotected_capacity = sorted(
            (
                item
                for item in closed_nonhard
                if item.metadata.session_id not in selected_sessions
                and item.metadata.session_id not in soft_protected
            ),
            key=lambda item: (item.metadata.updated_at, item.metadata.session_id),
        )
        soft_capacity = sorted(
            (
                item
                for item in closed_nonhard
                if item.metadata.session_id not in selected_sessions
                and item.metadata.session_id in soft_protected
            ),
            key=lambda item: (item.metadata.updated_at, item.metadata.session_id),
        )
        for item in (*unprotected_capacity, *soft_capacity):
            if projected_count <= policy.max_sessions and projected_bytes <= policy.max_total_bytes:
                break
            session_id = item.metadata.session_id
            reasons = ["capacity_limit"]
            reasons.extend(
                f"soft_protection_overridden:{reason}"
                for reason in soft_protected.get(session_id, ())
            )
            selected_sessions[session_id] = reasons
            projected_count -= 1
            projected_bytes -= item.bytes - fixed_bytes_by_session.get(session_id, 0)

        planned_fixed_frames = [
            frame
            for frame in fixed_frames
            if frame.session_id not in selected_sessions
        ]
        ordered_selected_sessions = sorted(
            selected_sessions,
            key=lambda session_id: (
                by_id[session_id].metadata.updated_at,
                session_id,
            ),
        )
        ordered_fixed_frames = sorted(
            planned_fixed_frames,
            key=lambda frame: (
                by_id[frame.session_id].metadata.updated_at,
                frame.session_id,
                frame.relative_path,
            ),
        )
        actions: list[EvidenceCleanupAction] = []
        reclaimed = 0
        for frame in ordered_fixed_frames:
            performed = False
            if not dry_run:
                try:
                    reclaimed_now = _remove_managed_fixed_frame(
                        self.root,
                        by_id[frame.session_id],
                        frame,
                    )
                except OSError as exc:
                    errors.append(f"{frame.session_id}/{frame.relative_path}: {exc}")
                else:
                    performed = True
                    reclaimed += reclaimed_now
            actions.append(
                EvidenceCleanupAction(
                    session_id=frame.session_id,
                    bytes=frame.bytes,
                    reasons=tuple(
                        f"issue_fixed:{signature}@registry_v{fixed_version}"
                        for signature, fixed_version in frame.fixed_issues
                    ),
                    performed=performed,
                    target="error_frame",
                    relative_path=frame.relative_path,
                )
            )

        for session_id in ordered_selected_sessions:
            item = by_id[session_id]
            performed = False
            if not dry_run:
                try:
                    reclaimed_now = _remove_managed_session(self.root, item)
                except OSError as exc:
                    errors.append(f"{session_id}: {exc}")
                else:
                    performed = True
                    reclaimed += reclaimed_now
            actions.append(
                EvidenceCleanupAction(
                    session_id=session_id,
                    bytes=item.bytes,
                    reasons=tuple(selected_sessions[session_id]),
                    performed=performed,
                )
            )

        if dry_run:
            actual_count = before_count
            actual_bytes = before_bytes
        else:
            actual_managed, _actual_invalid, _actual_ignored = self._inventory()
            actual_count = len(actual_managed)
            actual_bytes = sum(item.bytes for item in actual_managed)
        remaining_protected = {
            session_id: (
                *hard_protected.get(session_id, ()),
                *soft_protected.get(session_id, ()),
            )
            for session_id in set(hard_protected) | set(soft_protected)
            if session_id not in selected_sessions
        }
        planned_reclaimed = sum(
            by_id[session_id].bytes for session_id in selected_sessions
        ) + sum(frame.bytes for frame in planned_fixed_frames)
        return EvidenceCleanupReport(
            evaluated_at=value,
            fixed_registry_version=registry.version,
            policy=policy,
            dry_run=dry_run,
            before_session_count=before_count,
            before_total_bytes=before_bytes,
            projected_after_session_count=projected_count,
            projected_after_total_bytes=projected_bytes,
            actual_after_session_count=actual_count,
            actual_after_total_bytes=actual_bytes,
            planned_reclaimed_bytes=planned_reclaimed,
            reclaimed_bytes=reclaimed,
            projected_limits_satisfied=(
                projected_count <= policy.max_sessions and projected_bytes <= policy.max_total_bytes
            ),
            limits_satisfied=(
                actual_count <= policy.max_sessions and actual_bytes <= policy.max_total_bytes
            ),
            actions=tuple(actions),
            protected_sessions=tuple(
                ProtectedEvidenceSession(
                    session_id=session_id,
                    bytes=by_id[session_id].bytes,
                    reasons=tuple(reasons),
                )
                for session_id, reasons in sorted(remaining_protected.items())
            ),
            invalid_entries=tuple(invalid),
            ignored_entries=tuple(ignored),
            errors=tuple(errors),
        )

    def _write(self, metadata: EvidenceSession) -> None:
        path = self.session_path(metadata.session_id) / EVIDENCE_SESSION_METADATA
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metadata.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        path.chmod(0o600)

    def _inventory(
        self,
    ) -> tuple[list[_ManagedSession], list[str], list[str]]:
        if not self.root.exists():
            return [], [], []
        managed: list[_ManagedSession] = []
        invalid: list[str] = []
        ignored: list[str] = []
        for path in sorted(self.root.iterdir()):
            if path.is_symlink() or not path.is_dir():
                ignored.append(path.name)
                continue
            metadata_path = path / EVIDENCE_SESSION_METADATA
            if metadata_path.is_symlink() or not metadata_path.is_file():
                ignored.append(path.name)
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = EvidenceSession.from_payload(payload)
                if metadata.session_id != path.name:
                    raise LiveEvidenceError("metadata session_id does not match directory")
                size = _directory_size(path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                invalid.append(f"{path.name}: {exc}")
                continue
            managed.append(_ManagedSession(path, metadata, size))
        return managed, invalid, ignored


def load_fixed_issue_registry(path: Path) -> FixedIssueRegistry:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveEvidenceError(f"cannot read fixed issue registry: {path}") from exc
    return FixedIssueRegistry.from_payload(payload)


def save_fixed_issue_registry(path: Path, registry: FixedIssueRegistry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(registry.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unresolved_signatures(
    metadata: EvidenceSession,
    registry: FixedIssueRegistry,
    *,
    supported_runtime_versions: frozenset[str] | None = None,
) -> set[str]:
    if not metadata.registry_trusted:
        return set(metadata.issue_versions)
    return {
        signature
        for signature, observed_version in metadata.issue_versions.items()
        if not registry.resolves(
            signature,
            observed_version,
            supported_runtime_versions=supported_runtime_versions,
        )
    }


def _fixed_frame_candidates(
    item: _ManagedSession,
    registry: FixedIssueRegistry,
    *,
    supported_runtime_versions: frozenset[str] | None,
) -> tuple[list[_ManagedFixedFrame], list[str]]:
    """Find only registry-resolved screenshots explicitly named by the log."""

    metadata = item.metadata
    if not metadata.registry_trusted:
        return [], []
    resolved: dict[str, FixedIssue] = {}
    for signature, observed_version in metadata.issue_versions.items():
        fixed_issue = registry.resolving_issue(
            signature,
            observed_version,
            supported_runtime_versions=supported_runtime_versions,
        )
        if fixed_issue is not None:
            resolved[signature] = fixed_issue
    if not resolved:
        return [], []
    if metadata.sealed_event_count <= 0:
        return [], [f"{metadata.session_id}: legacy or unsealed event history; keeping frames"]

    events_path = item.path / "events.jsonl"
    try:
        event_sources = list_live_log_sources(events_path)
    except (OSError, ValueError) as exc:
        return [], [f"{metadata.session_id}/events.jsonl: {exc}"]

    references: dict[Path, set[str | None]] = {}
    errors: list[str] = []
    event_count = 0
    last_event: object = None
    try:
        for record in iter_live_log_records(event_sources):
            payload = record.payload
            event = payload.get("event")
            event_count += 1
            sequence = payload.get("scan_event_index")
            if type(sequence) is not int or sequence != event_count:
                errors.append(f"{record.source}:{record.line_number}: event sequence is incomplete")
            if event_count == 1 and event != "scan_session_started":
                errors.append(f"{record.source}: missing scan_session_started")
            last_event = event
            if event == "live_log_history_pruned":
                errors.append(
                    f"{record.source}:{record.line_number}: historical frame references were pruned"
                )
                continue
            if event not in {
                "error_frame_saved", "diagnostic_issue_recorded",
                "diagnostic_frame_saved", "diagnostic_frame_promoted",
            }:
                continue
            if event in {"diagnostic_frame_saved", "diagnostic_frame_promoted"}:
                if payload.get("kind") == "uncertain":
                    continue
            raw_path = payload.get("path")
            if event == "diagnostic_issue_recorded" and raw_path is None:
                continue
            location = f"{record.source}:{record.line_number}"
            if not isinstance(raw_path, str):
                errors.append(f"{location}: error frame path must be a string")
                continue
            candidate = _resolve_managed_frame_path(item, raw_path)
            if candidate is None:
                errors.append(f"{location}: error frame path is outside the managed frames directory")
                continue
            raw_signature = payload.get("issue_signature")
            event_signature = (
                raw_signature
                if event in {"error_frame_saved", "diagnostic_issue_recorded"}
                and isinstance(raw_signature, str)
                else None
            )
            references.setdefault(candidate, set()).add(event_signature)
        if list_live_log_sources(events_path) != event_sources:
            raise ValueError("event log slices changed during inspection")
        if event_count != metadata.sealed_event_count or last_event != "scan_session_stopped":
            errors.append(f"{metadata.session_id}: event history does not match its close seal")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    if errors:
        # A malformed/missing reference may describe an unresolved issue on
        # the same image. Never remove images from an incomplete inspection.
        return [], errors

    candidates: list[_ManagedFixedFrame] = []
    for path, signatures in sorted(references.items(), key=lambda item_: item_[0].as_posix()):
        resolved_signatures = tuple(
            sorted(signature for signature in signatures if signature is not None)
        )
        if (
            len(resolved_signatures) != len(signatures)
            or not resolved_signatures
            or any(signature not in resolved for signature in resolved_signatures)
        ):
            continue
        try:
            if path.is_symlink() or not path.is_file():
                continue
            identity = _stat_identity(path)
            relative_path = path.relative_to(item.path.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        candidates.append(
            _ManagedFixedFrame(
                session_id=metadata.session_id,
                path=path,
                relative_path=relative_path,
                bytes=identity[2],
                fixed_issues=tuple(
                    (
                        signature,
                        resolved[signature].fixed_in_registry_version,
                    )
                    for signature in resolved_signatures
                ),
                stat_identity=identity,
                event_sources=event_sources,
            )
        )
    return candidates, errors


def _resolve_managed_frame_path(item: _ManagedSession, raw_path: str) -> Path | None:
    if not raw_path.strip():
        return None
    frames_path = item.path / "frames"
    if frames_path.is_symlink() or not frames_path.is_dir():
        return None
    frames_resolved = frames_path.resolve()
    supplied = Path(raw_path)
    if ".." in supplied.parts:
        return None
    attempts = (supplied,) if supplied.is_absolute() else (supplied, item.path / supplied)
    for candidate in attempts:
        try:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        if (
            resolved.parent == frames_resolved
            and resolved.suffix.lower() == ".png"
        ):
            return resolved
    return None


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise OSError("managed evidence artifact is not a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _remove_managed_fixed_frame(
    root: Path,
    item: _ManagedSession,
    frame: _ManagedFixedFrame,
) -> int:
    root_resolved = root.resolve()
    session_path = item.path
    if session_path.is_symlink() or session_path.resolve().parent != root_resolved:
        raise OSError("refusing to remove evidence outside the managed root")
    metadata_path = session_path / EVIDENCE_SESSION_METADATA
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = EvidenceSession.from_payload(payload)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise OSError("refusing to remove frame from unreadable evidence session") from exc
    if metadata != item.metadata or metadata.status is EvidenceSessionStatus.ACTIVE:
        raise OSError("refusing to remove frame from changed or active evidence session")
    events_path = session_path / "events.jsonl"
    try:
        unchanged_sources = list_live_log_sources(events_path) == frame.event_sources
    except ValueError as exc:
        raise OSError("refusing to remove frame whose event log became unsafe") from exc
    if not unchanged_sources:
        raise OSError("refusing to remove frame whose event log changed")
    frames_path = session_path / "frames"
    if (
        frames_path.is_symlink()
        or frame.path.parent != frames_path.resolve()
        or frame.path.is_symlink()
        or _stat_identity(frame.path) != frame.stat_identity
    ):
        raise OSError("refusing to remove changed or unmanaged error frame")
    frame.path.unlink()
    return frame.bytes


def _remove_managed_session(root: Path, item: _ManagedSession) -> int:
    root_resolved = root.resolve()
    path = item.path
    if path.is_symlink() or path.resolve().parent != root_resolved:
        raise OSError("refusing to remove evidence outside the managed root")
    metadata_path = path / EVIDENCE_SESSION_METADATA
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = EvidenceSession.from_payload(payload)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise OSError("refusing to remove unreadable evidence session") from exc
    if metadata != item.metadata:
        raise OSError("refusing to remove evidence whose metadata changed")
    if metadata.session_id != path.name or metadata.status is EvidenceSessionStatus.ACTIVE:
        raise OSError("refusing to remove changed or active evidence session")
    removed_bytes = _directory_size(path)
    shutil.rmtree(path)
    return removed_bytes


def _directory_size(path: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        base = Path(directory)
        directory_names[:] = [name for name in directory_names if not (base / name).is_symlink()]
        for name in file_names:
            candidate = base / name
            try:
                total += candidate.lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _validate_session_id(session_id: str) -> None:
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise LiveEvidenceError(f"unsafe evidence session_id: {session_id!r}")


def _validate_signature(signature: str) -> None:
    if not _SAFE_SIGNATURE.fullmatch(signature):
        raise LiveEvidenceError(f"unsafe issue signature: {signature!r}")


def _validate_timestamp(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
        raise LiveEvidenceError(f"{name} must be a non-negative timestamp")


def _object_mapping(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise LiveEvidenceError(f"{name} must be a JSON object")
    return payload


def _mapping_bool(
    payload: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise LiveEvidenceError(f"{key} must be a boolean")
    return value


def _require_active(metadata: EvidenceSession) -> None:
    if metadata.status is not EvidenceSessionStatus.ACTIVE:
        raise LiveEvidenceError("cannot update a closed evidence session")


def _owner_process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = [
    "EVIDENCE_SESSION_METADATA",
    "EVIDENCE_SESSION_SCHEMA_VERSION",
    "FIXED_ISSUE_REGISTRY_SCHEMA_VERSION",
    "EvidenceCleanupAction",
    "EvidenceCleanupReport",
    "EvidenceIssue",
    "EvidenceRetentionPolicy",
    "EvidenceRound",
    "EvidenceSession",
    "EvidenceSessionStatus",
    "FixedIssue",
    "FixedIssueRegistry",
    "LiveEvidenceError",
    "LiveEvidenceStore",
    "ProtectedEvidenceSession",
    "load_fixed_issue_registry",
    "save_fixed_issue_registry",
    "stable_issue_signature",
]
