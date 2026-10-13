"""Read-only inventory and retention classification for legacy live evidence.

Legacy files have no trustworthy deletion manifest. They are never adopted by
the managed-session cleaner, and this module intentionally has no delete API.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TypedDict


class _StorageEntry(TypedDict):
    path: str
    category: str
    evidence_split: str | None
    bytes: int
    file_count: int
    retention: str
    automatic_deletion_allowed: bool


def _size_inventory(root: Path) -> tuple[int, int, list[str]]:
    size = 0
    count = 0
    warnings: list[str] = []
    pending = [root]
    while pending:
        path = pending.pop()
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                warnings.append(f"symlink_skipped:{path}")
            elif stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
            elif stat.S_ISREG(info.st_mode):
                size += info.st_size
                count += 1
        except OSError as exc:
            warnings.append(f"unreadable:{path}:{type(exc).__name__}")
    return size, count, warnings


def inventory_live_storage(root: Path) -> dict[str, object]:
    if root.is_symlink():
        raise ValueError("storage root must not be a symlink")
    if not root.is_dir():
        raise ValueError(f"live storage directory does not exist: {root}")
    entries: list[_StorageEntry] = []
    warnings: list[str] = []
    for category in ("recordings", "errors", "evidence"):
        directory = root / category
        if directory.is_symlink():
            warnings.append(f"symlink_skipped:{directory}")
            continue
        if not directory.exists():
            continue
        targets = sorted(directory.iterdir()) if category != "errors" else [directory]
        for target in targets:
            size, count, issues = _size_inventory(target)
            warnings.extend(issues)
            split = "unclassified"
            if category == "recordings" and target.is_dir() and not target.is_symlink():
                metadata = target / "session.json"
                try:
                    if metadata.is_symlink() or metadata.stat().st_size > 1024 * 1024:
                        raise ValueError("unsafe or oversized session metadata")
                    payload = json.loads(metadata.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("session metadata must be an object")
                    split = str(payload.get("evidence_split") or "unclassified")
                except (OSError, ValueError) as exc:
                    warnings.append(f"metadata_unverified:{metadata}:{type(exc).__name__}")
            retention = {
                "errors": "retain_legacy_errors_pending_manual_review",
                "evidence": "managed_policy_only_do_not_manually_delete",
            }.get(category, {
                "acceptance": "retain_acceptance_source_and_seals",
                "development": "review_development_after_fix_verification",
            }.get(split, "retain_unclassified_until_provenance_confirmed"))
            entries.append({
                "path": target.as_posix(),
                "category": category,
                "evidence_split": split if category == "recordings" else None,
                "bytes": size,
                "file_count": count,
                "retention": retention,
                "automatic_deletion_allowed": False,
            })
    return {
        "schema_version": "phase6-live-storage-inventory-v1",
        "read_only": True,
        "root": root.as_posix(),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "entries": sorted(entries, key=lambda entry: entry["bytes"], reverse=True),
        "warnings": warnings,
        "policy": [
            "Do not relabel development or unclassified recordings as acceptance.",
            "Retain original acceptance recordings, annotations and integrity seals together.",
            "Legacy error PNGs are not eligible for managed-session automatic cleanup.",
            "Review old development sessions against fixed issues before explicitly archiving them.",
            "Active or pinned managed sessions remain protected by the managed retention policy.",
        ],
    }
