from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inventory_live_storage import main
from src.reporting.live_storage import inventory_live_storage


def test_inventory_classifies_sources_without_modifying_them(tmp_path: Path, capsys) -> None:
    original: dict[Path, bytes] = {}
    for name, split in (("old", None), ("debug", "development"), ("formal", "acceptance")):
        session = tmp_path / "recordings" / name
        session.mkdir(parents=True)
        metadata = session / "session.json"
        metadata.write_text(json.dumps({"evidence_split": split}), encoding="utf-8")
        original[metadata] = metadata.read_bytes()
    errors = tmp_path / "errors"
    errors.mkdir()
    error = errors / "legacy.png"
    error.write_bytes(b"original evidence")
    original[error] = error.read_bytes()
    assert main(["--root", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["read_only"] is True
    assert report["total_bytes"] == sum(len(content) for content in original.values())
    rows = {Path(row["path"]).name: row for row in report["entries"]}
    assert rows["formal"]["retention"] == "retain_acceptance_source_and_seals"
    assert "provenance" in rows["old"]["retention"]
    assert all(not row["automatic_deletion_allowed"] for row in rows.values())
    assert {path: path.read_bytes() for path in original} == original


def test_inventory_does_not_follow_symlink_or_trust_bad_metadata(tmp_path: Path) -> None:
    live = tmp_path / "live"
    session = live / "recordings" / "bad"
    session.mkdir(parents=True)
    (session / "session.json").write_text("not JSON", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private").write_bytes(b"not in scope")
    (live / "errors").symlink_to(outside, target_is_directory=True)
    report = inventory_live_storage(live)
    assert report["total_bytes"] == len(b"not JSON")
    assert len(report["warnings"]) == 2
    alias = tmp_path / "alias"
    alias.symlink_to(live, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        inventory_live_storage(alias)
