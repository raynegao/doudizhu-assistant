from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reporting.live_diagnostics import analyze_live_log
from src.reporting.live_evidence import EVIDENCE_SESSION_METADATA, LiveEvidenceStore
from src.reporting.live_log_reader import list_live_log_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 6 runtime reliability without loading the log at once."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/live_assistant.jsonl"),
    )
    parser.add_argument(
        "--latest-evidence",
        action="store_true",
        help=(
            "Analyze the latest user-authorized scan session. "
            "Falls back to --log when no managed session exists."
        ),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("data/live_game/evidence"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/live-diagnostics/report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = (
        _latest_evidence_log(args.evidence_root) or args.log if args.latest_evidence else args.log
    )
    report = analyze_live_log(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    return 0


def _latest_evidence_log(root: Path) -> Path | None:
    """Resolve the latest managed session without trusting paths from JSON."""

    pointer = root / "latest-session.json"
    if pointer.is_symlink():
        raise ValueError(f"latest evidence pointer cannot be a symlink: {pointer}")
    if not pointer.is_file():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{pointer} must contain a JSON object")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str):
        raise ValueError(f"{pointer} is missing session_id")
    store = LiveEvidenceStore(root)
    session = store.session_path(session_id)
    if not session.exists():
        return None
    if session.is_symlink() or not session.is_dir():
        raise ValueError(f"latest evidence session is not a plain directory: {session}")
    root_resolved = root.resolve()
    session_resolved = session.resolve()
    if session_resolved.parent != root_resolved:
        raise ValueError(f"latest evidence session is outside evidence root: {session}")
    metadata = session / EVIDENCE_SESSION_METADATA
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError(f"latest evidence metadata is not a plain file: {metadata}")
    if metadata.resolve().parent != session_resolved:
        raise ValueError(f"latest evidence metadata is outside its session: {metadata}")
    store.load_session(session_id)
    events = session / "events.jsonl"
    if events.is_symlink() or (events.exists() and not events.is_file()):
        raise ValueError(f"latest evidence log is not a plain file: {events}")
    if events.resolve().parent != session_resolved:
        raise ValueError(f"latest evidence log is outside its session: {events}")
    if not list_live_log_sources(events):
        raise ValueError(f"latest evidence session has no retained log slices: {events}")
    return events


if __name__ == "__main__":
    raise SystemExit(main())
