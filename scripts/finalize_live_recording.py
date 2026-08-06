from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.record_live_game import _validate_session_name
from src.capture.recording_integrity import finalize_recording_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an existing Phase 6 recording and mark it complete."
    )
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path("data/live_game/recordings"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        session_name = _validate_session_name(args.session)
        inspection = finalize_recording_session(args.recordings_root / session_name)
    except ValueError as exc:
        raise SystemExit(f"cannot finalize recording: {exc}") from exc
    payload = {
        "session": session_name,
        "recorded_frames": len(inspection.manifest_rows),
        "complete_game": inspection.metadata.get("complete_game"),
        "integrity_valid": inspection.valid,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"session_metadata: {inspection.session_dir / 'session.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
