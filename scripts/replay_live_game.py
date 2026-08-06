from __future__ import annotations

import argparse
import time
from pathlib import Path

from scripts.annotate_live_session import validate_sealed_annotation
from src.capture.recorded_window import RecordedWindowFrameSource
from src.capture.recording_integrity import (
    REPLAY_SCHEMA_VERSION,
    inspect_recording_session,
    runtime_versions,
    sha256_directory,
    sha256_file,
    sha256_json_payload,
    sha256_python_implementation,
    write_json_atomic,
)
from src.config.live_layout import live_layout_from_dict, load_live_layout
from src.pipeline.live_runtime import LiveGameRuntime, format_live_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one recorded Phase 6 full-window session."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Defaults to the immutable config snapshot stored in the session.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/live-replay"),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing replay events/provenance output for this session.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspection = inspect_recording_session(args.manifest.parent)
    if not inspection.valid:
        raise SystemExit(
            "recording integrity validation failed: " + "; ".join(inspection.issues)
        )
    config_snapshot = inspection.session_dir / str(
        inspection.metadata["config_snapshot"]
    )
    config_path = args.config or config_snapshot
    config_sha256 = sha256_file(config_path)
    if config_sha256 != inspection.metadata.get("config_sha256"):
        raise SystemExit("replay config does not match the recording config snapshot")
    annotation_path = inspection.session_dir / "annotation.json"
    if inspection.metadata.get("evidence_split") == "acceptance":
        try:
            validate_sealed_annotation(inspection.session_dir)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                "acceptance replay requires a valid blind annotation sealed "
                f"before replay: {exc}"
            ) from exc
    base = load_live_layout(config_path)
    payload = base.to_payload()
    events_path = args.output_dir / "events.jsonl"
    provenance_path = args.output_dir / "replay.json"
    for output in (events_path, provenance_path):
        if output.exists() and not args.overwrite:
            raise SystemExit(
                f"replay output already exists: {output}; use --overwrite to replace it"
            )
        if output.exists():
            output.unlink()
    payload["log_file"] = events_path.as_posix()
    payload["error_frames_dir"] = (args.output_dir / "errors").as_posix()
    config = live_layout_from_dict(payload)
    source = RecordedWindowFrameSource(args.manifest, app_name=config.app_name)
    max_frames = source.frame_count
    if args.max_frames is not None:
        max_frames = min(max_frames, args.max_frames)
    runtime = LiveGameRuntime(
        config,
        frame_source=source,
        sleeper=lambda _: None,
        log_every_scene_frame=True,
    )
    last = None
    try:
        for snapshot in runtime.run_loop(max_frames=max_frames):
            last = snapshot
            if not args.quiet:
                print(format_live_snapshot(snapshot))
                print()
    finally:
        runtime.close()
    if last is None:
        raise SystemExit("recorded replay produced no frames")
    if not events_path.is_file():
        raise SystemExit("recorded replay did not produce an events log")
    model_path = config.model_path.resolve()
    if not model_path.is_file():
        raise SystemExit(f"replay model is missing: {model_path}")
    templates_path = config.templates_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    replay_report = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "session": inspection.session_dir.name,
        "created_at": time.time(),
        "manifest": _fingerprint(args.manifest),
        "config": _fingerprint(config_path),
        "model": _fingerprint(model_path),
        "templates": {
            "path": templates_path.as_posix(),
            "sha256": sha256_directory(templates_path),
        },
        "events_log": _fingerprint(events_path),
        "implementation": {
            "project_root": project_root.as_posix(),
            "sha256": sha256_python_implementation(project_root),
            "runtime_versions": runtime_versions(),
        },
        "replayed_frames": max_frames,
        "final_mode": last.tracker_update.mode.value,
    }
    if annotation_path.is_file():
        replay_report["annotation"] = _fingerprint(annotation_path)
    replay_report["report_sha256"] = sha256_json_payload(replay_report)
    write_json_atomic(provenance_path, replay_report)
    print(f"replayed_frames: {max_frames}")
    print(f"events_log: {config.log_file}")
    print(f"provenance: {provenance_path}")
    print(f"final_mode: {last.tracker_update.mode.value}")
    return 0


def _fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
    }


if __name__ == "__main__":
    raise SystemExit(main())
