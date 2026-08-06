from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from src.capture.recorded_window import RecordedWindowFrameSource
from src.capture.recording_integrity import (
    inspect_recording_session,
    sha256_file,
    sha256_json_payload,
    write_json_atomic,
)
from src.logic.rules import Play, PlayType, can_beat
from src.state.cards import FULL_DECK, CardSet
from src.state.events import DEFAULT_TURN_ORDER, PlayerSeat

ACTION_EVENTS = {"play_observed", "pass_observed"}
WORKBOOK_SCHEMA_VERSION = "phase6-blind-annotation-workbook-v1"
ANNOTATION_SCHEMA_VERSION = "phase6-blind-annotation-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or seal blind Phase 6 annotations before replay output exists."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--replays-root",
        type=Path,
        default=Path("runs/live-replay"),
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--seal", action="store_true")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--thumbnail-width", type=int, default=420)
    return parser


def assert_blind_annotation_preconditions(
    session_dir: Path,
    *,
    replays_root: Path,
) -> None:
    inspection = inspect_recording_session(session_dir)
    if not inspection.valid:
        raise ValueError(
            "recording integrity validation failed: "
            + "; ".join(inspection.issues)
        )
    if inspection.metadata.get("evidence_split") != "acceptance":
        raise ValueError("blind annotation requires an acceptance recording")
    if inspection.metadata.get("complete_game") is not True:
        raise ValueError("blind annotation requires a finalized complete game")
    replay_dir = replays_root / session_dir.name
    if replay_dir.exists() and any(replay_dir.iterdir()):
        raise ValueError(
            "blind annotation must be sealed before replay output exists: "
            f"{replay_dir}"
        )


def prepare_annotation_workbook(
    session_dir: Path,
    *,
    columns: int = 4,
    rows: int = 4,
    thumbnail_width: int = 420,
) -> dict[str, object]:
    if columns <= 0 or rows <= 0 or thumbnail_width < 160:
        raise ValueError("invalid annotation sheet layout")
    inspection = inspect_recording_session(session_dir)
    if not inspection.valid:
        raise ValueError("recording session is invalid")
    manifest_path = session_dir / "manifest.jsonl"
    sheet_dir = session_dir / "annotation_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for old_sheet in sheet_dir.glob("page-*.png"):
        old_sheet.unlink()

    rows_per_page = columns * rows
    sheet_records: list[dict[str, object]] = []
    frame_source = RecordedWindowFrameSource(manifest_path)
    try:
        for page_index, offset in enumerate(
            range(0, len(inspection.manifest_rows), rows_per_page),
            start=1,
        ):
            page_rows = inspection.manifest_rows[offset : offset + rows_per_page]
            sheet = _render_sheet(
                frame_source,
                page_rows,
                columns=columns,
                rows=rows,
                thumbnail_width=thumbnail_width,
            )
            sheet_path = sheet_dir / f"page-{page_index:03d}.png"
            sheet.save(sheet_path, compress_level=1)
            sheet_records.append({
                "page": page_index,
                "path": sheet_path.relative_to(session_dir).as_posix(),
                "sha256": sha256_file(sheet_path),
                "first_frame_id": _strict_int(
                    page_rows[0].get("frame_id"),
                    "frame_id",
                ),
                "last_frame_id": _strict_int(
                    page_rows[-1].get("frame_id"),
                    "frame_id",
                ),
            })
    finally:
        frame_source.close()

    workbook: dict[str, object] = {
        "schema_version": WORKBOOK_SCHEMA_VERSION,
        "session": session_dir.name,
        "generated_at": time.time(),
        "prediction_inputs_used": False,
        "manifest": _fingerprint(manifest_path),
        "frame_count": len(inspection.manifest_rows),
        "layout": {
            "columns": columns,
            "rows": rows,
            "thumbnail_width": thumbnail_width,
        },
        "sheets": sheet_records,
    }
    workbook["report_sha256"] = sha256_json_payload(workbook)
    write_json_atomic(session_dir / "annotation-workbook.json", workbook)
    return workbook


def validate_annotation_bundle(
    session_dir: Path,
) -> dict[str, int]:
    inspection = inspect_recording_session(session_dir)
    if not inspection.valid:
        raise ValueError("recording session is invalid")
    manifest_frame_ids = {
        _strict_int(row.get("frame_id"), "frame_id")
        for row in inspection.manifest_rows
    }
    expected_events_path = session_dir / "expected-events.jsonl"
    expected_scenes_path = session_dir / "expected-scenes.jsonl"
    event_rows = _read_jsonl(expected_events_path)
    scene_rows = _read_jsonl(expected_scenes_path)

    actions = [row for row in event_rows if row.get("event") in ACTION_EVENTS]
    results = [
        row for row in event_rows
        if row.get("event") == "round_result_detected"
    ]
    if len(actions) == 0:
        raise ValueError("expected-events.jsonl contains no actions")
    if len(actions) + len(results) != len(event_rows):
        raise ValueError("expected-events.jsonl contains unsupported events")
    if len(results) != 1 or event_rows[-1] is not results[0]:
        raise ValueError("expected-events.jsonl must end with exactly one result")

    played_cards: Counter[str] = Counter()
    previous_actor: PlayerSeat | None = None
    trick_target: Play | None = None
    consecutive_passes = 0
    for expected_sequence, action in enumerate(actions, start=1):
        if action.get("sequence_no") != expected_sequence:
            raise ValueError("expected action sequences must be contiguous")
        try:
            actor = PlayerSeat(str(action["actor"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid actor at sequence {expected_sequence}") from exc
        if previous_actor is not None:
            previous_index = DEFAULT_TURN_ORDER.index(previous_actor)
            expected_actor = DEFAULT_TURN_ORDER[
                (previous_index + 1) % len(DEFAULT_TURN_ORDER)
            ]
            if actor is not expected_actor:
                raise ValueError(
                    f"actor at sequence {expected_sequence} breaks turn order"
                )
        cards = _annotation_cards(action, sequence_no=expected_sequence)
        event_name = action.get("event")
        if event_name == "pass_observed" and cards:
            raise ValueError(f"pass at sequence {expected_sequence} has cards")
        if event_name == "pass_observed":
            if trick_target is None:
                raise ValueError(
                    f"pass at sequence {expected_sequence} cannot lead a trick"
                )
            consecutive_passes += 1
            if consecutive_passes == 2:
                trick_target = None
                consecutive_passes = 0
        if event_name == "play_observed":
            if not cards:
                raise ValueError(f"play at sequence {expected_sequence} has no cards")
            play = Play.parse(cards.cards)
            if play.type is PlayType.INVALID:
                raise ValueError(
                    f"play at sequence {expected_sequence} is not a legal card type"
                )
            if trick_target is not None and not can_beat(play, trick_target):
                raise ValueError(
                    f"play at sequence {expected_sequence} does not beat the trick"
                )
            trick_target = play
            consecutive_passes = 0
            played_cards.update(cards.cards)
        action["_normalized_actor"] = actor.value
        action["_normalized_cards"] = cards.to_list()
        previous_actor = actor

    deck_counts = Counter(FULL_DECK)
    if any(count > deck_counts[rank] for rank, count in played_cards.items()):
        raise ValueError("annotated plays contain more cards than a 54-card deck")

    if len(scene_rows) != len(actions):
        raise ValueError("expected-scenes.jsonl must contain one row per action")
    previous_frame_id = 0
    previous_remaining: dict[str, int] | None = None
    for expected_sequence, (action, scene) in enumerate(
        zip(actions, scene_rows, strict=True),
        start=1,
    ):
        if scene.get("after_sequence_no") != expected_sequence:
            raise ValueError("scene annotations must follow action sequence order")
        frame_id = _strict_int(scene.get("frame_id"), "frame_id")
        if frame_id not in manifest_frame_ids:
            raise ValueError(f"scene frame {frame_id} is not in the manifest")
        if frame_id <= previous_frame_id:
            raise ValueError("scene frame ids must be strictly increasing")
        current_remaining = _annotation_remaining(scene, expected_sequence)
        if previous_remaining is not None:
            expected_remaining = dict(previous_remaining)
            actor = str(action["_normalized_actor"])
            cards = list(action["_normalized_cards"])
            if action.get("event") == "play_observed":
                expected_remaining[actor] -= len(cards)
            if current_remaining != expected_remaining:
                raise ValueError(
                    f"remaining counts at sequence {expected_sequence} do not "
                    "match the annotated action"
                )
        previous_frame_id = frame_id
        previous_remaining = current_remaining

    result = results[0]
    try:
        winner = PlayerSeat(str(result["winner"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("round result winner is invalid") from exc
    if result.get("outcome") not in {"victory", "defeat"}:
        raise ValueError("round result outcome is invalid")
    expected_outcome = "victory" if winner is PlayerSeat.SELF else "defeat"
    if result.get("outcome") != expected_outcome:
        raise ValueError("round result outcome does not match the winner")
    final_action = actions[-1]
    if final_action.get("event") != "play_observed":
        raise ValueError("the final annotated action must be a play")
    if final_action["_normalized_actor"] != winner.value:
        raise ValueError("round winner must be the final action actor")
    assert previous_remaining is not None
    if previous_remaining[winner.value] != 0:
        raise ValueError("round winner must have zero remaining cards")

    return {
        "frame_count": len(inspection.manifest_rows),
        "action_count": len(actions),
        "scene_count": len(scene_rows),
        "result_count": len(results),
    }


def seal_annotation_bundle(session_dir: Path) -> dict[str, object]:
    summary = validate_annotation_bundle(session_dir)
    manifest_path = session_dir / "manifest.jsonl"
    events_path = session_dir / "expected-events.jsonl"
    scenes_path = session_dir / "expected-scenes.jsonl"
    workbook_path = session_dir / "annotation-workbook.json"
    workbook = _read_json_object(workbook_path)
    if workbook.get("schema_version") != WORKBOOK_SCHEMA_VERSION:
        raise ValueError("annotation workbook is missing or unsupported")
    if workbook.get("prediction_inputs_used") is not False:
        raise ValueError("annotation workbook is not blind")
    if _mapping(workbook.get("manifest")).get("sha256") != sha256_file(
        manifest_path
    ):
        raise ValueError("annotation workbook does not match the manifest")
    annotation: dict[str, object] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "session": session_dir.name,
        "completed_at": time.time(),
        "annotation_mode": "blind_without_replay_predictions",
        "prediction_inputs_used": False,
        "manifest": _fingerprint(manifest_path),
        "workbook": _fingerprint(workbook_path),
        "expected_events": _fingerprint(events_path),
        "expected_scenes": _fingerprint(scenes_path),
        "summary": summary,
    }
    annotation["report_sha256"] = sha256_json_payload(annotation)
    write_json_atomic(session_dir / "annotation.json", annotation)
    return annotation


def validate_sealed_annotation(session_dir: Path) -> dict[str, object]:
    """Verify that a blind annotation still matches its immutable inputs."""

    summary = validate_annotation_bundle(session_dir)
    annotation_path = session_dir / "annotation.json"
    annotation = _read_json_object(annotation_path)
    if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("annotation seal is missing or unsupported")
    if annotation.get("session") != session_dir.name:
        raise ValueError("annotation seal session does not match the directory")
    if annotation.get("annotation_mode") != "blind_without_replay_predictions":
        raise ValueError("annotation seal was not created in blind mode")
    if annotation.get("prediction_inputs_used") is not False:
        raise ValueError("annotation seal declares prediction-assisted labels")
    completed_at = annotation.get("completed_at")
    if (
        isinstance(completed_at, bool)
        or not isinstance(completed_at, (int, float))
    ):
        raise ValueError("annotation seal completion time is invalid")
    if annotation.get("summary") != summary:
        raise ValueError("annotation seal summary no longer matches the labels")

    workbook_path = session_dir / "annotation-workbook.json"
    inputs = (
        ("manifest", session_dir / "manifest.jsonl"),
        ("workbook", workbook_path),
        ("expected_events", session_dir / "expected-events.jsonl"),
        ("expected_scenes", session_dir / "expected-scenes.jsonl"),
    )
    for name, path in inputs:
        _validate_fingerprint(annotation.get(name), path, name)

    workbook = _read_json_object(workbook_path)
    if workbook.get("schema_version") != WORKBOOK_SCHEMA_VERSION:
        raise ValueError("annotation workbook is missing or unsupported")
    if workbook.get("session") != session_dir.name:
        raise ValueError("annotation workbook session does not match")
    if workbook.get("prediction_inputs_used") is not False:
        raise ValueError("annotation workbook declares prediction-assisted labels")
    generated_at = workbook.get("generated_at")
    if (
        isinstance(generated_at, bool)
        or not isinstance(generated_at, (int, float))
        or float(generated_at) > float(completed_at)
    ):
        raise ValueError("annotation workbook time is invalid")
    _validate_fingerprint(
        workbook.get("manifest"),
        session_dir / "manifest.jsonl",
        "workbook manifest",
    )
    sheets = workbook.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise ValueError("annotation workbook contains no contact sheets")
    for expected_page, raw_sheet in enumerate(sheets, start=1):
        sheet = _mapping(raw_sheet)
        if sheet.get("page") != expected_page:
            raise ValueError("annotation workbook page numbers are not contiguous")
        sheet_path = _safe_session_child(session_dir, sheet.get("path"))
        if sheet.get("sha256") != sha256_file(sheet_path):
            raise ValueError(f"annotation contact sheet changed: {sheet_path}")
    return annotation


def _render_sheet(
    frame_source: RecordedWindowFrameSource,
    records: tuple[dict[str, object], ...],
    *,
    columns: int,
    rows: int,
    thumbnail_width: int,
) -> Image.Image:
    raw_size = records[0].get("image_size")
    if not isinstance(raw_size, list) or len(raw_size) != 2:
        raise ValueError("recorded frame image_size is missing")
    width, height = (int(value) for value in raw_size)
    aspect = height / width
    thumbnail_height = round(thumbnail_width * aspect)
    label_height = 28
    sheet = Image.new(
        "RGB",
        (
            columns * thumbnail_width,
            rows * (thumbnail_height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        frame_id = _strict_int(record.get("frame_id"), "frame_id")
        timestamp = float(str(record.get("timestamp")))
        source = frame_source.capture(frame_id).image
        thumbnail = ImageOps.fit(
            source.convert("RGB"),
            (thumbnail_width, thumbnail_height),
            method=Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        x = column * thumbnail_width
        y = row * (thumbnail_height + label_height)
        sheet.paste(thumbnail, (x, y + label_height))
        draw.text(
            (x + 6, y + 6),
            (
                f"frame {frame_id:06d}  "
                f"t={timestamp:.3f}"
            ),
            fill="black",
        )
    return sheet


def _annotation_cards(
    action: Mapping[str, object],
    *,
    sequence_no: int,
) -> CardSet:
    raw_cards = action.get("cards", [])
    if not isinstance(raw_cards, (list, tuple, str)):
        raise ValueError(f"invalid cards at sequence {sequence_no}")
    try:
        return CardSet.parse(raw_cards)
    except ValueError as exc:
        raise ValueError(f"invalid cards at sequence {sequence_no}: {exc}") from exc


def _annotation_remaining(
    scene: Mapping[str, object],
    sequence_no: int,
) -> dict[str, int]:
    raw = scene.get("remaining")
    if not isinstance(raw, Mapping) or set(raw) != {
        seat.value for seat in PlayerSeat
    }:
        raise ValueError(
            f"remaining counts at sequence {sequence_no} must contain all seats"
        )
    remaining = {
        str(seat): _strict_int(value, f"remaining {seat}")
        for seat, value in raw.items()
    }
    if any(not 0 <= value <= 20 for value in remaining.values()):
        raise ValueError(f"remaining counts at sequence {sequence_no} are invalid")
    return remaining


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"missing annotation file: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        rows.append(payload)
    return rows


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing annotation workbook: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    sealed = dict(payload)
    report_sha256 = sealed.pop("report_sha256", None)
    if report_sha256 != sha256_json_payload(sealed):
        raise ValueError("annotation workbook checksum is invalid")
    return payload


def _fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
    }


def _validate_fingerprint(value: object, path: Path, label: str) -> None:
    fingerprint = _mapping(value)
    resolved = path.resolve()
    if fingerprint.get("path") != resolved.as_posix():
        raise ValueError(f"{label} path does not match the session")
    if not resolved.is_file():
        raise ValueError(f"{label} file is missing: {resolved}")
    if fingerprint.get("sha256") != sha256_file(resolved):
        raise ValueError(f"{label} checksum changed after annotation sealing")


def _safe_session_child(session_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("annotation contact sheet path is invalid")
    root = session_dir.resolve()
    candidate = (root / value).resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_file():
        raise ValueError("annotation contact sheet path is outside the session")
    return candidate


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        assert_blind_annotation_preconditions(
            args.session_dir,
            replays_root=args.replays_root,
        )
        if args.prepare:
            result = prepare_annotation_workbook(
                args.session_dir,
                columns=args.columns,
                rows=args.rows,
                thumbnail_width=args.thumbnail_width,
            )
            output = args.session_dir / "annotation-workbook.json"
        else:
            result = seal_annotation_bundle(args.session_dir)
            output = args.session_dir / "annotation.json"
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
