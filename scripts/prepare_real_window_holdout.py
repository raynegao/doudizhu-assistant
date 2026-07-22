"""Crop and label one real-window hand ROI as an independent holdout session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.crop_hand_roi_cards import crop_hand_roi_cards, parse_crop_size
from scripts.evaluate_real_window_holdout import load_holdout_manifest, sha256_file
from src.state.cards import normalize_rank
from src.vision.card_classifier import CARD_CLASSES


SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def prepare_holdout_session(
    *,
    roi_path: Path,
    output_root: Path,
    source_id: str,
    labels: Sequence[str],
    count: int,
    start_x: int,
    start_y: int,
    step_x: int,
    crop_size: tuple[int, int],
    manifest_path: Path | None = None,
) -> dict[str, object]:
    if not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError(
            "source-id must use letters, digits, dots, underscores, or hyphens"
        )
    normalized_labels = tuple(normalize_rank(label) for label in labels)
    if len(normalized_labels) != count:
        raise ValueError(f"expected {count} labels, got {len(normalized_labels)}")
    unsupported = [label for label in normalized_labels if label not in CARD_CLASSES]
    if unsupported:
        raise ValueError(f"unsupported labels: {unsupported}")
    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI image does not exist: {roi_path}")

    manifest_path = manifest_path or output_root / "manifest.jsonl"
    existing_records = (
        load_holdout_manifest(manifest_path)
        if manifest_path.is_file() and manifest_path.stat().st_size
        else []
    )
    if any(record["source_id"] == source_id for record in existing_records):
        raise ValueError(f"source-id already exists in holdout manifest: {source_id}")
    roi_sha256 = sha256_file(roi_path)
    if any(record.get("roi_sha256") == roi_sha256 for record in existing_records):
        raise ValueError("this ROI image is already registered in the holdout manifest")

    session_dir = output_root / "sessions" / source_id
    if session_dir.exists() and any(session_dir.iterdir()):
        raise ValueError(f"session directory is not empty: {session_dir}")
    crop_dir = session_dir / "crops"
    metadata = crop_hand_roi_cards(
        roi_path=roi_path,
        output_dir=crop_dir,
        count=count,
        start_x=start_x,
        start_y=start_y,
        step_x=step_x,
        crop_size=crop_size,
    )
    crop_paths = [crop_dir / str(item["filename"]) for item in metadata]
    crop_hashes = [sha256_file(path) for path in crop_paths]
    existing_hashes = {str(record["sha256"]) for record in existing_records}
    overlaps = [value for value in crop_hashes if value in existing_hashes]
    if overlaps:
        raise ValueError(
            f"holdout already contains {len(overlaps)} identical crop(s); use a new session"
        )

    contact_sheet_path = session_dir / "contact_sheet.png"
    build_contact_sheet(crop_paths, contact_sheet_path, normalized_labels)
    captured_at = datetime.now(timezone.utc).isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with manifest_path.open("a", encoding="utf-8") as handle:
        for index, (path, label, crop_sha256) in enumerate(
            zip(crop_paths, normalized_labels, crop_hashes, strict=True)
        ):
            record = {
                "image": Path(os.path.relpath(path, manifest_path.parent)).as_posix(),
                "label": label,
                "source_id": source_id,
                "crop_index": index,
                "sha256": crop_sha256,
                "roi_sha256": roi_sha256,
                "captured_at_utc": captured_at,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    summary = {
        "schema_version": "real-window-holdout-session-v1",
        "source_id": source_id,
        "roi": str(roi_path),
        "roi_sha256": roi_sha256,
        "crop_count": len(records),
        "labels": list(normalized_labels),
        "manifest": str(manifest_path),
        "contact_sheet": str(contact_sheet_path),
    }
    (session_dir / "session.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_contact_sheet(
    crop_paths: Sequence[Path],
    output_path: Path,
    labels: Sequence[str] | None = None,
) -> Path:
    if not crop_paths:
        raise ValueError("at least one crop is required for a contact sheet")
    thumb_size = (126, 150)
    gap = 16
    columns = min(6, len(crop_paths))
    rows = (len(crop_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * (thumb_size[0] + gap) + gap, rows * (thumb_size[1] + 44 + gap) + gap),
        "#101828",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(crop_paths):
        with Image.open(path) as image:
            preview = ImageOps.contain(
                ImageOps.exif_transpose(image).convert("RGB"),
                thumb_size,
                Image.Resampling.BICUBIC,
            )
        column = index % columns
        row = index // columns
        left = gap + column * (thumb_size[0] + gap)
        upper = gap + row * (thumb_size[1] + 44 + gap)
        sheet.paste(preview, (left, upper))
        label = labels[index] if labels is not None else "?"
        draw.text(
            (left, upper + thumb_size[1] + 10),
            f"{index:02d}  {label}",
            font=font,
            fill="#f9fafb",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _parse_labels(value: str) -> list[str]:
    return [part for part in value.split() if part]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop, label, hash, and register one real-window holdout ROI."
    )
    parser.add_argument("--roi", required=True, help="Fresh real-window hand ROI image.")
    parser.add_argument("--source-id", required=True, help="Unique capture/session identifier.")
    parser.add_argument("--output-root", default="data/real_window_holdout")
    parser.add_argument("--manifest")
    parser.add_argument("--labels", help="Space-separated ranks in left-to-right crop order.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-x", type=int, default=0)
    parser.add_argument("--start-y", type=int, default=0)
    parser.add_argument("--step-x", type=int, required=True)
    parser.add_argument("--crop-size", type=parse_crop_size, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    labels_value = args.labels
    if labels_value is None:
        labels_value = input(
            f"Enter {args.count} ranks from left to right (e.g. A K Q J 10): "
        )
    try:
        summary = prepare_holdout_session(
            roi_path=Path(args.roi),
            output_root=Path(args.output_root),
            source_id=args.source_id,
            labels=_parse_labels(labels_value),
            count=args.count,
            start_x=args.start_x,
            start_y=args.start_y,
            step_x=args.step_x,
            crop_size=args.crop_size,
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
