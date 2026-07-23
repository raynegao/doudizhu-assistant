from __future__ import annotations

import argparse
from pathlib import Path

from src.capture.screen_geometry import MacWindowCapture
from src.pipeline.live_layout import (
    LiveLayoutConfig,
    load_live_layout,
    render_layout_preview,
    render_roi_contact_sheet,
    save_live_layout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the Phase 6 macOS live-game layout."
    )
    parser.add_argument("--app-name", default="斗地主")
    parser.add_argument("--base-config", type=Path)
    parser.add_argument(
        "--save-config",
        type=Path,
        default=Path("configs/live_game.local.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/live_game/calibration"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = (
        load_live_layout(args.base_config)
        if args.base_config is not None
        else LiveLayoutConfig(app_name=args.app_name)
    )
    if config.app_name != args.app_name:
        payload = config.to_payload()
        payload["app_name"] = args.app_name
        from src.pipeline.live_layout import live_layout_from_dict

        config = live_layout_from_dict(payload)
    frame = MacWindowCapture(args.app_name).capture(frame_id=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "live_layout_preview.png"
    contact_path = args.output_dir / "live_layout_contact_sheet.png"
    render_layout_preview(frame.image, config).save(preview_path)
    render_roi_contact_sheet(frame.image, config).save(contact_path)
    save_live_layout(args.save_config, config)
    print(f"window: {frame.window.window_name}")
    print(f"logical_box: {frame.window.window_box}")
    print(f"pixel_box: {frame.pixel_box}")
    print(
        f"screen_scale: {frame.geometry.scale_x:.3f}x"
        f"{frame.geometry.scale_y:.3f}"
    )
    print(f"config: {args.save_config}")
    print(f"preview: {preview_path}")
    print(f"contact_sheet: {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
