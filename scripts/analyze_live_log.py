from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reporting.live_diagnostics import analyze_live_log


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
        "--output",
        type=Path,
        default=Path("runs/live-diagnostics/report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_live_log(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
