from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.reporting.showcase import (
    ShowcaseSettings,
    build_showcase_report,
    write_showcase_artifacts,
)


DEFAULT_EVENT_FILES = tuple(sorted(Path("examples/phase5").glob("*.jsonl")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the reproducible Phase 5 portfolio showcase artifacts."
    )
    parser.add_argument(
        "--events-file",
        action="append",
        dest="event_files",
        help="Phase 4 JSONL replay; repeat to benchmark multiple scenarios.",
    )
    parser.add_argument("--output-dir", default="runs/showcase")
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--vision-metrics-file",
        default="docs/evidence/card_cnn_eval.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    event_files = tuple(Path(path) for path in args.event_files or DEFAULT_EVENT_FILES)
    try:
        report = build_showcase_report(ShowcaseSettings(
            event_files=event_files,
            simulations=args.simulations,
            max_depth=args.max_depth,
            seed=args.seed,
            repeats=args.repeats,
            top_k=args.top_k,
            max_candidates=args.max_candidates,
            vision_metrics_file=Path(args.vision_metrics_file),
        ))
        paths = write_showcase_artifacts(report, Path(args.output_dir))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Phase 5 showcase input error: {exc}", file=sys.stderr)
        return 2

    print(f"Phase 5 showcase: {report['overall_status']}")
    print(f"Scenarios: {len(report['phase4']['scenarios'])}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
