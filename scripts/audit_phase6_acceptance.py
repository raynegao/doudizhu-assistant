from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reporting.live_acceptance import (
    Phase6AcceptanceThresholds,
    audit_phase6_acceptance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit independent Phase 6 sessions and acceptance metrics."
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path("data/live_game/recordings"),
    )
    parser.add_argument(
        "--replays-root",
        type=Path,
        default=Path("runs/live-replay"),
    )
    parser.add_argument(
        "--card-holdout-report",
        type=Path,
        default=Path("runs/real-window-holdout/report.json"),
    )
    parser.add_argument("--skip-card-holdout", action="store_true")
    parser.add_argument("--min-sessions", type=int, default=5)
    parser.add_argument("--round-success-rate", type=float, default=0.80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase6-acceptance/report.json"),
    )
    parser.add_argument("--require-thresholds", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_phase6_acceptance(
        args.recordings_root,
        args.replays_root,
        card_holdout_report=args.card_holdout_report,
        require_card_holdout=not args.skip_card_holdout,
        thresholds=Phase6AcceptanceThresholds(
            min_sessions=args.min_sessions,
            round_success_rate=args.round_success_rate,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if args.require_thresholds and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
