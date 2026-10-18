"""Measure a decision agent against a baseline in seeded self-play.

Example (quick, deterministic):

    python -m scripts.run_decision_arena --games 30 --subject monte_carlo \
        --opponent heuristic --simulations 64 --workers 4 \
        --output-dir runs/decision-arena/mc64-vs-heuristic
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.logic.arena import CONFIGURATIONS, AgentSpec, MatchSettings, run_match
from src.logic.monte_carlo import RULESET_VERSION

AGENT_KINDS = ("random", "rule", "heuristic", "shaped", "monte_carlo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=30, help="Deals; each is played in three configurations.")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--subject", choices=AGENT_KINDS, default="monte_carlo")
    parser.add_argument("--opponent", choices=AGENT_KINDS, default="heuristic")
    parser.add_argument("--simulations", type=int, default=128, help="Monte Carlo worlds per decision.")
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--allocation", choices=("uniform", "successive_halving"), default="uniform")
    parser.add_argument("--rollout-policy", choices=("heuristic", "shaped"), default="heuristic")
    parser.add_argument(
        "--opponent-settings",
        type=json.loads,
        default=None,
        help="JSON Monte Carlo settings for a monte_carlo opponent (default: same as the subject).",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/decision-arena/latest"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    monte_carlo = {
        "simulations": args.simulations,
        "max_depth": args.max_depth,
        "max_candidates": args.max_candidates,
        "allocation": args.allocation,
        "rollout_policy": args.rollout_policy,
    }
    settings = MatchSettings(
        games=args.games,
        seed=args.seed,
        subject=AgentSpec(args.subject, monte_carlo if args.subject == "monte_carlo" else {}),
        opponent=AgentSpec(
            args.opponent,
            (args.opponent_settings if args.opponent_settings is not None else monte_carlo)
            if args.opponent == "monte_carlo" else {},
        ),
        workers=args.workers,
    )
    started = time.perf_counter()
    report, deals = run_match(settings)
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    report["created_at"] = time.time()
    report["ruleset_version"] = RULESET_VERSION
    # Imported late: the capture package pulls in the full runtime (PyTorch),
    # which must not be loaded into every spawned arena worker.
    from src.capture.recording_integrity import sha256_python_implementation

    report["implementation_sha256"] = sha256_python_implementation(Path(__file__).resolve().parents[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "games.jsonl").open("w", encoding="utf-8") as handle:
        for deal in deals:
            for configuration in CONFIGURATIONS:
                handle.write(json.dumps(
                    {"configuration": configuration, **deal[configuration].to_payload()}, ensure_ascii=False,
                ) + "\n")
    print(f"subject={args.subject} opponent={args.opponent} games={args.games} seed={args.seed}")
    for configuration in CONFIGURATIONS:
        block = report["configurations"][configuration]
        low, high = block["landlord_win_rate_ci95"]
        print(
            f"  {configuration:40s} landlord {block['landlord_wins']}/{block['games']} = "
            f"{block['landlord_win_rate']:.3f} [{low:.3f}, {high:.3f}]  plies {block['average_plies']}  "
            f"subject {block['subject_mean_decision_ms']} ms/decision"
        )
        monte_carlo = block.get("subject_monte_carlo")
        if monte_carlo:
            print(
                f"      MC: {monte_carlo['decisions']} decisions, top-2 CI overlap "
                f"{monte_carlo['top2_ci_overlap_rate']:.1%}, best CI width "
                f"{monte_carlo['mean_best_ci_width']:.3f}, samples {monte_carlo['mean_best_samples']}, "
                f"legal candidates {monte_carlo['mean_legal_candidates']}"
            )
    for role in ("subject_as_landlord", "subject_as_farmers"):
        paired = report["paired_vs_reference"][role]
        print(
            f"  paired {role:20s} mean {paired['mean']:+.3f} [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}] "
            f"(+{paired['wins_gained']} / -{paired['wins_lost']} of {paired['n']} deals)"
        )
    print(f"  elapsed {report['elapsed_s']} s -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
