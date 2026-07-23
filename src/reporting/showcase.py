from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, Mapping

from src.logic.action_validation import validate_observed_action
from src.logic.decision import recommend_action
from src.logic.monte_carlo import MonteCarloSettings, recommend_phase4
from src.state.game_state import GameStateSnapshot
from src.state.replay import load_event_replay


@dataclass(frozen=True)
class ShowcaseSettings:
    event_files: tuple[Path, ...]
    simulations: int = 20
    max_depth: int = 20
    seed: int = 20260721
    repeats: int = 3
    top_k: int = 3
    max_candidates: int = 8
    vision_metrics_file: Path | None = Path("docs/evidence/card_cnn_eval.json")

    def __post_init__(self) -> None:
        if not self.event_files:
            raise ValueError("at least one Phase 4 event replay is required")
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.top_k <= 0 or self.max_candidates <= 0:
            raise ValueError("top_k and max_candidates must be positive")
        if self.top_k > self.max_candidates:
            raise ValueError("top_k cannot exceed max_candidates")


def build_showcase_report(
    settings: ShowcaseSettings,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    started = clock()
    phase1 = _run_phase1(clock)
    scenarios = tuple(_run_phase4_scenario(path, settings) for path in settings.event_files)
    vision_metrics = _load_vision_metrics(settings.vision_metrics_file)

    checks = [
        {
            "name": "phase1_deterministic_legal_recommendation",
            "passed": phase1["status"] == "passed",
            "evidence": phase1["recommended_action"],
        },
        *(
            check
            for scenario in scenarios
            for check in scenario["checks"]
        ),
    ]
    overall_passed = all(bool(check["passed"]) for check in checks)
    elapsed_ms = round((clock() - started) * 1000, 3)
    return {
        "schema_version": "phase5-showcase-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "overall_status": "passed" if overall_passed else "failed",
        "duration_ms": elapsed_ms,
        "phase1": phase1,
        "phase4": {
            "settings": {
                "simulations": settings.simulations,
                "max_depth": settings.max_depth,
                "seed": settings.seed,
                "repeats": settings.repeats,
                "top_k": settings.top_k,
                "max_candidates": settings.max_candidates,
                "time_budget_ms": 0,
            },
            "scenarios": list(scenarios),
        },
        "capabilities": [
            {
                "phase": "Phase 1",
                "status": "verified_this_run",
                "evidence": "deterministic rule recommendation",
            },
            {
                "phase": "Phase 2",
                "status": vision_metrics["status"],
                "evidence": vision_metrics,
            },
            {
                "phase": "Phase 3/3.5",
                "status": "not_run",
                "evidence": "macOS window capture and local model assets are required",
            },
            {
                "phase": "Phase 4",
                "status": "verified_this_run" if all(
                    scenario["status"] == "passed" for scenario in scenarios
                ) else "failed",
                "evidence": f"{len(scenarios)} tracked event replay scenarios",
            },
            {
                "phase": "Phase 5A",
                "status": "verified_this_run" if overall_passed else "failed",
                "evidence": "JSON, Markdown, and self-contained HTML report",
            },
        ],
        "acceptance_checks": checks,
        "limitations": [
            "Phase 4 uses a uniform remaining-card model, not exact opponent cards.",
            "The current ruleset is a documented standard subset and not every platform variant.",
            "Phase 3 does not yet observe opponent plays, passes, or remaining-card counters automatically.",
            "Phase 2 metrics are historical local small-sample evidence, not real-window generalization accuracy.",
            "Latency is machine-specific evidence and is not a cross-platform pass/fail threshold.",
            "The project never performs automatic clicks or unattended game play.",
        ],
        "reproduce": {
            "command": (
                f"{Path(sys.executable).name} -m scripts.run_phase5_showcase "
                f"--simulations {settings.simulations} --max-depth {settings.max_depth} "
                f"--seed {settings.seed} --repeats {settings.repeats}"
            ),
            "event_files": [_display_path(path) for path in settings.event_files],
        },
    }


def write_showcase_artifacts(
    report: Mapping[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "report.json",
        "markdown": output_dir / "summary.md",
        "html": output_dir / "index.html",
    }
    paths["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(_render_markdown(report), encoding="utf-8")
    paths["html"].write_text(_render_html(report), encoding="utf-8")
    return paths


def _run_phase1(clock: Callable[[], float]) -> dict[str, object]:
    state = GameStateSnapshot.from_inputs(
        "3 3 4 4 5 5 6 6 7 8 9 SJ BJ",
        "5 5",
    )
    started = clock()
    first = recommend_action(state)
    second = recommend_action(state)
    elapsed_ms = round((clock() - started) * 1000, 3)
    passed = first.action == second.action and first.action in first.candidates
    return {
        "status": "passed" if passed else "failed",
        "input_hand": state.hand.to_list(),
        "last_play": state.last_play.to_list(),
        "candidate_count": len(first.candidates),
        "recommended_action": first.action.to_list(),
        "reason": first.reason,
        "deterministic": first.action == second.action,
        "elapsed_ms": elapsed_ms,
    }


def _run_phase4_scenario(
    path: Path,
    settings: ShowcaseSettings,
) -> dict[str, object]:
    replay = load_event_replay(path, validator=validate_observed_action)
    monte_carlo = MonteCarloSettings(
        simulations=settings.simulations,
        max_depth=settings.max_depth,
        time_budget_ms=0,
        seed=settings.seed,
        top_k=settings.top_k,
        max_candidates=settings.max_candidates,
        min_rollouts_per_action=min(8, settings.simulations),
    )
    results = tuple(
        recommend_phase4(replay.state, monte_carlo)
        for _ in range(settings.repeats)
    )
    fingerprints = tuple(_decision_fingerprint(result.to_log_payload()) for result in results)
    first = results[0]
    latencies = [result.elapsed_ms for result in results]
    win_rates = [evaluation.estimated_win_rate for evaluation in first.rankings]
    checks = [
        {
            "name": f"{path.stem}_state_decision_ready",
            "passed": replay.state.decision_ready,
            "evidence": f"revision={replay.state.revision}",
        },
        {
            "name": f"{path.stem}_all_worlds_completed",
            "passed": all(
                result.completed_simulations == settings.simulations
                for result in results
            ),
            "evidence": f"{first.completed_simulations}/{settings.simulations}",
        },
        {
            "name": f"{path.stem}_fixed_seed_deterministic",
            "passed": len(set(fingerprints)) == 1,
            "evidence": fingerprints[0],
        },
        {
            "name": f"{path.stem}_top_k_sorted",
            "passed": win_rates == sorted(win_rates, reverse=True),
            "evidence": win_rates,
        },
    ]
    return {
        "name": path.stem,
        "fixture": _display_path(path),
        "round_id": replay.state.round_id,
        "landlord": replay.state.landlord.value,
        "event_count": replay.event_count,
        "state_revision": replay.state.revision,
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "decision_fingerprint": fingerprints[0],
        "deterministic_across_repeats": len(set(fingerprints)) == 1,
        "recommended_action": first.action.to_list(),
        "candidate_count": first.all_candidate_count,
        "evaluated_candidate_count": first.evaluated_candidate_count,
        "completed_simulations": first.completed_simulations,
        "requested_simulations": first.requested_simulations,
        "latency_ms": {
            "runs": [round(value, 3) for value in latencies],
            "min": round(min(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "worlds_per_second": round(
            first.completed_simulations / max(first.elapsed_ms / 1000, 1e-9),
            3,
        ),
        "reason": first.reason,
        "warnings": list(dict.fromkeys((*replay.warnings, *first.warnings))),
        "top_k": [evaluation.to_log_payload() for evaluation in first.rankings],
        "checks": checks,
    }


def _decision_fingerprint(payload: Mapping[str, object]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"elapsed_ms", "opponent_estimate"}
    }
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _load_vision_metrics(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {
            "status": "not_available",
            "source": _display_path(path) if path is not None else None,
            "scope": "historical local metrics were not bundled into this run",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("vision metrics evidence must be a JSON object")
    return {
        "status": "historical_evidence",
        "source": _display_path(path),
        "scope": payload.get("scope", "local small-sample evaluation"),
        "splits": payload.get("splits", {}),
        "error_count": payload.get("error_count"),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _render_markdown(report: Mapping[str, object]) -> str:
    phase1 = report["phase1"]
    phase4 = report["phase4"]
    lines = [
        "# doudizhu-assistant Phase 5 Showcase",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Git commit: `{report['git_commit']}`",
        f"- Generated at: `{report['generated_at_utc']}`",
        f"- Total showcase time: `{report['duration_ms']} ms`",
        "",
        "## Phase 1 deterministic rule baseline",
        "",
        f"- Recommended action: `{_cards(phase1['recommended_action'])}`",
        f"- Candidate count: `{phase1['candidate_count']}`",
        f"- Deterministic: `{phase1['deterministic']}`",
        f"- Reason: {phase1['reason']}",
        "",
        "## Phase 4 replay scenarios",
        "",
    ]
    for scenario in phase4["scenarios"]:
        lines.extend((
            f"### {scenario['name']}",
            "",
            f"- Status: **{scenario['status']}**",
            f"- Role: `{scenario['landlord']}` is landlord",
            f"- Recommendation: `{_cards(scenario['recommended_action'])}`",
            f"- Worlds: `{scenario['completed_simulations']}/{scenario['requested_simulations']}`",
            f"- Median latency: `{scenario['latency_ms']['median']} ms`",
            f"- Decision fingerprint: `{scenario['decision_fingerprint']}`",
            "",
            "| Rank | Action | Strategy score | Estimated result | Risks |",
            "|---:|---|---:|---:|---|",
        ))
        for rank, evaluation in enumerate(scenario["top_k"], start=1):
            estimated = evaluation["estimated_win_rate"]
            estimated_text = "n/a" if estimated is None else f"{estimated:.1%}"
            lines.append(
                f"| {rank} | `{_cards(evaluation['action'])}` | "
                f"{evaluation['strategy_score']:.3f} | {estimated_text} | "
                f"{', '.join(evaluation['risk_flags']) or 'none'} |"
            )
        lines.append("")
    lines.extend((
        "## Important limitations",
        "",
        *(f"- {item}" for item in report["limitations"]),
        "",
        "## Reproduce",
        "",
        "```bash",
        str(report["reproduce"]["command"]),
        "```",
        "",
    ))
    return "\n".join(lines)


def _render_html(report: Mapping[str, object]) -> str:
    phase1 = report["phase1"]
    scenarios = report["phase4"]["scenarios"]
    status_class = "ok" if report["overall_status"] == "passed" else "bad"
    scenario_sections = []
    for scenario in scenarios:
        rows = []
        for rank, evaluation in enumerate(scenario["top_k"], start=1):
            estimated = evaluation["estimated_win_rate"]
            estimated_text = "n/a" if estimated is None else f"{estimated:.1%}"
            risks = ", ".join(evaluation["risk_flags"]) or "none"
            rows.append(
                "<tr>"
                f"<td>{rank}</td><td><code>{escape(_cards(evaluation['action']))}</code></td>"
                f"<td>{evaluation['strategy_score']:.3f}</td><td>{estimated_text}</td>"
                f"<td>{escape(risks)}</td></tr>"
            )
        scenario_sections.append(
            f"<section class='panel'><div class='panel-head'><div>"
            f"<p class='eyebrow'>PHASE 4 · {escape(str(scenario['landlord']).upper())} LANDLORD</p>"
            f"<h2>{escape(str(scenario['name']))}</h2></div>"
            f"<span class='pill ok'>{escape(str(scenario['status']))}</span></div>"
            "<div class='metrics'>"
            f"<div><span>Recommendation</span><strong>{escape(_cards(scenario['recommended_action']))}</strong></div>"
            f"<div><span>Sampled worlds</span><strong>{scenario['completed_simulations']}/{scenario['requested_simulations']}</strong></div>"
            f"<div><span>Median latency</span><strong>{scenario['latency_ms']['median']} ms</strong></div>"
            f"<div><span>Fingerprint</span><strong class='mono'>{scenario['decision_fingerprint']}</strong></div>"
            "</div><div class='table-wrap'><table><thead><tr>"
            "<th>#</th><th>Action</th><th>Score</th><th>Estimated result</th><th>Risks</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
        )
    limitations = "".join(
        f"<li>{escape(str(item))}</li>" for item in report["limitations"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>doudizhu-assistant · Phase 5 Showcase</title>
  <style>
    :root {{ --bg:#08111f; --panel:#101d30; --line:#21334d; --text:#f4f7fb; --muted:#9fb0c8; --cyan:#40d9d0; --gold:#ffca68; --green:#4ade80; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at 80% 0,#123153 0,transparent 36%),var(--bg); color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1120px,calc(100% - 32px)); margin:auto; padding:56px 0 72px; }}
    h1 {{ font-size:clamp(38px,7vw,76px); letter-spacing:-.055em; line-height:.96; max-width:850px; margin:14px 0 20px; }} h2 {{ margin:0; font-size:25px; }}
    .eyebrow {{ color:var(--cyan); font-weight:800; letter-spacing:.14em; font-size:12px; margin:0 0 8px; }} .lede {{ color:var(--muted); max-width:720px; font-size:18px; }}
    .pill {{ display:inline-flex; padding:7px 12px; border:1px solid var(--line); border-radius:999px; text-transform:uppercase; font-size:11px; font-weight:800; letter-spacing:.1em; }} .ok {{ color:var(--green); border-color:#1f6842; background:#0b2b21; }} .bad {{ color:#fb7185; }}
    .hero-meta,.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:30px; }} .hero-meta div,.metrics div {{ padding:17px; border:1px solid var(--line); background:#0c1829cc; border-radius:14px; }}
    span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:5px; }} strong {{ font-size:18px; }} .mono,code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .flow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:34px 0; }} .flow div {{ text-align:center; padding:14px 8px; border:1px solid #22506b; color:var(--cyan); background:#0b1e30; border-radius:12px; font-weight:700; }}
    .panel {{ margin-top:18px; padding:25px; background:linear-gradient(145deg,#11223a,#0d192a); border:1px solid var(--line); border-radius:20px; box-shadow:0 16px 55px #0004; }} .panel-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
    .table-wrap {{ overflow:auto; margin-top:20px; }} table {{ border-collapse:collapse; width:100%; min-width:700px; }} th,td {{ padding:12px 10px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
    .limitations {{ border-left:3px solid var(--gold); }} li {{ margin:8px 0; color:#c9d5e6; }} footer {{ color:var(--muted); margin-top:28px; font-size:12px; }}
    @media(max-width:760px) {{ .hero-meta,.metrics {{ grid-template-columns:1fr 1fr; }} .flow {{ grid-template-columns:1fr; }} .panel {{ padding:18px; }} }}
  </style>
</head>
<body><main>
  <header><p class="eyebrow">REAL-TIME AI SYSTEM · REPRODUCIBLE OFFLINE EVIDENCE</p>
    <span class="pill {status_class}">{escape(str(report['overall_status']))}</span>
    <h1>Dou Dizhu assistant, from pixels to decisions.</h1>
    <p class="lede">A layered Python system for card recognition, observable game state, legal-action generation, and uncertainty-aware Monte Carlo recommendations.</p>
    <div class="hero-meta"><div><span>Git commit</span><strong class="mono">{escape(str(report['git_commit']))}</strong></div><div><span>Showcase time</span><strong>{report['duration_ms']} ms</strong></div><div><span>Phase 1 action</span><strong>{escape(_cards(phase1['recommended_action']))}</strong></div><div><span>Scenarios</span><strong>{len(scenarios)}</strong></div></div>
  </header>
  <div class="flow"><div>Screen / Replay</div><div>Vision</div><div>Observable State</div><div>Rules + Monte Carlo</div><div>Top-K Evidence</div></div>
  {''.join(scenario_sections)}
  <section class="panel limitations"><p class="eyebrow">HONEST BOUNDARIES</p><h2>What this report does not claim</h2><ul>{limitations}</ul></section>
  <footer>Generated at {escape(str(report['generated_at_utc']))} · schema {escape(str(report['schema_version']))} · no external assets or network required.</footer>
</main></body></html>
"""


def _cards(cards: object) -> str:
    if isinstance(cards, (list, tuple)):
        return " ".join(str(card) for card in cards) if cards else "pass"
    return str(cards)


__all__ = [
    "ShowcaseSettings",
    "build_showcase_report",
    "write_showcase_artifacts",
]
