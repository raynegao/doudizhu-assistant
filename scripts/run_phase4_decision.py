from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from src.config.settings import ConfigManager
from src.logic.action_validation import validate_observed_action
from src.logic.monte_carlo import MonteCarloSettings, Phase4DecisionResult, recommend_phase4
from src.state.cards import CardParseError
from src.state.events import DEFAULT_TURN_ORDER, ObservedAction, PlayerSeat
from src.state.game_tracker import GameStateTracker, GameStateTransitionError, StateUpdateStatus
from src.state.observable_state import ObservableGameState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 observable-state Monte Carlo recommendations."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hand", help="Current self hand for a direct state snapshot.")
    source.add_argument("--events-file", help="JSONL file beginning with game_started.")
    parser.add_argument("--round-id", default="manual-round")
    parser.add_argument("--config", default="configs/app.example.yaml")
    parser.add_argument("--landlord", choices=_seat_choices(), default="self")
    parser.add_argument("--turn-order", default="self,right,left")
    parser.add_argument("--left-count", type=int)
    parser.add_argument("--right-count", type=int)
    parser.add_argument("--played-cards", default="")
    parser.add_argument("--last-play", default="")
    parser.add_argument("--last-player", choices=_seat_choices())
    parser.add_argument("--consecutive-passes", type=int, choices=(0, 1), default=0)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument(
        "--time-budget-ms",
        type=int,
        default=None,
        help="0 disables the wall-clock deadline for deterministic offline evaluation.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--log-file", default="logs/phase4_decision.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.events_file:
            state, replay_warnings = _replay_events(Path(args.events_file))
        else:
            state = _build_direct_state(args)
            replay_warnings = ()
        config = ConfigManager(Path(args.config)).load().monte_carlo
        settings = MonteCarloSettings(
            simulations=_coalesce(args.simulations, config.simulations),
            max_depth=_coalesce(args.max_depth, config.max_depth),
            time_budget_ms=_coalesce(args.time_budget_ms, config.time_budget_ms),
            seed=_coalesce(args.seed, config.seed),
            top_k=_coalesce(args.top_k, config.top_k),
            max_candidates=_coalesce(args.max_candidates, config.max_candidates),
            min_rollouts_per_action=config.min_rollouts_per_action,
        )
        result = recommend_phase4(state, settings)
        if replay_warnings:
            result = replace(
                result,
                warnings=tuple(dict.fromkeys((*replay_warnings, *result.warnings))),
            )
    except (CardParseError, GameStateTransitionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Phase 4 输入错误: {exc}", file=sys.stderr)
        return 2

    print(format_phase4_result(state, result))
    _write_log(Path(args.log_file), state, result)
    return 0


def _build_direct_state(args: argparse.Namespace) -> ObservableGameState:
    remaining: dict[str, int] = {}
    if args.left_count is not None:
        remaining["left"] = args.left_count
    if args.right_count is not None:
        remaining["right"] = args.right_count
    last_player = args.last_player
    if args.last_play and last_player is None:
        raise ValueError("--last-player is required when --last-play is provided")
    return ObservableGameState.from_inputs(
        args.hand,
        round_id=args.round_id,
        landlord=args.landlord,
        current_actor="self",
        turn_order=_parse_turn_order(args.turn_order),
        remaining_cards=remaining,
        played_cards=args.played_cards,
        last_play=args.last_play,
        last_player=last_player,
        consecutive_passes=args.consecutive_passes,
    )


def _replay_events(path: Path) -> tuple[ObservableGameState, tuple[str, ...]]:
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not payloads or payloads[0].get("event") != "game_started":
        raise ValueError("events file must begin with a game_started object")
    start = payloads[0]
    remaining_payload = start.get("remaining_cards")
    if not isinstance(remaining_payload, dict):
        raise ValueError("game_started.remaining_cards must be an object")
    state = ObservableGameState.from_inputs(
        start.get("hand", ()),
        round_id=str(start.get("round_id", path.stem)),
        landlord=str(start.get("landlord", "self")),
        current_actor=str(start.get("current_actor", "self")),
        turn_order=start.get("turn_order", [seat.value for seat in DEFAULT_TURN_ORDER]),
        remaining_cards={str(seat): int(count) for seat, count in remaining_payload.items()},
        played_cards=start.get("played_cards", ()),
        last_play=start.get("last_play", ()),
        last_player=start.get("last_player"),
        consecutive_passes=int(start.get("consecutive_passes", 0)),
        state_confidence=float(start.get("state_confidence", 1.0)),
    )
    tracker = GameStateTracker(state, validator=validate_observed_action)
    warnings: list[str] = []
    for payload in payloads[1:]:
        event = ObservedAction.from_payload(payload)
        result = tracker.apply(event)
        if result.status is StateUpdateStatus.REJECTED:
            raise GameStateTransitionError(result.message)
        if result.status in {StateUpdateStatus.DEFERRED, StateUpdateStatus.DUPLICATE}:
            warnings.append(f"{result.status.value}: {event.event_id}: {result.message}")
            warnings.extend(result.warnings)
    return tracker.state, tuple(warnings)


def format_phase4_result(
    state: ObservableGameState,
    result: Phase4DecisionResult,
) -> str:
    lines = [
        "Dou Dizhu Phase 4 Decision",
        (
            f"round={state.round_id} revision={state.revision} "
            f"current={state.current_actor.value} landlord={state.landlord.value}"
        ),
        (
            f"unknown_cards={len(state.unknown_cards)} seed={result.seed} "
            f"sampled_worlds={result.completed_simulations}/{result.requested_simulations} "
            f"latency={result.elapsed_ms:.1f}ms"
        ),
        (
            f"candidates={result.all_candidate_count} "
            f"evaluated={result.evaluated_candidate_count}"
        ),
        "Top-K 推荐:",
    ]
    for index, evaluation in enumerate(result.rankings, start=1):
        estimated = (
            f"{evaluation.estimated_win_rate:.1%}"
            if evaluation.estimated_win_rate is not None
            else "n/a"
        )
        terminal = (
            f"{evaluation.terminal_win_rate:.1%}"
            if evaluation.terminal_win_rate is not None
            else "n/a"
        )
        lines.append(
            f"{index}. {evaluation.action} | score={evaluation.strategy_score:.3f} "
            f"estimated_win={estimated} "
            f"terminal_win={terminal} terminal_rate={evaluation.terminal_rate:.1%} "
            f"simulations={evaluation.simulations}"
        )
        if evaluation.risk_flags:
            lines.append(f"   risks: {', '.join(evaluation.risk_flags)}")
    lines.extend((f"推荐动作: {result.action}", f"推荐理由: {result.reason}"))
    if result.warnings:
        lines.append("WARNING:")
        lines.extend(f"  {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _write_log(
    path: Path,
    state: ObservableGameState,
    result: Phase4DecisionResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "phase4_recommendation",
        "state": state.to_log_payload(),
        **result.to_log_payload(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
        )


def _seat_choices() -> list[str]:
    return [seat.value for seat in PlayerSeat]


def _parse_turn_order(value: str) -> tuple[PlayerSeat, ...]:
    return tuple(PlayerSeat(part.strip()) for part in value.split(",") if part.strip())


def _coalesce(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
