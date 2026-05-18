from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.logic.decision import recommend_action
from src.logic.rules import Play, legal_actions
from src.state.cards import CardParseError
from src.state.game_state import GameStateSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dou Dizhu Phase 1 rule-engine MVP.")
    parser.add_argument("--hand", required=True, help="手牌，例如: '3 3 4 5 6 7 SJ BJ'")
    parser.add_argument("--last-play", default="", help="上一手牌；留空表示主动出牌")
    parser.add_argument("--log-file", default="logs/phase1.jsonl", help="JSONL 日志路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        state = GameStateSnapshot.from_inputs(args.hand, args.last_play)
        previous = Play.parse(state.last_play.cards)
        actions = legal_actions(state.hand, previous)
        decision = recommend_action(state)
    except (CardParseError, ValueError) as exc:
        print(f"输入错误: {exc}", file=sys.stderr)
        return 2

    _print_result(state, actions, decision)
    _write_log(Path(args.log_file), state, decision)
    return 0


def _print_result(state: GameStateSnapshot, actions, decision) -> None:
    non_pass = [action for action in actions if not action.is_pass]
    print(f"当前手牌: {state.hand}")
    print(f"上一手牌: {state.last_play}")
    print(f"候选动作数: {len(non_pass)}")
    print("可行动作:")
    for index, action in enumerate(actions, start=1):
        print(f"{index}. {action} ({action.type.value})")
    print(f"推荐动作: {decision.action}")
    print(f"推荐理由: {decision.reason}")


def _write_log(path: Path, state: GameStateSnapshot, decision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "recommendation",
        **state.to_log_payload(),
        **decision.to_log_payload(),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
