from __future__ import annotations

from dataclasses import dataclass

from src.logic.rules import Play, PlayType, legal_actions
from src.state.game_state import GameStateSnapshot


@dataclass(frozen=True)
class DecisionResult:
    action: Play
    candidates: tuple[Play, ...]
    reason: str
    warnings: tuple[str, ...] = ()

    def to_log_payload(self) -> dict[str, object]:
        return {
            "candidate_count": len([candidate for candidate in self.candidates if not candidate.is_pass]),
            "recommended_action": self.action.to_list(),
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def recommend_action(state: GameStateSnapshot) -> DecisionResult:
    previous = Play.parse(state.last_play.cards)
    candidates = tuple(legal_actions(state.hand, previous))
    non_pass = [candidate for candidate in candidates if not candidate.is_pass]

    if not non_pass:
        return DecisionResult(
            action=Play.parse(()),
            candidates=candidates,
            reason="没有可压过上一手的牌，建议过牌。",
        )

    if previous.type is PlayType.BOMB:
        rockets = [candidate for candidate in non_pass if candidate.type is PlayType.ROCKET]
        if rockets:
            return DecisionResult(
                action=rockets[0],
                candidates=candidates,
                reason="上一手是炸弹，火箭是唯一更高优先级的压制选择。",
            )

    exact_finish = [candidate for candidate in non_pass if len(candidate.cards) == len(state.hand)]
    if exact_finish:
        best_finish = _lowest_cost(exact_finish)
        return DecisionResult(
            action=best_finish,
            candidates=candidates,
            reason="该动作可以一次出完当前手牌，优先结束本轮。",
        )

    normal_actions = [
        candidate
        for candidate in non_pass
        if candidate.type not in {PlayType.BOMB, PlayType.ROCKET}
    ]
    if normal_actions:
        action = _lowest_cost(normal_actions)
        if previous.is_pass:
            reason = "主动出牌时选择较小且结构简单的动作，保留更强控制牌。"
        else:
            reason = "选择最小的普通可压制动作，尽量保留炸弹、火箭和高牌。"
        return DecisionResult(action=action, candidates=candidates, reason=reason)

    action = _lowest_cost(non_pass)
    if action.type is PlayType.ROCKET:
        reason = "只有火箭等强控制牌可用，使用它压制当前牌型。"
    else:
        reason = "只有炸弹等强控制牌可用，使用最低成本的强牌压制。"
    return DecisionResult(action=action, candidates=candidates, reason=reason)


def _lowest_cost(actions: list[Play]) -> Play:
    return sorted(actions, key=_decision_sort_key)[0]


def _decision_sort_key(play: Play) -> tuple[int, int, int, int]:
    type_penalty = {
        PlayType.SINGLE: 1,
        PlayType.PAIR: 2,
        PlayType.TRIPLE: 3,
        PlayType.TRIPLE_SINGLE: 4,
        PlayType.TRIPLE_PAIR: 5,
        PlayType.STRAIGHT: 6,
        PlayType.PAIR_STRAIGHT: 7,
        PlayType.PLANE: 8,
        PlayType.PLANE_SINGLE: 9,
        PlayType.PLANE_PAIR: 10,
        PlayType.FOUR_TWO_SINGLE: 11,
        PlayType.FOUR_TWO_PAIR: 12,
        PlayType.BOMB: 50,
        PlayType.ROCKET: 60,
        PlayType.PASS: 99,
        PlayType.INVALID: 100,
    }
    main_value = -1 if play.main_rank is None else _rank_value(play.main_rank)
    return (type_penalty[play.type], len(play.cards), main_value, play.combo_size)


def _rank_value(rank: str) -> int:
    from src.state.cards import RANK_VALUE

    return RANK_VALUE[rank]


__all__ = ["DecisionResult", "recommend_action"]
