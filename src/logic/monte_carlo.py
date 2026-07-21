from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Mapping, Protocol, Sequence

from src.logic.action_validation import validate_decision_state
from src.logic.decision import recommend_action
from src.logic.opponent_model import (
    OpponentDeal,
    OpponentEstimate,
    OpponentModelError,
    UniformOpponentModel,
)
from src.logic.rules import Play, PlayType, can_beat, legal_actions
from src.state.cards import RANK_VALUE, CardSet, sort_cards
from src.state.events import PlayerSeat
from src.state.observable_state import ObservableGameState


RULESET_VERSION = "phase4-standard-subset-v1"


@dataclass(frozen=True)
class MonteCarloSettings:
    simulations: int = 200
    max_depth: int = 50
    time_budget_ms: int = 500
    seed: int = 20260721
    top_k: int = 3
    max_candidates: int = 12
    min_rollouts_per_action: int = 8

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.time_budget_ms < 0:
            raise ValueError("time_budget_ms cannot be negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.top_k > self.max_candidates:
            raise ValueError("top_k cannot exceed max_candidates")
        if self.min_rollouts_per_action <= 0:
            raise ValueError("min_rollouts_per_action must be positive")


@dataclass(frozen=True)
class RolloutOutcome:
    expected_result: float
    terminal_result: float | None
    plies: int
    margin: float


@dataclass(frozen=True)
class ActionEvaluation:
    action: Play
    strategy_score: float
    estimated_win_rate: float | None
    terminal_win_rate: float | None
    terminal_rate: float
    simulations: int
    average_plies: float
    average_margin: float
    reason_factors: tuple[str, ...]
    risk_flags: tuple[str, ...]

    def to_log_payload(self) -> dict[str, object]:
        return {
            "action": self.action.to_list(),
            "play_type": self.action.type.value,
            "strategy_score": self.strategy_score,
            "estimated_win_rate": self.estimated_win_rate,
            "terminal_win_rate": self.terminal_win_rate,
            "terminal_rate": self.terminal_rate,
            "simulations": self.simulations,
            "average_plies": self.average_plies,
            "average_margin": self.average_margin,
            "reason_factors": list(self.reason_factors),
            "risk_flags": list(self.risk_flags),
        }


@dataclass(frozen=True)
class Phase4DecisionResult:
    action: Play
    rankings: tuple[ActionEvaluation, ...]
    all_candidate_count: int
    evaluated_candidate_count: int
    requested_simulations: int
    completed_simulations: int
    seed: int
    elapsed_ms: float
    reason: str
    warnings: tuple[str, ...]
    opponent_estimate: OpponentEstimate | None
    model_version: str = "phase4-monte-carlo-v1"
    ruleset_version: str = RULESET_VERSION

    @property
    def top_k(self) -> tuple[ActionEvaluation, ...]:
        return self.rankings

    def to_log_payload(self) -> dict[str, object]:
        return {
            "candidate_count": self.all_candidate_count,
            "evaluated_candidate_count": self.evaluated_candidate_count,
            "recommended_action": self.action.to_list(),
            "requested_simulations": self.requested_simulations,
            "completed_simulations": self.completed_simulations,
            "seed": self.seed,
            "elapsed_ms": self.elapsed_ms,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "model_version": self.model_version,
            "ruleset_version": self.ruleset_version,
            "top_k": [evaluation.to_log_payload() for evaluation in self.rankings],
            "opponent_estimate": (
                self.opponent_estimate.to_log_payload()
                if self.opponent_estimate is not None
                else None
            ),
        }


class RolloutPolicy(Protocol):
    def choose_action(
        self,
        *,
        player: PlayerSeat,
        hand: CardSet,
        previous: Play,
        last_player: PlayerSeat | None,
        landlord: PlayerSeat,
        remaining_cards: Mapping[PlayerSeat, int],
    ) -> Play: ...


class HeuristicRolloutPolicy:
    def choose_action(
        self,
        *,
        player: PlayerSeat,
        hand: CardSet,
        previous: Play,
        last_player: PlayerSeat | None,
        landlord: PlayerSeat,
        remaining_cards: Mapping[PlayerSeat, int],
    ) -> Play:
        actions = legal_actions(hand, previous)
        non_pass = [action for action in actions if not action.is_pass]
        if not non_pass:
            return Play.parse(())

        exact_finish = [action for action in non_pass if len(action.cards) == len(hand)]
        if exact_finish:
            return sorted(exact_finish, key=_stable_action_key)[0]

        pass_action = next((action for action in actions if action.is_pass), None)
        if (
            pass_action is not None
            and last_player is not None
            and _same_team(player, last_player, landlord)
            and player is not landlord
        ):
            return pass_action

        normal = [
            action
            for action in non_pass
            if action.type not in {PlayType.BOMB, PlayType.ROCKET}
        ]
        if normal:
            return sorted(
                normal,
                key=lambda action: (-len(action.cards), *_stable_action_key(action)),
            )[0]

        urgent = any(
            count <= 2
            for seat, count in remaining_cards.items()
            if not _same_team(player, seat, landlord)
        )
        if urgent:
            return sorted(non_pass, key=_stable_action_key)[0]
        if pass_action is not None:
            return pass_action
        return sorted(non_pass, key=_stable_action_key)[0]


class MonteCarloEvaluator:
    def __init__(
        self,
        *,
        opponent_model: UniformOpponentModel | None = None,
        policy: RolloutPolicy | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.opponent_model = opponent_model or UniformOpponentModel()
        self.policy = policy or HeuristicRolloutPolicy()
        self.clock = clock

    def evaluate(
        self,
        state: ObservableGameState,
        settings: MonteCarloSettings,
    ) -> Phase4DecisionResult:
        started = self.clock()
        validate_decision_state(state)
        self.opponent_model.validate_state(state)
        previous = Play.parse(state.trick_target.cards)
        all_candidates = tuple(legal_actions(state.self_hand, previous))
        if not all_candidates:
            raise ValueError("no legal actions are available")
        candidates = _select_candidates(
            state,
            all_candidates,
            max_candidates=settings.max_candidates,
        )

        deadline = (
            started + settings.time_budget_ms / 1000
            if settings.time_budget_ms > 0
            else math.inf
        )
        outcomes: dict[Play, list[RolloutOutcome]] = {
            candidate: [] for candidate in candidates
        }
        completed_worlds: list[OpponentDeal] = []
        rng = random.Random(settings.seed)
        deadline_reached = False

        for _ in range(settings.simulations):
            world = self.opponent_model.sample(state, rng)
            for candidate in candidates:
                outcomes[candidate].append(
                    _simulate_candidate(
                        state,
                        world,
                        candidate,
                        max_depth=settings.max_depth,
                        policy=self.policy,
                    )
                )
            completed_worlds.append(world)
            if self.clock() >= deadline:
                deadline_reached = True
                break

        completed = len(completed_worlds)
        warnings = list(state.warnings)
        if len(candidates) < len(all_candidates):
            warnings.append(
                f"candidate_pruned: evaluated {len(candidates)} of {len(all_candidates)} legal actions"
            )
        if completed < settings.simulations:
            warnings.append(
                f"time_budget_exhausted: completed {completed} of {settings.simulations} sampled worlds"
            )
        elif deadline_reached and settings.time_budget_ms > 0:
            warnings.append("time_budget_overrun: final complete world crossed the deadline")
        if completed < settings.min_rollouts_per_action:
            warnings.append(
                f"low_sample_count: only {completed} rollouts completed per action"
            )
        warnings.extend(("uniform_opponent_model", "rule_subset_only"))

        evaluations = tuple(
            _aggregate_action(
                state,
                candidate,
                values,
                global_risks=tuple(warnings),
            )
            for candidate, values in outcomes.items()
        )
        ordered = tuple(sorted(evaluations, key=_evaluation_sort_key))
        top = ordered[: min(settings.top_k, len(ordered))]
        best = top[0]
        reason = _recommendation_reason(best, completed)
        estimate = self.opponent_model.summarize(state, tuple(completed_worlds))
        elapsed_ms = (self.clock() - started) * 1000
        return Phase4DecisionResult(
            action=best.action,
            rankings=top,
            all_candidate_count=len(all_candidates),
            evaluated_candidate_count=len(candidates),
            requested_simulations=settings.simulations,
            completed_simulations=completed,
            seed=settings.seed,
            elapsed_ms=elapsed_ms,
            reason=reason,
            warnings=tuple(dict.fromkeys(warnings)),
            opponent_estimate=estimate,
        )


def recommend_phase4(
    state: ObservableGameState,
    settings: MonteCarloSettings | None = None,
    *,
    evaluator: MonteCarloEvaluator | None = None,
) -> Phase4DecisionResult:
    settings = settings or MonteCarloSettings()
    validate_decision_state(state)
    engine = evaluator or MonteCarloEvaluator()
    try:
        return engine.evaluate(state, settings)
    except OpponentModelError as exc:
        fallback = recommend_action(state.to_decision_snapshot())
        evaluation = ActionEvaluation(
            action=fallback.action,
            strategy_score=0.0,
            estimated_win_rate=None,
            terminal_win_rate=None,
            terminal_rate=0.0,
            simulations=0,
            average_plies=0.0,
            average_margin=0.0,
            reason_factors=("deterministic_fallback",),
            risk_flags=("incomplete_observable_state",),
        )
        warning = f"monte_carlo_unavailable: {exc}"
        return Phase4DecisionResult(
            action=fallback.action,
            rankings=(evaluation,),
            all_candidate_count=len(fallback.candidates),
            evaluated_candidate_count=1,
            requested_simulations=settings.simulations,
            completed_simulations=0,
            seed=settings.seed,
            elapsed_ms=0.0,
            reason=f"状态信息不足，回退到确定性策略：{fallback.reason}",
            warnings=tuple(dict.fromkeys((*state.warnings, warning))),
            opponent_estimate=None,
        )


def _simulate_candidate(
    state: ObservableGameState,
    deal: OpponentDeal,
    candidate: Play,
    *,
    max_depth: int,
    policy: RolloutPolicy,
) -> RolloutOutcome:
    hands: dict[PlayerSeat, CardSet] = {PlayerSeat.SELF: state.self_hand}
    hands.update({seat: hand for seat, hand in deal.hands})
    remaining = {seat: len(hand) for seat, hand in hands.items()}
    current = PlayerSeat.SELF
    previous = Play.parse(state.trick_target.cards)
    last_player = state.trick_leader
    passes = state.consecutive_passes

    def apply(action: Play) -> PlayerSeat | None:
        nonlocal current, previous, last_player, passes
        actor = current
        missing = Counter(action.cards) - Counter(hands[actor].cards)
        action_is_legal = (
            bool(previous.cards)
            if action.is_pass
            else (
                not missing
                and action.type is not PlayType.INVALID
                and (not previous.cards or can_beat(action, previous))
            )
        )
        if not action_is_legal:
            raise RuntimeError(
                f"rollout policy returned an illegal action for {actor.value}: {action}"
            )
        if action.is_pass:
            passes += 1
            if passes == 2:
                if last_player is None:
                    raise RuntimeError("rollout trick lost its last non-pass player")
                current = last_player
                previous = Play.parse(())
                last_player = None
                passes = 0
            else:
                current = state.next_player(actor)
            return None

        hand_counts = Counter(hands[actor].cards)
        hand_counts.subtract(action.cards)
        hands[actor] = CardSet(
            sort_cards(card for rank in hand_counts for card in [rank] * hand_counts[rank])
        )
        remaining[actor] = len(hands[actor])
        previous = action
        last_player = actor
        passes = 0
        if remaining[actor] == 0:
            return actor
        current = state.next_player(actor)
        return None

    winner = apply(candidate)
    if winner is not None:
        return _terminal_outcome(
            state,
            winner=winner,
            plies=1,
            remaining=remaining,
        )

    for depth in range(1, max_depth):
        action = policy.choose_action(
            player=current,
            hand=hands[current],
            previous=previous,
            last_player=last_player,
            landlord=state.landlord,
            remaining_cards=remaining,
        )
        winner = apply(action)
        if winner is not None:
            return _terminal_outcome(
                state,
                winner=winner,
                plies=depth + 1,
                remaining=remaining,
            )

    margin = _team_margin(state, remaining)
    expected = max(0.0, min(1.0, 0.5 + 0.5 * margin))
    return RolloutOutcome(
        expected_result=expected,
        terminal_result=None,
        plies=max_depth,
        margin=margin,
    )


def _terminal_outcome(
    state: ObservableGameState,
    *,
    winner: PlayerSeat,
    plies: int,
    remaining: Mapping[PlayerSeat, int],
) -> RolloutOutcome:
    won = _same_team(PlayerSeat.SELF, winner, state.landlord)
    return RolloutOutcome(
        expected_result=1.0 if won else 0.0,
        terminal_result=1.0 if won else 0.0,
        plies=plies,
        margin=_team_margin(state, remaining),
    )


def _team_margin(
    state: ObservableGameState,
    remaining: Mapping[PlayerSeat, int],
) -> float:
    def progress(seat: PlayerSeat) -> float:
        baseline = state.remaining_for(seat)
        if baseline <= 0:
            return 1.0
        value = (baseline - remaining[seat]) / baseline
        return max(0.0, min(1.0, value))

    landlord_progress = progress(state.landlord)
    farmer_progress = max(
        progress(seat) for seat in remaining if seat is not state.landlord
    )
    if state.landlord is PlayerSeat.SELF:
        return landlord_progress - farmer_progress
    return farmer_progress - landlord_progress


def _aggregate_action(
    state: ObservableGameState,
    action: Play,
    outcomes: Sequence[RolloutOutcome],
    *,
    global_risks: tuple[str, ...],
) -> ActionEvaluation:
    if not outcomes:
        raise ValueError("cannot aggregate zero rollout outcomes")
    simulations = len(outcomes)
    estimated_win_rate = sum(value.expected_result for value in outcomes) / simulations
    terminal_values = [
        value.terminal_result
        for value in outcomes
        if value.terminal_result is not None
    ]
    terminal_rate = len(terminal_values) / simulations
    terminal_win_rate = (
        sum(terminal_values) / len(terminal_values) if terminal_values else None
    )
    average_plies = sum(value.plies for value in outcomes) / simulations
    average_margin = sum(value.margin for value in outcomes) / simulations

    shed_score = len(action.cards) / max(1, len(state.self_hand))
    residual = Counter(state.self_hand.cards) - Counter(action.cards)
    residual_hand = CardSet(
        sort_cards(card for rank in residual for card in [rank] * residual[rank])
    )
    if not residual_hand:
        structure_score = 1.0
    else:
        residual_actions = legal_actions(residual_hand, Play.parse(()))
        longest = max((len(candidate.cards) for candidate in residual_actions), default=0)
        structure_score = longest / len(residual_hand)
    preserves_control = 0.0 if action.type in {PlayType.BOMB, PlayType.ROCKET} else 1.0
    if not residual_hand:
        preserves_control = 1.0
    strategy_score = max(
        0.0,
        min(
            1.0,
            0.80 * estimated_win_rate
            + 0.10 * shed_score
            + 0.05 * structure_score
            + 0.05 * preserves_control,
        ),
    )

    reasons: list[str] = []
    if not residual_hand:
        reasons.append("exact_finish")
    if len(action.cards) >= max(1, len(state.self_hand) // 2):
        reasons.append("sheds_many_cards")
    if preserves_control:
        reasons.append("preserves_bomb_or_rocket")
    risks = [risk for risk in global_risks if risk in {
        "uniform_opponent_model",
        "rule_subset_only",
    }]
    if terminal_rate < 1.0:
        risks.append("depth_limit_reached")
    if action.type in {PlayType.BOMB, PlayType.ROCKET} and residual_hand:
        risks.append("bomb_commitment")

    return ActionEvaluation(
        action=action,
        strategy_score=round(strategy_score, 6),
        estimated_win_rate=round(estimated_win_rate, 6),
        terminal_win_rate=(
            round(terminal_win_rate, 6) if terminal_win_rate is not None else None
        ),
        terminal_rate=round(terminal_rate, 6),
        simulations=simulations,
        average_plies=round(average_plies, 3),
        average_margin=round(average_margin, 3),
        reason_factors=tuple(reasons),
        risk_flags=tuple(dict.fromkeys(risks)),
    )


def _select_candidates(
    state: ObservableGameState,
    candidates: tuple[Play, ...],
    *,
    max_candidates: int,
) -> tuple[Play, ...]:
    if len(candidates) <= max_candidates:
        return candidates

    ordered: list[Play] = []

    def add(action: Play | None) -> None:
        if action is not None and action not in ordered and len(ordered) < max_candidates:
            ordered.append(action)

    exact = [action for action in candidates if len(action.cards) == len(state.self_hand)]
    for action in sorted(exact, key=_stable_action_key):
        add(action)
    add(recommend_action(state.to_decision_snapshot()).action)
    add(next((action for action in candidates if action.is_pass), None))
    for play_type in (PlayType.BOMB, PlayType.ROCKET):
        strong = [action for action in candidates if action.type is play_type]
        add(sorted(strong, key=_stable_action_key)[0] if strong else None)

    normal = [
        action
        for action in candidates
        if action.type not in {PlayType.PASS, PlayType.BOMB, PlayType.ROCKET}
    ]
    for action in sorted(
        normal,
        key=lambda value: (-len(value.cards), *_stable_action_key(value)),
    ):
        add(action)
    for action in sorted(candidates, key=_stable_action_key):
        add(action)
    return tuple(ordered)


def _same_team(
    left: PlayerSeat,
    right: PlayerSeat,
    landlord: PlayerSeat,
) -> bool:
    return (left is landlord) == (right is landlord)


def _stable_action_key(play: Play) -> tuple[int, int, int, tuple[int, ...]]:
    type_order = list(PlayType).index(play.type)
    main_value = RANK_VALUE[play.main_rank] if play.main_rank else -1
    cards = tuple(RANK_VALUE[card] for card in play.cards)
    return (type_order, len(play.cards), main_value, cards)


def _evaluation_sort_key(
    evaluation: ActionEvaluation,
) -> tuple[float, float, float, tuple[int, int, int, tuple[int, ...]]]:
    return (
        -evaluation.strategy_score,
        -(
            evaluation.estimated_win_rate
            if evaluation.estimated_win_rate is not None
            else -1.0
        ),
        -evaluation.average_margin,
        _stable_action_key(evaluation.action),
    )


def _recommendation_reason(evaluation: ActionEvaluation, simulations: int) -> str:
    if evaluation.estimated_win_rate is None:
        raise ValueError("simulated recommendation must include an estimated win rate")
    action_text = str(evaluation.action)
    reason = (
        f"在每个候选共用的 {simulations} 组对手牌样本中，{action_text} 的估计己方胜率为 "
        f"{evaluation.estimated_win_rate:.1%}，策略分为 {evaluation.strategy_score:.3f}。"
    )
    if "exact_finish" in evaluation.reason_factors:
        reason += " 该动作可以直接出完当前手牌。"
    elif "sheds_many_cards" in evaluation.reason_factors:
        reason += " 该动作能一次减少较多手牌。"
    if "bomb_commitment" in evaluation.risk_flags:
        reason += " 风险：会消耗炸弹或火箭控制力。"
    return reason


__all__ = [
    "ActionEvaluation",
    "HeuristicRolloutPolicy",
    "MonteCarloEvaluator",
    "MonteCarloSettings",
    "Phase4DecisionResult",
    "RULESET_VERSION",
    "RolloutOutcome",
    "RolloutPolicy",
    "recommend_phase4",
]
