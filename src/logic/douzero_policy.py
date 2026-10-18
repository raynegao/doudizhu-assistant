"""Greedy DouZero policy over this project's legal actions.

The network definitions mirror DouZero's ``LandlordLstmModel`` /
``FarmerLstmModel`` (Kwai Inc., Apache-2.0) layer for layer and name for
name, so the published ``landlord.ckpt`` / ``landlord_down.ckpt`` /
``landlord_up.ckpt`` state dicts load unchanged. The policy never generates
moves itself: candidates come from ``src.logic.rules.legal_actions`` and the
model only ranks them, so the existing legality guarantees stay intact.

This module imports PyTorch; ``src.logic`` itself does not import it, and
the arena only loads this module when a DouZero agent is requested.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn

from src.logic.action_validation import validate_decision_state
from src.logic.douzero_features import FARMER_FEATURES, LANDLORD_FEATURES, encode, view_from_state
from src.logic.monte_carlo import ActionEvaluation, MonteCarloSettings, Phase4DecisionResult
from src.logic.rules import Play, can_beat, legal_actions
from src.state.observable_state import ObservableGameState

DOUZERO_MODEL_VERSION = "douzero-wp-greedy-v1"

CHECKPOINT_FILES = {
    "landlord": "landlord.ckpt",
    "landlord_down": "landlord_down.ckpt",
    "landlord_up": "landlord_up.ckpt",
}


class LstmModel(nn.Module):
    """DouZero's LSTM + six dense layers; parameter names match the checkpoints."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(features + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        return self.dense6(x)


def build_model(position: str) -> LstmModel:
    """A fresh DouZero network for ``position`` (random weights)."""
    if position not in CHECKPOINT_FILES:
        raise ValueError(f"unknown DouZero position: {position}")
    return LstmModel(LANDLORD_FEATURES if position == "landlord" else FARMER_FEATURES)


def load_model(position: str, checkpoint: Path) -> LstmModel:
    """Load a DouZero state dict, ignoring keys the network does not define (as DouZero does)."""
    model = build_model(position)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    expected = model.state_dict()
    filtered = {key: value for key, value in state.items() if key in expected}
    missing = set(expected) - set(filtered)
    if missing:
        raise ValueError(f"{checkpoint} lacks DouZero parameters: {sorted(missing)[:4]}...")
    expected.update(filtered)
    model.load_state_dict(expected)
    model.eval()
    return model


class DouZeroPolicy:
    """Score every legal action with the position's network and play the best."""

    def __init__(self, checkpoint_dir: Path | str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self._models: dict[str, LstmModel] = {}

    def model_for(self, position: str) -> LstmModel:
        if position not in self._models:
            path = self.checkpoint_dir / CHECKPOINT_FILES[position]
            if not path.is_file():
                raise FileNotFoundError(f"DouZero checkpoint missing: {path}")
            self._models[position] = load_model(position, path)
        return self._models[position]

    def score(self, state: ObservableGameState, actions: Sequence[Play]) -> list[float]:
        view = view_from_state(state)
        z_batch, x_batch = encode(view, actions)
        model = self.model_for(view.position)
        with torch.no_grad():
            values = model(torch.from_numpy(z_batch), torch.from_numpy(x_batch))
        return [float(value) for value in values.flatten().tolist()]

    def choose(self, state: ObservableGameState) -> Play:
        actions = legal_actions(state.self_hand, Play.parse(state.trick_target.cards))
        if not actions:
            raise ValueError("no legal actions are available")
        if len(actions) == 1:
            return actions[0]
        scores = self.score(state, actions)
        best = max(range(len(actions)), key=lambda index: (scores[index], -index))
        return actions[best]


def recommend_douzero(
    state: ObservableGameState,
    settings: MonteCarloSettings,
    *,
    policy: DouZeroPolicy,
) -> Phase4DecisionResult:
    """Rank every legal action by the DouZero value; ``top_k`` from ``settings``.

    DouZero-WP was trained on a +1/-1 outcome, so ``(value + 1) / 2`` is shown
    as a model-internal win estimate without an interval. It is not calibrated
    and the flags say so; the rule-proven finish keeps its exact treatment.
    """
    started = time.perf_counter()
    validate_decision_state(state)
    previous = Play.parse(state.trick_target.cards)
    actions = legal_actions(state.self_hand, previous)
    if not actions:
        raise ValueError("no legal actions are available")
    scores = policy.score(state, actions) if len(actions) > 1 else [0.0]
    hand_counts = Counter(state.self_hand.cards)
    finishes = [
        not action.is_pass
        and Counter(action.cards) == hand_counts
        and (not previous.cards or can_beat(action, previous))
        for action in actions
    ]
    # A rule-proven finish outranks any network value, as in the Monte Carlo path.
    order = sorted(range(len(actions)), key=lambda index: (not finishes[index], -scores[index], index))
    rankings = []
    for index in order[: settings.top_k]:
        action = actions[index]
        exact_finish = finishes[index]
        estimate = 1.0 if exact_finish else min(1.0, max(0.0, (scores[index] + 1.0) / 2.0))
        rankings.append(ActionEvaluation(
            action=action,
            strategy_score=round(estimate, 6),
            estimated_win_rate=round(estimate, 6),
            terminal_win_rate=None,
            terminal_rate=0.0,
            simulations=0,
            average_plies=0.0,
            average_margin=0.0,
            reason_factors=("exact_finish",) if exact_finish else ("douzero_argmax",),
            risk_flags=("learned_policy", "douzero_value_not_calibrated"),
            win_rate_estimate_kind="rule_proven_finish" if exact_finish else "douzero_value",
            win_rate_interval_method="rule_proven_finish" if exact_finish else None,
            win_rate_ci95_low=1.0 if exact_finish else None,
            win_rate_ci95_high=1.0 if exact_finish else None,
        ))
    best = rankings[0]
    reason = (
        f"DouZero 预训练网络对 {len(actions)} 个合法动作打分，{best.action} 的估值最高"
        f"（折算 {best.estimated_win_rate:.1%}，未校准）。"
    )
    if "exact_finish" in best.reason_factors:
        reason += " 该动作可以直接出完当前手牌。"
    return Phase4DecisionResult(
        action=best.action,
        rankings=tuple(rankings),
        all_candidate_count=len(actions),
        evaluated_candidate_count=len(actions),
        requested_simulations=0,
        completed_simulations=0,
        seed=settings.seed,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        reason=reason,
        warnings=tuple(dict.fromkeys((
            *state.warnings, "learned_policy_douzero_wp", "douzero_value_not_calibrated",
        ))),
        opponent_estimate=None,
        model_version=DOUZERO_MODEL_VERSION,
    )


def douzero_decision_fn(
    checkpoint_dir: Path | str,
) -> Callable[[ObservableGameState, MonteCarloSettings], Phase4DecisionResult]:
    """A live ``DecisionFn`` that loads the checkpoints on first use."""
    policy = DouZeroPolicy(checkpoint_dir)

    def decide(state: ObservableGameState, settings: MonteCarloSettings) -> Phase4DecisionResult:
        return recommend_douzero(state, settings, policy=policy)

    return decide


__all__ = [
    "CHECKPOINT_FILES",
    "DOUZERO_MODEL_VERSION",
    "DouZeroPolicy",
    "LstmModel",
    "build_model",
    "douzero_decision_fn",
    "load_model",
    "recommend_douzero",
]
