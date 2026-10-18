"""The DouZero policy loads DouZero-shaped checkpoints and only ever plays legal actions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.logic.douzero_policy import CHECKPOINT_FILES, DouZeroPolicy, build_model, load_model
from src.logic.rules import Play, legal_actions
from src.state.observable_state import ObservableGameState

torch = pytest.importorskip("torch")


def _write_random_checkpoints(directory: Path, seed: int = 1) -> None:
    torch.manual_seed(seed)
    for position, name in CHECKPOINT_FILES.items():
        torch.save(build_model(position).state_dict(), directory / name)


def test_models_match_douzero_parameter_names_and_shapes() -> None:
    landlord = build_model("landlord").state_dict()
    farmer = build_model("landlord_up").state_dict()
    assert landlord["dense1.weight"].shape == (512, 373 + 128)
    assert farmer["dense1.weight"].shape == (512, 484 + 128)
    assert landlord["lstm.weight_ih_l0"].shape == (512, 162)
    assert landlord["dense6.weight"].shape == (1, 512)
    assert set(landlord) == {
        "lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
        *(f"dense{i}.{kind}" for i in range(1, 7) for kind in ("weight", "bias")),
    }
    with pytest.raises(ValueError):
        build_model("dealer")


def test_checkpoints_load_and_extra_keys_are_ignored(tmp_path: Path) -> None:
    _write_random_checkpoints(tmp_path)
    state = torch.load(tmp_path / "landlord.ckpt", weights_only=True)
    state["unused.extra"] = torch.zeros(1)
    torch.save(state, tmp_path / "landlord.ckpt")
    model = load_model("landlord", tmp_path / "landlord.ckpt")
    assert not model.training
    torch.save({"dense1.weight": torch.zeros(512, 501)}, tmp_path / "broken.ckpt")
    with pytest.raises(ValueError):
        load_model("landlord", tmp_path / "broken.ckpt")


def test_policy_plays_a_legal_action_for_every_position(tmp_path: Path) -> None:
    _write_random_checkpoints(tmp_path)
    policy = DouZeroPolicy(tmp_path)
    for landlord in ("self", "right", "left"):
        hand = "3 3 4 5 6 7 8 K A 2" if landlord != "self" else "3 3 4 5 6 7 8 K A 2 2 9 9 Q Q J 10 10 BJ 5"
        state = ObservableGameState.from_inputs(hand, landlord=landlord, current_actor="self")
        play = policy.choose(state)
        assert play in legal_actions(state.self_hand, Play.parse(()))
        scores = policy.score(state, legal_actions(state.self_hand, Play.parse(())))
        assert len(scores) == len(legal_actions(state.self_hand, Play.parse(())))
    # Following with a single legal answer (pass) skips the network.
    state = ObservableGameState.from_inputs("3", landlord="self", current_actor="self", last_play="2", last_player="right")
    assert policy.choose(state).is_pass
    with pytest.raises(FileNotFoundError):
        DouZeroPolicy(tmp_path / "missing").choose(ObservableGameState.from_inputs("3 4", landlord="self"))


def test_recommend_douzero_builds_a_phase4_result(tmp_path: Path) -> None:
    from src.logic.douzero_policy import (
        DOUZERO_MODEL_VERSION,
        douzero_decision_fn,
        recommend_douzero,
    )
    from src.logic.monte_carlo import MonteCarloSettings

    _write_random_checkpoints(tmp_path)
    policy = DouZeroPolicy(tmp_path)
    state = ObservableGameState.from_inputs("3 3 4 5 6 7 8 K A 2", landlord="self")
    result = recommend_douzero(state, MonteCarloSettings(top_k=3, seed=5), policy=policy)
    assert result.model_version == DOUZERO_MODEL_VERSION
    assert len(result.rankings) == 3 and result.action == result.rankings[0].action
    scores = [evaluation.estimated_win_rate for evaluation in result.rankings]
    assert scores == sorted(scores, reverse=True) and all(0.0 <= s <= 1.0 for s in scores)
    assert result.rankings[0].win_rate_estimate_kind == "douzero_value"
    assert result.rankings[0].win_rate_ci95_low is None
    assert "douzero_value_not_calibrated" in result.warnings and result.completed_simulations == 0
    assert "DouZero" in result.reason and result.seed == 5
    legal = {play for play in legal_actions(state.self_hand, Play.parse(()))}
    assert all(evaluation.action in legal for evaluation in result.rankings)
    # An exact finish stays rule-proven at 100% regardless of the network.
    finish = ObservableGameState.from_inputs("9 9", landlord="self", current_actor="self", last_play="8 8", last_player="left")
    proven = recommend_douzero(finish, MonteCarloSettings(top_k=2), policy=policy)
    assert proven.action.cards == ("9", "9") and proven.rankings[0].estimated_win_rate == 1.0
    assert proven.rankings[0].win_rate_estimate_kind == "rule_proven_finish"
    decide = douzero_decision_fn(tmp_path)
    assert decide(state, MonteCarloSettings()).action in legal


def test_live_layout_and_runtime_select_the_douzero_backend(tmp_path: Path) -> None:
    from src.config.live_layout import LiveLayoutConfig, live_layout_from_dict
    from src.logic.monte_carlo import recommend_phase4
    from src.pipeline.live_runtime import select_decision_fn

    assert LiveLayoutConfig().decision_backend == "monte_carlo"
    config = live_layout_from_dict({"decision_backend": "douzero", "douzero_checkpoint_dir": str(tmp_path)})
    assert config.to_payload()["decision_backend"] == "douzero"
    assert select_decision_fn(LiveLayoutConfig(), recommend_phase4) is recommend_phase4
    custom = lambda state, settings: None  # noqa: E731
    assert select_decision_fn(config, custom) is custom
    _write_random_checkpoints(tmp_path)
    chosen = select_decision_fn(config, recommend_phase4)
    assert chosen is not recommend_phase4
    state = ObservableGameState.from_inputs("3 4 5 6 7", landlord="self")
    from src.logic.monte_carlo import MonteCarloSettings
    assert chosen(state, MonteCarloSettings()).rankings[0].action in legal_actions(state.self_hand, Play.parse(()))
    with pytest.raises(ValueError):
        LiveLayoutConfig(decision_backend="random")
    empty = live_layout_from_dict({"decision_backend": "douzero", "douzero_checkpoint_dir": str(tmp_path / "none")})
    with pytest.raises(FileNotFoundError):
        select_decision_fn(empty, recommend_phase4)
