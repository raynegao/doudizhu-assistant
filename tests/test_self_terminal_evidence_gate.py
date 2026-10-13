"""No inference, images, or new Scene fields are needed by the last-play gate."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from src.logic.action_validation import validate_observed_action
from src.state.cards import CardSet
from src.state.events import ObservedAction, PlayerSeat, RoundPhase
from src.state.game_tracker import reduce_observed_action
from src.state.observable_state import ObservableGameState
from src.tracking.self_terminal_evidence import SelfTerminalEvidenceGate
from src.vision.scene_types import SceneObservation, VisualCard


def _state(hand: str = "BJ", target: str = "") -> ObservableGameState:
    return ObservableGameState.from_inputs(
        hand, round_id="self-terminal", current_actor=PlayerSeat.SELF,
        landlord=PlayerSeat.SELF, last_play=target,
        last_player=PlayerSeat.LEFT if target else None,
        remaining_cards={PlayerSeat.SELF: len(CardSet.parse(hand)),
                         PlayerSeat.LEFT: 8, PlayerSeat.RIGHT: 9},
        hidden_played_count=54 - len(CardSet.parse(hand)) - len(CardSet.parse(target)) - 17,
    )


def _cards(hand: str = "BJ", confidence: float = 0.97) -> tuple[VisualCard, ...]:
    return tuple(VisualCard(rank, confidence, (20 * i, 0, 20 * i + 30, 60))
                 for i, rank in enumerate(hand.split()))


def _scene(frame: int) -> SceneObservation:
    return SceneObservation(frame, float(frame), (0, 0, 200, 100), (), (), None, 0.0, 0.97)


def _observe(
    gate: SelfTerminalEvidenceGate,
    frame: int,
    *,
    state: ObservableGameState | None = None,
    cards: tuple[VisualCard, ...] | None = None,
    scene: SceneObservation | None = None,
    complete: bool = True,
    empty_hand_verified: bool = True,
    epoch: int | None = None,
    stability_frames: int = 3,
    confidence_threshold: float = 0.9,
) -> ObservedAction | None:
    return gate.observe(
        state=_state() if state is None else state,
        scene=_scene(frame) if scene is None else scene,
        cards=_cards() if cards is None else cards,
        complete=complete, empty_hand_verified=empty_hand_verified,
        epoch=epoch, stability_frames=stability_frames,
        confidence_threshold=confidence_threshold,
    )


@pytest.mark.parametrize("hand,target", [("BJ", "SJ"), ("SJ BJ", "2 2 2 2"),
                                        ("9 9", "8 8"), ("3 3", "")])
@pytest.mark.parametrize("required", [1, 2, 3, 5])
def test_exactly_n_fresh_observations_finish_only_self(hand: str, target: str, required: int) -> None:
    state = _state(hand, target)
    gate = SelfTerminalEvidenceGate()
    for frame in range(1, required):
        assert _observe(gate, frame, state=state, cards=_cards(hand),
                        stability_frames=required) is None
    event = _observe(gate, required, state=state, cards=_cards(hand),
                     stability_frames=required)
    assert event is not None
    assert event.cards == state.self_hand
    assert event.confidence == 0.97
    assert event.source == "live_self_table_terminal"
    assert event.actor is PlayerSeat.SELF
    assert event.sequence_no == state.last_sequence_no + 1
    updated = reduce_observed_action(state, event, validator=validate_observed_action).state
    assert updated.phase is RoundPhase.FINISHED
    assert updated.winner is PlayerSeat.SELF
    assert state.phase is RoundPhase.PLAYING  # Preview never mutates the caller.


@pytest.mark.parametrize("hand,target", [("3 4", ""), ("3 3", "A A"), ("SJ", "BJ")])
def test_matching_whole_hand_still_must_be_legal(hand: str, target: str) -> None:
    gate = SelfTerminalEvidenceGate()
    for frame in range(1, 7):
        assert _observe(gate, frame, state=_state(hand, target), cards=_cards(hand)) is None


@pytest.mark.parametrize("cards", [(), _cards("SJ"), _cards("BJ BJ"), _cards("BJ A"),
                                   _cards("invalid"), _cards("A A A A A")])
def test_missing_extra_wrong_or_invalid_cards_fail_closed(cards: tuple[VisualCard, ...]) -> None:
    gate = SelfTerminalEvidenceGate()
    for frame in range(1, 5):
        assert _observe(gate, frame, cards=cards) is None


@pytest.mark.parametrize("confidence", [0.899999, -0.1, 1.01, float("nan"),
                                        float("inf"), -float("inf"), True])
def test_every_slot_requires_finite_in_range_confidence(confidence: float) -> None:
    gate = SelfTerminalEvidenceGate()
    cards = (_cards("A")[0], replace(_cards("A")[0], confidence=confidence))
    for frame in range(1, 5):
        assert _observe(gate, frame, state=_state("A A"), cards=cards) is None


@pytest.mark.parametrize("problem", ["incomplete", "unverified_empty", "remaining_hand",
                                     "turn_true", "geometry_unknown", "wrong_actor",
                                     "uncertain", "finished", "inconsistent_turn"])
def test_ineligible_frame_breaks_support(problem: str) -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 1) is None
    assert _observe(gate, 2) is None
    state = _state()
    scene = _scene(3)
    if problem == "remaining_hand":
        scene = replace(scene, self_hand=_cards())
    elif problem == "turn_true":
        scene = replace(scene, self_turn=True, self_turn_confidence=0.0)
    elif problem == "geometry_unknown":
        scene = replace(scene, warnings=("self_hand_geometry_unavailable: test",))
    elif problem == "wrong_actor":
        state = replace(state, current_actor=PlayerSeat.RIGHT)
    elif problem == "uncertain":
        state = replace(state, phase=RoundPhase.UNCERTAIN)
    elif problem == "finished":
        state = _state("")
    elif problem == "inconsistent_turn":
        state = replace(_state("BJ", "A"), trick_leader=PlayerSeat.RIGHT)
    assert _observe(gate, 3, state=state, scene=scene,
                    complete=problem != "incomplete",
                    empty_hand_verified=problem != "unverified_empty") is None
    assert _observe(gate, 4) is None
    assert _observe(gate, 5) is None
    assert _observe(gate, 6) is not None


@pytest.mark.parametrize("stale_frame", [2, 1, 0])
def test_duplicate_or_reversed_frame_interrupts_and_preserves_watermark(stale_frame: int) -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 1) is None
    assert _observe(gate, 2) is None
    assert _observe(gate, stale_frame) is None
    assert _observe(gate, 2) is None
    assert _observe(gate, 3) is None
    assert _observe(gate, 4) is None
    assert _observe(gate, 5) is not None


def test_reset_and_bad_frame_do_not_make_old_source_ids_fresh() -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 10) is None
    assert _observe(gate, 11) is None
    gate.reset()
    assert _observe(gate, 9) is None
    assert _observe(gate, 10) is None
    assert _observe(gate, 11) is None
    assert _observe(gate, 20, complete=False) is None
    assert _observe(gate, 19) is None
    assert _observe(gate, 20) is None
    assert _observe(gate, 21) is None
    assert _observe(gate, 22) is None
    assert _observe(gate, 23) is not None


def test_epoch_is_a_monotonic_frame_floor_and_cannot_be_cleared_by_reset() -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 10, epoch=10) is None
    assert _observe(gate, 11, epoch=10) is None
    assert _observe(gate, 12, epoch=20) is None
    gate.reset()
    assert _observe(gate, 19, epoch=20) is None
    assert _observe(gate, 20, epoch=10) is None
    assert _observe(gate, 21, epoch=None) is None
    assert _observe(gate, 22, epoch=20) is None
    assert _observe(gate, 23, epoch=20) is None
    assert _observe(gate, 24, epoch=20) is not None


@pytest.mark.parametrize("change", ["revision", "round", "old_hand", "sequence", "target",
                                    "epoch", "required", "threshold", "turn"])
def test_new_state_or_eligibility_context_cannot_borrow_support(change: str) -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 1) is None
    assert _observe(gate, 2) is None
    state = _state()
    scene = _scene(3)
    if change == "revision":
        state = replace(state, revision=1)
    elif change == "round":
        state = replace(state, round_id="next")
    elif change == "old_hand":
        state = _state("A")
    elif change == "sequence":
        state = replace(state, last_sequence_no=5)
    elif change == "target":
        state = _state("BJ", "A")
    elif change == "turn":
        scene = replace(scene, self_turn=False)
    assert _observe(
        gate, 3, state=state, scene=scene,
        cards=_cards("A") if change == "old_hand" else _cards(),
        epoch=3 if change == "epoch" else None,
        stability_frames=4 if change == "required" else 3,
        confidence_threshold=0.8 if change == "threshold" else 0.9,
    ) is None


@pytest.mark.parametrize("change", ["order", "geometry"])
def test_slot_fingerprint_preserves_order_and_boxes(change: str) -> None:
    gate = SelfTerminalEvidenceGate()
    cards = _cards("SJ BJ")
    state = _state("SJ BJ")
    assert _observe(gate, 1, state=state, cards=cards) is None
    assert _observe(gate, 2, state=state, cards=cards) is None
    changed = tuple(reversed(cards)) if change == "order" else (
        replace(cards[0], box=(1, 0, 31, 60)), cards[1],
    )
    assert _observe(gate, 3, state=state, cards=changed) is None
    assert _observe(gate, 4, state=state, cards=changed) is None
    assert _observe(gate, 5, state=state, cards=changed) is not None


def test_confidence_is_minimum_of_all_original_supporting_slot_scores() -> None:
    gate = SelfTerminalEvidenceGate()
    for frame, confidence in ((1, 0.912345678901), (2, 0.98)):
        assert _observe(gate, frame, cards=_cards(confidence=confidence)) is None
    event = _observe(gate, 3, cards=_cards(confidence=0.95))
    assert event is not None
    assert event.confidence == 0.912345678901


@pytest.mark.parametrize("required", [1, 3])
def test_emitted_revision_is_not_emitted_again_even_after_reset(required: int) -> None:
    gate = SelfTerminalEvidenceGate()
    for frame in range(1, required):
        assert _observe(gate, frame, stability_frames=required) is None
    assert _observe(gate, required, stability_frames=required) is not None
    gate.reset()
    for frame in range(required, 3 * required + 2):
        assert _observe(gate, frame, stability_frames=required) is None


@pytest.mark.parametrize("required", [0, -1, True, 1.5])
def test_invalid_stability_is_rejected(required: int) -> None:
    with pytest.raises(ValueError, match="invalid self-terminal"):
        _observe(SelfTerminalEvidenceGate(), 1, stability_frames=required)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="invalid self-terminal"):
        _observe(SelfTerminalEvidenceGate(), 1, confidence_threshold=threshold)


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_invalid_epoch_is_rejected(epoch: int) -> None:
    with pytest.raises(ValueError, match="invalid self-terminal"):
        _observe(SelfTerminalEvidenceGate(), 1, epoch=epoch)


@pytest.mark.parametrize("turn", [None, False])
@pytest.mark.parametrize("confidence", [0.9, 1.0])
def test_explicit_evidence_does_not_rewrite_unknown_controls_or_raise_scores(
    turn: bool | None, confidence: float,
) -> None:
    gate = SelfTerminalEvidenceGate()
    for frame in (1, 4):
        scene = replace(_scene(frame), self_turn=turn, confidence=0.0)
        assert _observe(gate, frame, scene=scene, cards=_cards(confidence=confidence)) is None
        assert scene.self_turn is turn
    scene = replace(_scene(7), self_turn=turn, confidence=0.0)
    event = _observe(gate, 7, scene=scene, cards=_cards(confidence=confidence))
    assert event is not None
    assert event.confidence == confidence
    assert scene.self_turn is turn


@pytest.mark.parametrize("frame", [-1, True, 1.5])
def test_malformed_source_frame_interrupts_support(frame: int) -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 10) is None
    assert _observe(gate, 11) is None
    assert _observe(gate, frame) is None
    assert _observe(gate, 11) is None
    assert _observe(gate, 12) is None
    assert _observe(gate, 13) is None
    assert _observe(gate, 14) is not None


def test_invalid_configuration_also_interrupts_support() -> None:
    gate = SelfTerminalEvidenceGate()
    assert _observe(gate, 1) is None
    assert _observe(gate, 2) is None
    with pytest.raises(ValueError):
        _observe(gate, 3, confidence_threshold=float("nan"))
    assert _observe(gate, 4) is None
    assert _observe(gate, 5) is None
    assert _observe(gate, 6) is not None


def test_importing_gate_does_not_load_inference_or_pipeline() -> None:
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import src.tracking.self_terminal_evidence; "
            "assert 'torch' not in sys.modules; "
            "assert 'src.vision.scene_recognizer' not in sys.modules; "
            "assert 'src.pipeline.live_runtime' not in sys.modules"
        )], capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
