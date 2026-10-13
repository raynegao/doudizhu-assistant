from __future__ import annotations

import json
import pickle
from dataclasses import FrozenInstanceError

import pytest

from src.state.events import PlayerSeat
from src.tracking.observation_evidence import _action_confidence, _seat_fingerprint
from src.vision.scene_types import (
    CardFusionEvidence,
    SeatObservation,
    VisualCard,
    VisualSignal,
)


def test_legacy_visual_card_payload_and_constructor_are_unchanged() -> None:
    card = VisualCard("BJ", 0.97234567, (1, 2, 3, 4))
    assert card.to_log_payload() == {
        "rank": "BJ", "confidence": 0.972346, "box": [1, 2, 3, 4],
    }


def test_fusion_evidence_keeps_raw_score_and_rejected_reference_separate() -> None:
    evidence = CardFusionEvidence(
        raw_rank="BJ", raw_confidence=0.99123456789,
        reference_rank="SJ", reference_confidence=0.9659883720930232,
        fusion_reason="joker_reference_color_conflict",
        joker_color="red", joker_color_reason="supported_rank_ink",
    )
    card = VisualCard("BJ", evidence.raw_confidence, (1, 2, 3, 4), evidence=evidence)
    payload = json.loads(json.dumps(card.to_log_payload(), allow_nan=False))
    assert payload["rank"] == "BJ"
    assert payload["evidence"] == {
        "raw_rank": "BJ", "raw_confidence": 0.99123456789,
        "reference_rank": "SJ", "reference_confidence": 0.9659883720930232,
        "fusion_reason": "joker_reference_color_conflict",
        "joker_color": "red", "joker_color_reason": "supported_rank_ink",
    }
    with pytest.raises(FrozenInstanceError):
        evidence.raw_rank = "SJ"  # type: ignore[misc]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_diagnostics_remain_valid_json_without_fabricating_scores(score) -> None:
    evidence = CardFusionEvidence("BJ", score, "SJ", score, "invalid_model_score")
    payload = json.loads(json.dumps(evidence.to_log_payload(), allow_nan=False))
    assert payload["raw_confidence"] is None
    assert payload["reference_confidence"] is None


def test_no_reference_is_explicitly_distinct_from_zero_confidence_reference() -> None:
    missing = CardFusionEvidence("3", 0.7).to_log_payload()
    zero = CardFusionEvidence("3", 0.7, "4", 0.0).to_log_payload()
    assert missing["reference_rank"] is None and missing["reference_confidence"] is None
    assert zero["reference_rank"] == "4" and zero["reference_confidence"] == 0.0


def test_evidence_survives_the_process_serialization_boundary() -> None:
    original = VisualCard(
        "SJ", 0.0, (1, 2, 31, 42),
        CardFusionEvidence("SJ", 0.91, "SJ", 0.99, "joker_color_conflict", "red"),
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert restored.to_log_payload() == original.to_log_payload()
    assert restored.rank == "SJ" and restored.confidence == 0.0


def test_diagnostic_rank_and_score_cannot_override_observation_semantics() -> None:
    evidence = CardFusionEvidence("SJ", 0.99, "SJ", 0.999, "rejected_reference", "red")
    plain = SeatObservation(
        PlayerSeat.LEFT, VisualSignal.UNKNOWN, (VisualCard("BJ", 0.0, (1, 2, 31, 42)),),
    )
    logged = SeatObservation(
        PlayerSeat.LEFT, VisualSignal.UNKNOWN,
        (VisualCard("BJ", 0.0, (1, 2, 31, 42), evidence),),
    )
    assert logged.card_set == plain.card_set
    assert _seat_fingerprint(logged) == _seat_fingerprint(plain)
    assert _action_confidence(logged) == _action_confidence(plain) == 0.0
