from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as module
from src.config.live_layout import LiveLayoutConfig
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.rank_glyph import extract_rank_glyph, read_joker_color
from src.vision.scene_recognizer import (
    SceneRecognizer,
    SeatObservation,
    SeatRole,
    TemplateMatch,
    VisualCard,
    VisualSignal,
    _resolve_card_prediction,
)


def _prediction(rank: str, confidence: float = 0.99) -> CardPrediction:
    return CardPrediction(rank, confidence, {rank: confidence})


def _joker_card(rank: str) -> Image.Image:
    image = Image.new("RGB", (126, 210), "white")
    color = (160, 25, 20) if rank == "BJ" else (20, 20, 20)
    draw = ImageDraw.Draw(image)
    draw.line((28, 18, 76, 18), fill=color, width=7)
    draw.line((66, 18, 66, 82, 56, 98, 35, 98, 24, 84), fill=color, width=7)
    return image


def _cards(ranks: list[str]) -> tuple[VisualCard, ...]:
    return tuple(
        VisualCard(rank=rank, confidence=0.999, box=(index, 0, index + 1, 10))
        for index, rank in enumerate(ranks)
    )


def _recognizer() -> SceneRecognizer:
    recognizer = object.__new__(SceneRecognizer)
    recognizer.config = LiveLayoutConfig()
    recognizer._base_rank_references = {"K": [frozenset({1, 2})]}
    recognizer._rank_references = {
        "K": [frozenset({1, 2})],
        "5": [frozenset({7, 8})],
    }
    recognizer._reference_hand_fingerprint = ("K", "5")
    return recognizer


def _state(recognizer: SceneRecognizer):
    return deepcopy(recognizer._rank_references), recognizer._reference_hand_fingerprint


@pytest.mark.parametrize("confidence", [0.2, 0.695995, 0.999])
@pytest.mark.parametrize("reference_rank", ["SJ", "BJ"])
def test_normal_j_cannot_be_promoted_by_first_letter_reference(
    confidence: float, reference_rank: str,
) -> None:
    assert _resolve_card_prediction(
        _prediction("J", confidence), (reference_rank, 0.999),
    ) == ("J", confidence)


@pytest.mark.parametrize("rank", ["SJ", "BJ"])
def test_confident_raw_joker_still_survives_normal_j_reference(rank: str) -> None:
    image = _joker_card(rank)
    color = read_joker_color(image, extract_rank_glyph(image), LiveLayoutConfig().joker_color)
    assert _resolve_card_prediction(
        _prediction(rank), ("J", 0.999), joker_color=color,
    ) == (rank, 0.99)


def test_false_raw_joker_can_still_be_corrected_to_a() -> None:
    assert _resolve_card_prediction(_prediction("BJ", 0.972), ("A", 0.95)) == ("A", 0.95)


_VALID = ["SJ", "2", "2", "A", "K", "Q", "Q", "Q", "J", "10", "9", "8", "7", "5", "4", "4", "3"]
_TWO_SJ = ["SJ", "2", "2", "A", "K", "Q", "Q", "Q", "SJ", "10", "9", "8", "7", "5", "4", "4", "3"]
_EIGHT_TENS = ["SJ", "2", "2", "A", "10", "10", "10", "10", "8", "10", "10", "10", "10", "5", "4", "4", "3"]


@pytest.mark.parametrize(
    ("observed", "raw"),
    [
        (_TWO_SJ, _TWO_SJ), (_EIGHT_TENS, _EIGHT_TENS),
        (_VALID, _TWO_SJ), (_TWO_SJ, _VALID),
        ([], []), (_VALID, _VALID[:-1]), (["not-a-rank"], ["3"]),
        (_VALID + ["3"] * 4, _VALID + ["3"] * 4),
    ],
)
def test_invalid_whole_sample_cannot_reset_or_update_state(
    monkeypatch, observed: list[str], raw: list[str],
) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)

    def forbidden_signature(_image):
        raise AssertionError("Invalid sample reached signature learning")

    monkeypatch.setattr(module, "_rank_glyph_signature", forbidden_signature)
    recognizer._remember_rank_references(
        Image.new("RGB", (50, 20)), _cards(observed),
        raw_predictions=tuple(_prediction(rank) for rank in raw), reset=True,
    )
    assert _state(recognizer) == before


@pytest.mark.parametrize(
    "raw", [None, _prediction("J", 0.695995), _prediction("K"), _prediction("J", float("nan"))],
)
def test_template_only_agreement_is_not_independent_learning_evidence(monkeypatch, raw) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda _: frozenset({11, 12}))
    recognizer._remember_rank_references(
        Image.new("RGB", (50, 20)), _cards(["J"]),
        raw_predictions=() if raw is None else (raw,), reset=True,
    )
    assert _state(recognizer) == before


def test_only_high_confidence_raw_agreement_is_learned(monkeypatch) -> None:
    recognizer = _recognizer()
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda _: frozenset({11, 12}))
    hand_image = Image.new("RGB", (252, 210), "white")
    hand_image.paste(_joker_card("SJ"), (126, 0))
    cards = (
        VisualCard("J", 0.999, (0, 0, 126, 210)),
        VisualCard("SJ", 0.999, (126, 0, 252, 210)),
    )
    recognizer._remember_rank_references(
        hand_image, cards,
        raw_predictions=(_prediction("J", 0.695995), _prediction("SJ")), reset=True,
    )
    assert recognizer._rank_references == {
        "K": [frozenset({1, 2})], "SJ": [extract_rank_glyph(_joker_card("SJ")).signature],
    }
    assert recognizer._reference_hand_fingerprint == ("J", "SJ")
    assert recognizer._base_rank_references == {"K": [frozenset({1, 2})]}
    before = _state(recognizer)
    recognizer._remember_rank_references(
        hand_image, cards,
        raw_predictions=(_prediction("J", 0.695995), _prediction("SJ")), reset=True,
    )
    assert _state(recognizer) == before


def test_no_signature_does_not_reset_references_or_fingerprint(monkeypatch) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda _: frozenset())
    recognizer._remember_rank_references(
        Image.new("RGB", (50, 20)), _cards(["J"]),
        raw_predictions=(_prediction("J"),), reset=True,
    )
    assert _state(recognizer) == before


@pytest.mark.parametrize("confidence", [0.0, 0.69, float("nan"), float("inf"), -0.1, 1.1])
def test_one_rejected_slot_prevents_whole_hand_reset_even_if_others_are_learnable(
    monkeypatch, confidence: float,
) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda _: frozenset({11, 12}))
    cards = (
        VisualCard(rank="J", confidence=confidence, box=(0, 0, 1, 10)),
        VisualCard(rank="A", confidence=0.999, box=(1, 0, 2, 10)),
    )
    recognizer._remember_rank_references(
        Image.new("RGB", (50, 20)), cards,
        raw_predictions=(_prediction("J"), _prediction("A")), reset=True,
    )
    assert _state(recognizer) == before


def test_raw_high_score_cannot_learn_a_slot_below_learning_threshold(monkeypatch) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda _: frozenset({11, 12}))
    recognizer._remember_rank_references(
        Image.new("RGB", (50, 20)),
        (VisualCard(rank="J", confidence=0.75, box=(0, 0, 1, 10)),),
        raw_predictions=(_prediction("J"),), reset=True,
    )
    assert _state(recognizer) == before


def test_hand_observation_exports_original_not_calibrated_predictions(monkeypatch) -> None:
    recognizer = _recognizer()
    raw = _prediction("J", 0.695995)
    recognizer.predictor = lambda _: (raw,)
    monkeypatch.setattr(module, "infer_hand_geometry", lambda _image, **_kwargs: module.HandGeometry(1, ((0, 0, 20, 20),), True))
    monkeypatch.setattr(recognizer, "_match_rank_reference", lambda *args, **kwargs: ("SJ", 0.999))
    exported = [_prediction("5")]
    cards = recognizer._observe_hand(
        Image.new("RGB", (20, 20)), expected_count=1, visible_count=1,
        raw_predictions_out=exported,
    )
    assert exported == [raw]
    assert exported[0] is raw
    assert [(card.rank, card.confidence) for card in cards] == [("J", 0.695995)]
    monkeypatch.setattr(module, "infer_hand_geometry", lambda _image, **_kwargs: module.HandGeometry(None, (), False))
    assert recognizer._observe_hand(
        Image.new("RGB", (20, 20)), expected_count=1, visible_count=1,
        raw_predictions_out=exported,
    ) == ()
    assert exported == []


def test_seed_without_visible_count_preserves_reference_state(monkeypatch) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)
    monkeypatch.setattr(module, "infer_hand_geometry", lambda _image, **_kwargs: module.HandGeometry(None, (), False))
    assert recognizer.seed_hand_references(Image.new("RGB", (1000, 600))) == ()
    assert _state(recognizer) == before


@pytest.mark.parametrize("route", ["observe", "seed"])
@pytest.mark.parametrize("ranks", [_TWO_SJ, _EIGHT_TENS])
def test_complete_invalid_hand_does_not_reset_at_either_entrypoint(monkeypatch, route, ranks) -> None:
    recognizer = _recognizer()
    before = _state(recognizer)
    recognizer.predictor = lambda crops: tuple(_prediction(rank) for rank in ranks)
    recognizer._round_result_references = {}
    recognizer.remaining_reader = lambda _: {}
    recognizer.templates = SimpleNamespace(classify=lambda *args: TemplateMatch(None, 0.0))
    monkeypatch.setattr(module, "infer_hand_geometry", lambda _image, **_kwargs: module.HandGeometry(
        len(ranks), tuple((index, 0, index + 1, 10) for index in range(len(ranks))), True,
    ))
    monkeypatch.setattr(recognizer, "_match_rank_reference", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_classify_round_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_classify_role_badge", lambda _: (SeatRole.FARMER, 0.99))
    monkeypatch.setattr(module, "_classify_turn_controls", lambda _: (False, 0.99))
    monkeypatch.setattr(module, "_resolve_roles_from_hand_count", lambda *args, **kwargs: {
        PlayerSeat.SELF: (SeatRole.FARMER, 0.99),
        PlayerSeat.RIGHT: (SeatRole.LANDLORD, 0.99),
        PlayerSeat.LEFT: (SeatRole.FARMER, 0.99),
    })
    monkeypatch.setattr(recognizer, "_observe_seat", lambda image, seat, **kwargs: SeatObservation(
        seat=seat, signal=VisualSignal.NEUTRAL,
    ))
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda _: frozenset({11, 12}))
    image = Image.new("RGB", (1000, 600), "navy")
    if route == "observe":
        result = recognizer.observe(SimpleNamespace(
            image=image, frame_id=1, timestamp=1.0, pixel_box=(0, 0, 1000, 600),
        ))
        assert len(result.self_hand) == 17
    else:
        assert recognizer.seed_hand_references(image) == ()
    assert _state(recognizer) == before
