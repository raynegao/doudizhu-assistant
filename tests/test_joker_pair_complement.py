"""One color-conflicting Joker beside a confirmed partner takes the complementary identity.

Independent acceptance session acceptance-20261014-003 (frozen ``04f83e90``):
the left seat's rocket was segmented as two overlapping corners. The right
corner carried red bleed from the big Joker underneath, so the raw model said
``BJ`` at 0.46 while its own rank ink was strictly black. Per-card fusion
zeroed that slot, both slots read ``BJ`` and the seat stayed unknown until the
round was marked uncertain. A deck holds one SJ and one BJ: the slot beside a
confirmed BJ can only be SJ, and the measured ink agrees.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

import src.vision.scene_recognizer as scene_module
from src.config.live_layout import LiveLayoutConfig, NormalizedBox
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.joker_pair import resolve_joker_pair
from src.vision.rank_glyph import JokerColorEvidence
from src.vision.scene_recognizer import SceneRecognizer
from src.vision.scene_types import (
    CardFusionEvidence,
    SeatRole,
    TemplateMatch,
    VisualCard,
    VisualSignal,
)


def _card(rank: str, confidence: float, *, raw: str, reason: str, color: str,
          color_reason: str, box: tuple[int, int, int, int] = (0, 0, 10, 20)) -> VisualCard:
    return VisualCard(rank, confidence, box, CardFusionEvidence(
        raw_rank=raw, raw_confidence=confidence or 0.46, fusion_reason=reason,
        joker_color=color, joker_color_reason=color_reason,
    ))


def _confirmed(rank: str = "BJ", confidence: float = 0.99995) -> VisualCard:
    color = "red" if rank == "BJ" else "black"
    return _card(rank, confidence, raw=rank, reason="joker_color_consistent_raw_preserved",
                 color=color, color_reason=f"{color}_rank_ink")


def _conflict(raw: str = "BJ", color: str = "black",
              color_reason: str = "black_rank_ink") -> VisualCard:
    return _card(raw, 0.0, raw=raw, reason="joker_color_conflict",
                 color=color, color_reason=color_reason, box=(8, 0, 18, 20))


def _evidence(color: str, reason: str, fraction: float = 0.93) -> JokerColorEvidence:
    return JokerColorEvidence(
        color, reason, pixel_count=120,
        red_fraction=fraction if color == "red" else 0.0,
        black_fraction=fraction if color == "black" else 0.0,
        background_fraction=0.9,
    )


RED = _evidence("red", "red_rank_ink", 0.98)
BLACK = _evidence("black", "black_rank_ink", 0.93)


def test_conflicting_joker_beside_confirmed_big_joker_becomes_small_joker() -> None:
    cards = (_confirmed("BJ"), _conflict("BJ", "black"))
    resolved = resolve_joker_pair(cards, (RED, BLACK))
    assert resolved[0] == cards[0]
    second = resolved[1]
    assert second.rank == "SJ"
    assert second.confidence == pytest.approx(0.93)
    assert second.box == cards[1].box
    assert second.evidence is not None
    assert second.evidence.raw_rank == "BJ"
    assert second.evidence.raw_confidence == pytest.approx(0.46)
    assert second.evidence.fusion_reason == "joker_pair_complement"
    assert second.evidence.joker_color == "black"
    assert second.evidence.joker_color_reason == "black_rank_ink"


@pytest.mark.parametrize("color_reason", ["black_rank_ink", "black_rank_ink_antialiased_edge"])
def test_both_strict_black_ink_readings_support_the_complement(color_reason: str) -> None:
    cards = (_confirmed("BJ"), _conflict("BJ", "black", color_reason))
    resolved = resolve_joker_pair(cards, (RED, _evidence("black", color_reason, 0.88)))
    assert resolved[1].rank == "SJ" and resolved[1].confidence == pytest.approx(0.88)


def test_symmetric_red_conflict_beside_confirmed_small_joker_becomes_big_joker() -> None:
    cards = (_conflict("SJ", "red", "red_rank_ink"), _confirmed("SJ", 0.97))
    resolved = resolve_joker_pair(cards, (RED, BLACK))
    assert resolved[0].rank == "BJ"
    assert resolved[0].confidence == pytest.approx(min(0.97, 0.98))
    assert resolved[1] == cards[1]


def test_confidence_never_exceeds_the_partner() -> None:
    cards = (_confirmed("BJ", 0.91), _conflict("BJ", "black"))
    resolved = resolve_joker_pair(cards, (RED, _evidence("black", "black_rank_ink", 0.99)))
    assert resolved[1].confidence == pytest.approx(0.91)


def test_order_of_the_pair_does_not_matter() -> None:
    cards = (_conflict("BJ", "black"), _confirmed("BJ"))
    resolved = resolve_joker_pair(cards, (BLACK, RED))
    assert [card.rank for card in resolved] == ["SJ", "BJ"]


@pytest.mark.parametrize("kind", [
    "both_conflict", "partner_weak", "partner_reason", "same_ink_as_partner",
    "ink_unknown", "ink_not_strict", "three_jokers", "ordinary_card_with_joker_reference",
    "no_color_evidence",
])
def test_unsupported_pairs_are_returned_unchanged(kind: str) -> None:
    cards: tuple[VisualCard, ...]
    colors: tuple[JokerColorEvidence | None, ...]
    if kind == "both_conflict":
        cards, colors = (_conflict("BJ", "black"), _conflict("BJ", "black")), (BLACK, BLACK)
    elif kind == "partner_weak":
        cards, colors = (_confirmed("BJ", 0.89), _conflict("BJ", "black")), (RED, BLACK)
    elif kind == "partner_reason":
        weak = replace(_confirmed("BJ"), evidence=replace(
            _confirmed("BJ").evidence, fusion_reason="joker_reference_conflict_raw_preserved"))
        cards, colors = (weak, _conflict("BJ", "black")), (RED, BLACK)
    elif kind == "same_ink_as_partner":
        # Two red Joker inks cannot be a BJ/SJ pair; keep the conflict slot at zero.
        cards, colors = (_confirmed("BJ"), _conflict("SJ", "red", "red_rank_ink")), (RED, RED)
    elif kind == "ink_unknown":
        cards = (_confirmed("BJ"), _conflict("BJ", "unknown", "rank_ink_ambiguous"))
        colors = (RED, _evidence("unknown", "rank_ink_ambiguous", 0.0))
    elif kind == "ink_not_strict":
        cards = (_confirmed("BJ"), _conflict("BJ", "black", "rank_background_ambiguous"))
        colors = (RED, _evidence("black", "rank_background_ambiguous"))
    elif kind == "three_jokers":
        cards = (_confirmed("BJ"), _conflict("BJ", "black"), _conflict("BJ", "black"))
        colors = (RED, BLACK, BLACK)
    elif kind == "ordinary_card_with_joker_reference":
        ordinary = _card("K", 0.0, raw="K", reason="joker_reference_requires_raw_identity",
                         color="black", color_reason="black_rank_ink")
        cards, colors = (_confirmed("BJ"), ordinary), (RED, None)
    else:
        cards, colors = (_confirmed("BJ"), _conflict("BJ", "black")), (RED, None)
    assert resolve_joker_pair(cards, colors) == cards


def test_non_joker_cards_are_passed_through_beside_a_resolved_pair() -> None:
    plain = VisualCard("5", 0.97, (20, 0, 30, 20), CardFusionEvidence("5", 0.97))
    cards = (plain, _confirmed("BJ"), _conflict("BJ", "black"))
    resolved = resolve_joker_pair(cards, (None, RED, BLACK))
    assert resolved[0] == plain and resolved[2].rank == "SJ"


def test_mismatched_evidence_length_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        resolve_joker_pair((_confirmed("BJ"),), (RED, BLACK))


def _recognizer() -> SceneRecognizer:
    instance = object.__new__(SceneRecognizer)
    rois = dict(LiveLayoutConfig().rois)
    rois["left_play"] = rois["left_pass"] = NormalizedBox(0.0, 0.0, 1.0, 1.0)
    instance.config = LiveLayoutConfig(rois=rois)
    instance._rank_references = {}
    instance._base_rank_references = {}
    instance._reference_hand_fingerprint = ()
    instance.templates = SimpleNamespace(classify=lambda *_: TemplateMatch(None, 0.0))
    return instance


def test_table_observation_resolves_the_overlapped_rocket(monkeypatch) -> None:
    instance = _recognizer()
    instance.predictor = lambda crops: (
        CardPrediction("BJ", 0.99995, {"BJ": 0.99995}),
        CardPrediction("BJ", 0.4622, {"BJ": 0.4622, "SJ": 0.41}),
    )
    monkeypatch.setattr(scene_module, "segment_card_regions", lambda *_args, **_kwargs: (
        scene_module.CardRegion((0, 0, 126, 210), scene_module.CardRegionKind.RANK_CROP),
        scene_module.CardRegion((90, 0, 216, 210), scene_module.CardRegionKind.RANK_CROP),
    ))
    monkeypatch.setattr(scene_module, "_classify_pass_notice", lambda _: 0.0)
    monkeypatch.setattr(instance, "_match_rank_reference", lambda *_args, **_kwargs: None)
    readings = iter((RED, BLACK))
    monkeypatch.setattr(scene_module, "read_joker_color", lambda *_args, **_kwargs: next(readings))
    image = Image.new("RGB", (220, 210), (245, 245, 245))
    result = instance._observe_seat(image, PlayerSeat.LEFT,
                                    role=(SeatRole.LANDLORD, 0.99), remaining=(1, 0.99, True))
    assert result.signal is VisualSignal.PLAY
    assert [(card.rank, round(card.confidence, 5)) for card in result.cards] == [
        ("BJ", 0.99995), ("SJ", 0.93),
    ]
    assert result.cards[1].evidence.fusion_reason == "joker_pair_complement"
    assert result.cards[1].evidence.raw_rank == "BJ"
    assert result.confidence == pytest.approx(0.93)
