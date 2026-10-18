"""A table corner cut by the next card is reported, never classified.

Regression from the 2026-10-18 replay of acceptance-20261017-004 F299: while
the right seat's ``Q Q Q Q`` was still sliding into place, the leftmost Q was
covered down to 34 px of its 89 px crop and the retrained CNN read the
remaining stroke as a 6 at 0.81. The verified four-card counter drop then
confirmed ``6 Q Q Q`` in that single frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

import src.vision.scene_recognizer as scene_module
from src.config.live_layout import LiveLayoutConfig, NormalizedBox
from src.config.table_crops import TableRankCropConfig
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.scene_recognizer import SceneRecognizer, _visible_corner_widths
from src.vision.scene_types import SeatRole, TemplateMatch, VisualSignal

SLIDING = ((398, 84, 487, 244), (432, 84, 521, 244), (505, 84, 594, 244), (582, 84, 671, 244))
SETTLED = ((350, 84, 439, 244), (422, 84, 511, 244), (510, 84, 599, 244), (582, 84, 671, 244))


def _recognizer() -> SceneRecognizer:
    instance = object.__new__(SceneRecognizer)
    rois = dict(LiveLayoutConfig().rois)
    rois["right_play"] = rois["right_pass"] = NormalizedBox(0.0, 0.0, 1.0, 1.0)
    instance.config = LiveLayoutConfig(rois=rois)
    instance._rank_references = {}
    instance._base_rank_references = {}
    instance._reference_hand_fingerprint = ()
    instance.templates = SimpleNamespace(classify=lambda *_: TemplateMatch(None, 0.0))
    return instance


def _observe(monkeypatch, boxes, predictions):
    instance = _recognizer()
    instance.predictor = lambda crops: tuple(CardPrediction(rank, score, {rank: score}) for rank, score in predictions)
    monkeypatch.setattr(scene_module, "segment_card_regions", lambda *_args, **_kwargs: tuple(
        scene_module.CardRegion(box, scene_module.CardRegionKind.RANK_CROP) for box in boxes
    ))
    monkeypatch.setattr(scene_module, "_classify_pass_notice", lambda _: 0.0)
    monkeypatch.setattr(instance, "_match_rank_reference", lambda *_args, **_kwargs: None)
    image = Image.new("RGB", (776, 499), (245, 245, 245))
    return instance._observe_seat(image, PlayerSeat.RIGHT,
                                  role=(SeatRole.FARMER, 0.99), remaining=(13, 0.99, True))


def test_visible_corner_widths_use_the_nearest_same_row_neighbour() -> None:
    assert _visible_corner_widths(SLIDING) == (34, 73, 77, None)
    assert _visible_corner_widths(SETTLED) == (72, 88, 72, None)
    # A card wrapped below the last slot shares its x start and is not a neighbour.
    assert _visible_corner_widths(((0, 0, 89, 160), (0, 170, 89, 330))) == (None, None)
    assert _visible_corner_widths(()) == ()


def test_a_corner_cut_inside_its_glyph_zone_is_not_a_reading(monkeypatch) -> None:
    result = _observe(monkeypatch, SLIDING, (("6", 0.806), ("Q", 0.944), ("Q", 0.936), ("Q", 0.942)))
    assert result.signal is VisualSignal.UNKNOWN
    assert result.confidence == 0.0
    first = result.cards[0]
    assert first.rank == "6" and first.confidence == 0.0
    assert first.evidence is not None
    assert first.evidence.geometry_reason == "partial_corner:34/89"
    assert first.evidence.raw_rank == "6" and first.evidence.raw_confidence == pytest.approx(0.806)
    assert [round(card.confidence, 3) for card in result.cards[1:]] == [0.944, 0.936, 0.942]
    assert all(card.evidence.geometry_reason is None for card in result.cards[1:])
    assert "geometry_reason" in first.evidence.to_log_payload()
    assert "geometry_reason" not in result.cards[1].evidence.to_log_payload()


def test_settled_overlaps_keep_their_readings(monkeypatch) -> None:
    result = _observe(monkeypatch, SETTLED, (("Q", 0.936), ("Q", 0.935), ("Q", 0.937), ("Q", 0.942)))
    assert result.signal is VisualSignal.PLAY
    assert [card.rank for card in result.cards] == ["Q"] * 4
    assert result.confidence == pytest.approx(0.935)
    assert all(card.evidence.geometry_reason is None for card in result.cards)


def test_minimum_visible_width_ratio_is_validated_and_defaulted() -> None:
    assert TableRankCropConfig.from_dict({}).minimum_visible_width_ratio == 0.60
    assert TableRankCropConfig.from_dict({"minimum_visible_width_ratio": 0.5}).minimum_visible_width_ratio == 0.5
    assert "minimum_visible_width_ratio" in TableRankCropConfig().to_payload()
    for bad in (0.0, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            TableRankCropConfig(minimum_visible_width_ratio=bad)
