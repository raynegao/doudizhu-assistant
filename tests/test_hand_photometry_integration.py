from __future__ import annotations

import json
import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as module
from src.config.live_layout import LiveLayoutConfig, NormalizedBox
from src.state.events import PlayerSeat
from src.tracking.observation_evidence import _action_confidence, _seat_fingerprint
from src.vision.card_classifier import CardPrediction
from src.vision.rank_glyph import extract_rank_glyph, read_joker_color
from src.vision.scene_types import (
    CardFusionEvidence,
    CardPhotometryEvidence,
    SeatObservation,
    SeatRole,
    TemplateMatch,
    VisualCard,
    VisualSignal,
)

SIZE = (100, 160)
BOX = (0, 0, *SIZE)


def _crop(*, red=False, gray=True):
    image = Image.new("RGB", SIZE, (125, 125, 125) if gray else (250, 250, 250))
    ink = (80, 12, 10) if red else (10, 10, 10)
    if not gray:
        ink = tuple(value * 2 for value in ink)
    ImageDraw.Draw(image).line(
        ((24, 15), (66, 15), (66, 64), (60, 78), (44, 82), (30, 75)),
        fill=ink, width=8,
    )
    return image


def _prediction(rank, confidence=0.99):
    return CardPrediction(rank, confidence, {rank: confidence})


def _instance(predictions):
    # No constructor, model loading, capture or template filesystem access.
    instance = object.__new__(module.SceneRecognizer)
    instance.config = LiveLayoutConfig()
    instance.predictor = lambda _images: predictions
    instance._rank_references = {"K": [frozenset({1, 2})], "5": [frozenset({8})]}
    instance._base_rank_references = {"K": [frozenset({1, 2})]}
    instance._reference_hand_fingerprint = ("K", "5")
    instance._match_rank_reference = lambda *_args, **_kwargs: None
    return instance


def _state(instance):
    return deepcopy(instance._rank_references), instance._reference_hand_fingerprint


@pytest.mark.parametrize("rank", ["J", "SJ", "BJ"])
def test_classifier_glyph_reference_color_and_learning_share_prepared_pixels(monkeypatch, rank):
    original = _crop(red=rank == "BJ")
    original_bytes = original.tobytes()
    expected = _crop(red=rank == "BJ", gray=False).tobytes()
    prediction = _prediction(rank)
    instance = _instance((prediction,))
    seen = {name: [] for name in ("predictor", "glyph", "reference", "color", "learning")}
    preparations = []
    prepare = module.prepare_hand_rank_crop

    def prepare_spy(image, **kwargs):
        preparations.append(image.tobytes())
        return prepare(image, **kwargs)

    def predictor(images):
        seen["predictor"].extend(image.tobytes() for image in images)
        return (prediction,)

    def glyph_spy(image):
        seen["glyph"].append(image.tobytes())
        return extract_rank_glyph(image)

    def reference_spy(image, **kwargs):
        seen["reference"].append(image.tobytes())
        assert kwargs["glyph"].signature == extract_rank_glyph(image).signature
        return rank, 0.98

    def color_spy(image, glyph, config):
        seen["color"].append(image.tobytes())
        return read_joker_color(image, glyph, config)

    def learning_spy(image):
        seen["learning"].append(image.tobytes())
        return extract_rank_glyph(image).signature

    monkeypatch.setattr(module, "prepare_hand_rank_crop", prepare_spy)
    monkeypatch.setattr(module, "extract_rank_glyph", glyph_spy)
    monkeypatch.setattr(module, "read_joker_color", color_spy)
    monkeypatch.setattr(module, "_rank_glyph_signature", learning_spy)
    instance.predictor = predictor
    instance._match_rank_reference = reference_spy
    raw, prepared = [], []
    cards = instance._observe_hand(
        original, expected_count=1, geometry=module.HandGeometry(1, (BOX,), True),
        raw_predictions_out=raw, prepared_crops_out=prepared,
    )
    assert raw == [prediction] and prepared[0].tobytes() == expected
    assert cards[0].rank == rank and cards[0].confidence >= 0.98
    assert cards[0].evidence.input_photometry.reason == "neutral_gray_paper_gain"
    instance._remember_rank_references(
        original, cards, raw_predictions=raw, prepared_crops=prepared, reset=True,
    )
    assert preparations == [original_bytes]  # No second normalization while learning.
    assert original.tobytes() == original_bytes
    assert seen["predictor"] == seen["reference"] == [expected]
    assert seen["glyph"] and all(value == expected for value in seen["glyph"])
    if rank == "J":
        assert seen["learning"] == [expected] and seen["color"] == []
    else:
        assert seen["color"] == [expected, expected] and seen["learning"] == []
        assert len(seen["glyph"]) == 2  # Recognition and atomic Joker learning precheck.
    assert instance._rank_references[rank] == [extract_rank_glyph(prepared[0]).signature]
    assert instance._reference_hand_fingerprint == (rank,)


@pytest.mark.parametrize("entry", ["observe", "seed"])
def test_both_production_entry_points_forward_exact_prepared_inputs(monkeypatch, entry):
    ranks = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"] if entry == "observe" else ["SJ"]
    image = Image.new("RGB", (SIZE[0] * len(ranks), SIZE[1]))
    boxes = tuple((i * SIZE[0], 0, (i + 1) * SIZE[0], SIZE[1]) for i in range(len(ranks)))
    for box in boxes:
        image.paste(_crop(), box[:2])
    instance = _instance(tuple(_prediction(rank) for rank in ranks))
    rois = dict(instance.config.rois)
    rois["self_hand"] = NormalizedBox(0, 0, 1, 1)
    instance.config = replace(instance.config, rois=rois)
    instance.templates = SimpleNamespace(classify=lambda *_args, **_kwargs: TemplateMatch(None, 0))
    instance.remaining_reader = None
    instance._round_result_references = {}
    monkeypatch.setattr(module, "infer_hand_geometry", lambda *_args, **_kwargs: module.HandGeometry(len(ranks), boxes, True))
    monkeypatch.setattr(module, "_classify_round_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_classify_role_badge", lambda _image: (SeatRole.FARMER, 0.99))
    monkeypatch.setattr(instance, "_observe_seat", lambda _image, seat, **_kwargs: SeatObservation(seat, VisualSignal.NEUTRAL))
    consumed, learned = [], []
    original_remember = instance._remember_rank_references

    def predictor(crops):
        consumed.extend(crops)
        return tuple(_prediction(rank) for rank in ranks)

    def remember(hand, cards, **kwargs):
        learned.extend(kwargs["prepared_crops"])
        assert kwargs["reset"] is True
        return original_remember(hand, cards, **kwargs)

    instance.predictor = predictor
    monkeypatch.setattr(instance, "_remember_rank_references", remember)
    if entry == "seed":
        cards = instance.seed_hand_references(image)
    else:
        frame = SimpleNamespace(image=image, frame_id=1, timestamp=1.0, pixel_box=(0, 0, *image.size))
        cards = instance.observe(frame).self_hand
    assert [card.rank for card in cards] == ranks
    assert len(consumed) == len(learned) == len(ranks)
    assert all(left is right for left, right in zip(consumed, learned, strict=True))
    assert all(crop.tobytes() == _crop(gray=False).tobytes() for crop in consumed)
    assert instance._reference_hand_fingerprint == tuple(ranks)


@pytest.mark.parametrize("case", ["unknown", "clipped", "mixed", "flat_gray"])
@pytest.mark.parametrize("strong_reference", [False, True])
def test_refusal_retains_slots_and_raw_but_cannot_learn(case, strong_reference):
    first = _crop()
    if case == "mixed":
        ImageDraw.Draw(first).rectangle((60, 0, 99, 159), fill=(250, 250, 250))
    elif case == "flat_gray":
        first = Image.new("RGB", SIZE, (125, 125, 125))
    image = Image.new("RGB", (200, 160))
    image.paste(first, (0, 0))
    image.paste(_crop(), (100, 0))
    boxes = (BOX, (100, 0, 200, 160))
    geometry = module.HandGeometry(
        None if case == "unknown" else 2, boxes, case not in {"unknown", "clipped"},
        (0,) if case == "clipped" else (), "unknown" if case == "unknown" else "",
    )
    predictions = (_prediction("8", 0.999), _prediction("4", 0.999))
    instance = _instance(predictions)
    if strong_reference:
        instance._match_rank_reference = lambda *_args, **_kwargs: ("8", 0.999)
    before = _state(instance)
    raw, prepared = [], []
    cards = instance._observe_hand(
        image, expected_count=2, geometry=geometry,
        raw_predictions_out=raw, prepared_crops_out=prepared,
    )
    assert len(cards) == len(raw) == len(prepared) == 2
    assert raw == list(predictions)
    assert [card.rank for card in cards] == (["8", "8"] if strong_reference else ["8", "4"])
    assert cards[0].box == BOX and cards[0].confidence == 0
    assert not cards[0].evidence.input_photometry.usable
    assert cards[0].evidence.raw_confidence == 0.999
    assert prepared[0].tobytes() == first.tobytes()
    if case == "unknown":
        assert cards[1].confidence == 0
    else:
        assert cards[1].confidence == 0.999
    instance._remember_rank_references(
        image, cards, raw_predictions=raw, prepared_crops=prepared, reset=True,
    )
    assert _state(instance) == before


def test_normalized_fusion_score_cannot_replace_independent_raw_learning_evidence():
    instance = _instance((_prediction("J", 0.695995),))
    instance._match_rank_reference = lambda *_args, **_kwargs: ("J", 0.999)
    before = _state(instance)
    raw, prepared = [], []
    image = _crop()
    cards = instance._observe_hand(
        image, expected_count=1, geometry=module.HandGeometry(1, (BOX,), True),
        raw_predictions_out=raw, prepared_crops_out=prepared,
    )
    assert cards[0].confidence == 0.999
    assert cards[0].evidence.input_photometry.reason == "neutral_gray_paper_gain"
    assert cards[0].evidence.raw_confidence == 0.695995
    instance._remember_rank_references(
        image, cards, raw_predictions=raw, prepared_crops=prepared, reset=True,
    )
    assert _state(instance) == before


@pytest.mark.parametrize("fused", [("SJ", "SJ"), ("SJ", "3")])
def test_prepared_pixels_cannot_bypass_physical_whole_hand_validation(monkeypatch, fused):
    instance = _instance(())
    before = _state(instance)
    image = Image.new("RGB", (200, 160))
    boxes = (BOX, (100, 0, 200, 160))

    def forbidden(_image):
        raise AssertionError("Physically invalid whole hand reached glyph learning")

    monkeypatch.setattr(module, "extract_rank_glyph", forbidden)
    instance._remember_rank_references(
        image, tuple(VisualCard(rank, 0.999, box) for rank, box in zip(fused, boxes, strict=True)),
        raw_predictions=(_prediction("SJ"), _prediction("SJ")),
        prepared_crops=(_crop(gray=False), _crop(gray=False)), reset=True,
    )
    assert _state(instance) == before


@pytest.mark.parametrize("prepared", [[], [Image.new("RGB", (99, 160))], [Image.new("RGB", SIZE)] * 2])
def test_bad_prepared_length_or_size_never_changes_reference_state(monkeypatch, prepared):
    instance = _instance((_prediction("J"),))
    before = _state(instance)

    def forbidden(_image):
        raise AssertionError("Invalid prepared inputs reached learning")

    monkeypatch.setattr(module, "_rank_glyph_signature", forbidden)
    instance._remember_rank_references(
        _crop(), (VisualCard("J", 0.99, BOX),), raw_predictions=(_prediction("J"),),
        prepared_crops=prepared, reset=True,
    )
    assert _state(instance) == before


@pytest.mark.parametrize("rank", ["J", "8"])
@pytest.mark.parametrize("joker", ["SJ", "BJ"])
def test_gray_normalization_cannot_promote_ordinary_raw_to_joker(rank, joker):
    instance = _instance((_prediction(rank),))
    instance._match_rank_reference = lambda *_args, **_kwargs: (joker, 0.999)
    cards = instance._observe_hand(_crop(), expected_count=1, geometry=module.HandGeometry(1, (BOX,), True))
    assert cards[0].rank == rank
    assert cards[0].confidence == (0.99 if rank == "J" else 0)
    assert cards[0].evidence.input_photometry.reason == "neutral_gray_paper_gain"


def test_legacy_learning_without_prepared_images_keeps_original_pixels(monkeypatch):
    instance = _instance((_prediction("J"),))
    image = _crop()
    observed = []
    monkeypatch.setattr(module, "_rank_glyph_signature", lambda crop: observed.append(crop.tobytes()) or frozenset({30}))
    instance._remember_rank_references(
        image, (VisualCard("J", 0.99, BOX),), raw_predictions=(_prediction("J"),), reset=True,
    )
    assert observed == [image.tobytes()]
    assert instance._rank_references["J"] == [frozenset({30})]


def test_empty_geometry_clears_reused_output_buffers_without_prediction():
    instance = _instance(())
    instance.predictor = lambda _images: pytest.fail("Empty geometry must not reach predictor")
    raw, prepared = [_prediction("8")], [_crop()]
    assert instance._observe_hand(
        _crop(), expected_count=1, geometry=module.HandGeometry(None, (), False),
        raw_predictions_out=raw, prepared_crops_out=prepared,
    ) == ()
    assert raw == prepared == []


def test_photometry_diagnostics_roundtrip_without_changing_tracking_semantics():
    photo = CardPhotometryEvidence("neutral_gray_paper_gain", 2.0, 125, 0.9)
    raw = CardFusionEvidence("J", 0.99)
    annotated = replace(raw, input_photometry=photo)
    assert "input_photometry" not in raw.to_log_payload()
    plain = SeatObservation(PlayerSeat.SELF, VisualSignal.UNKNOWN, (VisualCard("J", 0, BOX, raw),))
    detailed = replace(plain, cards=(replace(plain.cards[0], evidence=annotated),))
    restored = pickle.loads(pickle.dumps(detailed))
    assert restored == detailed
    assert json.loads(json.dumps(restored.to_log_payload(), allow_nan=False))["cards"][0]["evidence"]["input_photometry"] == photo.to_log_payload()
    assert _seat_fingerprint(plain) == _seat_fingerprint(detailed)
    assert _action_confidence(plain) == _action_confidence(detailed) == 0
    with pytest.raises(FrozenInstanceError):
        photo.gain = 3


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_photometry_metrics_do_not_emit_nonstandard_json(invalid):
    photo = CardPhotometryEvidence("diagnostic_invalid", invalid, paper_fraction=invalid)
    payload = json.loads(json.dumps(photo.to_log_payload(), allow_nan=False))
    assert payload["gain"] is None and payload["paper_fraction"] is None
