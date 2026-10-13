from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as scene_module
from src.config.joker_color import JokerColorConfig
from src.config.live_layout import LiveLayoutConfig, NormalizedBox, live_layout_from_dict
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.rank_glyph import extract_rank_glyph, read_joker_color
from src.vision.scene_recognizer import SceneRecognizer, _fuse_card_prediction
from src.vision.scene_types import SeatRole, TemplateMatch, VisualCard, VisualSignal


def _rank_card(color: str, *, rank: str = "J", paper: int = 245) -> Image.Image:
    image = Image.new("RGB", (126, 210), (paper, paper, paper))
    draw = ImageDraw.Draw(image)
    ink = (160, 25, 20) if color == "red" else (20, 20, 20)
    if rank == "J":
        draw.line((28, 18, 76, 18), fill=ink, width=7)
        draw.line((66, 18, 66, 82, 56, 98, 35, 98, 24, 84), fill=ink, width=7)
    elif rank == "A":
        draw.line((24, 98, 51, 18, 80, 98), fill=ink, width=7)
        draw.line((35, 66, 67, 66), fill=ink, width=7)
    else:
        raise AssertionError(rank)
    return image


def _color(image: Image.Image):
    return read_joker_color(image, extract_rank_glyph(image), JokerColorConfig())


def _prediction(rank: str, confidence: float = 0.99) -> CardPrediction:
    return CardPrediction(rank, confidence, {rank: confidence})


@pytest.mark.parametrize("color", ["red", "black"])
def test_color_uses_original_primary_rank_ink(color: str) -> None:
    image = _rank_card(color)
    evidence = _color(image)
    assert evidence.color == color
    assert evidence.pixel_count >= 40
    assert evidence.background_fraction >= 0.7
    assert extract_rank_glyph(image).signature == extract_rank_glyph(image.convert("L")).signature


def test_red_body_and_landlord_corner_cannot_turn_black_rank_red() -> None:
    image = _rank_card("black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((82, 105, 125, 209), fill=(170, 20, 20))
    draw.polygon(((86, 0), (125, 0), (125, 74)), fill=(170, 20, 20))
    assert _color(image).color == "black"
    assert _color(image).red_fraction == 0.0


@pytest.mark.parametrize("kind", ["blank", "red_body_only", "mixed", "occluded", "gray_selected"])
def test_weak_or_masked_rank_color_is_unknown(kind: str) -> None:
    image = _rank_card("red")
    if kind in {"blank", "red_body_only"}:
        image = Image.new("RGB", image.size, "white")
        if kind == "red_body_only":
            ImageDraw.Draw(image).rectangle((90, 110, 125, 209), fill=(170, 20, 20))
    elif kind == "mixed":
        points = sorted(extract_rank_glyph(image).pixels)
        for point in points[:len(points) // 2]:
            image.putpixel(point, (20, 20, 20))
    elif kind == "occluded":
        ImageDraw.Draw(image).rectangle((10, 40, 84, 70), fill=(85, 85, 85))
    else:
        image = _rank_card("black", paper=155)
    assert _color(image).color == "unknown"


@pytest.mark.parametrize("rank,color", [("BJ", "red"), ("SJ", "black")])
@pytest.mark.parametrize("reference", [None, ("SJ", 0.999), ("BJ", 0.999), ("J", 0.999)])
def test_matching_color_preserves_raw_joker_rank_and_confidence(rank, color, reference) -> None:
    result_rank, confidence, evidence = _fuse_card_prediction(
        _prediction(rank, 0.973), reference, joker_color=_color(_rank_card(color)),
    )
    assert (result_rank, confidence) == (rank, 0.973)
    assert evidence.raw_confidence == 0.973
    assert evidence.joker_color == color


@pytest.mark.parametrize("rank,color", [("BJ", "black"), ("SJ", "red")])
def test_conflicting_color_does_not_flip_joker_identity(rank: str, color: str) -> None:
    result_rank, confidence, evidence = _fuse_card_prediction(
        _prediction(rank), ("BJ" if rank == "SJ" else "SJ", 0.999),
        joker_color=_color(_rank_card(color)),
    )
    assert (result_rank, confidence) == (rank, 0.0)
    assert evidence.fusion_reason == "joker_color_conflict"


@pytest.mark.parametrize("color", [None, _color(Image.new("RGB", (126, 210), "white"))])
def test_absent_color_cannot_be_bypassed_by_high_confidence_raw_or_reference(color) -> None:
    rank, confidence, evidence = _fuse_card_prediction(
        _prediction("BJ", 0.999), ("BJ", 0.999), joker_color=color,
    )
    assert (rank, confidence) == ("BJ", 0.0)
    assert evidence.fusion_reason == "joker_color_unavailable"


def test_ordinary_red_j_is_not_promoted_to_big_joker() -> None:
    rank, confidence, evidence = _fuse_card_prediction(
        _prediction("J", 0.65), ("BJ", 0.999), joker_color=_color(_rank_card("red")),
    )
    assert (rank, confidence) == ("J", 0.65)
    assert evidence.fusion_reason == "normal_j_preserved"


@pytest.mark.parametrize("rank", ["A", "K", "10"])
@pytest.mark.parametrize("reference_rank", ["SJ", "BJ"])
@pytest.mark.parametrize("color", ["red", "black"])
def test_ordinary_rank_cannot_gain_joker_identity_from_reference(rank, reference_rank, color) -> None:
    result_rank, confidence, evidence = _fuse_card_prediction(
        _prediction(rank), (reference_rank, 0.999), joker_color=_color(_rank_card(color)),
    )
    assert (result_rank, confidence) == (rank, 0.0)
    assert evidence.fusion_reason == "joker_reference_requires_raw_identity"


@pytest.mark.parametrize("raw_rank,reference", [("J", None), ("A", None), ("J", ("BJ", 0.99))])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_invalid_raw_confidence_keeps_rank_slot_and_valid_json(raw_rank, reference, invalid) -> None:
    rank, confidence, evidence = _fuse_card_prediction(_prediction(raw_rank, invalid), reference)
    assert (rank, confidence) == (raw_rank, 0.0)
    assert evidence.fusion_reason.endswith("confidence_invalid")
    json.dumps(VisualCard(rank, confidence, (0, 0, 126, 210), evidence).to_log_payload(), allow_nan=False)


@pytest.mark.parametrize("raw_rank", ["K", "BJ"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_reference_confidence_is_not_emitted(raw_rank, invalid) -> None:
    rank, confidence, evidence = _fuse_card_prediction(_prediction(raw_rank), ("A", invalid))
    assert (rank, confidence) == ("A", 0.0)
    assert evidence.fusion_reason.endswith("confidence_invalid")
    json.dumps(VisualCard(rank, confidence, (0, 0, 126, 210), evidence).to_log_payload(), allow_nan=False)


@pytest.mark.parametrize("color", ["red", "black"])
def test_strong_a_shape_correction_precedes_joker_identity_gate(color: str) -> None:
    rank, confidence, evidence = _fuse_card_prediction(
        _prediction("BJ"), ("A", 0.95), joker_color=_color(_rank_card(color, rank="A")),
    )
    assert (rank, confidence) == ("A", 0.95)
    assert evidence.fusion_reason == "non_joker_reference"


def _recognizer() -> SceneRecognizer:
    instance = object.__new__(SceneRecognizer)
    instance.config = LiveLayoutConfig()
    instance._rank_references = {"K": [frozenset({1, 2})]}
    instance._base_rank_references = deepcopy(instance._rank_references)
    instance._reference_hand_fingerprint = ("K",)
    return instance


@pytest.mark.parametrize("color", ["red", "black"])
def test_table_general_a_reference_is_not_replaced_by_joker_only(monkeypatch, color: str) -> None:
    instance = _recognizer()
    rois = dict(instance.config.rois)
    rois["right_play"] = rois["right_pass"] = NormalizedBox(0.0, 0.0, 1.0, 1.0)
    instance.config = LiveLayoutConfig(rois=rois)
    instance.templates = SimpleNamespace(classify=lambda *_: TemplateMatch(None, 0.0))
    instance.predictor = lambda _: (_prediction("BJ"),)
    image = _rank_card(color, rank="A")
    monkeypatch.setattr(scene_module, "segment_card_regions", lambda *_args, **_kwargs: (
        scene_module.CardRegion((0, 0, 126, 210), scene_module.CardRegionKind.RANK_CROP),
    ))
    monkeypatch.setattr(scene_module, "_classify_pass_notice", lambda _: 0.0)

    def match(_image, **kwargs):
        assert kwargs.get("allowed_ranks") is None, "Joker-only displaced a valid A reference"
        return "A", 0.95

    monkeypatch.setattr(instance, "_match_rank_reference", match)
    result = instance._observe_seat(image, PlayerSeat.RIGHT,
                                    role=(SeatRole.LANDLORD, 0.99), remaining=(17, 0.99, True))
    assert result.signal is VisualSignal.PLAY
    assert [(card.rank, card.confidence) for card in result.cards] == [("A", 0.95)]
    assert result.cards[0].evidence.fusion_reason == "non_joker_reference"


@pytest.mark.parametrize("color,expected_confidence", [("red", 0.99), ("black", 0.0)])
def test_hand_connection_retains_slot_and_exports_exact_fusion_reason(monkeypatch, color, expected_confidence) -> None:
    instance = _recognizer()
    instance.predictor = lambda _: (_prediction("BJ"),)
    image = _rank_card(color)
    geometry = scene_module.HandGeometry(1, ((0, 0, 126, 210),), True)
    monkeypatch.setattr(instance, "_match_rank_reference", lambda *_args, **_kwargs: ("SJ", 0.995))
    cards = instance._observe_hand(image, expected_count=1, geometry=geometry)
    assert len(cards) == 1 and cards[0].box == (0, 0, 126, 210)
    assert (cards[0].rank, cards[0].confidence) == ("BJ", expected_confidence)
    assert cards[0].evidence.raw_rank == "BJ"
    assert cards[0].evidence.reference_rank == "SJ"
    assert cards[0].evidence.joker_color == color


@pytest.mark.parametrize("path", ["hand", "table"])
def test_both_connections_reject_non_joker_raw_promoted_by_reference(monkeypatch, path) -> None:
    instance = _recognizer()
    instance.predictor = lambda _: (_prediction("K"),)
    monkeypatch.setattr(instance, "_match_rank_reference", lambda *_args, **_kwargs: ("BJ", 0.999))
    image = _rank_card("black")
    if path == "hand":
        geometry = scene_module.HandGeometry(1, ((0, 0, 126, 210),), True)
        cards = instance._observe_hand(image, expected_count=1, geometry=geometry)
    else:
        rois = dict(instance.config.rois)
        rois["right_play"] = rois["right_pass"] = NormalizedBox(0.0, 0.0, 1.0, 1.0)
        instance.config = LiveLayoutConfig(rois=rois)
        instance.templates = SimpleNamespace(classify=lambda *_: TemplateMatch(None, 0.0))
        monkeypatch.setattr(scene_module, "segment_card_regions", lambda *_args, **_kwargs: (
            scene_module.CardRegion((0, 0, 126, 210), scene_module.CardRegionKind.RANK_CROP),
        ))
        monkeypatch.setattr(scene_module, "_classify_pass_notice", lambda _: 0.0)
        seat = instance._observe_seat(image, PlayerSeat.RIGHT,
                                     role=(SeatRole.LANDLORD, 0.99), remaining=(17, 0.99, True))
        assert seat.signal is VisualSignal.UNKNOWN
        cards = seat.cards
    assert len(cards) == 1 and cards[0].box == (0, 0, 126, 210)
    assert (cards[0].rank, cards[0].confidence) == ("K", 0.0)
    assert cards[0].evidence.reference_rank == "BJ"


def test_geometry_rejection_keeps_color_conflict_reason(monkeypatch) -> None:
    instance = _recognizer()
    instance.predictor = lambda _: (_prediction("BJ"),)
    monkeypatch.setattr(instance, "_match_rank_reference", lambda *_args, **_kwargs: None)
    geometry = scene_module.HandGeometry(1, ((0, 0, 126, 210),), False, clipped_slots=(0,))
    card = instance._observe_hand(_rank_card("black"), expected_count=1, geometry=geometry)[0]
    assert card.confidence == 0.0
    assert card.evidence.fusion_reason == "joker_color_conflict;geometry_clipped"
    assert card.evidence.joker_color_reason == "black_rank_ink"


@pytest.mark.parametrize("kind", ["opposite", "gray_selected", "blank"])
def test_direct_learning_cannot_bypass_color_gate(kind: str) -> None:
    instance = _recognizer()
    before = deepcopy(instance._rank_references), instance._reference_hand_fingerprint
    image = _rank_card("black", paper=155 if kind == "gray_selected" else 245)
    if kind == "blank":
        image = Image.new("RGB", (126, 210), "white")
    instance._remember_rank_references(
        image, (VisualCard("BJ", 0.99, (0, 0, 126, 210)),),
        raw_predictions=(_prediction("BJ"),), reset=True,
    )
    assert (instance._rank_references, instance._reference_hand_fingerprint) == before


@pytest.mark.parametrize("raw_confidence", [0.8, float("nan")])
@pytest.mark.parametrize("kind", ["opposite", "gray_selected", "blank"])
def test_weak_raw_joker_cannot_skip_whole_hand_color_precheck(raw_confidence, kind) -> None:
    instance = _recognizer()
    before = deepcopy(instance._rank_references), instance._reference_hand_fingerprint
    image = Image.new("RGB", (252, 210), "white")
    if kind != "blank":
        image.paste(_rank_card("black", paper=155 if kind == "gray_selected" else 245), (0, 0))
    image.paste(_rank_card("black", rank="A"), (126, 0))
    instance._remember_rank_references(
        image, (VisualCard("BJ", 0.99, (0, 0, 126, 210)),
                VisualCard("A", 0.99, (126, 0, 252, 210))),
        raw_predictions=(_prediction("BJ", raw_confidence), _prediction("A")), reset=True,
    )
    assert (instance._rank_references, instance._reference_hand_fingerprint) == before


def test_valid_joker_learning_still_accepts_matching_raw_rgb() -> None:
    instance = _recognizer()
    instance._remember_rank_references(
        _rank_card("red"), (VisualCard("BJ", 0.99, (0, 0, 126, 210)),),
        raw_predictions=(_prediction("BJ"),), reset=True,
    )
    assert "BJ" in instance._rank_references
    assert instance._reference_hand_fingerprint == ("BJ",)


def test_identical_hand_fingerprint_skips_color_and_signature_work(monkeypatch) -> None:
    instance = _recognizer()
    instance._reference_hand_fingerprint = ("BJ",)
    before = deepcopy(instance._rank_references), instance._reference_hand_fingerprint

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Identical hand must return before glyph/color work")

    monkeypatch.setattr(scene_module, "extract_rank_glyph", forbidden)
    monkeypatch.setattr(scene_module, "read_joker_color", forbidden)
    monkeypatch.setattr(scene_module, "_rank_glyph_signature", forbidden)
    instance._remember_rank_references(
        _rank_card("red"), (VisualCard("BJ", 0.99, (0, 0, 126, 210)),),
        raw_predictions=(_prediction("BJ"),), reset=True,
    )
    assert (instance._rank_references, instance._reference_hand_fingerprint) == before


def test_first_joker_learning_reuses_color_precheck_signature(monkeypatch) -> None:
    instance = _recognizer()
    image = _rank_card("red")
    expected = extract_rank_glyph(image).signature
    calls = []

    def extract_once(crop):
        calls.append(crop.size)
        return extract_rank_glyph(crop)

    def forbidden(_image):
        raise AssertionError("Joker signature must reuse its color precheck component")

    monkeypatch.setattr(scene_module, "extract_rank_glyph", extract_once)
    monkeypatch.setattr(scene_module, "_rank_glyph_signature", forbidden)
    instance._remember_rank_references(
        image, (VisualCard("BJ", 0.99, (0, 0, 126, 210)),),
        raw_predictions=(_prediction("BJ"),), reset=True,
    )
    assert calls == [(126, 210)]
    assert instance._rank_references["BJ"] == [expected]


def test_public_seed_cannot_accept_color_rejected_joker(monkeypatch) -> None:
    instance = _recognizer()
    instance.predictor = lambda _: (_prediction("BJ"),)
    rois = dict(instance.config.rois)
    rois["self_hand"] = NormalizedBox(0.0, 0.0, 1.0, 1.0)
    instance.config = LiveLayoutConfig(rois=rois)
    before = deepcopy(instance._rank_references), instance._reference_hand_fingerprint
    monkeypatch.setattr(instance, "_match_rank_reference", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scene_module, "infer_hand_geometry", lambda *_args, **_kwargs:
                        scene_module.HandGeometry(1, ((0, 0, 126, 210),), True))
    assert instance.seed_hand_references(_rank_card("black")) == ()
    assert (instance._rank_references, instance._reference_hand_fingerprint) == before


def test_color_config_round_trip_and_old_config_default() -> None:
    config = LiveLayoutConfig(joker_color=JokerColorConfig(min_color_fraction=0.95))
    assert live_layout_from_dict(config.to_payload()).joker_color == config.joker_color
    assert live_layout_from_dict({}).joker_color == JokerColorConfig()


@pytest.mark.parametrize("payload", [None, [], {"typo": 1}, {"min_component_pixels": True},
                                      {"min_color_fraction": float("nan")},
                                      {"red_channel_margin": -1}, {"min_background_fraction": 0.5}])
def test_color_config_rejects_invalid_or_silent_typo(payload) -> None:
    with pytest.raises(ValueError):
        JokerColorConfig.from_dict(payload)


@pytest.mark.parametrize("name", list(JokerColorConfig().to_payload()))
def test_all_color_config_fields_reject_boolean(name: str) -> None:
    with pytest.raises(ValueError):
        JokerColorConfig.from_dict({name: True})
    with pytest.raises(ValueError):
        JokerColorConfig(**{name: True})


@pytest.mark.parametrize("name", ["min_color_fraction", "min_background_fraction"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), 1.01])
def test_color_config_fractions_reject_nonfinite_and_out_of_range(name, invalid) -> None:
    with pytest.raises(ValueError):
        JokerColorConfig.from_dict({name: invalid})
    with pytest.raises(ValueError):
        JokerColorConfig(**{name: invalid})
