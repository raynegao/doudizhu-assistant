from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import src.vision.scene_recognizer as module
from src.config.live_layout import LiveLayoutConfig, NormalizedBox, live_layout_from_dict
from src.config.table_crops import TableRankCropConfig
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.card_regions import CardRegion, CardRegionKind
from src.vision.scene_types import CardFusionEvidence, SeatRole, TemplateMatch


@pytest.mark.parametrize("field", ["body_height_ratio", "width_height_ratio", "visible_corner_height_ratio"])
@pytest.mark.parametrize("bad", [True, "0.5", None, -1, 0, 1.1, float("nan"), float("inf"), 10**1000])
def test_crop_ratio_must_be_valid_finite_number(field, bad):
    with pytest.raises(ValueError, match="table_rank_crop"):
        TableRankCropConfig.from_dict({field: bad})


@pytest.mark.parametrize("field", ["minimum_height", "minimum_width"])
@pytest.mark.parametrize("bad", [True, "24", 1.5, None, -1, 0])
def test_crop_minimum_must_be_positive_integer(field, bad):
    with pytest.raises(ValueError, match="table_rank_crop"):
        TableRankCropConfig.from_dict({field: bad})


@pytest.mark.parametrize("bad", [None, [], "default", {"extra": 1}])
def test_bad_config_object_rejected(bad):
    with pytest.raises(ValueError, match="table_rank_crop"):
        TableRankCropConfig.from_dict(bad)


def test_live_layout_roundtrip_preserves_explicit_crop_contract():
    crop = TableRankCropConfig(body_height_ratio=0.5, width_height_ratio=0.7)
    config = LiveLayoutConfig(table_rank_crop=crop)
    assert live_layout_from_dict(json.loads(json.dumps(config.to_payload()))).table_rank_crop == crop
    assert live_layout_from_dict({}).table_rank_crop == TableRankCropConfig()
    with pytest.raises(ValueError, match="table_rank_crop"):
        LiveLayoutConfig(table_rank_crop={})


@pytest.mark.parametrize("kind", list(CardRegionKind))
def test_actual_input_boxes_are_explicit_and_not_double_cropped(kind):
    region = CardRegion((100, 20, 269, 260), kind)
    expected = {
        CardRegionKind.FULL_CARD: (100, 20, 177, 148),
        CardRegionKind.VISIBLE_CORNER: (100, 20, 269, 183),
        CardRegionKind.RANK_CROP: region.box,
    }[kind]
    assert region.classifier_box(TableRankCropConfig()) == expected
    pixels = np.arange(300 * 300 * 3, dtype=np.uint8).reshape(300, 300, 3)
    image = Image.fromarray(pixels)
    assert region.classifier_crop(image, TableRankCropConfig()).tobytes() == image.crop(expected).tobytes()


def test_crop_contract_independent_of_predicted_rank_and_bounded_by_body():
    config = TableRankCropConfig(minimum_height=1000, minimum_width=1000)
    region = CardRegion((1, 2, 18, 27), CardRegionKind.FULL_CARD)
    assert region.classifier_box(config) == region.box
    with pytest.raises(ValueError, match="outside"):
        region.classifier_crop(Image.new("RGB", (17, 26)), config)


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (1, 1, 10, 0), (-1, 0, 10, 10), (False, 0, 10, 10), (0, 0, 10), [0, 0, 10, 10]])
def test_invalid_region_cannot_create_padded_or_empty_prediction(box):
    with pytest.raises(ValueError, match="card region"):
        CardRegion(box, CardRegionKind.FULL_CARD)


def test_geometry_kind_is_not_inferred_from_box_aspect():
    box = (10, 10, 100, 170)
    full = CardRegion(box, CardRegionKind.FULL_CARD)
    already_cropped = CardRegion(box, CardRegionKind.RANK_CROP)
    assert full.classifier_box(TableRankCropConfig()) != already_cropped.classifier_box(TableRankCropConfig())
    with pytest.raises(ValueError, match="geometry kind"):
        CardRegion(box, "full_card")


@pytest.mark.parametrize("overlapped", [False, True])
def test_segmentation_tags_source_branch_while_compatibility_boxes_stay_identical(overlapped):
    image = Image.new("RGB", (780, 420), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    for left in (14, 88) if overlapped else (14, 400):
        draw.rounded_rectangle((left, 91, left + 168, 332), radius=8, fill="white", outline=(75, 75, 75), width=2)
    regions = module.segment_card_regions(image, table_edges=True)
    assert len(regions) == 2
    expected = CardRegionKind.VISIBLE_CORNER if overlapped else CardRegionKind.FULL_CARD
    assert all(region.kind is expected for region in regions)
    assert tuple(region.box for region in regions) == module.segment_card_boxes(image, table_edges=True)


def test_stacked_corner_rois_keep_legacy_model_preprocessing():
    image = Image.new("RGB", (780, 420), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    starts = tuple(14 + index * 74 for index in range(9))
    for left in starts:
        draw.rounded_rectangle((left, 84, left + 168, 324), radius=8, fill="white", outline=(75, 75, 75), width=2)
    draw.rounded_rectangle((starts[-1], 194, starts[-1] + 168, 414), radius=8, fill="white", outline=(75, 75, 75), width=2)
    regions = module.segment_card_regions(image, table_edges=True)
    assert len(regions) == 10
    assert all(region.kind is CardRegionKind.VISIBLE_CORNER for region in regions)
    assert all(region.classifier_crop(image, TableRankCropConfig()).tobytes()
               == module._card_rank_crop(image.crop(region.box)).tobytes() for region in regions)


@pytest.mark.parametrize("rank", ["3", "10", "J", "A", "SJ", "BJ"])
def test_observe_seat_sends_branch_correct_pixels_and_logs_input_without_touching_score(rank, monkeypatch):
    image = Image.new("RGB", (400, 300), "white")
    regions = (CardRegion((0, 0, 169, 240), CardRegionKind.FULL_CARD),
               CardRegion((200, 0, 290, 160), CardRegionKind.RANK_CROP))
    config = TableRankCropConfig(body_height_ratio=0.5, width_height_ratio=0.6)
    rois = dict(LiveLayoutConfig().rois)
    rois["left_play"] = rois["left_pass"] = NormalizedBox(0, 0, 1, 1)
    instance = object.__new__(module.SceneRecognizer)
    instance.config = LiveLayoutConfig(rois=rois, table_rank_crop=config)
    instance.templates = SimpleNamespace(classify=lambda *_: TemplateMatch(None, 0.0))
    received = []

    def predict(crops):
        received.extend(crops)
        return tuple(CardPrediction(rank, 0.912345678, {rank: 0.912345678}) for _ in crops)

    instance.predictor = predict
    instance._rank_references = {}
    monkeypatch.setattr(module, "segment_card_regions", lambda *_args, **_kwargs: regions)
    monkeypatch.setattr(module, "_classify_pass_notice", lambda _: 0.0)
    observation = instance._observe_seat(image, PlayerSeat.LEFT, role=(SeatRole.FARMER, 0.99), remaining=(10, 0.99, True))
    assert [crop.size for crop in received] == [(72, 120), (90, 160)]
    assert [card.box for card in observation.cards] == [region.box for region in regions]
    for card, region in zip(observation.cards, regions, strict=True):
        assert card.evidence.raw_confidence == 0.912345678
        assert card.evidence.input_geometry == region.kind.value
        assert card.evidence.input_box == region.classifier_box(config)
        # No color in this white synthetic image: original Joker identity gate
        # still rejects; a geometry change cannot boost the score or bypass it.
        assert card.confidence == (0 if rank in {"SJ", "BJ"} else 0.912345678)


def test_input_diagnostics_do_not_modify_old_payload_or_tracking_fingerprint():
    from src.tracking.observation_evidence import _seat_fingerprint
    from src.vision.scene_types import SeatObservation, VisualCard, VisualSignal

    original = CardFusionEvidence("A", 0.9)
    annotated = replace(original, input_geometry="full_card", input_box=(1, 2, 30, 40))
    assert "input_geometry" not in original.to_log_payload()
    assert annotated.to_log_payload()["input_box"] == [1, 2, 30, 40]
    before = SeatObservation(PlayerSeat.LEFT, VisualSignal.PLAY, (VisualCard("A", 0.9, (1, 2, 50, 90), original),))
    after = replace(before, cards=(replace(before.cards[0], evidence=annotated),))
    assert _seat_fingerprint(before) == _seat_fingerprint(after)


def test_pure_card_geometry_import_does_not_load_model_or_rules():
    completed = subprocess.run([sys.executable, "-c", "import sys; import src.vision.card_regions; assert 'torch' not in sys.modules; assert not any(x.startswith(('src.logic', 'src.state', 'src.pipeline')) for x in sys.modules)"], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
