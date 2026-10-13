from __future__ import annotations

import json
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.config.hand_geometry import ShortHandRecoveryConfig
from src.config.live_layout import LiveLayoutConfig, live_layout_from_dict
from src.config.self_table import SelfTableConfig
from src.vision.card_regions import CardRegion, CardRegionKind
from src.vision.scene_recognizer import (
    _self_table_body_bounds,
    _self_table_rank_regions,
    segment_card_regions,
)


def table(count=2, *, scale=1, dx=0, dy=0):
    image = Image.new("RGB", (780, 440), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    for index in range(count):
        x, y = 60 + index * 74 + dx, 80 + dy
        draw.rounded_rectangle((x, y, x + 168, y + 241), radius=8,
                               fill=(235, 235, 235), outline=(125, 125, 125), width=2)
        draw.text((x + 15, y + 12), "J", font=ImageFont.load_default(size=38), fill=(0, 0, 0))
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)


def certify(image, regions=None, config=None):
    regions = segment_card_regions(image, table_edges=True) if regions is None else regions
    return _self_table_body_bounds(image, regions, config=config or SelfTableConfig(),
                                   edge_config=ShortHandRecoveryConfig())


@pytest.mark.parametrize("count", [1, 2, 3, 5])
@pytest.mark.parametrize("offset", [(0, 0), (20, -20), (100, 30)])
def test_complete_fans_have_proven_body_height_and_no_cross_slot_inputs(count, offset):
    image = table(count, dx=offset[0], dy=offset[1])
    regions = segment_card_regions(image, table_edges=True)
    assert len(regions) == count
    bounds = certify(image, regions)
    assert bounds is not None
    config = LiveLayoutConfig()
    inputs = _self_table_rank_regions(regions, bounds, config)
    assert len(inputs) == count
    for index, region in enumerate(inputs):
        assert region.kind is CardRegionKind.RANK_CROP
        assert region.box[3] - region.box[1] == round((bounds[3] - bounds[1]) * .535)
        if index + 1 < count:
            assert region.box[2] <= inputs[index + 1].box[0]


@pytest.mark.parametrize("which", ["top", "bottom", "left", "right"])
def test_clipped_body_never_claims_complete(which):
    image = table()
    boxes = {"top": (0, 100, 780, 440), "bottom": (0, 0, 780, 300),
             "left": (80, 0, 780, 440), "right": (0, 0, 260, 440)}
    assert certify(image.crop(boxes[which])) is None


@pytest.mark.parametrize("kind", ["button", "glyph", "empty", "gray", "solid_white"])
def test_non_card_ui_is_rejected(kind):
    image = Image.new("RGB", (780, 440), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    if kind == "button":
        draw.rounded_rectangle((110, 100, 670, 300), radius=70, fill=(235, 175, 55))
    if kind in {"button", "glyph"}:
        draw.text((260, 155), "PASS", font=ImageFont.load_default(size=90), fill="white")
    if kind in {"gray", "solid_white"}:
        draw.rectangle((0, 0, 779, 439), fill=(135, 135, 135) if kind == "gray" else "white")
    assert certify(image) is None


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reversed", "shifted_top"])
def test_all_physical_seams_must_correspond_to_exact_proposed_slots(mutation):
    image = table(3)
    regions = segment_card_regions(image, table_edges=True)
    assert len(regions) == 3 and certify(image, regions)
    if mutation == "missing":
        regions = (regions[0], regions[2])
    elif mutation == "extra":
        regions = (regions[0], CardRegion((95, 80, 160, 220), CardRegionKind.VISIBLE_CORNER), *regions[1:])
    elif mutation == "duplicate":
        regions = (*regions, regions[-1])
    elif mutation == "reversed":
        regions = tuple(reversed(regions))
    else:
        box = regions[0].box
        regions = (CardRegion((box[0], box[1] + 15, box[2], box[3]), regions[0].kind), *regions[1:])
    assert certify(image, regions) is None


def test_middle_seam_erasure_fails_even_when_both_corners_remain():
    image = table()
    regions = segment_card_regions(image, table_edges=True)
    draw = ImageDraw.Draw(image)
    draw.rectangle((130, 120, 138, 285), fill=(235, 235, 235))
    assert certify(image, regions) is None


def test_short_lower_band_obstruction_cannot_hide_an_unrepresented_real_seam():
    image = table(3)
    draw = ImageDraw.Draw(image)
    draw.rectangle((130, 180, 140, 190), fill=(235, 235, 235))
    draw.rectangle((130, 286, 140, 302), fill=(235, 235, 235))
    regions = segment_card_regions(image, table_edges=True)
    assert len(regions) == 2  # Existing segmenter missed the partly obscured seam.
    assert certify(image, regions) is None  # Certifier must not fabricate a third slot.


def test_model_crop_uses_explicit_custom_ratio_not_observation_corner_height():
    image = table()
    regions = segment_card_regions(image, table_edges=True)
    bounds = certify(image, regions)
    assert bounds is not None
    config = LiveLayoutConfig()
    config = replace(config, table_rank_crop=replace(config.table_rank_crop, body_height_ratio=.4))
    inputs = _self_table_rank_regions(regions, bounds, config)
    assert all(region.box[3] - region.box[1] == round((bounds[3] - bounds[1]) * .4) for region in inputs)


def test_new_configuration_is_explicit_and_preserves_legacy_snapshots():
    payload = {"self_table": {"paper_min_channel": 185}}
    before = json.dumps(payload, sort_keys=True)
    config = live_layout_from_dict(payload)
    assert config.self_table.paper_min_channel == 185
    assert json.dumps(payload, sort_keys=True) == before
    assert live_layout_from_dict(config.to_payload()).self_table == config.self_table
    assert live_layout_from_dict({}).self_table == SelfTableConfig()
    assert certify(table(), config=SelfTableConfig(maximum_input_pixels=100)) is None


@pytest.mark.parametrize("values", [{"paper_min_channel": True}, {"paper_max_chroma": 256},
    {"minimum_paper_fraction": float("nan")}, {"minimum_row_fraction": 0},
    {"maximum_input_pixels": -1}, {"bogus": 1}])
def test_invalid_config_rejected(values):
    with pytest.raises(ValueError):
        SelfTableConfig.from_dict(values)
