from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import src.vision.scene_recognizer as scene_recognizer_module

from src.pipeline.live_layout import LiveLayoutConfig
from src.state.cards import RANKS
from src.state.events import PlayerSeat
from src.vision.card_classifier import CardPrediction
from src.vision.scene_recognizer import (
    CvRemainingReader,
    RemainingTextMatch,
    SceneRecognizer,
    SeatRole,
    TemplateMatcher,
    _classify_role_badge,
    _classify_turn_controls,
    _decode_glyph_signature,
    _encode_glyph_signature,
    _glyph_similarity,
    _load_builtin_rank_references,
    _load_builtin_remaining_references,
    _rank_glyph_signature,
    _regularize_overlapping_hand_boxes,
    _remaining_glyph_signature,
    _remaining_signature_hole_count,
    _resolve_card_prediction,
    _resolve_roles_from_hand_count,
    _resolve_remaining,
    infer_visible_hand_count,
    infer_overlapping_hand_boxes,
    segment_card_boxes,
)


def _pattern(position: str) -> Image.Image:
    image = Image.new("L", (80, 40), 0)
    draw = ImageDraw.Draw(image)
    if position == "left":
        draw.rectangle((5, 5, 30, 35), fill=255)
    else:
        draw.rectangle((50, 5, 75, 35), fill=255)
    return image


def test_template_matcher_chooses_real_image_label(tmp_path: Path) -> None:
    for label, position in (("pass", "left"), ("neutral", "right")):
        directory = tmp_path / "pass" / label
        directory.mkdir(parents=True)
        _pattern(position).save(directory / "sample.png")

    matcher = TemplateMatcher(tmp_path)
    match = matcher.classify("pass", _pattern("left"))

    assert match.label == "pass"
    assert match.confidence == 1.0
    assert set(matcher.available_labels("pass")) == {"neutral", "pass"}


def test_role_badge_classifier_is_independent_of_seat_background() -> None:
    landlord = Image.new("RGB", (235, 115), (43, 73, 145))
    farmer = Image.new("RGB", (223, 115), (43, 73, 145))
    ImageDraw.Draw(landlord).rectangle(
        (55, 42, 174, 54),
        fill=(242, 177, 70),
    )
    ImageDraw.Draw(farmer).rectangle(
        (52, 42, 171, 54),
        fill=(244, 244, 244),
    )

    landlord_role, landlord_confidence = _classify_role_badge(landlord)
    farmer_role, farmer_confidence = _classify_role_badge(farmer)

    assert landlord_role is SeatRole.LANDLORD
    assert farmer_role is SeatRole.FARMER
    assert landlord_confidence >= 0.9
    assert farmer_confidence >= 0.9


def test_role_badge_classifier_rejects_background_without_glyphs() -> None:
    role, confidence = _classify_role_badge(
        Image.new("RGB", (223, 115), (43, 73, 145))
    )

    assert role is SeatRole.UNKNOWN
    assert confidence == 0.0


def test_turn_control_classifier_uses_yellow_buttons_not_blue_background() -> None:
    inactive = Image.new("RGB", (500, 240), (43, 73, 145))
    active = inactive.copy()
    draw = ImageDraw.Draw(active)
    draw.rounded_rectangle(
        (45, 55, 225, 205),
        radius=20,
        fill=(242, 166, 31),
    )
    draw.rounded_rectangle(
        (275, 55, 455, 205),
        radius=20,
        fill=(242, 166, 31),
    )

    active_value, active_confidence = _classify_turn_controls(active)
    inactive_value, inactive_confidence = _classify_turn_controls(inactive)

    assert active_value is True
    assert active_confidence >= 0.94
    assert inactive_value is False
    assert inactive_confidence >= 0.95


def test_segment_card_boxes_finds_separated_face_up_cards() -> None:
    image = Image.new("RGB", (300, 160), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 20, 85, 140), fill="white")
    draw.rectangle((120, 20, 180, 140), fill="white")
    draw.rectangle((215, 20, 275, 140), fill="white")

    boxes = segment_card_boxes(image)

    assert len(boxes) == 3
    assert boxes[0][0] <= 25
    assert boxes[-1][2] >= 275


def test_segment_card_boxes_finds_overlapped_cards() -> None:
    image = Image.new("RGB", (420, 240), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    for left in (20, 85, 150, 215):
        draw.rectangle(
            (left, 25, left + 140, 210),
            fill="white",
            outline=(75, 75, 75),
            width=2,
        )

    boxes = segment_card_boxes(image)

    assert len(boxes) == 4
    assert abs(boxes[0][0] - 20) <= 2
    assert [box[0] for box in boxes[1:]] == [85, 150, 215]


def test_segment_card_boxes_splits_tight_pair_in_single_white_region() -> None:
    image = Image.new("RGB", (780, 420), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    for left in (14, 88):
        draw.rounded_rectangle(
            (left, 91, left + 168, 332),
            radius=8,
            fill="white",
            outline=(75, 75, 75),
            width=2,
        )

    boxes = segment_card_boxes(image)

    assert len(boxes) == 2
    assert abs(boxes[0][0] - 14) <= 2
    assert abs(boxes[1][0] - 88) <= 2


def test_segment_card_boxes_ignores_separate_button_below_landlord_card() -> None:
    image = Image.new("RGB", (420, 500), (45, 75, 145))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((14, 90, 183, 332), radius=8, fill="white")
    draw.polygon(
        ((80, 90), (183, 90), (183, 220)),
        fill=(190, 35, 35),
    )
    draw.rectangle((14, 480, 183, 499), fill="white")

    boxes = segment_card_boxes(image)

    assert len(boxes) == 1
    assert boxes[0][1] <= 90
    assert boxes[0][3] < 360


def test_infer_overlapping_hand_boxes_uses_visible_white_extent() -> None:
    image = Image.new("RGB", (600, 220), (30, 60, 130))
    ImageDraw.Draw(image).rectangle((40, 10, 560, 215), fill="white")

    boxes = infer_overlapping_hand_boxes(image, 17)

    assert len(boxes) == 17
    assert boxes[0][0] == 40
    assert boxes[-1][2] <= image.width
    assert all(box[3] <= image.height for box in boxes)


def test_infer_overlapping_hand_boxes_accepts_centered_short_hand() -> None:
    image = Image.new("RGB", (900, 260), (30, 60, 130))
    draw = ImageDraw.Draw(image)
    starts = (330, 390, 450)
    for left in starts:
        draw.rounded_rectangle(
            (left, 10, left + 150, 240),
            radius=6,
            fill="white",
            outline=(90, 90, 90),
            width=2,
        )

    assert infer_visible_hand_count(image, maximum=20) == 3
    boxes = infer_overlapping_hand_boxes(image, 3)

    assert len(boxes) == 3
    assert all(
        abs(box[0] - expected) <= 3
        for box, expected in zip(boxes, starts)
    )


def test_infer_overlapping_hand_boxes_accepts_one_and_two_card_endgames() -> None:
    for starts in ((375,), (345, 415)):
        image = Image.new("RGB", (900, 260), (30, 60, 130))
        draw = ImageDraw.Draw(image)
        for left in starts:
            draw.rounded_rectangle(
                (left, 10, left + 150, 240),
                radius=6,
                fill="white",
                outline=(90, 90, 90),
                width=2,
            )

        assert infer_visible_hand_count(image, maximum=20) == len(starts)
        boxes = infer_overlapping_hand_boxes(image, len(starts))

        assert len(boxes) == len(starts)
        assert all(
            abs(box[0] - expected) <= 3
            for box, expected in zip(boxes, starts)
        )


def test_visible_count_rejects_two_internal_edges_inside_one_card(
    monkeypatch,
) -> None:
    image = Image.new("RGB", (900, 420), (30, 60, 130))
    ImageDraw.Draw(image).rounded_rectangle(
        (310, 10, 590, 405),
        radius=8,
        fill="white",
        outline=(80, 80, 80),
        width=2,
    )
    false_edges = (
        (310, 10, 460, 277),
        (364, 10, 514, 277),
    )
    monkeypatch.setattr(
        scene_recognizer_module,
        "_segment_overlapping_card_boxes",
        lambda *_args, **_kwargs: false_edges,
    )

    assert infer_visible_hand_count(image, maximum=20) == 1


def test_infer_overlapping_hand_boxes_keeps_card_tops_above_notice() -> None:
    image = Image.new("RGB", (900, 260), (30, 60, 130))
    draw = ImageDraw.Draw(image)
    for index in range(16):
        left = 15 + index * 48
        draw.rounded_rectangle(
            (left, 10, left + 150, 240),
            radius=6,
            fill="white",
            outline=(90, 90, 90),
            width=2,
        )
    # The client displays this kind of full-width notice after an opponent
    # plays a combination the user cannot beat.
    draw.rectangle((0, 82, image.width, 170), fill=(75, 75, 85))

    boxes = infer_overlapping_hand_boxes(image, 16)

    assert len(boxes) == 16
    assert all(box[1] <= 12 for box in boxes)
    assert all(box[3] > 100 for box in boxes)


def test_pass_notice_classifier_uses_upper_text_band_not_lower_card_body() -> None:
    neutral = Image.new("RGB", (330, 184), (45, 75, 145))
    notice = neutral.copy()
    draw = ImageDraw.Draw(notice)
    for left in (35, 55, 95, 120, 155, 180):
        draw.rectangle((left, 35, left + 12, 62), fill=(240, 240, 230))
    lower_card = neutral.copy()
    ImageDraw.Draw(lower_card).rounded_rectangle(
        (80, 88, 250, 183),
        radius=8,
        fill="white",
    )

    assert scene_recognizer_module._classify_pass_notice(neutral) == 0.0
    assert scene_recognizer_module._classify_pass_notice(notice) >= 0.82
    assert scene_recognizer_module._classify_pass_notice(lower_card) == 0.0


def test_structural_pass_confidence_is_used_by_seat_observation(
    tmp_path: Path,
) -> None:
    config = LiveLayoutConfig(templates_dir=tmp_path)
    image = Image.new("RGB", (1000, 600), (45, 75, 145))
    left, top, right, bottom = config.roi("right_pass").to_pixel_box(image.size)
    draw = ImageDraw.Draw(image)
    for offset in (15, 30, 48, 66, 86, 104):
        draw.rectangle(
            (
                left + offset,
                top + 12,
                left + offset + 8,
                top + 27,
            ),
            fill=(240, 240, 230),
        )
    recognizer = SceneRecognizer(
        config,
        predictor=lambda _: (),
        remaining_reader=lambda _: {},
    )

    observation = recognizer._observe_seat(  # noqa: SLF001
        image,
        PlayerSeat.RIGHT,
        role=(SeatRole.FARMER, 0.99),
        remaining=(17, 0.99, True),
    )

    assert observation.signal.value == "pass"
    assert observation.confidence >= config.pass_threshold
    assert observation.pass_confidence >= config.pass_threshold


def test_infer_visible_hand_count_reads_overlapping_card_edges() -> None:
    image = Image.new("RGB", (900, 260), (30, 60, 130))
    draw = ImageDraw.Draw(image)
    for index in range(14):
        left = 20 + index * 50
        draw.rounded_rectangle(
            (left, 10, left + 150, 240),
            radius=6,
            fill="white",
            outline=(90, 90, 90),
            width=2,
        )

    assert infer_visible_hand_count(image, maximum=17) == 14


def test_hand_grid_removes_joker_glyph_edge_before_and_after_play() -> None:
    def boxes(starts: list[int]) -> tuple[tuple[int, int, int, int], ...]:
        return tuple((start, 10, start + 126, 220) for start in starts)

    initial_card_starts = [0, *range(102, 2040, 102)]
    after_play_card_starts = [0, *range(107, 2033, 107)]

    initial = _regularize_overlapping_hand_boxes(
        boxes([0, 26, *initial_card_starts[1:]])
    )
    after_play = _regularize_overlapping_hand_boxes(
        boxes([0, 27, *after_play_card_starts[1:]])
    )

    assert len(initial) == 20
    assert [box[0] for box in initial] == initial_card_starts
    assert len(after_play) == 19
    assert [box[0] for box in after_play] == after_play_card_starts


def test_confident_cnn_joker_is_not_overridden_by_normal_j_reference() -> None:
    prediction = CardPrediction(
        rank="BJ",
        confidence=0.999,
        probabilities={"BJ": 0.999},
    )

    assert _resolve_card_prediction(
        prediction,
        ("J", 0.997),
    ) == ("BJ", 0.999)


def test_low_confidence_joker_can_still_be_corrected_by_reference() -> None:
    prediction = CardPrediction(
        rank="BJ",
        confidence=0.62,
        probabilities={"BJ": 0.62},
    )

    assert _resolve_card_prediction(
        prediction,
        ("J", 0.98),
    ) == ("J", 0.98)


def test_rank_reference_still_corrects_non_joker_prediction() -> None:
    prediction = CardPrediction(
        rank="Q",
        confidence=0.71,
        probabilities={"Q": 0.71},
    )

    assert _resolve_card_prediction(
        prediction,
        ("K", 0.97),
    ) == ("K", 0.97)


def test_seed_hand_references_accepts_safe_midgame_scan(
    monkeypatch,
) -> None:
    ranks = ("BJ", "2", "A", "K", "K", "K", "J", "9", "5")
    monkeypatch.setattr(
        scene_recognizer_module,
        "infer_visible_hand_count",
        lambda _image, maximum: len(ranks),
    )
    monkeypatch.setattr(
        scene_recognizer_module,
        "infer_overlapping_hand_boxes",
        lambda _image, count: tuple(
            (index * 10, 0, index * 10 + 9, 20)
            for index in range(count)
        ),
    )
    recognizer = SceneRecognizer(
        LiveLayoutConfig(),
        predictor=lambda crops: tuple(
            CardPrediction(
                rank=rank,
                confidence=0.99,
                probabilities={rank: 0.99},
            )
            for rank, _crop in zip(ranks, crops, strict=True)
        ),
        remaining_reader=lambda _: {},
    )

    cards = recognizer.seed_hand_references(
        Image.new("RGB", (1000, 600), "navy")
    )

    assert tuple(card.rank for card in cards) == ranks


def test_twenty_visible_cards_override_misleading_role_templates() -> None:
    roles = {
        PlayerSeat.SELF: (SeatRole.FARMER, 0.92),
        PlayerSeat.LEFT: (SeatRole.LANDLORD, 0.93),
        PlayerSeat.RIGHT: (SeatRole.FARMER, 0.97),
    }

    resolved = _resolve_roles_from_hand_count(
        roles,
        visible_hand_count=20,
    )

    assert resolved == {
        PlayerSeat.SELF: (SeatRole.LANDLORD, 1.0),
        PlayerSeat.LEFT: (SeatRole.FARMER, 1.0),
        PlayerSeat.RIGHT: (SeatRole.FARMER, 1.0),
    }


def _synthetic_rank_card(scale: float, *, rank: str) -> Image.Image:
    width, height = round(126 * scale), round(210 * scale)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    stroke = max(3, round(7 * scale))
    if rank == "6":
        draw.ellipse(
            (
                round(22 * scale),
                round(18 * scale),
                round(78 * scale),
                round(100 * scale),
            ),
            outline="black",
            width=stroke,
        )
        draw.line(
            (
                round(25 * scale),
                round(55 * scale),
                round(50 * scale),
                round(18 * scale),
            ),
            fill="black",
            width=stroke,
        )
    else:
        draw.ellipse(
            (
                round(22 * scale),
                round(15 * scale),
                round(78 * scale),
                round(60 * scale),
            ),
            outline="black",
            width=stroke,
        )
        draw.ellipse(
            (
                round(22 * scale),
                round(55 * scale),
                round(78 * scale),
                round(105 * scale),
            ),
            outline="black",
            width=stroke,
        )
    return image


def test_rank_glyph_signature_is_scale_stable_and_rank_specific() -> None:
    small_six = _rank_glyph_signature(_synthetic_rank_card(1.0, rank="6"))
    large_six = _rank_glyph_signature(_synthetic_rank_card(1.35, rank="6"))
    large_eight = _rank_glyph_signature(_synthetic_rank_card(1.35, rank="8"))

    assert _glyph_similarity(small_six, large_six) >= 0.75
    assert _glyph_similarity(small_six, large_eight) < 0.70


def test_scene_recognizer_preloads_real_rank_glyph_templates(
    tmp_path: Path,
) -> None:
    template_dir = tmp_path / "rank" / "6"
    template_dir.mkdir(parents=True)
    _synthetic_rank_card(1.0, rank="6").save(template_dir / "sample.png")
    recognizer = SceneRecognizer(
        LiveLayoutConfig(templates_dir=tmp_path),
        predictor=lambda _: (),
        remaining_reader=lambda _: {},
    )

    match = recognizer._match_rank_reference(  # noqa: SLF001
        _synthetic_rank_card(1.2, rank="6"),
        minimum_similarity=0.68,
        minimum_margin=0.05,
    )

    assert match is not None
    assert match[0] == "6"
    assert match[1] >= 0.9


def test_builtin_rank_glyph_asset_is_complete_and_decodable() -> None:
    references = _load_builtin_rank_references()

    assert set(references) == set(RANKS)
    assert all(references[rank] for rank in RANKS)
    sample = references["Q"][0]
    assert _decode_glyph_signature(
        _encode_glyph_signature(sample)
    ) == sample


def test_native_text_count_overrides_unverified_whole_roi_template() -> None:
    count, confidence, verified = _resolve_remaining(
        template_count=16,
        template_confidence=0.979,
        text_match=RemainingTextMatch(count=13, confidence=1.0),
        template_threshold=0.78,
    )

    assert count == 13
    assert confidence == 1.0
    assert verified is True


def _render_counter(value: str) -> Image.Image:
    remaining = _load_builtin_remaining_references()
    ranks = _load_builtin_rank_references()
    image = Image.new("RGB", (180, 100), (43, 73, 145))
    left = 15
    for digit in value:
        signature = (remaining.get(digit) or ranks[digit])[0]
        for index in signature:
            y, x = divmod(index, 64)
            image.putpixel((left + x, 18 + y), (245, 180, 50))
        left += 70
    return image


def test_cv_remaining_reader_uses_bundled_digits_without_local_templates() -> None:
    references = _load_builtin_remaining_references()
    reader = CvRemainingReader(template_images=())

    assert set(references) == {"0", "1", "2"}
    for expected in (1, 9, 10, 12, 16, 17, 20):
        match = reader.read(_render_counter(str(expected)))
        assert match.count == expected
        assert match.confidence >= 0.80


def test_cv_remaining_reader_ignores_thin_animated_yellow_streak() -> None:
    reader = CvRemainingReader(template_images=())
    image = _render_counter("12")
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 20, 5, 53), fill=(245, 180, 50))

    match = reader.read(image)

    assert match.count == 12
    assert match.confidence >= 0.80


def test_remaining_digit_topology_distinguishes_five_from_six() -> None:
    open_five = Image.new("L", (32, 48), 0)
    five_draw = ImageDraw.Draw(open_five)
    five_draw.line((5, 5, 25, 5), fill=255, width=4)
    five_draw.line((5, 5, 5, 25), fill=255, width=4)
    five_draw.line((5, 25, 24, 25), fill=255, width=4)
    five_draw.line((24, 25, 24, 42), fill=255, width=4)
    five_draw.line((5, 42, 24, 42), fill=255, width=4)
    closed_six = open_five.copy()
    ImageDraw.Draw(closed_six).line(
        (5, 25, 5, 42),
        fill=255,
        width=4,
    )

    five_signature = _remaining_glyph_signature(
        np.asarray(open_five, dtype=np.uint8) > 0
    )
    six_signature = _remaining_glyph_signature(
        np.asarray(closed_six, dtype=np.uint8) > 0
    )

    assert _remaining_signature_hole_count(five_signature) == 0
    assert _remaining_signature_hole_count(six_signature) == 1


def test_similar_whole_roi_template_is_not_verified_without_text() -> None:
    count, confidence, verified = _resolve_remaining(
        template_count=16,
        template_confidence=0.979,
        text_match=None,
        template_threshold=0.78,
    )

    assert count == 16
    assert confidence == 0.979
    assert verified is False
