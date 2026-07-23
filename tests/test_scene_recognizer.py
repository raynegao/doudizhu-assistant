from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.vision.scene_recognizer import (
    RemainingTextMatch,
    TemplateMatcher,
    _resolve_remaining,
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


def test_infer_overlapping_hand_boxes_uses_visible_white_extent() -> None:
    image = Image.new("RGB", (600, 220), (30, 60, 130))
    ImageDraw.Draw(image).rectangle((40, 10, 560, 215), fill="white")

    boxes = infer_overlapping_hand_boxes(image, 17)

    assert len(boxes) == 17
    assert boxes[0][0] == 40
    assert boxes[-1][2] <= image.width
    assert all(box[3] <= image.height for box in boxes)


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
