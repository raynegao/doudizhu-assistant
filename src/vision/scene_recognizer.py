from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageOps, ImageStat
import torch

from src.capture.screen_geometry import CapturedWindow
from src.pipeline.live_layout import LiveLayoutConfig
from src.state.cards import CardSet
from src.state.events import PlayerSeat
from src.vision.card_classifier import (
    CardPrediction,
    load_checkpoint,
    predict_tensors,
    preprocess_image,
    select_device,
)


class VisualSignal(str, Enum):
    NEUTRAL = "neutral"
    PLAY = "play"
    PASS = "pass"
    UNKNOWN = "unknown"


class SeatRole(str, Enum):
    LANDLORD = "landlord"
    FARMER = "farmer"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TemplateMatch:
    label: str | None
    confidence: float
    template: str | None = None


@dataclass(frozen=True)
class VisualCard:
    rank: str
    confidence: float
    box: tuple[int, int, int, int]

    def to_log_payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "confidence": round(self.confidence, 6),
            "box": list(self.box),
        }


@dataclass(frozen=True)
class SeatObservation:
    seat: PlayerSeat
    signal: VisualSignal
    cards: tuple[VisualCard, ...] = ()
    remaining_count: int | None = None
    role: SeatRole = SeatRole.UNKNOWN
    confidence: float = 0.0
    pass_confidence: float = 0.0
    remaining_confidence: float = 0.0
    role_confidence: float = 0.0

    @property
    def card_set(self) -> CardSet:
        return CardSet.parse(card.rank for card in self.cards)

    def to_log_payload(self) -> dict[str, object]:
        return {
            "seat": self.seat.value,
            "signal": self.signal.value,
            "cards": [card.to_log_payload() for card in self.cards],
            "remaining_count": self.remaining_count,
            "role": self.role.value,
            "confidence": round(self.confidence, 6),
            "pass_confidence": round(self.pass_confidence, 6),
            "remaining_confidence": round(self.remaining_confidence, 6),
            "role_confidence": round(self.role_confidence, 6),
        }


@dataclass(frozen=True)
class SceneObservation:
    frame_id: int
    timestamp: float
    window_pixel_box: tuple[int, int, int, int]
    self_hand: tuple[VisualCard, ...]
    seats: tuple[SeatObservation, ...]
    self_turn: bool | None
    self_turn_confidence: float
    confidence: float
    warnings: tuple[str, ...] = ()

    def seat(self, seat: PlayerSeat) -> SeatObservation:
        for observation in self.seats:
            if observation.seat is seat:
                return observation
        raise KeyError(seat)

    @property
    def self_hand_set(self) -> CardSet:
        return CardSet.parse(card.rank for card in self.self_hand)

    def to_log_payload(self) -> dict[str, object]:
        return {
            "event": "scene_observation",
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "window_pixel_box": list(self.window_pixel_box),
            "self_hand": [card.to_log_payload() for card in self.self_hand],
            "seats": [seat.to_log_payload() for seat in self.seats],
            "self_turn": self.self_turn,
            "self_turn_confidence": round(self.self_turn_confidence, 6),
            "confidence": round(self.confidence, 6),
            "warnings": list(self.warnings),
        }


class TemplateMatcher:
    """Small real-image template matcher with no external OCR dependency."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, tuple[tuple[str, Path, Image.Image], ...]] = {}

    def available_labels(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted({label for label, _, _ in self._templates(kind)}))

    def classify(
        self,
        kind: str,
        image: Image.Image,
        *,
        allowed_labels: Iterable[str] | None = None,
    ) -> TemplateMatch:
        allowed = set(allowed_labels) if allowed_labels is not None else None
        candidates = [
            item
            for item in self._templates(kind)
            if allowed is None or item[0] in allowed
        ]
        if not candidates:
            return TemplateMatch(label=None, confidence=0.0)
        scored = [
            (_template_similarity(image, template), label, path)
            for label, path, template in candidates
        ]
        confidence, label, path = max(scored, key=lambda item: item[0])
        return TemplateMatch(
            label=label,
            confidence=confidence,
            template=path.as_posix(),
        )

    def _templates(self, kind: str) -> tuple[tuple[str, Path, Image.Image], ...]:
        if kind in self._cache:
            return self._cache[kind]
        directory = self.root / kind
        items: list[tuple[str, Path, Image.Image]] = []
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                label = path.parent.name if path.parent != directory else path.stem.split("__", 1)[0]
                try:
                    image = Image.open(path).convert("L")
                except OSError:
                    continue
                items.append((label, path, image.copy()))
        self._cache[kind] = tuple(items)
        return self._cache[kind]


CardPredictionFn = Callable[[Sequence[Image.Image]], Sequence[CardPrediction]]


class SceneRecognizer:
    def __init__(
        self,
        config: LiveLayoutConfig,
        *,
        predictor: CardPredictionFn | None = None,
        template_matcher: TemplateMatcher | None = None,
    ) -> None:
        self.config = config
        self.templates = template_matcher or TemplateMatcher(config.templates_dir)
        if predictor is None:
            device = select_device(config.device_name)
            model, classes, image_size = load_checkpoint(config.model_path, device=device)

            def model_predictor(crops: Sequence[Image.Image]) -> Sequence[CardPrediction]:
                if not crops:
                    return ()
                tensors = torch.stack([
                    preprocess_image(crop, image_size=image_size)
                    for crop in crops
                ])
                return predict_tensors(model, tensors, classes=classes, device=device)

            self.predictor = model_predictor
        else:
            self.predictor = predictor

    def observe(self, frame: CapturedWindow) -> SceneObservation:
        warnings: list[str] = []
        roles: dict[PlayerSeat, tuple[SeatRole, float]] = {}
        remaining: dict[PlayerSeat, tuple[int | None, float]] = {}

        for seat in PlayerSeat:
            role_match = self.templates.classify("role", self.config.crop(frame.image, f"{seat.value}_role"))
            role = _parse_role(role_match, self.config.template_threshold)
            roles[seat] = role, role_match.confidence
            if role is SeatRole.UNKNOWN:
                warnings.append(f"{seat.value}_role_unavailable")

        self_role = roles[PlayerSeat.SELF][0]
        expected_hand_count = 20 if self_role is SeatRole.LANDLORD else 17
        self_hand = self._observe_initial_hand(
            self.config.crop(frame.image, "self_hand"),
            expected_count=expected_hand_count,
        )
        if len(self_hand) != expected_hand_count:
            warnings.append(
                f"self_hand_not_initial: recognized={len(self_hand)} expected={expected_hand_count}"
            )

        for seat in PlayerSeat:
            count_crop = self.config.crop(frame.image, f"{seat.value}_remaining")
            match = self.templates.classify("remaining", count_crop)
            count = _parse_remaining(match, self.config.template_threshold)
            confidence = match.confidence
            if seat is PlayerSeat.SELF and len(self_hand) in {17, 20}:
                count = len(self_hand)
                confidence = _minimum_card_confidence(self_hand)
            remaining[seat] = count, confidence
            if count is None:
                warnings.append(f"{seat.value}_remaining_unavailable")

        seats = tuple(
            self._observe_seat(
                frame.image,
                seat,
                role=roles[seat],
                remaining=remaining[seat],
            )
            for seat in PlayerSeat
        )
        turn_match = self.templates.classify("turn", self.config.crop(frame.image, "self_turn"))
        self_turn: bool | None
        if turn_match.confidence < self.config.template_threshold:
            self_turn = None
            warnings.append("self_turn_unavailable")
        else:
            self_turn = turn_match.label == "active"

        confidences = [
            confidence
            for seat in seats
            for confidence in (
                seat.confidence,
                seat.remaining_confidence,
                seat.role_confidence,
            )
            if confidence > 0
        ]
        if self_hand:
            confidences.append(_minimum_card_confidence(self_hand))
        if turn_match.confidence > 0:
            confidences.append(turn_match.confidence)
        scene_confidence = min(confidences) if confidences else 0.0
        return SceneObservation(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            window_pixel_box=frame.pixel_box,
            self_hand=self_hand,
            seats=seats,
            self_turn=self_turn,
            self_turn_confidence=turn_match.confidence,
            confidence=scene_confidence,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _observe_seat(
        self,
        image: Image.Image,
        seat: PlayerSeat,
        *,
        role: tuple[SeatRole, float],
        remaining: tuple[int | None, float],
    ) -> SeatObservation:
        play_crop = self.config.crop(image, f"{seat.value}_play")
        pass_crop = self.config.crop(image, f"{seat.value}_pass")
        pass_match = self.templates.classify("pass", pass_crop)
        if (
            pass_match.label == "pass"
            and pass_match.confidence >= self.config.pass_threshold
        ):
            signal = VisualSignal.PASS
            cards: tuple[VisualCard, ...] = ()
            confidence = pass_match.confidence
        else:
            boxes = segment_card_boxes(play_crop)
            crops = [_card_rank_crop(play_crop.crop(box)) for box in boxes]
            predictions = self.predictor(crops)
            cards = tuple(
                VisualCard(
                    rank=prediction.rank,
                    confidence=prediction.confidence,
                    box=boxes[index],
                )
                for index, prediction in enumerate(predictions)
            )
            if cards and all(
                card.confidence >= self.config.confidence_threshold
                for card in cards
            ):
                signal = VisualSignal.PLAY
                confidence = _minimum_card_confidence(cards)
            elif cards:
                signal = VisualSignal.UNKNOWN
                confidence = _minimum_card_confidence(cards)
            else:
                signal = VisualSignal.NEUTRAL
                confidence = (
                    pass_match.confidence
                    if pass_match.label == "neutral"
                    else max(0.0, 1.0 - pass_match.confidence)
                )
        return SeatObservation(
            seat=seat,
            signal=signal,
            cards=cards,
            remaining_count=remaining[0],
            role=role[0],
            confidence=confidence,
            pass_confidence=pass_match.confidence,
            remaining_confidence=remaining[1],
            role_confidence=role[1],
        )

    def _observe_initial_hand(
        self,
        image: Image.Image,
        *,
        expected_count: int,
    ) -> tuple[VisualCard, ...]:
        boxes = infer_overlapping_hand_boxes(image, expected_count)
        if not boxes:
            return ()
        crops = [image.crop(box) for box in boxes]
        predictions = self.predictor(crops)
        return tuple(
            VisualCard(
                rank=prediction.rank,
                confidence=prediction.confidence,
                box=boxes[index],
            )
            for index, prediction in enumerate(predictions)
        )


def _template_similarity(image: Image.Image, template: Image.Image) -> float:
    target = ImageOps.autocontrast(image.convert("L"))
    reference = ImageOps.autocontrast(template.convert("L"))
    target = ImageOps.fit(target, reference.size, method=Image.Resampling.BILINEAR)
    difference = ImageStat.Stat(
        Image.frombytes(
            "L",
            reference.size,
            bytes(abs(left - right) for left, right in zip(target.tobytes(), reference.tobytes())),
        )
    ).mean[0]
    return max(0.0, min(1.0, 1.0 - difference / 255.0))


def _parse_role(match: TemplateMatch, threshold: float) -> SeatRole:
    if match.confidence < threshold or match.label not in {"landlord", "farmer"}:
        return SeatRole.UNKNOWN
    return SeatRole(match.label)


def _parse_remaining(match: TemplateMatch, threshold: float) -> int | None:
    if match.confidence < threshold or match.label is None:
        return None
    try:
        value = int(match.label)
    except ValueError:
        return None
    return value if 0 <= value <= 20 else None


def _minimum_card_confidence(cards: Sequence[VisualCard]) -> float:
    return min((card.confidence for card in cards), default=0.0)


def _is_card_white(pixel: tuple[int, ...]) -> bool:
    red, green, blue = pixel[:3]
    return max(red, green, blue) >= 175 and min(red, green, blue) >= 115


def _active_x_runs(
    image: Image.Image,
    *,
    min_column_ratio: float = 0.08,
    max_gap: int = 2,
) -> list[tuple[int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    minimum = max(3, round(height * min_column_ratio))
    active = [
        sum(_is_card_white(pixels[x, y]) for y in range(height)) >= minimum
        for x in range(width)
    ]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for x, value in enumerate(active):
        if value:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, x - gap + 1))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, width))
    return runs


def segment_card_boxes(image: Image.Image) -> tuple[tuple[int, int, int, int], ...]:
    """Locate separated or overlapped face-up cards on the blue table."""

    width, height = image.size
    minimum_width = max(10, round(width * 0.03))
    boxes: list[tuple[int, int, int, int]] = []
    rgb = image.convert("RGB")
    pixels = rgb.load()
    for left, right in _active_x_runs(rgb):
        if right - left < minimum_width:
            continue
        ys = [
            y
            for x in range(left, right)
            for y in range(height)
            if _is_card_white(pixels[x, y])
        ]
        if not ys:
            continue
        top = max(0, min(ys) - 2)
        bottom = min(height, max(ys) + 3)
        if bottom - top < max(12, round(height * 0.10)):
            continue
        box_width = right - left
        box_height = bottom - top
        aspect_ratio = box_width / box_height
        white_pixels = sum(
            _is_card_white(pixels[x, y])
            for x in range(left, right)
            for y in range(top, bottom)
        )
        white_ratio = white_pixels / max(1, box_width * box_height)
        if not 0.28 <= aspect_ratio <= 1.20 or white_ratio < 0.60:
            continue
        boxes.append((left, top, right, bottom))
    if boxes:
        return tuple(boxes)
    return _segment_overlapping_card_boxes(rgb)


def _segment_overlapping_card_boxes(
    image: Image.Image,
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = image.size
    x_runs = _active_x_runs(image)
    if not x_runs:
        return ()
    left, right = max(x_runs, key=lambda run: run[1] - run[0])
    row_counts = [
        sum(_is_card_white(image.getpixel((x, y))) for x in range(left, right))
        for y in range(height)
    ]
    minimum_row = max(8, round((right - left) * 0.25))
    row_runs = _boolean_runs(
        [count >= minimum_row for count in row_counts],
        max_gap=2,
    )
    row_runs = [
        run
        for run in row_runs
        if run[1] - run[0] >= max(18, round(height * 0.25))
    ]
    if not row_runs:
        return ()
    top, bottom = max(row_runs, key=lambda run: run[1] - run[0])
    card_height = bottom - top
    if card_height <= 0:
        return ()

    grayscale = image.convert("L")
    edge_scores = []
    for x in range(max(left + 1, 1), min(right + 1, width)):
        score = sum(
            abs(grayscale.getpixel((x, y)) - grayscale.getpixel((x - 1, y)))
            for y in range(top, bottom)
        ) / card_height
        edge_scores.append((x, score))
    if not edge_scores:
        return ()
    maximum_edge = max(score for _, score in edge_scores)
    threshold = max(14.0, maximum_edge * 0.34)
    edge_candidates = [x for x, score in edge_scores if score >= threshold]
    edge_groups = _integer_groups(edge_candidates, max_gap=2)
    score_by_x = dict(edge_scores)
    edges = [
        max(group, key=lambda x: score_by_x[x])
        for group in edge_groups
    ]
    internal_edges = [
        edge
        for edge in edges
        if edge > left + 3 and edge < right - 3
    ]
    starts = [left, *internal_edges]
    if len(starts) < 2:
        return ()

    rank_box_width = max(16, round(card_height * 0.38))
    rank_box_height = max(24, round(card_height * 0.68))
    boxes = [
        (
            start,
            top,
            min(width, start + rank_box_width),
            min(height, top + rank_box_height),
        )
        for start in starts
        if min(width, start + rank_box_width) - start >= 12
    ]
    return tuple(boxes)


def infer_overlapping_hand_boxes(
    image: Image.Image,
    count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    if count <= 0:
        return ()
    width, height = image.size
    rgb = image.convert("RGB")
    white_columns = _active_x_runs(rgb, min_column_ratio=0.15, max_gap=3)
    if not white_columns:
        return ()
    left = min(run[0] for run in white_columns)
    right = max(run[1] for run in white_columns)
    if right - left < width * 0.35:
        return ()
    row_counts = [
        sum(_is_card_white(rgb.getpixel((x, y))) for x in range(left, right))
        for y in range(height)
    ]
    row_runs = _boolean_runs(
        [count >= max(8, round((right - left) * 0.25)) for count in row_counts],
        max_gap=2,
    )
    if row_runs:
        start_y, card_bottom = max(
            row_runs,
            key=lambda run: run[1] - run[0],
        )
    else:
        start_y = max(0, round(height * 0.04))
        card_bottom = height
    full_card_height = max(24, card_bottom - start_y)
    full_card_width = max(16, round(full_card_height * 0.72))
    crop_height = max(24, round(full_card_height * 0.535))
    crop_width = max(16, round(crop_height * 0.60))
    if count == 1:
        step = 0
    else:
        step = max(1, int((right - left - full_card_width) / (count - 1)))
    boxes = []
    for index in range(count):
        x = left + index * step
        box = (
            max(0, min(x, width - crop_width)),
            start_y,
            max(0, min(x, width - crop_width)) + crop_width,
            min(height, start_y + crop_height),
        )
        boxes.append(box)
    return tuple(boxes)


def _boolean_runs(
    values: Sequence[bool],
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, value in enumerate(values):
        if value:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, index - gap + 1))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _integer_groups(
    values: Sequence[int],
    *,
    max_gap: int,
) -> list[tuple[int, ...]]:
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [tuple(group) for group in groups]


def _card_rank_crop(card: Image.Image) -> Image.Image:
    width, height = card.size
    return card.crop((0, 0, width, max(1, round(height * 0.68))))


__all__ = [
    "SceneObservation",
    "SceneRecognizer",
    "SeatObservation",
    "SeatRole",
    "TemplateMatch",
    "TemplateMatcher",
    "VisualCard",
    "VisualSignal",
    "infer_overlapping_hand_boxes",
    "segment_card_boxes",
]
