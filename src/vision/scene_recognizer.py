from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import subprocess
import tempfile
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.capture.screen_geometry import CapturedWindow
from src.config.live_layout import LiveLayoutConfig
from src.state.cards import RANKS, CardSet, normalize_rank
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
    remaining_verified: bool = True

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
            "remaining_verified": self.remaining_verified,
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

    def template_images(
        self,
        kind: str,
    ) -> tuple[tuple[str, Image.Image], ...]:
        """Return cached template images for feature-based recognizers."""

        return tuple(
            (label, image.copy())
            for label, _, image in self._templates(kind)
        )

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
                    # Keep colour here. Template similarity converts to
                    # grayscale itself, while feature readers such as the
                    # yellow remaining-count segmenter need the original RGB
                    # pixels.
                    image = Image.open(path).convert("RGB")
                except OSError:
                    continue
                items.append((label, path, image.copy()))
        self._cache[kind] = tuple(items)
        return self._cache[kind]


CardPredictionFn = Callable[[Sequence[Image.Image]], Sequence[CardPrediction]]


@dataclass(frozen=True)
class RemainingTextMatch:
    count: int | None
    confidence: float


class CvRemainingReader:
    """Read the yellow remaining-card digits directly from the table pixels.

    macOS Vision frequently ignores a single outlined digit because it does
    not consider it a text line.  The game counter is much more constrained:
    it is always one or two yellow glyphs in the range 1..20.  Segmenting that
    colour and comparing normalized glyph silhouettes is both faster and more
    reliable for the live client.
    """

    def __init__(
        self,
        *,
        template_images: Sequence[tuple[str, Image.Image]] = (),
    ) -> None:
        rank_references = _load_builtin_rank_references()
        self._digit_references = _load_builtin_remaining_references()
        for digit in "23456789":
            references = self._digit_references.setdefault(digit, [])
            references.extend(rank_references.get(digit, ()))
        # Whole-counter templates are still useful, but only as sources of
        # individual glyph shapes.  This removes the seat background and card
        # stack that made whole-ROI template matching confuse 3 with 16.
        for label, image in template_images:
            if not label.isdigit():
                continue
            masks = _remaining_digit_masks(image)
            if len(masks) != len(label):
                continue
            for digit, mask in zip(label, masks, strict=True):
                signature = _remaining_glyph_signature(mask)
                if not signature:
                    continue
                references = self._digit_references.setdefault(digit, [])
                if any(
                    _glyph_similarity(signature, existing) >= 0.98
                    for existing in references
                ):
                    continue
                references.append(signature)
        self._digit_reference_holes = {
            digit: tuple(
                (reference, _remaining_signature_hole_count(reference))
                for reference in references
            )
            for digit, references in self._digit_references.items()
        }

    def __call__(
        self,
        images: Mapping[PlayerSeat, Image.Image],
    ) -> Mapping[PlayerSeat, RemainingTextMatch]:
        matches: dict[PlayerSeat, RemainingTextMatch] = {}
        for seat, image in images.items():
            match = self.read(image)
            if match.count is not None:
                matches[seat] = match
        return matches

    def read(self, image: Image.Image) -> RemainingTextMatch:
        masks = _remaining_digit_masks(image)
        if not 1 <= len(masks) <= 2:
            return RemainingTextMatch(None, 0.0)

        digits: list[str] = []
        confidences: list[float] = []
        for index, mask in enumerate(masks):
            digit, confidence = self._classify_digit(
                mask,
                index=index,
                digit_count=len(masks),
                prefix=digits,
            )
            if digit is None:
                return RemainingTextMatch(None, confidence)
            digits.append(digit)
            confidences.append(confidence)

        try:
            count = int("".join(digits))
        except ValueError:
            return RemainingTextMatch(None, 0.0)
        if not 1 <= count <= 20:
            return RemainingTextMatch(None, min(confidences, default=0.0))
        return RemainingTextMatch(count, min(confidences))

    def _classify_digit(
        self,
        mask: np.ndarray,
        *,
        index: int,
        digit_count: int,
        prefix: Sequence[str],
    ) -> tuple[str | None, float]:
        signature = _remaining_glyph_signature(mask)
        allowed_digits = set("0123456789")
        if digit_count == 1:
            allowed_digits.discard("0")
        elif index == 0:
            # The only two-digit values are 10..20.
            allowed_digits = {"1", "2"}
        elif prefix and prefix[0] == "2":
            # Once the first glyph is 2 the value can only be 20.
            allowed_digits = {"0"}
        observed_holes = _remaining_signature_hole_count(signature)
        topology_references = {
            digit: tuple(
                reference
                for reference, hole_count in references
                if hole_count == observed_holes
            )
            for digit, references in self._digit_reference_holes.items()
            if digit in allowed_digits
        }
        has_topology_match = any(topology_references.values())
        scored = sorted(
            (
                max(
                    _glyph_similarity(signature, reference)
                    for reference in (
                        topology_references.get(digit, ())
                        if has_topology_match
                        else references
                    )
                ),
                digit,
            )
            for digit, references in self._digit_references.items()
            if (
                references
                and digit in allowed_digits
                and (
                    not has_topology_match
                    or topology_references.get(digit)
                )
            )
        )
        if scored:
            best_score, best_digit = scored[-1]
            runner_up = scored[-2][0] if len(scored) > 1 else 0.0
            if best_score >= 0.50 and best_score - runner_up >= 0.025:
                confidence = min(
                    0.995,
                    0.72 + max(0.0, best_score - 0.45) * 1.35,
                )
                return best_digit, confidence

        # 1 is the only counter digit without a card-rank counterpart.  It is
        # either the complete value 1 or the leading digit of 10..19.
        if index == 0 and (digit_count == 1 or digit_count == 2):
            return "1", 0.86
        # A second unmatched glyph after 2 can only be zero because the complete
        # counter is constrained to 1..20.
        if index == 1 and prefix and prefix[0] == "2":
            return "0", 0.80
        return None, scored[-1][0] if scored else 0.0


class MacVisionRemainingReader:
    """Read the two opponent counters with macOS Vision text recognition."""

    def __init__(
        self,
        *,
        source_path: Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.source_path = source_path or Path(__file__).with_name(
            "macos_vision_ocr.swift"
        )
        self._runner = runner
        self._binary_path: Path | None = None

    def __call__(
        self,
        images: Mapping[PlayerSeat, Image.Image],
    ) -> Mapping[PlayerSeat, RemainingTextMatch]:
        binary = self._ensure_binary()
        seats = [
            seat
            for seat in (PlayerSeat.LEFT, PlayerSeat.RIGHT)
            if seat in images
        ]
        if not seats:
            return {}
        with tempfile.TemporaryDirectory(
            prefix="doudizhu-remaining-ocr-"
        ) as temp_dir:
            paths: list[Path] = []
            for seat in seats:
                path = Path(temp_dir) / f"{seat.value}.png"
                images[seat].convert("RGB").save(path)
                paths.append(path)
            try:
                result = self._runner(
                    [str(binary), *(str(path) for path in paths)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                return {}
        matches: dict[PlayerSeat, RemainingTextMatch] = {}
        for seat, line in zip(seats, result.stdout.splitlines(), strict=False):
            parts = line.rsplit("\t", 2)
            if len(parts) != 3:
                continue
            try:
                count = int(parts[1])
                confidence = float(parts[2])
            except ValueError:
                continue
            if 0 <= count <= 20 and 0.0 <= confidence <= 1.0:
                matches[seat] = RemainingTextMatch(count, confidence)
        return matches

    def _ensure_binary(self) -> Path:
        if self._binary_path is not None:
            return self._binary_path
        source = self.source_path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()[:12]
        binary = Path(tempfile.gettempdir()) / (
            f"doudizhu-macos-vision-ocr-{digest}"
        )
        if not binary.exists():
            temporary = binary.with_name(f"{binary.name}.{os.getpid()}.tmp")
            try:
                self._runner(
                    [
                        "/usr/bin/swiftc",
                        "-O",
                        str(self.source_path),
                        "-o",
                        str(temporary),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
        self._binary_path = binary
        return binary


RemainingReader = Callable[
    [Mapping[PlayerSeat, Image.Image]],
    Mapping[PlayerSeat, RemainingTextMatch],
]


class SceneRecognizer:
    def __init__(
        self,
        config: LiveLayoutConfig,
        *,
        predictor: CardPredictionFn | None = None,
        template_matcher: TemplateMatcher | None = None,
        remaining_reader: RemainingReader | None = None,
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
        self._rank_references = _load_builtin_rank_references()
        self._reference_hand_fingerprint: tuple[str, ...] | None = None
        self._load_rank_reference_templates()
        if remaining_reader is not None:
            self.remaining_reader: RemainingReader | None = remaining_reader
        else:
            self.remaining_reader = CvRemainingReader(
                template_images=self.templates.template_images("remaining"),
            )

    def observe(self, frame: CapturedWindow) -> SceneObservation:
        warnings: list[str] = []
        raw_roles: dict[PlayerSeat, tuple[SeatRole, float]] = {}
        remaining: dict[PlayerSeat, tuple[int | None, float, bool]] = {}

        for seat in PlayerSeat:
            role_crop = self.config.crop(
                frame.image,
                f"{seat.value}_role",
            )
            badge_role, badge_confidence = _classify_role_badge(role_crop)
            if badge_role is not SeatRole.UNKNOWN:
                raw_roles[seat] = badge_role, badge_confidence
                continue
            role_match = self.templates.classify("role", role_crop)
            role = _parse_role(role_match, self.config.template_threshold)
            raw_roles[seat] = role, role_match.confidence

        self_hand_crop = self.config.crop(frame.image, "self_hand")
        visible_hand_count = infer_visible_hand_count(
            self_hand_crop,
            maximum=20,
        )
        roles = _resolve_roles_from_hand_count(
            raw_roles,
            visible_hand_count=visible_hand_count,
        )
        for seat, (role, _) in roles.items():
            if role is SeatRole.UNKNOWN:
                warnings.append(f"{seat.value}_role_unavailable")

        self_role = roles[PlayerSeat.SELF][0]
        expected_hand_count = 20 if self_role is SeatRole.LANDLORD else 17
        self_hand = self._observe_hand(
            self_hand_crop,
            expected_count=expected_hand_count,
            visible_count=visible_hand_count,
        )
        if not self_hand:
            warnings.append(
                f"self_hand_unavailable: expected_at_most={expected_hand_count}"
            )
        self._remember_rank_references(self_hand_crop, self_hand)

        count_crops = {
            seat: self.config.crop(frame.image, f"{seat.value}_remaining")
            for seat in PlayerSeat
        }
        ocr_matches: Mapping[PlayerSeat, RemainingTextMatch] = {}
        if self.remaining_reader is not None:
            try:
                ocr_matches = self.remaining_reader(count_crops)
            except (OSError, subprocess.SubprocessError, ValueError):
                warnings.append("remaining_ocr_unavailable")

        for seat in PlayerSeat:
            count_crop = count_crops[seat]
            match = self.templates.classify("remaining", count_crop)
            template_count = _parse_remaining(
                match,
                max(self.config.template_threshold, 0.94),
            )
            count, confidence, verified = _resolve_remaining(
                template_count=template_count,
                template_confidence=match.confidence,
                text_match=ocr_matches.get(seat),
                template_threshold=self.config.template_threshold,
            )
            if (
                seat is PlayerSeat.SELF
                and 0 < len(self_hand) <= expected_hand_count
            ):
                count = len(self_hand)
                confidence = _minimum_card_confidence(self_hand)
                verified = True
            remaining[seat] = count, confidence, verified
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
        turn_crop = self.config.crop(frame.image, "self_turn")
        self_turn, turn_confidence = _classify_turn_controls(turn_crop)
        if self_turn is None:
            turn_match = self.templates.classify("turn", turn_crop)
            if turn_match.confidence < self.config.template_threshold:
                self_turn = None
                turn_confidence = turn_match.confidence
                warnings.append("self_turn_unavailable")
            else:
                self_turn = turn_match.label == "active"
                turn_confidence = turn_match.confidence

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
        if turn_confidence > 0:
            confidences.append(turn_confidence)
        scene_confidence = min(confidences) if confidences else 0.0
        return SceneObservation(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            window_pixel_box=frame.pixel_box,
            self_hand=self_hand,
            seats=seats,
            self_turn=self_turn,
            self_turn_confidence=turn_confidence,
            confidence=scene_confidence,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def seed_hand_references(
        self,
        full_window_image: Image.Image,
    ) -> tuple[VisualCard, ...]:
        """Restore per-round rank references from a saved initial table frame."""

        hand_image = self.config.crop(full_window_image, "self_hand")
        visible_count = infer_visible_hand_count(hand_image, maximum=20)
        # A round can be initialized from a safe mid-game self-turn scan, so
        # its reference frame may contain any physically valid number of
        # cards.  Restricting recovery to opening 17/20-card frames caused a
        # recognition-worker restart after the first action to discard a valid
        # checkpoint and rescan the table as a new approximate round.
        if not 1 <= visible_count <= 20:
            return ()
        cards = self._observe_hand(
            hand_image,
            expected_count=visible_count,
            visible_count=visible_count,
        )
        try:
            card_set = CardSet.parse(card.rank for card in cards)
        except ValueError:
            return ()
        if len(card_set) != visible_count:
            return ()
        self._remember_rank_references(hand_image, cards)
        return cards

    def _observe_seat(
        self,
        image: Image.Image,
        seat: PlayerSeat,
        *,
        role: tuple[SeatRole, float],
        remaining: tuple[int | None, float, bool],
    ) -> SeatObservation:
        play_crop = self.config.crop(image, f"{seat.value}_play")
        pass_crop = self.config.crop(image, f"{seat.value}_pass")
        pass_match = self.templates.classify("pass", pass_crop)
        pass_notice_confidence = _classify_pass_notice(pass_crop)
        if (
            pass_notice_confidence >= self.config.pass_threshold
            or (
                pass_match.label == "pass"
                and pass_match.confidence >= self.config.pass_threshold
            )
        ):
            signal = VisualSignal.PASS
            cards: tuple[VisualCard, ...] = ()
            confidence = max(
                pass_notice_confidence,
                (
                    pass_match.confidence
                    if pass_match.label == "pass"
                    else 0.0
                ),
            )
        else:
            # The self action ROI overlaps the large control buttons in this
            # client and produces false card-like regions. Self plays are more
            # reliably derived from the stable hand difference in the tracker.
            boxes = (
                ()
                if seat is PlayerSeat.SELF
                else segment_card_boxes(play_crop)
            )
            crops = [_card_rank_crop(play_crop.crop(box)) for box in boxes]
            predictions = self.predictor(crops)
            observed_cards: list[VisualCard] = []
            for index, prediction in enumerate(predictions):
                reference_match = self._match_rank_reference(
                    play_crop.crop(boxes[index]),
                    # Table cards are larger and can have a slightly different
                    # antialiased glyph from the hand reference. A strong
                    # runner-up margin still makes 0.70 overlap reliable; the
                    # previous 0.78 floor missed a real red 10 and left the CNN
                    # to call it K/7.
                    minimum_similarity=0.70,
                    minimum_margin=0.12,
                )
                rank, confidence = _resolve_card_prediction(
                    prediction,
                    reference_match,
                )
                observed_cards.append(
                    VisualCard(
                        rank=rank,
                        confidence=confidence,
                        box=boxes[index],
                    )
                )
            cards = tuple(observed_cards)
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
            pass_confidence=max(
                pass_notice_confidence,
                (
                    pass_match.confidence
                    if pass_match.label == "pass"
                    else 0.0
                ),
            ),
            remaining_confidence=remaining[1],
            role_confidence=role[1],
            remaining_verified=remaining[2],
        )

    def _observe_hand(
        self,
        image: Image.Image,
        *,
        expected_count: int,
        visible_count: int | None = None,
    ) -> tuple[VisualCard, ...]:
        if visible_count is None:
            visible_count = infer_visible_hand_count(
                image,
                maximum=20,
            )
        count = visible_count or expected_count
        boxes = infer_overlapping_hand_boxes(image, count)
        if not boxes:
            return ()
        crops = [image.crop(box) for box in boxes]
        predictions = self.predictor(crops)
        observed: list[VisualCard] = []
        for index, prediction in enumerate(predictions):
            reference_match = self._match_rank_reference(
                crops[index],
                minimum_similarity=0.68,
                minimum_margin=0.05,
            )
            rank, confidence = _resolve_card_prediction(
                prediction,
                reference_match,
            )
            observed.append(
                VisualCard(
                    rank=rank,
                    confidence=confidence,
                    box=boxes[index],
                )
            )
        return tuple(observed)

    def _load_rank_reference_templates(self) -> None:
        for label, image in self.templates.template_images("rank"):
            try:
                rank = normalize_rank(label)
            except ValueError:
                continue
            signature = _rank_glyph_signature(image)
            if not signature:
                continue
            references = self._rank_references.setdefault(rank, [])
            if any(
                _glyph_similarity(signature, existing) >= 0.98
                for existing in references
            ):
                continue
            references.append(signature)
            del references[:-12]

    def _remember_rank_references(
        self,
        hand_image: Image.Image,
        cards: Sequence[VisualCard],
    ) -> None:
        fingerprint = tuple(card.rank for card in cards)
        if fingerprint == self._reference_hand_fingerprint:
            return
        self._reference_hand_fingerprint = fingerprint
        for card in cards:
            if card.confidence < max(0.85, self.config.confidence_threshold):
                continue
            signature = _rank_glyph_signature(hand_image.crop(card.box))
            if not signature:
                continue
            references = self._rank_references.setdefault(card.rank, [])
            if any(
                _glyph_similarity(signature, existing) >= 0.98
                for existing in references
            ):
                continue
            references.append(signature)
            del references[:-4]

    def _match_rank_reference(
        self,
        card_image: Image.Image,
        *,
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.12,
    ) -> tuple[str, float] | None:
        signature = _rank_glyph_signature(card_image)
        if not signature:
            return None
        scores = sorted(
            (
                max(
                    _glyph_similarity(signature, reference)
                    for reference in references
                ),
                rank,
            )
            for rank, references in self._rank_references.items()
            if references
        )
        if not scores:
            return None
        best_score, best_rank = scores[-1]
        runner_up = scores[-2][0] if len(scores) > 1 else 0.0
        # Large table cards can make a joker reference moderately similar to
        # a rounded numeral (notably 6), even though the correct per-round hand
        # glyph is an almost exact match.  At >=0.90 absolute overlap, a 0.025
        # lead is already stronger evidence than the cross-scale CNN trained
        # on compact hand crops.  Keep the wider configured margin for all
        # weaker matches.
        required_margin = (
            min(minimum_margin, 0.025)
            if best_score >= 0.90
            else minimum_margin
        )
        if (
            best_score < minimum_similarity
            or best_score - runner_up < required_margin
        ):
            return None
        # IoU/Jaccard between rasterized glyphs is intentionally conservative:
        # antialiasing and scale changes keep an otherwise unambiguous real
        # match around 0.70-0.90.  Once both the absolute and runner-up gates
        # have passed, calibrate that feature score into an observation
        # confidence instead of treating raw pixel overlap as probability.
        calibrated_confidence = 0.75 + best_score * 0.25
        return best_rank, min(
            0.999,
            max(self.config.confidence_threshold, calibrated_confidence),
        )


def _template_similarity(image: Image.Image, template: Image.Image) -> float:
    target = ImageOps.autocontrast(image.convert("L"))
    reference = ImageOps.autocontrast(template.convert("L"))
    target = ImageOps.fit(target, reference.size, method=Image.Resampling.BILINEAR)
    target_pixels = np.asarray(target, dtype=np.int16)
    reference_pixels = np.asarray(reference, dtype=np.int16)
    difference = float(np.abs(target_pixels - reference_pixels).mean())
    return max(0.0, min(1.0, 1.0 - difference / 255.0))


def _parse_role(match: TemplateMatch, threshold: float) -> SeatRole:
    if match.confidence < threshold or match.label not in {"landlord", "farmer"}:
        return SeatRole.UNKNOWN
    return SeatRole(match.label)


def _classify_role_badge(image: Image.Image) -> tuple[SeatRole, float]:
    """Classify the small role badge without depending on its screen seat.

    The current Mac client renders ``地主`` in gold and ``农民`` in white.
    Whole-ROI templates are dominated by the seat-specific avatar fragment and
    blue background, so a template recorded at the left seat can lose to a
    wrong-label template recorded at the self seat.  Restricting the signal to
    bright badge glyph pixels makes the same landlord template transferable
    across all three seats.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return SeatRole.UNKNOWN, 0.0
    height, width = rgb.shape[:2]
    top = max(0, round(height * 0.20))
    bottom = max(top + 1, round(height * 0.98))
    left = max(0, round(width * 0.03))
    right = max(left + 1, round(width * 0.95))
    badge = rgb[top:bottom, left:right]
    red, green, blue = (
        badge[:, :, 0],
        badge[:, :, 1],
        badge[:, :, 2],
    )
    yellow = (
        (red > 170)
        & (green > 100)
        & (blue < 160)
        & ((red - blue) > 45)
    )
    maximum = np.maximum.reduce((red, green, blue))
    minimum = np.minimum.reduce((red, green, blue))
    white = (
        (red > 180)
        & (green > 180)
        & (blue > 180)
        & ((maximum - minimum) < 35)
    )
    pixel_count = max(1, badge.shape[0] * badge.shape[1])
    yellow_ratio = float(yellow.sum()) / pixel_count
    white_ratio = float(white.sum()) / pixel_count
    minimum_glyph_ratio = 0.01
    if (
        yellow_ratio >= minimum_glyph_ratio
        and yellow_ratio >= white_ratio * 1.5
    ):
        confidence = min(0.999, 0.90 + yellow_ratio * 2.0)
        return SeatRole.LANDLORD, confidence
    if (
        white_ratio >= minimum_glyph_ratio
        and white_ratio >= yellow_ratio * 1.5
    ):
        confidence = min(0.999, 0.90 + white_ratio * 2.0)
        return SeatRole.FARMER, confidence
    return SeatRole.UNKNOWN, max(yellow_ratio, white_ratio)


def _classify_pass_notice(image: Image.Image) -> float:
    """Detect the client's white ``不出`` notice without a local template.

    The glyphs sit in the upper third of each pass ROI. Face-up table cards can
    overlap the lower half of the same ROI, so measuring the whole crop would
    turn ordinary plays into false passes. The neutral table has no pale
    foreground in this upper band; a conservative density gate therefore
    transfers across both opponent seats and small window movements.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return 0.0
    height, width = rgb.shape[:2]
    band_bottom = max(1, round(height * 0.36))
    band = rgb[:band_bottom]
    pale = (
        (band.min(axis=2) > 120)
        & (band.max(axis=2) > 190)
        & ((band.max(axis=2) - band.min(axis=2)) < 80)
    )
    density = float(pale.mean())
    if density < 0.025:
        return 0.0

    active_rows = int(
        np.count_nonzero(pale.sum(axis=1) >= max(3, width * 0.015))
    )
    active_columns = int(
        np.count_nonzero(pale.sum(axis=0) >= max(2, band_bottom * 0.08))
    )
    if (
        active_rows < max(5, round(band_bottom * 0.12))
        or active_columns < max(12, round(width * 0.08))
    ):
        return 0.0
    return min(0.995, 0.82 + (density - 0.025) * 3.2)


def _classify_turn_controls(
    image: Image.Image,
) -> tuple[bool | None, float]:
    """Detect the yellow play controls without relying on blue background.

    Whole-ROI active templates are mostly the unchanged table color.  With no
    explicit inactive template, a blank opponent-turn crop can therefore look
    deceptively similar to the active sample.  The current Mac client renders
    large orange/yellow Hint and Play buttons only on the local turn, so their
    saturated warm pixels are the transferable signal.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return None, 0.0
    red, green, blue = (
        rgb[:, :, 0],
        rgb[:, :, 1],
        rgb[:, :, 2],
    )
    yellow = (
        (red > 150)
        & (green > 80)
        & (blue < 110)
        & ((red - blue) > 65)
        & ((green - blue) > 25)
    )
    yellow_ratio = float(yellow.mean())
    if yellow_ratio >= 0.08:
        confidence = min(0.999, 0.94 + (yellow_ratio - 0.08) * 0.20)
        return True, confidence
    if yellow_ratio <= 0.02:
        confidence = min(0.999, 0.95 + (0.02 - yellow_ratio) * 2.0)
        return False, confidence
    return None, max(0.0, 1.0 - abs(yellow_ratio - 0.05) / 0.03)


def _resolve_roles_from_hand_count(
    roles: Mapping[PlayerSeat, tuple[SeatRole, float]],
    *,
    visible_hand_count: int | None,
) -> dict[PlayerSeat, tuple[SeatRole, float]]:
    resolved = dict(roles)
    # At the post-bidding start screen only the local landlord can visibly
    # hold 20 cards.  This structural signal is stronger than a whole-ROI role
    # template, whose mostly unchanged blue background can otherwise match the
    # wrong seat/label with deceptively high confidence.
    if visible_hand_count == 20:
        resolved[PlayerSeat.SELF] = (SeatRole.LANDLORD, 1.0)
        resolved[PlayerSeat.LEFT] = (SeatRole.FARMER, 1.0)
        resolved[PlayerSeat.RIGHT] = (SeatRole.FARMER, 1.0)
    return resolved


def _parse_remaining(match: TemplateMatch, threshold: float) -> int | None:
    if match.confidence < threshold or match.label is None:
        return None
    try:
        value = int(match.label)
    except ValueError:
        return None
    return value if 0 <= value <= 20 else None


def _resolve_remaining(
    *,
    template_count: int | None,
    template_confidence: float,
    text_match: RemainingTextMatch | None,
    template_threshold: float,
) -> tuple[int | None, float, bool]:
    if (
        text_match is not None
        and text_match.count is not None
        and text_match.confidence >= 0.50
    ):
        return text_match.count, text_match.confidence, True
    # Whole-ROI templates are useful as a best-effort display value, but an
    # unseen number can still look very similar because most pixels are the
    # unchanged seat background. Only near-exact matches may block tracking.
    return (
        template_count,
        template_confidence,
        bool(
            template_count is not None
            and template_confidence >= max(template_threshold, 0.995)
        ),
    )


def _remaining_digit_masks(image: Image.Image) -> tuple[np.ndarray, ...]:
    """Extract one or two filled yellow counter glyphs from a seat ROI."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return ()
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (
        (red > 145)
        & (green > 70)
        & (blue < 165)
        & ((red - blue) > 45)
        & ((green - blue) > 5)
    )
    active_columns = np.any(mask, axis=0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(active_columns):
        if active and start is None:
            start = index
        if start is not None and (
            not active or index == len(active_columns) - 1
        ):
            stop = index if not active else index + 1
            if stop - start >= 2:
                runs.append((start, stop))
            start = None

    glyphs: list[np.ndarray] = []
    minimum_height = max(8, round(image.height * 0.18))
    for left, right in runs:
        region = mask[:, left:right]
        active_rows = np.flatnonzero(np.any(region, axis=1))
        if active_rows.size == 0:
            continue
        top = int(active_rows[0])
        bottom = int(active_rows[-1]) + 1
        glyph = region[top:bottom]
        if glyph.shape[0] < minimum_height:
            continue
        # Animated decorations around the opponent avatar occasionally add a
        # tall but only a few pixels wide yellow streak to this ROI.  Treating
        # that streak as a third digit makes a stable value such as ``12``
        # disappear for one frame.  Real counter glyphs keep a substantially
        # wider bounding box, including the digit 1.
        minimum_width = max(3, round(glyph.shape[0] * 0.25))
        if glyph.shape[1] < minimum_width:
            continue
        glyphs.append(glyph)
    return tuple(glyphs)


def _remaining_glyph_signature(mask: np.ndarray) -> frozenset[int]:
    if mask.size == 0 or not bool(mask.any()):
        return frozenset()
    glyph = Image.fromarray(
        np.where(mask, 255, 0).astype(np.uint8),
        mode="L",
    )
    fitted = ImageOps.contain(
        glyph,
        (52, 52),
        method=Image.Resampling.NEAREST,
    )
    normalized = Image.new("L", (64, 64))
    normalized.paste(
        fitted,
        (
            (normalized.width - fitted.width) // 2,
            (normalized.height - fitted.height) // 2,
        ),
    )
    pixels = np.asarray(normalized, dtype=np.uint8)
    return frozenset(
        int(index)
        for index in np.flatnonzero(pixels.reshape(-1) > 0)
    )


def _remaining_signature_hole_count(signature: frozenset[int]) -> int:
    """Count enclosed background regions in a normalized counter glyph."""

    if not signature:
        return 0
    foreground = np.zeros((64, 64), dtype=bool)
    for index in signature:
        if 0 <= index < foreground.size:
            foreground[index // 64, index % 64] = True
    background = ~foreground
    visited = np.zeros_like(background)
    holes = 0
    for start_y in range(64):
        for start_x in range(64):
            if not background[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            size = 0
            touches_border = False
            while stack:
                y, x = stack.pop()
                size += 1
                if y in {0, 63} or x in {0, 63}:
                    touches_border = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_y = y + dy
                    next_x = x + dx
                    if (
                        0 <= next_y < 64
                        and 0 <= next_x < 64
                        and background[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
            if not touches_border and size >= 4:
                holes += 1
    return holes


def _minimum_card_confidence(cards: Sequence[VisualCard]) -> float:
    return min((card.confidence for card in cards), default=0.0)


def _rank_glyph_signature(image: Image.Image) -> frozenset[int]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    foreground = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if pixels[x, y] < 130
    }
    components: list[set[tuple[int, int]]] = []
    while foreground:
        start = foreground.pop()
        component = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    neighbor = (x + offset_x, y + offset_y)
                    if neighbor in foreground:
                        foreground.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        if len(component) >= 20:
            components.append(component)

    def bounds(
        component: set[tuple[int, int]],
    ) -> tuple[int, int, int, int]:
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1

    candidates = []
    for component in components:
        left, top, right, bottom = bounds(component)
        if (
            top < height * 0.45
            and left < width * 0.40
            and bottom - top > height * 0.12
            and right - left > width * 0.12
            and bottom - top < height * 0.70
            # Wide glyphs such as K can occupy about 73% of the visible card
            # crop.  The previous 70% ceiling discarded them completely,
            # which made a fresh mid-game scan depend on a lucky CNN result.
            and right - left < width * 0.85
        ):
            candidates.append((component, (left, top, right, bottom)))
    if not candidates:
        return frozenset()

    primary_component, primary_box = min(
        candidates,
        key=lambda item: (item[1][1], item[1][0]),
    )
    primary_top, primary_bottom = primary_box[1], primary_box[3]
    selected = set(primary_component)
    for component, box in candidates:
        if component is primary_component:
            continue
        overlap = min(primary_bottom, box[3]) - max(primary_top, box[1])
        if overlap > 0:
            selected.update(component)

    left = min(point[0] for point in selected)
    top = min(point[1] for point in selected)
    right = max(point[0] for point in selected) + 1
    bottom = max(point[1] for point in selected) + 1
    mask = Image.new("L", (right - left, bottom - top))
    mask_pixels = mask.load()
    for x, y in selected:
        mask_pixels[x - left, y - top] = 255
    fitted = ImageOps.contain(
        mask,
        (52, 52),
        method=Image.Resampling.NEAREST,
    )
    normalized = Image.new("L", (64, 64))
    normalized.paste(
        fitted,
        (
            (normalized.width - fitted.width) // 2,
            (normalized.height - fitted.height) // 2,
        ),
    )
    return frozenset(
        y * normalized.width + x
        for y in range(normalized.height)
        for x in range(normalized.width)
        if normalized.getpixel((x, y)) > 0
    )


def _glyph_similarity(
    left: frozenset[int],
    right: frozenset[int],
) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_RANK_GLYPH_GRID_SIZE = 64
_RANK_GLYPH_BYTE_COUNT = (_RANK_GLYPH_GRID_SIZE**2) // 8
_BUILTIN_RANK_GLYPHS = (
    Path(__file__).with_name("assets") / "rank_glyph_signatures.json"
)
_BUILTIN_REMAINING_GLYPHS = (
    Path(__file__).with_name("assets") / "remaining_glyph_signatures.json"
)


def _encode_glyph_signature(signature: frozenset[int]) -> str:
    bits = bytearray(_RANK_GLYPH_BYTE_COUNT)
    for index in signature:
        if not 0 <= index < _RANK_GLYPH_GRID_SIZE**2:
            raise ValueError(f"glyph signature index out of range: {index}")
        bits[index // 8] |= 1 << (index % 8)
    return base64.b64encode(bytes(bits)).decode("ascii")


def _decode_glyph_signature(encoded: str) -> frozenset[int]:
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    if len(raw) != _RANK_GLYPH_BYTE_COUNT:
        raise ValueError(
            f"invalid glyph signature size: {len(raw)} bytes"
        )
    return frozenset(
        byte_index * 8 + bit_index
        for byte_index, value in enumerate(raw)
        for bit_index in range(8)
        if value & (1 << bit_index)
    )


def _load_builtin_rank_references(
    path: Path = _BUILTIN_RANK_GLYPHS,
) -> dict[str, list[frozenset[int]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        payload.get("format") != "rank-glyph-signatures-v1"
        or payload.get("grid_size") != _RANK_GLYPH_GRID_SIZE
        or not isinstance(payload.get("ranks"), dict)
    ):
        return {}
    references: dict[str, list[frozenset[int]]] = {}
    for rank in RANKS:
        encoded_items = payload["ranks"].get(rank, ())
        if not isinstance(encoded_items, list):
            continue
        decoded: list[frozenset[int]] = []
        for encoded in encoded_items[:12]:
            if not isinstance(encoded, str):
                continue
            try:
                signature = _decode_glyph_signature(encoded)
            except (ValueError, binascii.Error):
                continue
            if signature:
                decoded.append(signature)
        if decoded:
            references[rank] = decoded
    return references


def _load_builtin_remaining_references(
    path: Path = _BUILTIN_REMAINING_GLYPHS,
) -> dict[str, list[frozenset[int]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        payload.get("format") != "remaining-glyph-signatures-zlib-v1"
        or payload.get("grid_size") != _RANK_GLYPH_GRID_SIZE
        or not isinstance(payload.get("digits"), dict)
    ):
        return {}
    references: dict[str, list[frozenset[int]]] = {}
    for digit in "0123456789":
        encoded_items = payload["digits"].get(digit, ())
        if not isinstance(encoded_items, list):
            continue
        decoded: list[frozenset[int]] = []
        for encoded in encoded_items[:12]:
            if not isinstance(encoded, str):
                continue
            try:
                compressed = base64.b64decode(
                    encoded.encode("ascii"),
                    validate=True,
                )
                raw = zlib.decompress(compressed)
                signature = _decode_glyph_signature(
                    base64.b64encode(raw).decode("ascii")
                )
            except (ValueError, binascii.Error, zlib.error):
                continue
            if signature:
                decoded.append(signature)
        if decoded:
            references[digit] = decoded
    return references


def _is_card_white(pixel: tuple[int, ...]) -> bool:
    red, green, blue = pixel[:3]
    return max(red, green, blue) >= 175 and min(red, green, blue) >= 115


def _card_white_mask(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return (
        (pixels.max(axis=2) >= 175)
        & (pixels.min(axis=2) >= 115)
    )


def _active_x_runs_from_mask(
    white_mask: np.ndarray,
    *,
    min_column_ratio: float,
    max_gap: int,
) -> list[tuple[int, int]]:
    height, width = white_mask.shape
    minimum = max(3, round(height * min_column_ratio))
    active = (white_mask.sum(axis=0) >= minimum).tolist()
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


def _active_x_runs(
    image: Image.Image,
    *,
    min_column_ratio: float = 0.08,
    max_gap: int = 2,
) -> list[tuple[int, int]]:
    return _active_x_runs_from_mask(
        _card_white_mask(image),
        min_column_ratio=min_column_ratio,
        max_gap=max_gap,
    )


def _single_portrait_card_box(
    image: Image.Image,
    *,
    white_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    """Recover one portrait card whose middle is hidden by a wide notice.

    A horizontal game notice splits the white body into top and bottom runs.
    Looking only at the longest run can then expose the landlord sash, suit,
    and glyph edges as several fake card starts.  Combining the earliest and
    latest durable white runs recovers the original portrait geometry.  A
    genuinely overlapped pair is almost square, so it deliberately fails the
    portrait aspect-ratio guard.
    """

    width, height = image.size
    if white_mask is None:
        white_mask = _card_white_mask(image)
    column_runs = [
        run
        for run in _active_x_runs_from_mask(
            white_mask,
            min_column_ratio=0.15,
            max_gap=3,
        )
        if run[1] - run[0] >= max(12, round(width * 0.03))
    ]
    if len(column_runs) != 1:
        return None

    left, right = column_runs[0]
    row_counts = white_mask[:, left:right].sum(axis=1)
    row_runs = _boolean_runs(
        [
            int(count) >= max(8, round((right - left) * 0.25))
            for count in row_counts
        ],
        max_gap=2,
    )
    durable_runs = [
        run
        for run in row_runs
        if run[1] - run[0] >= max(12, round(height * 0.10))
    ]
    if not durable_runs:
        return None

    top = min(run[0] for run in durable_runs)
    bottom = max(run[1] for run in durable_runs)
    span = bottom - top
    if (
        span <= 0
        or top > round(height * 0.28)
        or bottom < round(height * 0.70)
    ):
        return None
    aspect_ratio = (right - left) / span
    visible_ratio = sum(
        run_bottom - run_top
        for run_top, run_bottom in durable_runs
    ) / span
    if not 0.48 <= aspect_ratio <= 0.80 or visible_ratio < 0.32:
        return None
    return (
        left,
        max(0, top - 2),
        right,
        min(height, bottom + 3),
    )


def segment_card_boxes(image: Image.Image) -> tuple[tuple[int, int, int, int], ...]:
    """Locate separated or overlapped face-up cards on the blue table."""

    width, height = image.size
    minimum_width = max(10, round(width * 0.03))
    boxes: list[tuple[int, int, int, int]] = []
    rgb = image.convert("RGB")
    white_mask = _card_white_mask(rgb)
    single_card = _single_portrait_card_box(
        rgb,
        white_mask=white_mask,
    )
    if single_card is not None:
        return (single_card,)
    for left, right in _active_x_runs_from_mask(
        white_mask,
        min_column_ratio=0.08,
        max_gap=2,
    ):
        if right - left < minimum_width:
            continue
        row_counts = white_mask[:, left:right].sum(axis=1)
        row_runs = _boolean_runs(
            [
                int(count) >= max(4, round((right - left) * 0.25))
                for count in row_counts
            ],
            max_gap=2,
        )
        row_runs = [
            run
            for run in row_runs
            if run[1] - run[0] >= max(12, round(height * 0.10))
        ]
        if not row_runs:
            continue
        row_top, row_bottom = max(
            row_runs,
            key=lambda run: run[1] - run[0],
        )
        top = max(0, row_top - 2)
        bottom = min(height, row_bottom + 3)
        if bottom - top < max(12, round(height * 0.10)):
            continue
        box_width = right - left
        box_height = bottom - top
        aspect_ratio = box_width / box_height
        white_ratio = float(white_mask[top:bottom, left:right].mean())
        if not 0.28 <= aspect_ratio <= 1.20 or white_ratio < 0.60:
            continue
        boxes.append((left, top, right, bottom))
    overlapping_boxes = _segment_overlapping_card_boxes(
        rgb,
        white_mask=white_mask,
    )
    if len(boxes) == 1 and len(overlapping_boxes) > 1:
        left, top, right, bottom = boxes[0]
        merged_aspect_ratio = (right - left) / max(1, bottom - top)
        # Two tightly overlapped table cards form one continuous white region.
        # Its bounding box is almost square, while one ordinary face-up card is
        # substantially narrower than it is tall. Prefer the vertical-edge
        # starts in that case so a pair such as QQ/88 is classified card by
        # card instead of being sent to the CNN as one merged crop.
        if merged_aspect_ratio >= 0.82:
            return overlapping_boxes
    if boxes:
        return tuple(boxes)
    return overlapping_boxes


def infer_visible_hand_count(
    image: Image.Image,
    *,
    maximum: int = 20,
) -> int | None:
    if _single_portrait_card_box(image) is not None:
        return 1
    # The local client's hand is always overlapped. Trying separated-card
    # segmentation first scans the same large ROI twice and made each live
    # frame take multiple seconds.
    boxes = _segment_overlapping_card_boxes(image)
    boxes = _regularize_overlapping_hand_boxes(boxes)
    if len(boxes) == 2:
        # A lone landlord card carries a diagonal red/yellow sash.  Its upper
        # edge can be stronger than the card border and look like a second
        # overlapped-card start.  The connected-component path already
        # distinguishes one normal portrait card from a genuinely tight pair
        # by the merged body's aspect ratio, so consult it for this ambiguous
        # two-edge endgame case.
        separated_boxes = segment_card_boxes(image)
        if len(separated_boxes) == 1:
            boxes = separated_boxes
    if not boxes:
        boxes = segment_card_boxes(image)
    count = len(boxes)
    if 1 <= count <= maximum:
        return count
    return None


def _segment_overlapping_card_boxes(
    image: Image.Image,
    *,
    white_mask: np.ndarray | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = image.size
    if white_mask is None:
        white_mask = _card_white_mask(image)
    x_runs = _active_x_runs_from_mask(
        white_mask,
        min_column_ratio=0.08,
        max_gap=2,
    )
    if not x_runs:
        return ()
    left, right = max(x_runs, key=lambda run: run[1] - run[0])
    row_counts = white_mask[:, left:right].sum(axis=1)
    minimum_row = max(8, round((right - left) * 0.25))
    row_runs = _boolean_runs(
        [int(count) >= minimum_row for count in row_counts],
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

    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    edge_start = max(left + 1, 1)
    edge_stop = min(right + 1, width)
    differences = np.abs(
        grayscale[top:bottom, edge_start:edge_stop]
        - grayscale[top:bottom, edge_start - 1:edge_stop - 1]
    )
    mean_scores = differences.mean(axis=0)
    edge_scores = [
        (edge_start + index, float(score))
        for index, score in enumerate(mean_scores)
    ]
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


def _regularize_overlapping_hand_boxes(
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Remove strong glyph edges that masquerade as overlapped card starts.

    The local client lays the hand on an almost uniform horizontal grid.  A
    vertical ``JOKER`` glyph can be stronger than a card border and therefore
    introduce an extra start inside the first exposed card.  Map candidates to
    their nearest grid slot and retain only the closest edge per slot.  This is
    deliberately applied even when the raw count is below the configured
    maximum: after one real card is played, ``19 cards + 1 glyph edge`` would
    otherwise still look like a valid 20-card hand.
    """

    ordered = tuple(sorted(boxes, key=lambda box: box[0]))
    if len(ordered) < 3:
        return ordered
    starts = np.asarray([box[0] for box in ordered], dtype=np.float64)
    gaps = np.diff(starts)
    positive_gaps = gaps[gaps >= 4.0]
    if len(positive_gaps) < 2:
        return ordered

    # Genuine card steps occupy the upper half when one or two short glyph
    # gaps are mixed in.  Using only that half avoids shrinking the estimated
    # grid to the internal strokes of a Joker.
    gap_midpoint = float(np.median(positive_gaps))
    likely_card_gaps = positive_gaps[positive_gaps >= gap_midpoint]
    typical_step = float(np.median(likely_card_gaps))
    if typical_step < 8.0:
        return ordered

    origin = float(starts[0])
    tolerance = max(4.0, typical_step * 0.42)
    best_by_slot: dict[
        int,
        tuple[float, tuple[int, int, int, int]],
    ] = {}
    for box in ordered:
        offset = float(box[0]) - origin
        slot = max(0, int(round(offset / typical_step)))
        distance = abs(offset - slot * typical_step)
        if distance > tolerance:
            continue
        existing = best_by_slot.get(slot)
        if existing is None or distance < existing[0]:
            best_by_slot[slot] = (distance, box)

    regularized = tuple(
        value[1]
        for _, value in sorted(best_by_slot.items())
    )
    return regularized or ordered


def infer_overlapping_hand_boxes(
    image: Image.Image,
    count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    if count <= 0:
        return ()
    width, height = image.size
    rgb = image.convert("RGB")
    white_mask = _card_white_mask(rgb)
    white_columns = _active_x_runs_from_mask(
        white_mask,
        min_column_ratio=0.15,
        max_gap=3,
    )
    if not white_columns:
        return ()
    left = min(run[0] for run in white_columns)
    right = max(run[1] for run in white_columns)
    # Full 17/20-card hands span most of the configured ROI, but this client
    # centers and narrows the fan as cards are played. Requiring every hand to
    # occupy 35% of the ROI made a real three-card endgame disappear entirely.
    # Keep the wide-hand noise guard while scaling it down with the known
    # visible count.
    minimum_extent_ratio = min(0.35, max(0.04, 0.35 * count / 20.0))
    if right - left < width * minimum_extent_ratio:
        return ()
    row_counts = white_mask[:, left:right].sum(axis=1)
    row_runs = _boolean_runs(
        [
            int(row_count) >= max(8, round((right - left) * 0.25))
            for row_count in row_counts
        ],
        max_gap=2,
    )
    if row_runs:
        # In-game notices such as “您当前没有牌大过对方” can cover the
        # middle of every hand card and split the white card body into two
        # runs.  The old longest-run rule then started below the notice and
        # classified blank card bottoms.  Preserve the earliest card top and
        # the latest card bottom so rank crops stay anchored at the glyphs.
        early_runs = [
            run
            for run in row_runs
            if run[0] <= round(height * 0.25)
            and run[1] - run[0] >= max(12, round(height * 0.12))
        ]
        start_y = (
            min(run[0] for run in early_runs)
            if early_runs
            else max(row_runs, key=lambda run: run[1] - run[0])[0]
        )
        card_bottom = max(run[1] for run in row_runs)
    else:
        start_y = max(0, round(height * 0.04))
        card_bottom = height
    full_card_height = max(24, card_bottom - start_y)
    full_card_width = max(16, round(full_card_height * 0.72))
    crop_height = max(24, round(full_card_height * 0.535))
    crop_width = max(16, round(crop_height * 0.60))
    detected = _regularize_overlapping_hand_boxes(
        _segment_overlapping_card_boxes(
            rgb,
            white_mask=white_mask,
        )
    )
    detected_starts = (
        [box[0] for box in detected]
        if len(detected) == count
        else []
    )
    if detected_starts:
        starts = detected_starts
    elif count == 1:
        starts = [left]
    else:
        step = max(1, int((right - left - full_card_width) / (count - 1)))
        starts = [left + index * step for index in range(count)]
    boxes = []
    for x in starts:
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


def _resolve_card_prediction(
    prediction: CardPrediction,
    reference_match: tuple[str, float] | None,
    *,
    confident_joker_threshold: float = 0.90,
) -> tuple[str, float]:
    """Fuse CNN and glyph references without turning a Joker into a rank.

    Vertical Joker artwork contains a prominent letter ``J``.  The generic
    glyph matcher therefore occasionally reports an excellent normal-J match
    even though the full-card CNN confidently sees a Joker.  Preserve a
    confident CNN Joker against a non-Joker reference; all other predictions
    continue to benefit from the real-window reference correction.
    """

    if reference_match is None:
        return prediction.rank, prediction.confidence
    reference_rank, reference_confidence = reference_match
    if (
        prediction.rank in {"SJ", "BJ"}
        and prediction.confidence >= confident_joker_threshold
        and reference_rank not in {"SJ", "BJ"}
    ):
        return prediction.rank, prediction.confidence
    return reference_rank, reference_confidence


__all__ = [
    "CvRemainingReader",
    "SceneObservation",
    "SceneRecognizer",
    "MacVisionRemainingReader",
    "RemainingTextMatch",
    "SeatObservation",
    "SeatRole",
    "TemplateMatch",
    "TemplateMatcher",
    "VisualCard",
    "VisualSignal",
    "infer_visible_hand_count",
    "infer_overlapping_hand_boxes",
    "segment_card_boxes",
]
