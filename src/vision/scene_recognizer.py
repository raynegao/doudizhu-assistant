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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.capture.screen_geometry import CapturedWindow
from src.config.hand_geometry import ShortHandRecoveryConfig
from src.config.live_layout import LiveLayoutConfig
from src.config.self_table import SelfTableConfig
from src.state.cards import RANKS, CardSet, normalize_rank
from src.state.events import PlayerSeat
from src.vision.card_classifier import (
    CardPrediction,
    load_checkpoint,
    predict_tensors,
    preprocess_image,
    select_device,
)
from src.vision.card_regions import CardRegion, CardRegionKind
from src.vision.clipped_hand_edges import clipped_trailing_edges_supported
from src.vision.hand_empty import observe_empty_hand_domain
from src.vision.hand_photometry import prepare_hand_rank_crop
from src.vision.joker_pair import resolve_joker_pair
from src.vision.rank_glyph import (
    JokerColorEvidence,
    RankGlyphComponent,
    extract_rank_glyph,
    read_joker_color,
)
from src.vision.rectangular_card_edges import rectangular_slot_supported
from src.vision.scene_types import CardFusionEvidence, SelfTableObservation
from src.vision.scene_types import (
    RoundResultObservation as RoundResultObservation,
)
from src.vision.scene_types import (
    RoundResultOutcome as RoundResultOutcome,
)
from src.vision.scene_types import (
    SceneObservation as SceneObservation,
)
from src.vision.scene_types import (
    SeatObservation as SeatObservation,
)
from src.vision.scene_types import (
    SeatRole as SeatRole,
)
from src.vision.scene_types import (
    TemplateMatch as TemplateMatch,
)
from src.vision.scene_types import (
    VisualCard as VisualCard,
)
from src.vision.scene_types import (
    VisualSignal as VisualSignal,
)
from src.vision.turn_controls import observe_turn_controls


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

        return tuple((label, image.copy()) for label, _, image in self._templates(kind))

    def classify(
        self,
        kind: str,
        image: Image.Image,
        *,
        allowed_labels: Iterable[str] | None = None,
    ) -> TemplateMatch:
        allowed = set(allowed_labels) if allowed_labels is not None else None
        candidates = [
            item for item in self._templates(kind) if allowed is None or item[0] in allowed
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
                label = (
                    path.parent.name if path.parent != directory else path.stem.split("__", 1)[0]
                )
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
                if any(_glyph_similarity(signature, existing) >= 0.98 for existing in references):
                    continue
                references.append(signature)
        self._digit_reference_holes = {
            digit: tuple(
                (reference, _remaining_signature_hole_count(reference)) for reference in references
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
        # Classify a standalone 0 before read() rejects the out-of-range count;
        # excluding it here can force a clear zero into another digit or 1.
        if digit_count == 2 and index == 0:
            # The only two-digit values are 10..20.
            allowed_digits = {"1", "2"}
        elif digit_count == 2 and prefix and prefix[0] == "2":
            # Once the first glyph is 2 the value can only be 20.
            allowed_digits = {"0"}
        hole_components = _remaining_signature_holes(signature)
        if _remaining_zero_hole_incompatible(signature, hole_components):
            allowed_digits.discard("0")
        if not allowed_digits:
            # Geometry contradicting the only allowed suffix must not reach
            # the unmatched-after-2 fallback and turn a visible 29 into 20.
            return None, 0.0
        observed_holes = len(hole_components)
        topology_references = {
            digit: tuple(
                reference for reference, hole_count in references if hole_count == observed_holes
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
                        topology_references.get(digit, ()) if has_topology_match else references
                    )
                ),
                digit,
            )
            for digit, references in self._digit_references.items()
            if (
                references
                and digit in allowed_digits
                and (not has_topology_match or topology_references.get(digit))
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
        self.source_path = source_path or Path(__file__).with_name("macos_vision_ocr.swift")
        self._runner = runner
        self._binary_path: Path | None = None

    def __call__(
        self,
        images: Mapping[PlayerSeat, Image.Image],
    ) -> Mapping[PlayerSeat, RemainingTextMatch]:
        binary = self._ensure_binary()
        seats = [seat for seat in (PlayerSeat.LEFT, PlayerSeat.RIGHT) if seat in images]
        if not seats:
            return {}
        with tempfile.TemporaryDirectory(prefix="doudizhu-remaining-ocr-") as temp_dir:
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
        binary = Path(tempfile.gettempdir()) / (f"doudizhu-macos-vision-ocr-{digest}")
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
                tensors = torch.stack(
                    [preprocess_image(crop, image_size=image_size) for crop in crops]
                )
                return predict_tensors(model, tensors, classes=classes, device=device)

            self.predictor = model_predictor
        else:
            self.predictor = predictor
        self._rank_references = _load_builtin_rank_references()
        self._reference_hand_fingerprint: tuple[str, ...] | None = None
        self._load_rank_reference_templates()
        self._base_rank_references = {
            rank: list(references) for rank, references in self._rank_references.items()
        }
        self._round_result_references = _load_builtin_round_result_references()
        if remaining_reader is not None:
            self.remaining_reader: RemainingReader | None = remaining_reader
        else:
            self.remaining_reader = CvRemainingReader(
                template_images=self.templates.template_images("remaining"),
            )

    def observe(self, frame: CapturedWindow) -> SceneObservation:
        round_result = _classify_round_result(
            frame.image,
            references=self._round_result_references,
        )
        if round_result is not None:
            return SceneObservation(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                window_pixel_box=frame.pixel_box,
                self_hand=(),
                seats=tuple(
                    SeatObservation(
                        seat=seat,
                        signal=VisualSignal.NEUTRAL,
                    )
                    for seat in PlayerSeat
                ),
                self_turn=False,
                self_turn_confidence=round_result.confidence,
                confidence=round_result.confidence,
                warnings=("round_result_overlay",),
                round_result=round_result,
            )
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
        hand_geometry = infer_hand_geometry(
            self_hand_crop,
            maximum=20,
            recovery_config=self.config.short_hand_recovery,
        )
        visible_hand_count = hand_geometry.count
        if hand_geometry.reason:
            warnings.append(f"self_hand_geometry_unavailable: {hand_geometry.reason}")
        roles = _resolve_roles_from_hand_count(
            raw_roles,
            visible_hand_count=visible_hand_count,
        )
        for seat, (role, _) in roles.items():
            if role is SeatRole.UNKNOWN:
                warnings.append(f"{seat.value}_role_unavailable")

        self_role = roles[PlayerSeat.SELF][0]
        expected_hand_count = 20 if self_role is SeatRole.LANDLORD else 17
        raw_hand_predictions: list[CardPrediction] = []
        prepared_hand_crops: list[Image.Image] = []
        self_hand = self._observe_hand(
            self_hand_crop,
            expected_count=expected_hand_count,
            visible_count=visible_hand_count,
            raw_predictions_out=raw_hand_predictions,
            prepared_crops_out=prepared_hand_crops,
            geometry=hand_geometry,
        )
        if not self_hand:
            warnings.append(f"self_hand_unavailable: expected_at_most={expected_hand_count}")
        if len(self_hand) == expected_hand_count:
            # A count alone does not establish a valid opening. Validate the
            # whole hand and independent raw evidence before resetting state.
            self._remember_rank_references(
                self_hand_crop,
                self_hand,
                raw_predictions=raw_hand_predictions,
                prepared_crops=prepared_hand_crops,
                reset=True,
            )

        count_crops = {
            seat: self.config.crop(frame.image, f"{seat.value}_remaining") for seat in PlayerSeat
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
                and hand_geometry.count is not None
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
        turn_evidence = observe_turn_controls(turn_crop, config=self.config.turn_controls)
        self_turn, turn_confidence = turn_evidence.active, turn_evidence.confidence
        if self_turn is None:
            # A structural refusal cannot become a turn or a pass merely
            # because a mostly-blue whole-ROI template matches its background.
            warnings.extend(("self_turn_unavailable", f"self_turn_geometry: {turn_evidence.reason}"))

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
        if hand_geometry.reason and not hand_geometry.boxes:
            # Unknown gray geometry is not a verified empty hand. Preserve
            # that distinction even when unrelated turn controls look clear.
            self_turn = None
            turn_confidence = scene_confidence = 0.0
        empty_evidence = observe_empty_hand_domain(
            self_hand_crop, config=self.config.hand_empty,
            geometry_count=hand_geometry.count, has_candidates=bool(hand_geometry.boxes),
            geometry_reason=hand_geometry.reason,
        )
        empty_verified = empty_evidence.verified and not self_hand
        self_table = (
            self._observe_self_table(frame.image)
            if empty_verified and self_turn is not True else None
        )
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
            self_hand_empty_verified=empty_verified,
            self_table=self_table,
            self_hand_empty_evidence=empty_evidence,
        )

    def _observe_self_table(self, image: Image.Image) -> SelfTableObservation:
        left, top, right, bottom = self.config.rois["self_play"].to_pixel_box(image.size)
        hand_top = self.config.rois["self_hand"].to_pixel_box(image.size)[1]
        if top >= hand_top or bottom > hand_top:
            return SelfTableObservation(reason="table_roi_overlaps_hand")
        # Complete a bottom-clipped table body in the same captured window,
        # never inside the hand ROI or by adding a state-derived card slot.
        crop_box = left, top, right, hand_top
        crop = image.crop(crop_box)
        if crop.width * crop.height > self.config.self_table.maximum_input_pixels:
            return SelfTableObservation(reason="table_input_budget_exceeded", crop_box=crop_box)
        regions = segment_card_regions(crop, table_edges=True)
        bounds = _self_table_body_bounds(
            crop, regions, config=self.config.self_table, edge_config=self.config.short_hand_recovery,
        )
        if bounds is None:
            return SelfTableObservation(reason="table_body_unverified", crop_box=crop_box)
        inputs = _self_table_rank_regions(regions, bounds, self.config)
        cards = self._observe_table_regions(crop, inputs)
        if len(cards) != len(regions):
            return SelfTableObservation(cards=cards, reason="incomplete_prediction_slots", crop_box=crop_box)
        return SelfTableObservation(cards, True, "closed_paper_table_bodies", crop_box)

    def seed_hand_references(
        self,
        full_window_image: Image.Image,
    ) -> tuple[VisualCard, ...]:
        """Restore per-round rank references from a saved initial table frame."""

        hand_image = self.config.crop(full_window_image, "self_hand")
        hand_geometry = infer_hand_geometry(
            hand_image, maximum=20, recovery_config=self.config.short_hand_recovery,
        )
        visible_count = hand_geometry.count
        # A round can be initialized from a safe mid-game self-turn scan, so
        # its reference frame may contain any physically valid number of
        # cards.  Restricting recovery to opening 17/20-card frames caused a
        # recognition-worker restart after the first action to discard a valid
        # checkpoint and rescan the table as a new approximate round.
        if visible_count is None or not 1 <= visible_count <= 20:
            return ()
        raw_hand_predictions: list[CardPrediction] = []
        prepared_hand_crops: list[Image.Image] = []
        cards = self._observe_hand(
            hand_image,
            expected_count=visible_count,
            visible_count=visible_count,
            raw_predictions_out=raw_hand_predictions,
            prepared_crops_out=prepared_hand_crops,
            geometry=hand_geometry,
        )
        try:
            card_set = CardSet.parse(card.rank for card in cards)
        except ValueError:
            return ()
        if len(card_set) != visible_count:
            return ()
        if any(not self.config.confidence_threshold <= card.confidence <= 1.0 for card in cards):
            return ()
        self._remember_rank_references(
            hand_image,
            cards,
            raw_predictions=raw_hand_predictions,
            prepared_crops=prepared_hand_crops,
            reset=True,
        )
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
        if pass_notice_confidence >= self.config.pass_threshold or (
            pass_match.label == "pass" and pass_match.confidence >= self.config.pass_threshold
        ):
            signal = VisualSignal.PASS
            cards: tuple[VisualCard, ...] = ()
            confidence = max(
                pass_notice_confidence,
                (pass_match.confidence if pass_match.label == "pass" else 0.0),
            )
        else:
            # The self action ROI overlaps the large control buttons in this
            # client and produces false card-like regions. Self plays are more
            # reliably derived from the stable hand difference in the tracker.
            regions = (
                () if seat is PlayerSeat.SELF else segment_card_regions(play_crop, table_edges=True)
            )
            cards = self._observe_table_regions(play_crop, regions)
            if cards and all(card.confidence >= self.config.confidence_threshold for card in cards):
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
                (pass_match.confidence if pass_match.label == "pass" else 0.0),
            ),
            remaining_confidence=remaining[1],
            role_confidence=role[1],
            remaining_verified=remaining[2],
        )

    def _observe_table_regions(
        self, play_crop: Image.Image, regions: Sequence[CardRegion],
    ) -> tuple[VisualCard, ...]:
        boxes = tuple(region.box for region in regions)
        crops = [region.classifier_crop(play_crop, self.config.table_rank_crop)
                 for region in regions]
        predictions = self.predictor(crops)
        observed_cards: list[VisualCard] = []
        colors: list[JokerColorEvidence | None] = []
        for index, prediction in enumerate(predictions):
            card_image = play_crop.crop(boxes[index])
            glyph = extract_rank_glyph(card_image)
            reference_match = self._match_rank_reference(
                card_image,
                # Table cards are larger and can have a slightly different
                # antialiased glyph from the hand reference. A strong
                # runner-up margin still makes 0.70 overlap reliable; the
                # previous 0.78 floor missed a real red 10 and left the CNN
                # to call it K/7.
                minimum_similarity=0.70,
                minimum_margin=0.12,
                glyph=glyph,
            )
            if prediction.rank in {"SJ", "BJ"} and (
                reference_match is None or reference_match[0] in {"J", "SJ", "BJ"}
            ):
                # The vertical JOKER word is often closest to a normal J.
                # Compare Joker shapes for diagnostics, but let the raw
                # identity and independent rank ink guard the final output.
                joker_match = self._match_rank_reference(
                    card_image,
                    minimum_similarity=0.80,
                    minimum_margin=0.025,
                    allowed_ranks={"SJ", "BJ"},
                    glyph=glyph,
                )
                if joker_match is not None:
                    reference_match = joker_match
            color = (
                read_joker_color(card_image, glyph, self.config.joker_color)
                if prediction.rank in {"SJ", "BJ"} else None
            )
            colors.append(color)
            rank, confidence, evidence = _fuse_card_prediction(
                prediction,
                reference_match,
                joker_color=color,
            )
            evidence = replace(
                evidence, input_geometry=regions[index].kind.value,
                input_box=regions[index].classifier_box(self.config.table_rank_crop),
            )
            observed_cards.append(
                VisualCard(
                    rank=rank,
                    confidence=confidence,
                    box=boxes[index],
                    evidence=evidence,
                )
            )
        # Overlapping table Jokers can leave the upper corner reading as the
        # lower card's Joker; the deck then forces the complement.
        resolved = resolve_joker_pair(observed_cards, colors)
        # A corner whose right neighbour starts inside its glyph zone (cards
        # still sliding into place, or a far tighter overlap than this client
        # ever settles on) cannot be read: a partial Q can score as a 6. The
        # reading is kept for diagnostics with zero confidence; nothing is
        # re-classified or invented.
        visible_widths = _visible_corner_widths(boxes)
        gated: list[VisualCard] = []
        for card, visible in zip(resolved, visible_widths, strict=True):
            crop_width = card.box[2] - card.box[0]
            if (
                visible is not None
                and visible < self.config.table_rank_crop.minimum_visible_width_ratio * crop_width
            ):
                reason = f"partial_corner:{visible}/{crop_width}"
                card = replace(
                    card,
                    confidence=0.0,
                    evidence=(
                        replace(card.evidence, geometry_reason=reason)
                        if card.evidence is not None else None
                    ),
                )
            gated.append(card)
        return tuple(gated)

    def _observe_hand(
        self,
        image: Image.Image,
        *,
        expected_count: int,
        visible_count: int | None = None,
        raw_predictions_out: list[CardPrediction] | None = None,
        prepared_crops_out: list[Image.Image] | None = None,
        geometry: HandGeometry | None = None,
    ) -> tuple[VisualCard, ...]:
        if raw_predictions_out is not None:
            raw_predictions_out.clear()
        if prepared_crops_out is not None:
            prepared_crops_out.clear()
        if geometry is None:
            geometry = infer_hand_geometry(
                image, count=visible_count, recovery_config=self.config.short_hand_recovery,
            )
        boxes = geometry.boxes
        if not boxes:
            return ()
        prepared = [
            prepare_hand_rank_crop(
                image.crop(box),
                geometry_verified=(
                    geometry.count is not None
                    and (geometry.usable or bool(geometry.clipped_slots))
                    and index not in geometry.clipped_slots
                    and 0 <= box[0] < box[2] <= image.width
                    and 0 <= box[1] < box[3] <= image.height
                ),
                config=self.config.hand_photometry,
            )
            for index, box in enumerate(boxes)
        ]
        crops = [item.image for item in prepared]
        if prepared_crops_out is not None:
            prepared_crops_out.extend(crops)
        predictions = self.predictor(crops)
        if raw_predictions_out is not None:
            raw_predictions_out.extend(predictions)
        observed: list[VisualCard] = []
        for index, prediction in enumerate(predictions):
            glyph = extract_rank_glyph(crops[index])
            reference_match = self._match_rank_reference(
                crops[index],
                minimum_similarity=0.68,
                minimum_margin=0.05,
                glyph=glyph,
            )
            color = (
                read_joker_color(crops[index], glyph, self.config.joker_color)
                if prediction.rank in {"SJ", "BJ"} else None
            )
            rank, confidence, evidence = _fuse_card_prediction(
                prediction,
                reference_match,
                joker_color=color,
            )
            evidence = replace(evidence, input_photometry=prepared[index].evidence)
            if index in geometry.clipped_slots:
                confidence = 0.0
                evidence = replace(evidence, fusion_reason=f"{evidence.fusion_reason};geometry_clipped")
            elif not geometry.usable and not geometry.clipped_slots:
                confidence = 0.0
                evidence = replace(
                    evidence,
                    fusion_reason=(
                        f"{evidence.fusion_reason};geometry_unusable:{geometry.reason or 'unknown'}"
                    ),
                )
            elif not prepared[index].evidence.usable:
                confidence = 0.0
                evidence = replace(
                    evidence, fusion_reason=(
                        f"{evidence.fusion_reason};photometry_unusable:{prepared[index].evidence.reason}"
                    ),
                )
            observed.append(
                VisualCard(
                    rank=rank,
                    confidence=confidence,
                    box=boxes[index],
                    evidence=evidence,
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
            if any(_glyph_similarity(signature, existing) >= 0.98 for existing in references):
                continue
            references.append(signature)
            del references[:-12]

    def _remember_rank_references(
        self,
        hand_image: Image.Image,
        cards: Sequence[VisualCard],
        *,
        raw_predictions: Sequence[CardPrediction] = (),
        prepared_crops: Sequence[Image.Image] | None = None,
        reset: bool = False,
    ) -> None:
        # Reject the whole sample before changing references or its fingerprint.
        # In particular, 17 boxes can still contain eight 10s or two small Jokers.
        if not 1 <= len(cards) <= 20 or len(raw_predictions) != len(cards):
            return
        if prepared_crops is not None and (
            len(prepared_crops) != len(cards)
            or any(
                crop.size != (card.box[2] - card.box[0], card.box[3] - card.box[1])
                for crop, card in zip(prepared_crops, cards, strict=True)
            )
        ):
            return
        if any(
            not 0.0 < card.confidence <= 1.0
            or card.confidence < self.config.confidence_threshold
            for card in cards
        ):
            return
        try:
            CardSet.parse(card.rank for card in cards)
            CardSet.parse(prediction.rank for prediction in raw_predictions)
        except ValueError:
            return
        fingerprint = tuple(card.rank for card in cards)
        if fingerprint == self._reference_hand_fingerprint:
            return
        # Production passes the exact images used by the raw classifier. A
        # legacy direct call without prepared crops retains original pixels;
        # never estimate a second gain or learn a different exposure here.
        rank_crops = (
            prepared_crops if prepared_crops is not None
            else tuple(hand_image.crop(card.box) for card in cards)
        )
        # Validate every output Joker before selecting high-confidence learning
        # candidates. Otherwise a weak/NaN raw Joker could be skipped while a
        # different valid card reset the whole hand's references/fingerprint.
        joker_signatures: dict[int, frozenset[int]] = {}
        for index, (card, prediction) in enumerate(zip(cards, raw_predictions, strict=True)):
            if card.rank not in {"SJ", "BJ"}:
                continue
            if prediction.rank != card.rank:
                return
            card_image = rank_crops[index]
            glyph = extract_rank_glyph(card_image)
            color = read_joker_color(
                card_image, glyph, self.config.joker_color,
            )
            if color.color != ("red" if card.rank == "BJ" else "black"):
                return
            joker_signatures[index] = glyph.signature
        learned: list[tuple[str, frozenset[int]]] = []
        minimum_confidence = max(0.85, self.config.confidence_threshold)
        for index, (card, prediction) in enumerate(zip(cards, raw_predictions, strict=True)):
            # A calibrated template score is not an independent confirmation.
            # Learn only ranks supported by the original classifier itself.
            if not (
                prediction.rank == card.rank
                and card.confidence >= minimum_confidence
                and minimum_confidence <= prediction.confidence <= 1.0
            ):
                continue
            signature = (
                joker_signatures[index] if index in joker_signatures
                else _rank_glyph_signature(rank_crops[index])
            )
            if signature:
                learned.append((card.rank, signature))
        if not learned:
            return
        if reset:
            self._reset_rank_references()
        self._reference_hand_fingerprint = fingerprint
        for rank, signature in learned:
            references = self._rank_references.setdefault(rank, [])
            if any(_glyph_similarity(signature, existing) >= 0.98 for existing in references):
                continue
            references.append(signature)
            del references[:-4]

    def _reset_rank_references(self) -> None:
        self._rank_references = {
            rank: list(references) for rank, references in self._base_rank_references.items()
        }
        self._reference_hand_fingerprint = None

    def _match_rank_reference(
        self,
        card_image: Image.Image,
        *,
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.12,
        allowed_ranks: set[str] | None = None,
        glyph: RankGlyphComponent | None = None,
    ) -> tuple[str, float] | None:
        signature = glyph.signature if glyph is not None else _rank_glyph_signature(card_image)
        if not signature:
            return None
        scores = sorted(
            (
                max(_glyph_similarity(signature, reference) for reference in references),
                rank,
            )
            for rank, references in self._rank_references.items()
            if references and (allowed_ranks is None or rank in allowed_ranks)
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
        required_margin = min(minimum_margin, 0.025) if best_score >= 0.90 else minimum_margin
        if best_score < minimum_similarity or best_score - runner_up < required_margin:
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
    yellow = (red > 170) & (green > 100) & (blue < 160) & ((red - blue) > 45)
    maximum = np.maximum.reduce((red, green, blue))
    minimum = np.minimum.reduce((red, green, blue))
    white = (red > 180) & (green > 180) & (blue > 180) & ((maximum - minimum) < 35)
    pixel_count = max(1, badge.shape[0] * badge.shape[1])
    yellow_ratio = float(yellow.sum()) / pixel_count
    white_ratio = float(white.sum()) / pixel_count
    minimum_glyph_ratio = 0.01
    if yellow_ratio >= minimum_glyph_ratio and yellow_ratio >= white_ratio * 1.5:
        confidence = min(0.999, 0.90 + yellow_ratio * 2.0)
        return SeatRole.LANDLORD, confidence
    if white_ratio >= minimum_glyph_ratio and white_ratio >= yellow_ratio * 1.5:
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

    active_rows = int(np.count_nonzero(pale.sum(axis=1) >= max(3, width * 0.015)))
    active_columns = int(np.count_nonzero(pale.sum(axis=0) >= max(2, band_bottom * 0.08)))
    if active_rows < max(5, round(band_bottom * 0.12)) or active_columns < max(
        12, round(width * 0.08)
    ):
        return 0.0
    return min(0.995, 0.82 + (density - 0.025) * 3.2)


def _classify_turn_controls(
    image: Image.Image,
) -> tuple[bool | None, float]:
    """Legacy two-value API; use complete entities, never yellow area alone."""
    evidence = observe_turn_controls(image)
    return evidence.active, evidence.confidence


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
    if text_match is not None and text_match.count is not None and text_match.confidence >= 0.50:
        return text_match.count, text_match.confidence, True
    # Whole-ROI templates are useful as a best-effort display value, but an
    # unseen number can still look very similar because most pixels are the
    # unchanged seat background. Only near-exact matches may block tracking.
    return (
        template_count,
        template_confidence,
        bool(template_count is not None and template_confidence >= max(template_threshold, 0.995)),
    )


def _remaining_digit_masks(image: Image.Image) -> tuple[np.ndarray, ...]:
    """Extract one or two filled yellow counter glyphs from a seat ROI."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return ()
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (red > 145) & (green > 70) & (blue < 165) & ((red - blue) > 45) & ((green - blue) > 5)
    active_columns = np.any(mask, axis=0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(active_columns):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(active_columns) - 1):
            stop = index if not active else index + 1
            if stop - start >= 2:
                runs.append((start, stop))
            start = None

    candidates: list[tuple[np.ndarray, bool]] = []
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
        clipped = (
            left == 0 or right == mask.shape[1] or top == 0 or bottom == mask.shape[0]
        )
        candidates.append((glyph, clipped))
    return _trusted_counter_glyphs(candidates)


# Counter digits share one line height; a run this much shorter than the real
# digit is not part of the counter font.
_COUNTER_GLYPH_HEIGHT_TOLERANCE = 0.85


def _trusted_counter_glyphs(
    candidates: Sequence[tuple[np.ndarray, bool]],
) -> tuple[np.ndarray, ...]:
    """Drop ROI-clipped decorations; refuse a read that may hold a clipped digit.

    The calibrated counter badge sits inside its ROI, so a run touching the ROI
    border is never a complete counter glyph. The avatar or an animation
    bleeding over the border is shorter than the counter font, so such a run
    is discarded and the remaining glyphs are read. A clipped run as tall as a
    real digit may be a cut digit; the whole read then becomes unknown rather
    than a value that silently misses that digit.
    """
    unclipped = [glyph for glyph, clipped in candidates if not clipped]
    if not unclipped:
        return ()
    reference_height = max(glyph.shape[0] for glyph in unclipped)
    for glyph, clipped in candidates:
        if clipped and glyph.shape[0] >= reference_height * _COUNTER_GLYPH_HEIGHT_TOLERANCE:
            return ()
    return tuple(unclipped)


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
    return frozenset(int(index) for index in np.flatnonzero(pixels.reshape(-1) > 0))


def _remaining_signature_hole_count(signature: frozenset[int]) -> int:
    """Count enclosed background regions in a normalized counter glyph."""

    return len(_remaining_signature_holes(signature))


def _remaining_signature_holes(signature: frozenset[int]) -> tuple[frozenset[int], ...]:
    """Keep hole geometry from the same flood fill used for topology counts."""

    if not signature:
        return ()
    foreground = np.zeros((64, 64), dtype=bool)
    for index in signature:
        if 0 <= index < foreground.size:
            foreground[index // 64, index % 64] = True
    background = ~foreground
    visited = np.zeros_like(background)
    holes: list[frozenset[int]] = []
    for start_y in range(64):
        for start_x in range(64):
            if not background[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            pixels: list[int] = []
            touches_border = False
            while stack:
                y, x = stack.pop()
                pixels.append(y * 64 + x)
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
            if not touches_border and len(pixels) >= 4:
                holes.append(frozenset(pixels))
    return tuple(holes)


def _remaining_zero_hole_incompatible(
    signature: frozenset[int],
    holes: Sequence[frozenset[int]],
) -> bool:
    """Veto 0 when its only loop lies wholly above/below the glyph midpoint.

    A 9's upper loop and descending tail can overlap a 0 reference more than
    a cross-font 9 reference. Hole count alone cannot separate them. This is
    only a zero veto: the existing matcher must still establish another digit.
    Open/noisy or multi-hole glyphs retain the existing topology behavior.
    """

    if len(holes) != 1 or not signature:
        return False
    glyph_rows = [index // 64 for index in signature]
    hole_rows = [index // 64 for index in holes[0]]
    double_midpoint = min(glyph_rows) + max(glyph_rows)
    return (
        max(hole_rows) * 2 < double_midpoint
        or min(hole_rows) * 2 > double_midpoint
    )


def _minimum_card_confidence(cards: Sequence[VisualCard]) -> float:
    return min((card.confidence for card in cards), default=0.0)


def _visible_corner_widths(
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int | None, ...]:
    """Horizontal extent of each box before its nearest right neighbour in the same row.

    Pure geometry on the segmentation boxes: a neighbour shares the row when
    the two boxes overlap vertically by at least half of the shorter height,
    so a card wrapped below the last slot is not a neighbour. ``None`` means
    no right neighbour, i.e. the full crop is visible.
    """
    widths: list[int | None] = []
    for left, top, _right, bottom in boxes:
        nearest: int | None = None
        for other_left, other_top, _other_right, other_bottom in boxes:
            if other_left <= left:
                continue
            overlap = min(bottom, other_bottom) - max(top, other_top)
            shorter = max(1, min(bottom - top, other_bottom - other_top))
            if overlap * 2 < shorter:
                continue
            visible = other_left - left
            if nearest is None or visible < nearest:
                nearest = visible
        widths.append(nearest)
    return tuple(widths)


def _rank_glyph_signature(image: Image.Image) -> frozenset[int]:
    return extract_rank_glyph(image).signature


def _glyph_similarity(
    left: frozenset[int],
    right: frozenset[int],
) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_RANK_GLYPH_GRID_SIZE = 64
_RANK_GLYPH_BYTE_COUNT = (_RANK_GLYPH_GRID_SIZE**2) // 8
_BUILTIN_RANK_GLYPHS = Path(__file__).with_name("assets") / "rank_glyph_signatures.json"
_BUILTIN_REMAINING_GLYPHS = Path(__file__).with_name("assets") / "remaining_glyph_signatures.json"
_BUILTIN_ROUND_RESULT_GLYPHS = Path(__file__).with_name("assets") / "round_result_signatures.json"
_ROUND_RESULT_GRID_SIZE = (96, 48)
_ROUND_RESULT_BYTE_COUNT = (_ROUND_RESULT_GRID_SIZE[0] * _ROUND_RESULT_GRID_SIZE[1]) // 8


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
        raise ValueError(f"invalid glyph signature size: {len(raw)} bytes")
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
                signature = _decode_glyph_signature(base64.b64encode(raw).decode("ascii"))
            except (ValueError, binascii.Error, zlib.error):
                continue
            if signature:
                decoded.append(signature)
        if decoded:
            references[digit] = decoded
    return references


def _load_builtin_round_result_references(
    path: Path = _BUILTIN_ROUND_RESULT_GLYPHS,
) -> dict[RoundResultOutcome, list[frozenset[int]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        payload.get("format") != "round-result-signatures-v1"
        or payload.get("grid_size") != list(_ROUND_RESULT_GRID_SIZE)
        or not isinstance(payload.get("outcomes"), dict)
    ):
        return {}
    references: dict[RoundResultOutcome, list[frozenset[int]]] = {}
    for outcome in RoundResultOutcome:
        encoded_items = payload["outcomes"].get(outcome.value, ())
        if not isinstance(encoded_items, list):
            continue
        decoded: list[frozenset[int]] = []
        for encoded in encoded_items[:8]:
            if not isinstance(encoded, str):
                continue
            try:
                raw = base64.b64decode(encoded.encode("ascii"), validate=True)
            except (ValueError, binascii.Error):
                continue
            if len(raw) != _ROUND_RESULT_BYTE_COUNT:
                continue
            signature = frozenset(
                byte_index * 8 + bit_index
                for byte_index, value in enumerate(raw)
                for bit_index in range(8)
                if value & (1 << bit_index)
            )
            if signature:
                decoded.append(signature)
        if decoded:
            references[outcome] = decoded
    return references


def _classify_round_result(
    image: Image.Image,
    *,
    references: Mapping[RoundResultOutcome, Sequence[frozenset[int]]] | None = None,
) -> RoundResultObservation | None:
    """Detect the client's modal end-of-round panel before normal CV runs.

    A result panel has two large pale or bright-gold glyphs and a continue button.
    Requiring both structures avoids confusing ordinary white table cards or
    the yellow play controls with a terminal overlay.  The bundled signature
    identifies the observed ``失败`` and ``胜利`` artwork. Unknown pale titles
    retain the legacy unknown result; gold alone never implies victory.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return None
    height, width = rgb.shape[:2]
    title = rgb[
        round(height * 0.10) : round(height * 0.31),
        round(width * 0.37) : round(width * 0.63),
    ]
    button = rgb[
        round(height * 0.62) : round(height * 0.82),
        round(width * 0.37) : round(width * 0.63),
    ]
    if title.size == 0 or button.size == 0:
        return None
    title_white = (title.min(axis=2) > 180) & ((title.max(axis=2) - title.min(axis=2)) < 55)
    red, green, blue = button[:, :, 0], button[:, :, 1], button[:, :, 2]
    button_yellow = (
        (red > 150) & (green > 75) & (blue < 130) & ((red - blue) > 55) & ((green - blue) > 15)
    )
    button_white = (button.min(axis=2) > 180) & ((button.max(axis=2) - button.min(axis=2)) < 55)
    title_mask = title_white
    title_runs, title_height_ratio, title_density = _round_title_structure(title_mask)
    gold_title = False
    if len(title_runs) < 2 or title_height_ratio < 0.35 or not 0.05 <= title_density <= 0.36:
        # Reuse the old pale brightness and yellow channel-separation gates;
        # the dim modal background must not vote as bright title lettering.
        title_red, title_green, title_blue = title[:, :, 0], title[:, :, 1], title[:, :, 2]
        bright_gold = ((title_red > 180) & (title_green > 180) & (title_blue < 180)
                       & (title_red - title_blue > 55) & (title_green - title_blue > 15))
        title_mask = title_white | bright_gold
        title_runs, title_height_ratio, title_density = _round_title_structure(title_mask)
        gold_title = True
    yellow_density = float(button_yellow.mean())
    button_white_density = float(button_white.mean())
    if (
        len(title_runs) < 2
        or title_height_ratio < 0.35
        or not 0.05 <= title_density <= 0.36
        or yellow_density < 0.15
        or button_white_density > 0.10
    ):
        return None

    signature = _round_result_title_signature(title_mask)
    available = references if references is not None else _load_builtin_round_result_references()
    matches = [
        (_glyph_similarity(signature, reference), outcome)
        for outcome, outcome_references in available.items()
        for reference in outcome_references
    ]
    similarity, outcome = max(
        matches,
        default=(0.0, RoundResultOutcome.UNKNOWN),
    )
    if similarity < 0.62:
        outcome = RoundResultOutcome.UNKNOWN
    if gold_title and outcome is not RoundResultOutcome.VICTORY:
        return None
    structural_confidence = min(
        0.96,
        0.84 + min(0.08, yellow_density * 0.20) + min(0.04, title_density * 0.20),
    )
    confidence = (
        max(structural_confidence, min(0.995, 0.90 + similarity * 0.095))
        if outcome is not RoundResultOutcome.UNKNOWN
        else structural_confidence
    )
    return RoundResultObservation(outcome=outcome, confidence=confidence)


def _round_title_structure(title_white: np.ndarray) -> tuple[list[tuple[int, int]], float, float]:
    x_runs = _boolean_runs(
        (title_white.mean(axis=0) >= 0.10).tolist(),
        max_gap=max(2, round(title_white.shape[1] * 0.006)),
    )
    title_runs = [run for run in x_runs if run[1] - run[0] >= round(title_white.shape[1] * 0.15)]
    active_rows = np.flatnonzero(title_white.any(axis=1))
    title_height_ratio = (
        0.0
        if active_rows.size == 0
        else (int(active_rows[-1]) - int(active_rows[0]) + 1) / title_white.shape[0]
    )
    return title_runs, title_height_ratio, float(title_white.mean())


def _round_result_title_signature(mask: np.ndarray) -> frozenset[int]:
    if mask.size == 0 or not bool(mask.any()):
        return frozenset()
    active_rows = np.flatnonzero(mask.any(axis=1))
    active_columns = np.flatnonzero(mask.any(axis=0))
    if active_rows.size == 0 or active_columns.size == 0:
        return frozenset()
    cropped = mask[
        int(active_rows[0]) : int(active_rows[-1]) + 1,
        int(active_columns[0]) : int(active_columns[-1]) + 1,
    ]
    glyph = Image.fromarray(
        np.where(cropped, 255, 0).astype(np.uint8),
        mode="L",
    )
    fitted = ImageOps.contain(
        glyph,
        (92, 44),
        method=Image.Resampling.NEAREST,
    )
    normalized = Image.new("L", _ROUND_RESULT_GRID_SIZE)
    normalized.paste(
        fitted,
        (
            (normalized.width - fitted.width) // 2,
            (normalized.height - fitted.height) // 2,
        ),
    )
    pixels = np.asarray(normalized, dtype=np.uint8)
    return frozenset(int(index) for index in np.flatnonzero(pixels.reshape(-1) > 0))


def _is_card_white(pixel: tuple[int, ...]) -> bool:
    red, green, blue = pixel[:3]
    return max(red, green, blue) >= 175 and min(red, green, blue) >= 115


def _card_white_mask(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return (pixels.max(axis=2) >= 175) & (pixels.min(axis=2) >= 115)


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
        [int(count) >= max(8, round((right - left) * 0.25)) for count in row_counts],
        max_gap=2,
    )
    durable_runs = [run for run in row_runs if run[1] - run[0] >= max(12, round(height * 0.10))]
    if not durable_runs:
        return None

    top = min(run[0] for run in durable_runs)
    bottom = max(run[1] for run in durable_runs)
    span = bottom - top
    if span <= 0 or top > round(height * 0.28) or bottom < round(height * 0.70):
        return None
    aspect_ratio = (right - left) / span
    visible_ratio = sum(run_bottom - run_top for run_top, run_bottom in durable_runs) / span
    if not 0.48 <= aspect_ratio <= 0.80 or visible_ratio < 0.32:
        return None
    return (
        left,
        max(0, top - 2),
        right,
        min(height, bottom + 3),
    )


def segment_card_boxes(
    image: Image.Image,
    *,
    table_edges: bool = False,
) -> tuple[tuple[int, int, int, int], ...]:
    """Compatibility boxes for hand geometry and external labeling tools."""
    return tuple(region.box for region in segment_card_regions(image, table_edges=table_edges))


def segment_card_regions(
    image: Image.Image,
    *,
    table_edges: bool = False,
) -> tuple[CardRegion, ...]:
    """Tag full bodies vs overlapping corner ROIs before model preprocessing."""

    def rank_regions(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[CardRegion, ...]:
        return tuple(CardRegion(box, CardRegionKind.VISIBLE_CORNER) for box in boxes)

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
        return (CardRegion(single_card, CardRegionKind.FULL_CARD),)
    for left, right in _active_x_runs_from_mask(
        white_mask,
        min_column_ratio=0.08,
        max_gap=2,
    ):
        if right - left < minimum_width:
            continue
        row_counts = white_mask[:, left:right].sum(axis=1)
        row_runs = _boolean_runs(
            [int(count) >= max(4, round((right - left) * 0.25)) for count in row_counts],
            max_gap=2,
        )
        row_runs = [run for run in row_runs if run[1] - run[0] >= max(12, round(height * 0.10))]
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
            return rank_regions(_append_vertically_stacked_card_boxes(
                rgb,
                _filter_table_card_edges(rgb, overlapping_boxes) if table_edges else overlapping_boxes,
            ))
    if boxes:
        # Connected-component boxes already cover each full portrait card.
        # Horizontal wrap recovery is only meaningful for the overlapping
        # rank-crop path below; applying it here would turn a normal card's
        # bottom border into a false second card.
        return tuple(CardRegion(box, CardRegionKind.FULL_CARD) for box in boxes)
    return rank_regions(_append_vertically_stacked_card_boxes(
        rgb,
        _filter_table_card_edges(rgb, overlapping_boxes) if table_edges else overlapping_boxes,
    ))


def _filter_table_card_edges(
    image: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Reject short rank stems only after selecting a table overlap branch.

    A card border persists through the body and its lower half; the bold
    vertical stroke in a K can exceed the mean edge threshold without either.
    Hand fallbacks and their notice handling must not enable this filter.
    """

    original = tuple(boxes)
    if len(original) < 3:
        return original
    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    supported_boxes = []
    for box in original:
        left, top, _right, bottom = box
        physical_bottom = min(image.height, top + round((bottom - top) / 0.68))
        for x in range(max(1, left - 1), min(image.width, left + 2)):
            differences = np.abs(
                grayscale[top:physical_bottom, x] - grayscale[top:physical_bottom, x - 1]
            )
            if not len(differences):
                continue
            supported = differences >= 14
            runs = _boolean_runs(supported.tolist(), max_gap=2)
            longest = max((end - start for start, end in runs), default=0)
            if (
                longest >= len(supported) * 0.55
                and float(supported[len(supported) // 2 :].mean()) >= 0.50
            ):
                supported_boxes.append(box)
                break
    # Insufficient border evidence is not proof of a single card or empty
    # table. Keep the old candidate for the existing uncertainty/count gates.
    return tuple(supported_boxes) if len(supported_boxes) >= 2 else original


@dataclass(frozen=True)
class HandGeometry:
    """Hand-only geometry; unusable crops must not become trusted card ranks.

    Boxes may remain as diagnostic candidates while count is None: candidate
    cardinality is not independent evidence for the physical number of cards.
    """

    count: int | None
    boxes: tuple[tuple[int, int, int, int], ...]
    usable: bool
    clipped_slots: tuple[int, ...] = ()
    reason: str = ""


def infer_visible_hand_count(
    image: Image.Image,
    *,
    maximum: int = 20,
) -> int | None:
    return infer_hand_geometry(image, maximum=maximum).count


def infer_overlapping_hand_boxes(
    image: Image.Image,
    count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    if count <= 0:
        return ()
    return infer_hand_geometry(image, count=count).boxes


def infer_hand_geometry(
    image: Image.Image,
    *,
    maximum: int = 20,
    count: int | None = None,
    recovery_config: ShortHandRecoveryConfig | None = None,
) -> HandGeometry:
    """Resolve slots and rank crops together, without changing table masks.

    An explicit count retains the legacy white-fan fallback. Gray selections
    require independently visible card slots instead of inventing a grid.
    """

    if recovery_config is not None and not isinstance(recovery_config, ShortHandRecoveryConfig):
        raise ValueError("recovery_config must be a ShortHandRecoveryConfig")
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb)
    white = _card_white_mask(rgb)
    high, low = pixels.max(axis=2), pixels.min(axis=2)
    gray = (low >= 90) & (high < 175) & (high - low <= 18)
    gray_bands = [
        run
        for run in _active_x_runs_from_mask(gray, min_column_ratio=0.45, max_gap=3)
        if run[1] - run[0] >= max(12, round(rgb.height * 0.04))
    ]
    grayscale: np.ndarray | None = None
    horizontal_difference: np.ndarray | None = None
    input_crop_ratios = (0.535, 0.60)  # Primary white/gray crop contract.
    visible_count: int | None
    if gray_bands:
        # A neutral background cannot prove where a gray hand begins. Do not
        # reinterpret a full-ROI gray panel as one card or use expected_count.
        if any(left <= 2 or right >= rgb.width - 2 for left, right in gray_bands):
            return HandGeometry(None, (), False, reason="ambiguous_gray_background")
        mask = white | gray
        boxes = _gray_hand_boxes(rgb, mask, maximum=maximum)
        if not boxes:
            grayscale, horizontal_difference = _hand_edge_arrays(rgb)
            boxes = _recover_short_hand_boxes(
                rgb, mask, maximum=maximum, config=recovery_config,
                grayscale=grayscale, horizontal_difference=horizontal_difference,
            )
            input_crop_ratios = (recovery_config or ShortHandRecoveryConfig()).rank_crop_ratios
        if not boxes:
            return HandGeometry(None, (), False, reason="ambiguous_gray_hand")
        visible_count = len(boxes)
    else:
        mask = white
        visible_count = _infer_white_hand_count(rgb, maximum=maximum) if count is None else count
        boxes = _infer_white_hand_boxes(rgb, visible_count) if visible_count else ()
    if not boxes:
        return HandGeometry(visible_count, (), False)
    if horizontal_difference is None:
        grayscale, horizontal_difference = _hand_edge_arrays(rgb)
    geometry = _anchor_hand_slots(
        rgb, mask, boxes, visible_count, horizontal_difference=horizontal_difference,
        config=recovery_config, grayscale=grayscale, input_crop_ratios=input_crop_ratios,
    )
    if (
        not gray_bands
        and not geometry.usable
        and geometry.clipped_slots
        and geometry.count is not None
        and geometry.count > 3
    ):
        recovered = _recover_short_hand_boxes(
            rgb, mask, maximum=maximum, config=recovery_config,
            grayscale=grayscale, horizontal_difference=horizontal_difference,
        )
        if recovered:
            input_crop_ratios = (recovery_config or ShortHandRecoveryConfig()).rank_crop_ratios
            geometry = _anchor_hand_slots(
                rgb, mask, recovered, len(recovered), horizontal_difference=horizontal_difference,
                config=recovery_config, grayscale=grayscale, input_crop_ratios=input_crop_ratios,
            )
    if not gray_bands:
        assert grayscale is not None
        geometry = _verify_short_hand_origins(
            rgb, mask, geometry, maximum=maximum,
            config=recovery_config or ShortHandRecoveryConfig(),
            grayscale=grayscale, horizontal_difference=horizontal_difference,
            input_crop_ratios=input_crop_ratios,
        )
    return geometry


def _hand_edge_arrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    return grayscale, np.abs(np.diff(grayscale, axis=1))


def _verify_short_hand_origins(
    image: Image.Image,
    mask: np.ndarray,
    geometry: HandGeometry,
    *,
    maximum: int,
    config: ShortHandRecoveryConfig,
    grayscale: np.ndarray,
    horizontal_difference: np.ndarray,
    input_crop_ratios: tuple[float, float] = (0.535, 0.60),
) -> HandGeometry:
    """White, level 2–3-card fans: count agreement cannot certify origins.

    Missing old rim support only discredits the old crop. Replacement origins
    must be recovered independently before comparing candidate cardinality.
    """
    # A wrong x can also make vertical extent verification fail. Permit the
    # independent horizontal recovery to inspect these candidates, but never
    # convert their length into a confirmed count unless recovery and the
    # second full anchoring both prove the replacement slots.
    unconfirmed_extents = (
        geometry.count is None and not geometry.usable
        and geometry.reason == "hand_slot_extent_unverified:candidate_count_unconfirmed"
    )
    if (
        geometry.clipped_slots
        or not 2 <= len(geometry.boxes) <= config.maximum_slots
        or (not unconfirmed_extents and (
            not geometry.usable or geometry.count != len(geometry.boxes)
        ))
    ):
        return geometry
    runs = _active_x_runs_from_mask(mask, min_column_ratio=0.08, max_gap=3)
    if len(runs) != 1:
        return geometry
    left, right = runs[0]
    rows = _boolean_runs(
        (mask[:, left:right].sum(axis=1) >= max(8, round((right - left) * 0.25))).tolist(),
        max_gap=2,
    )
    rows = [run for run in rows if run[1] - run[0] >= max(18, round(image.height * 0.25))]
    if len(rows) != 1:
        return geometry
    top, bottom = rows[0]
    body_height = bottom - top
    alignment = max(4, round(body_height * config.alignment_ratio))
    if (
        top <= 0
        or bottom >= image.height - 2
        or not config.fan_aspect[0] <= (right - left) / body_height <= config.fan_aspect[1]
        or max(box[1] for box in geometry.boxes) - min(box[1] for box in geometry.boxes)
        > alignment
    ):
        return geometry
    band = horizontal_difference[
        top + round(body_height * config.lower_band[0]):
        top + round(body_height * config.lower_band[1])
    ]
    if not band.shape[0]:
        return geometry

    def support(x: int) -> float:
        strip = band[:, max(0, x - 4):min(band.shape[1], x + 3)]
        return float((strip.max(axis=1) >= config.minimum_difference).mean()) if strip.size else 0.0

    suspect = {
        index for index, box in enumerate(geometry.boxes)
        if support(box[0]) < config.minimum_edge_support
    }
    if not suspect:
        return geometry
    # Every failure after this point preserves candidates, not a trusted count
    # or rank. The existing observation entry sets all unusable slots to zero.
    unverified = HandGeometry(
        None, geometry.boxes, False,
        reason="horizontal_origin_unverified:candidate_count_unconfirmed",
    )
    recovered = _recover_short_hand_boxes(
        image, mask, maximum=maximum, config=config,
        grayscale=grayscale, horizontal_difference=horizontal_difference,
    )
    if len(recovered) != len(geometry.boxes):
        return unverified
    bounds = (left, top, right, bottom)
    top_bounds = (left, image.height - bottom, right, image.height - top)
    for new in recovered:
        if (
            support(new[0]) < config.minimum_edge_support
            or not _short_hand_corner(horizontal_difference, bounds, new[0], config)
            or not _short_hand_corner(horizontal_difference[::-1], top_bounds, new[0], config)
        ):
            return unverified
    adjusted = []
    for index, (old, new) in enumerate(zip(geometry.boxes, recovered, strict=True)):
        x, y, old_right, old_bottom = old
        width = old_right - x
        offset = new[0] - x
        if index not in suspect:
            if abs(offset) > alignment:
                return unverified
            adjusted.append(old)
        else:
            if abs(offset) >= width / 2 or offset == 0:
                return unverified
            adjusted.append((new[0], y, new[0] + width, old_bottom))
    anchored = _anchor_hand_slots(
        image, mask, tuple(adjusted), len(adjusted),
        horizontal_difference=horizontal_difference, config=config, grayscale=grayscale,
        input_crop_ratios=input_crop_ratios,
    )
    if (
        not anchored.usable
        or anchored.count != len(recovered)
        or anchored.boxes != tuple(adjusted)
    ):
        # A second anchoring pass must not turn horizontal repair into a new
        # vertical/crop-size interpretation or silently drop a candidate slot.
        return unverified
    return anchored


def _short_hand_corner(
    difference: np.ndarray,
    bounds: tuple[int, int, int, int],
    x: int,
    config: ShortHandRecoveryConfig,
) -> bool:
    """A sustained rim must turn right at the body endpoint, not stay a bar."""
    _, top, _, bottom = bounds
    height = bottom - top
    radius = max(8, round(height * config.corner_depth_ratio))
    frame_height, difference_width = difference.shape
    if bottom >= frame_height or bottom - radius < 0 or x < 4 or x + radius > difference_width:
        return False
    band = difference[bottom - radius : bottom, x - 4 : x + radius]
    offsets = band.argmax(axis=1).astype(float) - 4
    valid = band.max(axis=1) >= config.minimum_difference
    indexes = np.flatnonzero(valid)
    if (
        len(offsets) < 10
        or not len(indexes)
        or float(valid.mean()) < config.minimum_corner_support
        or len(offsets) - 1 - indexes[-1] > max(3, height * config.alignment_ratio)
    ):
        return False
    offsets = offsets[: indexes[-1] + 1]
    base = float(np.median(offsets[: max(3, len(offsets) // 3)]))
    bend = float(np.median(offsets[-3:])) - base
    minimum_bend = max(2.0, height * config.corner_bend_ratio)
    return bool(
        abs(base) <= 3
        and bend >= minimum_bend
        and np.count_nonzero(offsets >= base + minimum_bend) >= 2
        and float(np.mean(np.abs(np.diff(offsets)) <= max(2, height * 0.01)))
        >= config.minimum_corner_support
    )


def _self_table_regions_complete(
    image: Image.Image, regions: Sequence[CardRegion], *,
    config: SelfTableConfig, edge_config: ShortHandRecoveryConfig,
) -> bool:
    return _self_table_body_bounds(image, regions, config=config, edge_config=edge_config) is not None


def _self_table_rank_regions(
    regions: Sequence[CardRegion], bounds: tuple[int, int, int, int], config: LiveLayoutConfig,
) -> tuple[CardRegion, ...]:
    """Use proven body height, clipped only by the next physically exposed seam."""
    _, top, right, bottom = bounds
    inputs = []
    for index, region in enumerate(regions):
        x = region.box[0]
        exposed_right = regions[index + 1].box[0] if index + 1 < len(regions) else right
        body = CardRegion((x, top, right, bottom), CardRegionKind.FULL_CARD)
        left, upper, crop_right, lower = body.classifier_box(config.table_rank_crop)
        inputs.append(CardRegion((left, upper, min(crop_right, exposed_right), lower), CardRegionKind.RANK_CROP))
    return tuple(inputs)


def _self_table_body_bounds(
    image: Image.Image,
    regions: Sequence[CardRegion],
    *,
    config: SelfTableConfig,
    edge_config: ShortHandRecoveryConfig,
) -> tuple[int, int, int, int] | None:
    """Certify all proposed slots in a closed paper fan, never create slots.

    Button lettering can pass the table segmenter's white-area heuristic.
    Require both curved endpoints of every sustained seam and the outer
    right edge, using the unchanged card-corner predicate. A bottom-clipped
    panel or an unrepresented real seam cannot establish a complete group.
    """
    if (not regions or not image.width or not image.height
            or image.width * image.height > config.maximum_input_pixels):
        return None
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    mask = (rgb.min(axis=2) >= config.paper_min_channel) & (
        np.ptp(rgb, axis=2) <= config.paper_max_chroma
    )
    columns = [run for run in _active_x_runs_from_mask(
        mask, min_column_ratio=config.minimum_column_fraction, max_gap=3,
    ) if run[1] - run[0] >= max(12, round(image.width * 0.03))]
    if len(columns) != 1:
        return None
    left, right = columns[0]
    rows = [run for run in _boolean_runs(
        (mask[:, left:right].mean(axis=1) >= config.minimum_row_fraction).tolist(), max_gap=2,
    ) if run[1] - run[0] >= max(18, round(image.height * config.minimum_height_fraction))]
    if len(rows) != 1:
        return None
    top, bottom = rows[0]
    height = bottom - top
    alignment = max(4, round(height * edge_config.alignment_ratio))
    starts = [region.box[0] for region in regions]
    if (
        top <= alignment or bottom >= image.height - alignment
        or left <= alignment or right >= image.width - alignment
        or starts != sorted(set(starts)) or abs(starts[0] - left) > alignment
        or not edge_config.final_aspect[0] <= (right - starts[-1]) / height <= edge_config.final_aspect[1]
        or any(abs(region.box[1] - top) > alignment or region.box[2] > right + alignment
               or region.box[3] > bottom + alignment for region in regions)
    ):
        return None
    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    difference = np.abs(np.diff(grayscale, axis=1))
    bounds = left, top, right, bottom
    reversed_bounds = left, image.height - bottom, right, image.height - top

    def closed_seam(x: int) -> bool:
        if not (_short_hand_corner(difference, bounds, x, edge_config)
                and _short_hand_corner(difference[::-1], reversed_bounds, x, edge_config)):
            return False
        radius = max(8, round(height * edge_config.corner_depth_ratio))
        band = difference[top + radius:bottom - radius, max(0, x - 3):x + 3]
        return bool(band.size and float((band.max(axis=1) >= edge_config.minimum_difference).mean())
                    >= edge_config.minimum_edge_support)

    for index, x in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else right
        if stop <= x or float(mask[top:bottom, x:stop].mean()) < config.minimum_paper_fraction:
            return None
        if not closed_seam(x):
            return None
    # The last region's right coordinate can be only a visible corner. Prove
    # the real outside edge from paper pixels, not from its classifier box.
    mirrored = np.abs(np.diff(grayscale[:, ::-1], axis=1))
    mirrored_bounds = image.width - right, top, image.width - left, bottom
    mirror_top = image.width - right, image.height - bottom, image.width - left, image.height - top
    if not (_short_hand_corner(mirrored, mirrored_bounds, image.width - right, edge_config)
            and _short_hand_corner(mirrored[::-1], mirror_top, image.width - right, edge_config)):
        return None
    # A short obstruction in the legacy lower band can hide an omitted seam.
    # Discover candidates across the entire straight body; both endpoints
    # still need independent closure before any omission is considered real.
    radius = max(8, round(height * edge_config.corner_depth_ratio))
    band = difference[top + radius:bottom - radius]
    if not band.size:
        return None
    scores = (band >= edge_config.minimum_difference).mean(axis=0)
    groups = _integer_groups(
        (np.flatnonzero(scores >= edge_config.minimum_edge_support) + 1).tolist(), max_gap=2,
    )
    for group in groups:
        x = max(group, key=lambda candidate: float(scores[candidate - 1]))
        if left + alignment < x < right - alignment and closed_seam(x):
            if not any(abs(x - start) <= alignment for start in starts):
                return None
    return bounds


def _recover_short_hand_boxes(
    image: Image.Image,
    mask: np.ndarray,
    *,
    maximum: int,
    config: ShortHandRecoveryConfig | None = None,
    grayscale: np.ndarray | None = None,
    horizontal_difference: np.ndarray | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """Fail-closed fallback for failed gray fans or clipped white pseudo-grids.

    Card bodies must be aligned and fully visible at their lower edge. Rank
    tops may be clipped, but the caller must preserve that unusable geometry.
    No expected count, cards, classifier confidence or tracker state is used.
    """
    config = config or ShortHandRecoveryConfig()
    width, height = image.size
    runs = _active_x_runs_from_mask(mask, min_column_ratio=0.08, max_gap=3)
    if len(runs) != 1:
        return ()
    left, right = runs[0]
    rows = _boolean_runs(
        (mask[:, left:right].sum(axis=1) >= max(8, round((right - left) * 0.25))).tolist(),
        max_gap=2,
    )
    rows = [run for run in rows if run[1] - run[0] >= max(18, round(height * 0.25))]
    if len(rows) != 1:
        return ()
    top, bottom = rows[0]
    body_height = bottom - top
    if (
        bottom >= height - 2
        or not config.fan_aspect[0] <= (right - left) / body_height <= config.fan_aspect[1]
    ):
        return ()
    if grayscale is None:
        grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    difference = (
        np.abs(np.diff(grayscale, axis=1))
        if horizontal_difference is None else horizontal_difference
    )
    band = difference[
        top + round(body_height * config.lower_band[0]) : top
        + round(body_height * config.lower_band[1])
    ]
    if not band.shape[0]:
        return ()
    scores = (band >= config.minimum_difference).mean(axis=0)
    groups = _integer_groups(
        (np.flatnonzero(scores >= config.minimum_edge_support) + 1).tolist(), max_gap=2
    )
    # Treat every sustained interior rim as a candidate before inspecting its
    # closed endpoint. Do not estimate a grid from character-stroke distances.
    candidates = [
        max(group, key=lambda x: float(scores[x - 1]))
        for group in groups
        if left + 3 < min(group) and max(group) < right - 3
    ]
    bounds = (left, top, right, bottom)
    # Same horizontal differences, vertically reversed: no per-candidate image
    # conversion, full-ROI differencing or flipped-image allocation.
    flipped_difference = difference[::-1, :]
    top_bounds = (left, height - bottom, right, height - top)
    starts = [left]
    for x in candidates:
        if not _short_hand_corner(difference, bounds, x, config):
            continue
        if top > 0 and not _short_hand_corner(flipped_difference, top_bounds, x, config):
            # A bottom-curved decoration is not a complete card-side seam.
            continue
        starts.append(x)
    if not 1 <= len(starts) <= min(config.maximum_slots, maximum):
        return ()
    if not config.final_aspect[0] <= (right - starts[-1]) / body_height <= config.final_aspect[1]:
        return ()
    crop_height = max(24, round(body_height * config.rank_crop_ratios[0]))
    crop_width = max(16, round(crop_height * config.rank_crop_ratios[1]))
    # This narrow fallback must not claim rank slots too thin for its own
    # requested crop. Existing healthy dense fans never enter this branch.
    if any(second - first < crop_width for first, second in zip(starts, starts[1:], strict=False)):
        return ()
    boxes = tuple(
        (x, top, min(width, x + crop_width), min(height, top + crop_height)) for x in starts
    )
    tops, bottoms = [], []
    for index, (x, _, crop_right, crop_bottom) in enumerate(boxes):
        exposed_right = starts[index + 1] if index + 1 < len(starts) else crop_right
        margin = max(3, round((exposed_right - x) * 0.10))
        strip = mask[:, x + margin : exposed_right - margin]
        face_rows = _boolean_runs((strip.mean(axis=1) >= 0.50).tolist(), max_gap=2)
        face_rows = [run for run in face_rows if run[1] - run[0] >= max(6, round(height * 0.02))]
        if not face_rows:
            return ()
        tops.append(min(run[0] for run in face_rows))
        bottoms.append(max(run[1] for run in face_rows))
        ink = grayscale[top + margin : crop_bottom, x + margin : exposed_right - margin]
        if not ink.size or np.count_nonzero(ink < 80) < max(6, round(ink.size * 0.002)):
            return ()
    # A shared endpoint band is not evidence for staggered or occluded bodies.
    tolerance = max(4, round(body_height * config.alignment_ratio))
    if max(tops) - min(tops) > tolerance or max(bottoms) - min(bottoms) > tolerance:
        return ()
    if any(abs(value - bottom) > tolerance for value in bottoms):
        return ()
    return boxes



def _gray_hand_boxes(
    image: Image.Image,
    mask: np.ndarray,
    *,
    maximum: int,
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = image.size
    runs = _active_x_runs_from_mask(mask, min_column_ratio=0.08, max_gap=3)
    if not runs:
        return ()
    left, right = max(runs, key=lambda run: run[1] - run[0])
    rows = _boolean_runs(
        (mask[:, left:right].sum(axis=1) >= max(8, round((right - left) * 0.25))).tolist(),
        max_gap=2,
    )
    rows = [run for run in rows if run[1] - run[0] >= max(18, round(height * 0.25))]
    if not rows:
        return ()
    top, bottom = min(run[0] for run in rows), max(run[1] for run in rows)
    card_height = bottom - top
    grayscale = np.asarray(image.convert("L"), dtype=np.float64)
    edge_start, edge_stop = max(left + 1, 1), min(right + 1, width)
    current = grayscale[top:bottom, edge_start:edge_stop]
    previous = grayscale[top:bottom, edge_start - 1 : edge_stop - 1]
    local_level = np.maximum(40.0, (current.mean(axis=0) + previous.mean(axis=0)) / 2)
    scores = np.abs(current - previous).mean(axis=0) * 200.0 / local_level
    preliminary = _integer_groups(
        (np.flatnonzero(scores >= 14.0) + edge_start).tolist(), max_gap=2
    )
    peaks = [max(float(scores[x - edge_start]) for x in group) for group in preliminary]
    # A single white-to-gray transition must not suppress a weaker gray/white
    # card boundary. Use typical strong peaks rather than the maximum outlier.
    threshold = max(14.0, float(np.percentile(peaks, 75)) * 0.50) if peaks else 14.0
    groups = _integer_groups(
        (np.flatnonzero(scores >= threshold) + edge_start).tolist(), max_gap=2
    )
    edges = [max(group, key=lambda x: float(scores[x - edge_start])) for group in groups]
    if (right - left) / card_height <= 0.82:
        starts = [left]
    else:
        starts = [left, *(x for x in edges if left + 3 < x < right - 3)]
        starts = [
            box[0]
            for box in _regularize_overlapping_hand_boxes(
                tuple((x, top, x + 32, bottom) for x in starts)
            )
        ]
    if not 1 <= len(starts) <= maximum:
        return ()
    # The final exposed card must still have a portrait-sized body. A large
    # featureless gray dialog with a few text strokes is not an overlapped fan.
    if not 0.40 <= (right - starts[-1]) / card_height <= 0.90:
        return ()
    crop_height = max(24, round(card_height * 0.535))
    crop_width = max(16, round(crop_height * 0.60))
    boxes = tuple(
        (x, top, min(width, x + crop_width), min(height, top + crop_height)) for x in starts
    )
    for index, (x, y, crop_right, crop_bottom) in enumerate(boxes):
        strip_right = starts[index + 1] if index + 1 < len(starts) else crop_right
        margin = max(3, round((strip_right - x) * 0.10))
        ink = grayscale[
            min(crop_bottom, y + margin) : crop_bottom,
            x + margin : max(x + margin + 1, strip_right - margin),
        ]
        if ink.size == 0 or int(np.count_nonzero(ink < 80)) < max(6, round(ink.size * 0.002)):
            return ()
    return boxes


@dataclass(frozen=True)
class _HandSlotExtent:
    left: int
    top: int
    bottom: int


def _hand_slot_face_rows(
    mask: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    index: int,
) -> tuple[int, list[tuple[int, int]]]:
    left, _top, right, _bottom = boxes[index]
    strip_right = boxes[index + 1][0] if index + 1 < len(boxes) else right
    margin = max(3, round((strip_right - left) * 0.10))
    sample_left = min(strip_right - 2, left + margin)
    sample_right = max(sample_left + 1, strip_right - margin)
    if sample_left < 0 or sample_right > mask.shape[1] or sample_left >= sample_right:
        return strip_right, []
    rows = _boolean_runs(
        (mask[:, sample_left:sample_right].mean(axis=1) >= 0.50).tolist(), max_gap=2,
    )
    return strip_right, [run for run in rows if run[1] - run[0] >= max(6, round(mask.shape[0] * 0.02))]


def _hand_seed_extents_agree(
    mask: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    config: ShortHandRecoveryConfig,
    input_height_ratio: float = 0.535,
) -> bool:
    """Visible face endpoints may discredit even a geometrically unchanged seed."""
    for index, (_left, top, _right, bottom) in enumerate(boxes):
        _strip_right, rows = _hand_slot_face_rows(mask, boxes, index)
        if not rows:
            return False
        body_height = (bottom - top) / input_height_ratio
        tolerance = max(4, round(body_height * config.alignment_ratio))
        if abs(rows[0][0] - top) > tolerance or abs(rows[-1][1] - top - body_height) > tolerance:
            return False
    return True


def _certify_hand_slot_extents(
    mask: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    difference: np.ndarray,
    config: ShortHandRecoveryConfig,
    *,
    grayscale: np.ndarray | None = None,
) -> tuple[_HandSlotExtent, ...] | None:
    """Require each proposed slot's own closed top and bottom, not a longest edge.

    Regularization is only a contradiction check. Its returned slots are never
    used to create or delete candidates, or to borrow a count from game state.
    """
    if not boxes or _regularize_overlapping_hand_boxes(boxes) != boxes:
        return None
    height, width = mask.shape
    flipped = difference[::-1]
    extents: list[_HandSlotExtent] = []
    for index, (left, seed_top, _right, _seed_bottom) in enumerate(boxes):
        strip_right, rows = _hand_slot_face_rows(mask, boxes, index)
        if not rows or rows[0][0] == 0 or rows[-1][1] >= height - 2:
            return None
        face_top, bottom = rows[0][0], rows[-1][1]
        edge_left, edge_right = max(0, left - 4), min(width - 1, left + 3)
        if edge_left >= edge_right:
            return None
        edge_rows = _boolean_runs(
            (difference[:, edge_left:edge_right].max(axis=1) >= 20).tolist(), max_gap=2,
        )
        edge_rows = [run for run in edge_rows if run[1] - run[0] >= max(18, round(height * 0.25))]
        if grayscale is not None:
            # Square cards have no curved corner. Require their independent
            # paired closures, not a relaxed threshold on the rounded path.
            rectangular_extents = [
                (top, end)
                for top in sorted({face_top, seed_top, *(start for start, _ in edge_rows)})
                for end in sorted({bottom, *(stop for _, stop in edge_rows)})
                if rectangular_slot_supported(
                    grayscale, left=left, top=top, right=strip_right, bottom=end, config=config,
                )
            ]
            if rectangular_extents:
                tolerance = max(3, round((bottom - face_top) * config.alignment_ratio))
                if any(
                    max(values) - min(values) > tolerance
                    for values in zip(*rectangular_extents, strict=True)
                ):
                    return None
                # Prefer visible face endpoints within the certified band.
                top, end = min(rectangular_extents, key=lambda pair: (
                    abs(pair[0] - face_top) + abs(pair[1] - bottom), pair,
                ))
                extents.append(_HandSlotExtent(left, top, end))
                continue
        # A raised card can reveal a predecessor below itself. Find its own
        # closed bottom before using that endpoint to confirm the top.
        search_radius = max(8, round((bottom - face_top) * config.corner_depth_ratio))
        if not _short_hand_corner(difference, (left, face_top, strip_right, bottom), left, config):
            proposals = {
                value for _start, stop in edge_rows
                for value in range(stop, min(height - 2, stop + search_radius + 1))
            }
            accepted_bottoms = [
                value for value in sorted(proposals)
                if value - face_top >= max(24, round(height * 0.25))
                and _short_hand_corner(difference, (left, face_top, strip_right, value), left, config)
            ]
            tolerance = max(3, round((bottom - face_top) * config.alignment_ratio))
            if not accepted_bottoms or max(accepted_bottoms) - min(accepted_bottoms) > tolerance:
                return None
            bottom = min(accepted_bottoms)
        search_radius = max(8, round((bottom - face_top) * config.corner_depth_ratio))
        top_proposals = {face_top, seed_top}
        for start, _stop in edge_rows:
            top_proposals.update(range(max(face_top, start - search_radius), start + 1))
        accepted = []
        for top in sorted(top_proposals):
            if top <= 0 or bottom - top < max(24, round(height * 0.25)):
                continue
            if not _short_hand_corner(difference, (left, top, strip_right, bottom), left, config):
                continue
            top_bounds = (left, height - bottom, strip_right, height - top)
            if not _short_hand_corner(flipped, top_bounds, left, config):
                continue
            accepted.append(top)
            if top == face_top:
                break  # A proven first visible top outranks interior text.
        tolerance = max(3, round((bottom - face_top) * config.alignment_ratio))
        if not accepted:
            # The rounded rim can precede the first row with majority paper.
            # Only complete a failed candidate set within the existing endpoint
            # tolerance; both closures must still belong to this same slot x.
            for top in range(max(1, face_top - tolerance), face_top):
                if bottom - top < max(24, round(height * 0.25)):
                    continue
                if not _short_hand_corner(difference, (left, top, strip_right, bottom), left, config):
                    continue
                top_bounds = (left, height - bottom, strip_right, height - top)
                if _short_hand_corner(flipped, top_bounds, left, config):
                    accepted.append(top)
        if not accepted or max(accepted) - min(accepted) > tolerance:
            return None
        top = face_top if face_top in accepted else min(accepted)
        extents.append(_HandSlotExtent(left, top, bottom))
    return tuple(extents)


def _anchor_hand_slots(
    image: Image.Image,
    mask: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    count: int | None,
    *,
    horizontal_difference: np.ndarray | None = None,
    config: ShortHandRecoveryConfig | None = None,
    grayscale: np.ndarray | None = None,
    input_crop_ratios: tuple[float, float] = (0.535, 0.60),
) -> HandGeometry:
    """Validate vertical changes using the supplied crop's actual size contract.

    Primary boxes use the legacy ratios; recovered boxes explicitly carry
    their configured ratios. The recovery config alone cannot identify which
    source generated a particular input box.
    """
    if config is not None and not isinstance(config, ShortHandRecoveryConfig):
        raise ValueError("config must be a ShortHandRecoveryConfig")
    config = config or ShortHandRecoveryConfig()
    unknown = HandGeometry(None, boxes, False,
                           reason="hand_slot_extent_unverified:candidate_count_unconfirmed")
    if not boxes or count != len(boxes):
        return unknown
    if horizontal_difference is None:
        grayscale, horizontal_difference = _hand_edge_arrays(image)
    elif grayscale is None:
        grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    proposed = _propose_hand_slot_anchors(image, mask, boxes, count,
                                        horizontal_difference=horizontal_difference,
                                        input_crop_ratios=input_crop_ratios)
    if proposed.count == count and proposed.boxes == boxes:
        if _hand_seed_extents_agree(mask, boxes, config, input_crop_ratios[0]):
            return proposed
    if proposed.clipped_slots and proposed.count == count and len(proposed.boxes) == len(boxes):
        return proposed  # Preserve physical count, never invent clipped glyphs.
    if (proposed.clipped_slots and len(proposed.boxes) < len(boxes)
            and _regularize_overlapping_hand_boxes(proposed.boxes) == proposed.boxes):
        def corner_supported(left: int, top: int, right: int, bottom: int,
                             require_top: bool) -> bool:
            assert horizontal_difference is not None
            if not _short_hand_corner(horizontal_difference, (left, top, right, bottom), left, config):
                return False
            return not require_top or _short_hand_corner(
                horizontal_difference[::-1],
                (left, image.height - bottom, right, image.height - top), left, config,
            )

        assert grayscale is not None
        if clipped_trailing_edges_supported(
            grayscale, mask, candidate_boxes=boxes, retained_boxes=proposed.boxes,
            clipped_slots=proposed.clipped_slots, config=config, corner_supported=corner_supported,
        ):
            return proposed  # Only independently proved trailing right contours are removed.
    extents = _certify_hand_slot_extents(
        mask, boxes, horizontal_difference, config, grayscale=grayscale,
    )
    if extents is None:
        return unknown
    heights = [item.bottom - item.top for item in extents]
    body_height = int(np.median(heights))
    tolerance = max(4, round(body_height * config.alignment_ratio))
    if max(heights) - min(heights) > tolerance:
        return replace(unknown, reason="hand_slot_extent_mismatch:candidate_count_unconfirmed")
    # Alignment between cards says nothing about whether the incoming seed is
    # at the correct height. Preserve it only when *each* certified extent
    # also agrees; otherwise rebuild from the independently observed endpoints.
    if all(
        abs(box[1] - extent.top) <= tolerance
        and abs((box[3] - box[1]) / input_crop_ratios[0] - (extent.bottom - extent.top)) <= tolerance
        for box, extent in zip(boxes, extents, strict=True)
    ):
        return HandGeometry(count, boxes, True)
    if proposed.usable and proposed.count == count and len(proposed.boxes) == len(extents) and all(
        box[0] == extent.left and abs(box[1] - extent.top) <= tolerance
        and abs((box[3] - box[1]) / input_crop_ratios[0] - (extent.bottom - extent.top)) <= tolerance
        for box, extent in zip(proposed.boxes, extents, strict=True)
    ):
        return proposed
    crop_height = max(24, round(body_height * input_crop_ratios[0]))
    crop_width = max(16, round(crop_height * input_crop_ratios[1]))
    adjusted = tuple((item.left, item.top, min(image.width, item.left + crop_width),
                      min(image.height, item.top + crop_height)) for item in extents)
    return HandGeometry(count, adjusted, True)


def _propose_hand_slot_anchors(
    image: Image.Image,
    mask: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    count: int | None,
    *,
    horizontal_difference: np.ndarray | None = None,
    input_crop_ratios: tuple[float, float] = (0.535, 0.60),
) -> HandGeometry:
    """Propose legacy anchors; changed extents/deletions need independent proof."""

    height, width = mask.shape
    local_tops: list[int] = []
    local_bottoms: list[int] = []
    edge_heights: list[int] = []
    clipped: list[int] = []
    if horizontal_difference is None:
        _, horizontal_difference = _hand_edge_arrays(image)
    elevation = max(8, round(height * 0.04))
    for index, (left, top, right, _bottom) in enumerate(boxes):
        strip_right = boxes[index + 1][0] if index + 1 < len(boxes) else right
        margin = max(3, round((strip_right - left) * 0.10))
        sample_left = min(strip_right - 2, left + margin)
        sample_right = max(sample_left + 1, strip_right - margin)
        density = mask[:, sample_left:sample_right].mean(axis=1)
        rows = _boolean_runs((density >= 0.50).tolist(), max_gap=2)
        sustained = [run for run in rows if run[1] - run[0] >= max(6, round(height * 0.02))]
        local_top = min((run[0] for run in sustained), default=top)
        local_bottom = max((run[1] for run in sustained), default=_bottom)
        edge_left, edge_right = max(0, left - 4), min(width - 1, left + 3)
        edge_rows = _boolean_runs(
            (horizontal_difference[:, edge_left:edge_right].max(axis=1) >= 20).tolist(),
            max_gap=2,
        ) if edge_right > edge_left else []
        edge_rows = [run for run in edge_rows if run[1] - run[0] >= max(18, round(height * 0.25))]
        edge_top, edge_bottom = max(edge_rows, key=lambda run: run[1] - run[0]) if edge_rows else (local_top, local_bottom)
        edge_heights.append(edge_bottom - edge_top)
        # A raised neighboring card may remain visible behind this slot.
        # Its face is white too, but this card's own vertical border starts
        # at the lower top; do not anchor to the neighbor's exposed pixels.
        if abs(edge_top - local_top) >= elevation:
            local_top, local_bottom = edge_top, edge_bottom
            aligned_rows = [
                run for run in _boolean_runs((density >= 0.50).tolist(), max_gap=0)
                if edge_top - 2 <= run[0] <= edge_top + max(3, round(height * 0.02))
                and run[1] - run[0] >= max(6, round(height * 0.02))
            ]
            if aligned_rows:
                local_top, local_bottom = min(aligned_rows, key=lambda run: abs(run[0] - edge_top))
        local_tops.append(local_top)
        local_bottoms.append(local_bottom)
        # A rank can break the initial face run after just 5 rows. Checking
        # only long runs missed the first clipped frame of the selection.
        if len(density) >= 3 and bool(np.all(density[:3] >= 0.50)):
            clipped.append(index)
    raised = max(local_tops) - min(local_tops) >= elevation
    if raised or clipped:
        minimum_edge_height = round((boxes[0][3] - boxes[0][1]) / input_crop_ratios[0] * 0.60)
        kept = [index for index, span in enumerate(edge_heights) if span >= minimum_edge_height]
        # Short exposed borders below a raised card belong to the card
        # behind it, not another horizontal slot. No missing slots are added.
        if kept and len(kept) < len(boxes):
            boxes = tuple(boxes[index] for index in kept)
            count = len(boxes)
            local_tops = [local_tops[index] for index in kept]
            local_bottoms = [local_bottoms[index] for index in kept]
            clipped = [index for index, old_index in enumerate(kept) if old_index in clipped]
    adjusted = list(boxes)
    if raised and not clipped:
        # The lowest top identifies an unraised card. Use its own body, not
        # the union from the highest raised top to the lowest card bottom.
        unraised_top = max(local_tops)
        unraised = [
            index for index, top in enumerate(local_tops) if unraised_top - top < elevation
        ]
        body_height = int(np.median([local_bottoms[i] - local_tops[i] for i in unraised]))
        crop_height = max(24, round(body_height * input_crop_ratios[0]))
        crop_width = max(16, round(crop_height * input_crop_ratios[1]))
        adjusted = [
            (left, local_tops[index], min(width, left + crop_width),
             min(height, local_tops[index] + crop_height))
            for index, (left, _top, _right, _bottom) in enumerate(boxes)
        ]
    return HandGeometry(
        count, tuple(adjusted), not clipped, tuple(clipped),
        "rank_top_outside_input_roi" if clipped else "",
    )


def _infer_white_hand_count(
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
    row_runs = [run for run in row_runs if run[1] - run[0] >= max(18, round(height * 0.25))]
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
        - grayscale[top:bottom, edge_start - 1 : edge_stop - 1]
    )
    mean_scores = differences.mean(axis=0)
    edge_scores = [(edge_start + index, float(score)) for index, score in enumerate(mean_scores)]
    if not edge_scores:
        return ()
    maximum_edge = max(score for _, score in edge_scores)
    threshold = max(14.0, maximum_edge * 0.34)
    edge_candidates = [x for x, score in edge_scores if score >= threshold]
    edge_groups = _integer_groups(edge_candidates, max_gap=2)
    score_by_x = dict(edge_scores)
    edges = [max(group, key=lambda x: score_by_x[x]) for group in edge_groups]
    internal_edges = [edge for edge in edges if edge > left + 3 and edge < right - 3]
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


def _append_vertically_stacked_card_boxes(
    image: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Recover cards wrapped below the last horizontal table-card slot.

    The macOS client keeps most of a long combination on one horizontal row,
    then renders an additional same-rank card directly below the final slot.
    Both cards therefore share the same x start and the normal x-edge
    segmenter sees only one card.  The wrapped card still has a strong,
    full-width horizontal top border; rank/suit strokes do not span this
    conservative right-side band.
    """

    if not boxes:
        return ()
    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    image_height, image_width = grayscale.shape
    recovered: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom in sorted(boxes, key=lambda box: (box[0], box[1])):
        recovered.append((left, top, right, bottom))
        rank_width = right - left
        rank_height = bottom - top
        if rank_width < 12 or rank_height < 24:
            continue
        band_left = min(right - 2, left + round(rank_width * 0.35))
        band_right = min(image_width, right - 2)
        search_top = max(1, top + round(rank_height * 0.45))
        # The physical card bottom is around 1.45 rank-crop heights below the
        # top and is also a full-width edge. Stop before it so it cannot be
        # mistaken for the top of a wrapped card.
        search_bottom = min(
            image_height - 1,
            top + round(rank_height * 1.25),
        )
        if band_right - band_left < 8 or search_bottom <= search_top:
            continue
        differences = np.abs(
            grayscale[search_top:search_bottom, band_left:band_right]
            - grayscale[search_top - 1 : search_bottom - 1, band_left:band_right]
        )
        ratios = (differences >= 25).mean(axis=1)
        means = differences.mean(axis=1)
        candidates = [
            search_top + index
            for index, (ratio, mean) in enumerate(zip(ratios, means, strict=True))
            if float(ratio) >= 0.82 and float(mean) >= 28.0
        ]
        for group in _integer_groups(candidates, max_gap=2):
            stacked_top = max(
                group,
                key=lambda y: (
                    float(ratios[y - search_top]),
                    float(means[y - search_top]),
                ),
            )
            stacked_box = (
                left,
                stacked_top,
                right,
                min(image_height, stacked_top + rank_height),
            )
            if stacked_box[3] - stacked_box[1] >= 24:
                recovered.append(stacked_box)
            # The current client wraps at most one extra card per x slot.
            break
    return tuple(sorted(recovered, key=lambda box: (box[0], box[1])))


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

    regularized = tuple(value[1] for _, value in sorted(best_by_slot.items()))
    return regularized or ordered


def _infer_white_hand_boxes(
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
        [int(row_count) >= max(8, round((right - left) * 0.25)) for row_count in row_counts],
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
            if run[0] <= round(height * 0.25) and run[1] - run[0] >= max(12, round(height * 0.12))
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
    detected_starts = [box[0] for box in detected] if len(detected) == count else []
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
    joker_color: JokerColorEvidence | None = None,
) -> tuple[str, float]:
    rank, confidence, _ = _fuse_card_prediction(
        prediction, reference_match,
        confident_joker_threshold=confident_joker_threshold,
        joker_color=joker_color,
    )
    return rank, confidence


def _fuse_card_prediction(
    prediction: CardPrediction,
    reference_match: tuple[str, float] | None,
    *,
    confident_joker_threshold: float = 0.90,
    joker_color: JokerColorEvidence | None = None,
) -> tuple[str, float, CardFusionEvidence]:
    """Preserve shape corrections; color only guards an output that remains Joker."""
    reference_rank = reference_match[0] if reference_match is not None else None
    reference_confidence = reference_match[1] if reference_match is not None else None

    def result(rank: str, confidence: float, reason: str) -> tuple[str, float, CardFusionEvidence]:
        if not 0.0 <= confidence <= 1.0:
            confidence = 0.0
            reason = f"{reason};confidence_invalid"
        return rank, confidence, CardFusionEvidence(
            raw_rank=prediction.rank,
            raw_confidence=prediction.confidence,
            reference_rank=reference_rank,
            reference_confidence=reference_confidence,
            fusion_reason=reason,
            joker_color=joker_color.color if joker_color is not None else None,
            joker_color_reason=joker_color.reason if joker_color is not None else None,
        )

    if prediction.rank == "J" and reference_rank in {"SJ", "BJ"}:
        return result(prediction.rank, prediction.confidence, "normal_j_preserved")
    if prediction.rank not in {"SJ", "BJ"}:
        if reference_match is None:
            return result(prediction.rank, prediction.confidence, "raw_prediction")
        if reference_rank in {"SJ", "BJ"}:
            # Color cannot prove Joker identity for an ordinary rank either:
            # e.g. a black K must not become SJ merely through a gray template.
            return result(prediction.rank, 0.0, "joker_reference_requires_raw_identity")
        return result(*reference_match, "reference_rank")

    protected_j = (
        reference_rank == "J"
        and prediction.confidence >= confident_joker_threshold
    )
    if reference_match is not None and reference_rank not in {"SJ", "BJ"} and not protected_j:
        # A strong A (including black A) is a shape correction, not evidence for
        # a small Joker. Do not let the Joker color gate suppress that path.
        return result(*reference_match, "non_joker_reference")

    if joker_color is None or joker_color.color == "unknown":
        return result(prediction.rank, 0.0, "joker_color_unavailable")
    expected_color = "red" if prediction.rank == "BJ" else "black"
    if joker_color.color != expected_color:
        # Color is a conflict gate, never a way to flip SJ/BJ or remove a slot.
        return result(prediction.rank, 0.0, "joker_color_conflict")
    if not 0.0 <= prediction.confidence <= 1.0:
        return result(prediction.rank, 0.0, "joker_raw_confidence_invalid")
    if reference_rank in {"SJ", "BJ"} and reference_rank != prediction.rank:
        reason = "joker_reference_conflict_raw_preserved"
    elif protected_j:
        reason = "confident_joker_preserved"
    else:
        reason = "joker_color_consistent_raw_preserved"
    return result(prediction.rank, prediction.confidence, reason)


__all__ = [
    "CvRemainingReader",
    "HandGeometry",
    "SceneObservation",
    "SceneRecognizer",
    "MacVisionRemainingReader",
    "RoundResultObservation",
    "RoundResultOutcome",
    "RemainingTextMatch",
    "SeatObservation",
    "SeatRole",
    "TemplateMatch",
    "TemplateMatcher",
    "VisualCard",
    "VisualSignal",
    "infer_hand_geometry",
    "infer_visible_hand_count",
    "infer_overlapping_hand_boxes",
    "segment_card_boxes",
    "segment_card_regions",
]
