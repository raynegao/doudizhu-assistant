"""Immutable visual observations, independent of inference frameworks and GUI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src.state.cards import CardSet
from src.state.events import PlayerSeat


class VisualSignal(str, Enum):
    NEUTRAL = "neutral"
    PLAY = "play"
    PASS = "pass"
    UNKNOWN = "unknown"


class SeatRole(str, Enum):
    LANDLORD = "landlord"
    FARMER = "farmer"
    UNKNOWN = "unknown"


class RoundResultOutcome(str, Enum):
    VICTORY = "victory"
    DEFEAT = "defeat"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TemplateMatch:
    label: str | None
    confidence: float
    template: str | None = None


@dataclass(frozen=True)
class CardPhotometryEvidence:
    """Actual input transform, not a second prediction or confidence score."""

    reason: str
    gain: float = 1.0
    background_level: int | None = None
    paper_fraction: float | None = None
    usable: bool = True

    def to_log_payload(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "usable": self.usable,
            "gain": self.gain if math.isfinite(self.gain) else None,
            "background_level": self.background_level,
            "paper_fraction": (
                self.paper_fraction if self.paper_fraction is not None
                and math.isfinite(self.paper_fraction) else None
            ),
        }


@dataclass(frozen=True)
class CardFusionEvidence:
    """Diagnostic model/fusion provenance, never an independent card label.

    Retain finite scores at their original precision so a rejected template
    path can be reproduced. Non-finite model scores are recorded as unknown,
    not emitted as non-standard JSON NaN/Infinity or promoted to confidence.
    """

    raw_rank: str
    raw_confidence: float
    reference_rank: str | None = None
    reference_confidence: float | None = None
    fusion_reason: str = "unspecified"
    joker_color: str | None = None
    joker_color_reason: str | None = None
    input_geometry: str | None = None
    input_box: tuple[int, int, int, int] | None = None
    input_photometry: CardPhotometryEvidence | None = None
    # Set when segmentation geometry alone voids the reading (for example a
    # corner cut by the next card); the rank and scores above stay as read.
    geometry_reason: str | None = None

    def to_log_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "raw_rank": self.raw_rank,
            "raw_confidence": (
                self.raw_confidence if math.isfinite(self.raw_confidence) else None
            ),
            "reference_rank": self.reference_rank,
            "reference_confidence": (
                self.reference_confidence
                if self.reference_confidence is not None
                and math.isfinite(self.reference_confidence)
                else None
            ),
            "fusion_reason": self.fusion_reason,
            "joker_color": self.joker_color,
            "joker_color_reason": self.joker_color_reason,
        }
        if self.input_geometry is not None:
            payload["input_geometry"] = self.input_geometry
            payload["input_box"] = list(self.input_box) if self.input_box is not None else None
        if self.input_photometry is not None:
            payload["input_photometry"] = self.input_photometry.to_log_payload()
        if self.geometry_reason is not None:
            payload["geometry_reason"] = self.geometry_reason
        return payload


@dataclass(frozen=True)
class VisualCard:
    rank: str
    confidence: float
    box: tuple[int, int, int, int]
    evidence: CardFusionEvidence | None = None

    def to_log_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rank": self.rank,
            "confidence": round(self.confidence, 6),
            "box": list(self.box),
        }
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_log_payload()
        return payload


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
class RoundResultObservation:
    outcome: RoundResultOutcome
    confidence: float

    def to_log_payload(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class SelfTableObservation:
    """Dedicated table geometry evidence; not a generic self PLAY signal."""

    cards: tuple[VisualCard, ...] = ()
    complete: bool = False
    reason: str = "unavailable"
    crop_box: tuple[int, int, int, int] | None = None

    def to_log_payload(self) -> dict[str, object]:
        return {"cards": [card.to_log_payload() for card in self.cards],
                "complete": self.complete, "reason": self.reason,
                "crop_box": list(self.crop_box) if self.crop_box is not None else None}


@dataclass(frozen=True)
class HandEmptyEvidence:
    verified: bool = False
    reason: str = "unavailable"
    effective_box: tuple[int, int, int, int] | None = None
    excluded_bottom_px: int = 0
    blue_fraction: float | None = None

    def to_log_payload(self) -> dict[str, object]:
        return {"verified": self.verified, "reason": self.reason,
                "effective_box": list(self.effective_box) if self.effective_box is not None else None,
                "excluded_bottom_px": self.excluded_bottom_px, "blue_fraction": self.blue_fraction}


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
    round_result: RoundResultObservation | None = None
    self_hand_empty_verified: bool = False
    self_table: SelfTableObservation | None = None
    self_hand_empty_evidence: HandEmptyEvidence | None = None

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
            "self_hand_empty_verified": self.self_hand_empty_verified,
            "self_hand_empty_evidence": (
                self.self_hand_empty_evidence.to_log_payload()
                if self.self_hand_empty_evidence is not None else None
            ),
            "self_table": self.self_table.to_log_payload() if self.self_table is not None else None,
            "round_result": (
                self.round_result.to_log_payload() if self.round_result is not None else None
            ),
        }
