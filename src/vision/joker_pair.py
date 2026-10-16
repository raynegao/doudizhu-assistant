"""Deck-constrained identity for one color-conflicting Joker beside a confirmed partner.

Per-card fusion keeps a raw Joker whose measured rank ink contradicts it as a
zero-confidence slot; colour alone must never flip SJ/BJ. Two overlapping
Jokers on the table can still contaminate the upper corner crop with the lower
card's ink, so both corners come back as the same Joker. A deck holds exactly
one SJ and one BJ: next to a confirmed partner the other Joker-shaped slot can
only be the complement, and its own strict ink reading has to agree with that
complement. Nothing here promotes an ordinary rank or touches the hand path's
reference learning, which still requires raw and final identity to match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from src.vision.rank_glyph import JokerColorEvidence
from src.vision.scene_types import VisualCard

COMPLEMENT_REASON = "joker_pair_complement"
_COMPLEMENT = {"SJ": "BJ", "BJ": "SJ"}
_EXPECTED_INK = {"SJ": "black", "BJ": "red"}
_CONFIRMED_REASONS = frozenset({
    "joker_color_consistent_raw_preserved",
    "confident_joker_preserved",
})
_STRICT_INK_REASONS = frozenset({
    "red_rank_ink",
    "black_rank_ink",
    "black_rank_ink_antialiased_edge",
})


def resolve_joker_pair(
    cards: Sequence[VisualCard],
    colors: Sequence[JokerColorEvidence | None],
    *,
    confident_joker_threshold: float = 0.90,
) -> tuple[VisualCard, ...]:
    """Return ``cards`` with at most one conflict slot renamed to its partner's complement.

    ``colors`` holds the ink reading used for each card's fusion (``None`` for
    non-Joker predictions). The input is returned unchanged unless exactly two
    slots carry a raw Joker prediction, one of them is a confirmed Joker and the
    other is a ``joker_color_conflict`` slot whose strict ink colour is the
    complement's. The assigned confidence is bounded by both the partner's
    confidence and the measured fraction of the expected ink.
    """
    if len(cards) != len(colors):
        raise ValueError("joker pair resolution needs one color reading per card")
    joker_slots = [
        index for index, card in enumerate(cards)
        if card.evidence is not None and card.evidence.raw_rank in _COMPLEMENT
    ]
    if len(joker_slots) != 2:
        return tuple(cards)
    first, second = joker_slots
    for partner_index, conflict_index in ((first, second), (second, first)):
        resolved = _complement(
            cards[partner_index], cards[conflict_index], colors[conflict_index],
            confident_joker_threshold,
        )
        if resolved is not None:
            output = list(cards)
            output[conflict_index] = resolved
            return tuple(output)
    return tuple(cards)


def _complement(
    partner: VisualCard,
    conflict: VisualCard,
    ink: JokerColorEvidence | None,
    threshold: float,
) -> VisualCard | None:
    partner_evidence, conflict_evidence = partner.evidence, conflict.evidence
    if partner_evidence is None or conflict_evidence is None or ink is None:
        return None
    if (
        partner.rank not in _COMPLEMENT
        or partner.rank != partner_evidence.raw_rank
        or partner_evidence.fusion_reason not in _CONFIRMED_REASONS
        or partner_evidence.joker_color != _EXPECTED_INK[partner.rank]
        or not threshold <= partner.confidence <= 1.0
    ):
        return None
    complement = _COMPLEMENT[partner.rank]
    expected_ink = _EXPECTED_INK[complement]
    if (
        conflict_evidence.fusion_reason != "joker_color_conflict"
        or conflict.confidence != 0.0
        or ink.color != expected_ink
        or ink.reason not in _STRICT_INK_REASONS
        or conflict_evidence.joker_color != expected_ink
    ):
        return None
    fraction = ink.black_fraction if expected_ink == "black" else ink.red_fraction
    if not 0.0 < fraction <= 1.0:
        return None
    return replace(
        conflict,
        rank=complement,
        confidence=min(partner.confidence, fraction),
        evidence=replace(conflict_evidence, fusion_reason=COMPLEMENT_REASON),
    )


__all__ = ["COMPLEMENT_REASON", "resolve_joker_pair"]
