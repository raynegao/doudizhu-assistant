from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace

from src.pipeline.runtime import CardObservation


@dataclass(frozen=True)
class StabilizedObservations:
    raw: tuple[CardObservation, ...]
    stable: tuple[CardObservation, ...]
    stabilized: bool
    window_size: int


class ObservationStabilizer:
    def __init__(self, window_size: int = 3) -> None:
        if window_size <= 0:
            raise ValueError("stability-window must be positive")
        self.window_size = window_size
        self._history: deque[tuple[CardObservation, ...]] = deque(maxlen=window_size)

    def update(self, observations: tuple[CardObservation, ...]) -> StabilizedObservations:
        self._history.append(observations)
        if self.window_size == 1 or len(self._history) == 1:
            return StabilizedObservations(
                raw=observations,
                stable=observations,
                stabilized=False,
                window_size=self.window_size,
            )
        stable = tuple(self._stabilize_index(index, observation) for index, observation in enumerate(observations))
        return StabilizedObservations(
            raw=observations,
            stable=stable,
            stabilized=stable != observations,
            window_size=self.window_size,
        )

    def _stabilize_index(self, index: int, fallback: CardObservation) -> CardObservation:
        samples = [frame[index] for frame in self._history if index < len(frame)]
        if not samples:
            return fallback
        counts = Counter(sample.rank for sample in samples)
        average_confidence: dict[str, float] = defaultdict(float)
        for rank in counts:
            rank_samples = [sample for sample in samples if sample.rank == rank]
            average_confidence[rank] = sum(sample.confidence for sample in rank_samples) / len(rank_samples)
        best_rank = sorted(
            counts,
            key=lambda rank: (counts[rank], average_confidence[rank]),
            reverse=True,
        )[0]
        best_samples = [sample for sample in samples if sample.rank == best_rank]
        confidence = average_confidence[best_rank]
        return replace(best_samples[-1], confidence=confidence)
