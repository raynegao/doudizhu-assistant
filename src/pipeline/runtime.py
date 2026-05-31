from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image
import torch

from src.logic.decision import recommend_action
from src.state.cards import CardParseError, CardSet
from src.state.game_state import GameStateSnapshot
from src.vision.card_classifier import (
    CardPrediction,
    load_checkpoint,
    predict_tensors,
    preprocess_image,
    select_device,
)


DEFAULT_ROI_BOX = (380, 1110, 2555, 1515)
DEFAULT_CARD_COUNT = 15
DEFAULT_START_X = 0
DEFAULT_START_Y = 20
DEFAULT_STEP_X = 135
DEFAULT_CROP_SIZE = (126, 210)
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


class RuntimeCaptureError(RuntimeError):
    """Raised when the configured frame source cannot capture a frame."""


@dataclass(frozen=True)
class ScreenFrame:
    frame_id: int
    timestamp: float
    image: Image.Image
    roi_box: tuple[int, int, int, int]
    source: str


@dataclass(frozen=True)
class CardObservation:
    index: int
    rank: str
    confidence: float
    box: tuple[int, int, int, int]

    def to_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "rank": self.rank,
            "confidence": self.confidence,
            "box": list(self.box),
        }


@dataclass(frozen=True)
class RuntimeDecisionEvent:
    frame_id: int
    timestamp: float
    source: str
    roi_box: tuple[int, int, int, int]
    recognized_cards: tuple[str, ...]
    observations: tuple[CardObservation, ...]
    last_play: tuple[str, ...]
    candidate_count: int
    recommended_action: tuple[str, ...]
    reason: str
    warnings: tuple[str, ...]
    latency_ms: float
    error: str | None = None

    def to_log_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": "phase3_recommendation",
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "roi_box": list(self.roi_box),
            "recognized_cards": list(self.recognized_cards),
            "observations": [observation.to_payload() for observation in self.observations],
            "last_play": list(self.last_play),
            "candidate_count": self.candidate_count,
            "recommended_action": list(self.recommended_action),
            "reason": self.reason,
            "warnings": list(self.warnings),
            "latency_ms": self.latency_ms,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class RuntimeSettings:
    model_path: Path = Path("models/card_cnn.pt")
    device_name: str = "auto"
    roi_box: tuple[int, int, int, int] = DEFAULT_ROI_BOX
    count: int = DEFAULT_CARD_COUNT
    start_x: int = DEFAULT_START_X
    start_y: int = DEFAULT_START_Y
    step_x: int = DEFAULT_STEP_X
    crop_size: tuple[int, int] = DEFAULT_CROP_SIZE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    last_play: str = ""
    log_file: Path | None = Path("logs/phase3_runtime.jsonl")
    source: str = "mac_screen"


class FrameSource(Protocol):
    def capture(self, frame_id: int) -> ScreenFrame:
        ...


PredictionFn = Callable[[Sequence[Image.Image]], Sequence[CardPrediction]]


class MacScreenFrameSource:
    def __init__(
        self,
        roi_box: tuple[int, int, int, int],
        source: str = "mac_screen",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._roi_box = roi_box
        self._source = source
        self._clock = clock

    def capture(self, frame_id: int) -> ScreenFrame:
        try:
            from PIL import ImageGrab

            image = ImageGrab.grab(bbox=self._roi_box).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeCaptureError(
                "Mac screen capture failed. On macOS, grant Screen Recording permission "
                "to the terminal or Codex app, and verify that --roi-box intersects the active display. "
                f"Original error: {exc}"
            ) from exc

        if image.width <= 0 or image.height <= 0:
            raise RuntimeCaptureError("Mac screen capture returned an empty ROI image.")

        return ScreenFrame(
            frame_id=frame_id,
            timestamp=self._clock(),
            image=image,
            roi_box=self._roi_box,
            source=self._source,
        )


class Phase3Runtime:
    def __init__(
        self,
        settings: RuntimeSettings,
        frame_source: FrameSource | None = None,
        predictor: PredictionFn | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.frame_source = frame_source or MacScreenFrameSource(settings.roi_box, source=settings.source)
        self._sleeper = sleeper
        self._next_frame_id = 1
        if predictor is None:
            device = select_device(settings.device_name)
            model, classes, image_size = load_checkpoint(settings.model_path, device=device)
            self._predictor = _build_model_predictor(model, classes, image_size, device)
        else:
            self._predictor = predictor

    def run_once(self, frame_id: int | None = None) -> RuntimeDecisionEvent:
        current_frame_id = self._allocate_frame_id(frame_id)
        started_at = time.perf_counter()
        frame = self.frame_source.capture(current_frame_id)
        observations = self._observe_frame(frame)
        event = self._build_decision_event(
            frame=frame,
            observations=observations,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )
        if self.settings.log_file is not None:
            write_runtime_event(self.settings.log_file, event)
        return event

    def run_loop(self, max_frames: int | None = None, interval: float = 1.0) -> Iterator[RuntimeDecisionEvent]:
        produced = 0
        while max_frames is None or produced < max_frames:
            yield self.run_once()
            produced += 1
            if max_frames is None or produced < max_frames:
                self._sleeper(max(0.0, interval))

    def _allocate_frame_id(self, frame_id: int | None) -> int:
        if frame_id is not None:
            self._next_frame_id = max(self._next_frame_id, frame_id + 1)
            return frame_id
        current = self._next_frame_id
        self._next_frame_id += 1
        return current

    def _observe_frame(self, frame: ScreenFrame) -> tuple[CardObservation, ...]:
        crop_boxes = compute_crop_boxes(
            count=self.settings.count,
            start_x=self.settings.start_x,
            start_y=self.settings.start_y,
            step_x=self.settings.step_x,
            crop_size=self.settings.crop_size,
            image_size=frame.image.size,
        )
        crops = [frame.image.crop(box) for box in crop_boxes]
        predictions = self._predictor(crops)
        if len(predictions) != len(crop_boxes):
            raise ValueError(f"Predictor returned {len(predictions)} results for {len(crop_boxes)} crops.")
        return tuple(
            CardObservation(
                index=index,
                rank=prediction.rank,
                confidence=prediction.confidence,
                box=crop_boxes[index],
            )
            for index, prediction in enumerate(predictions)
        )

    def _build_decision_event(
        self,
        frame: ScreenFrame,
        observations: tuple[CardObservation, ...],
        latency_ms: float,
    ) -> RuntimeDecisionEvent:
        recognized_cards = tuple(observation.rank for observation in observations)
        warnings = [
            f"low-confidence card {observation.index:02d}: {observation.rank}={observation.confidence:.3f}"
            for observation in observations
            if observation.confidence < self.settings.confidence_threshold
        ]
        hand_text = " ".join(recognized_cards)
        try:
            state = GameStateSnapshot.from_inputs(hand_text, self.settings.last_play)
            decision = recommend_action(state)
            candidate_count = len([candidate for candidate in decision.candidates if not candidate.is_pass])
            last_play = state.last_play.to_list()
            recommended_action = tuple(decision.action.to_list())
            reason = decision.reason
            error = None
        except (CardParseError, ValueError) as exc:
            last_play = _parse_last_play_safely(self.settings.last_play)
            candidate_count = 0
            recommended_action = ()
            reason = f"无法生成推荐：{exc}"
            warnings.append(reason)
            error = str(exc)

        return RuntimeDecisionEvent(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            source=frame.source,
            roi_box=frame.roi_box,
            recognized_cards=recognized_cards,
            observations=observations,
            last_play=tuple(last_play),
            candidate_count=candidate_count,
            recommended_action=recommended_action,
            reason=reason,
            warnings=tuple(warnings),
            latency_ms=latency_ms,
            error=error,
        )


def compute_crop_boxes(
    count: int,
    start_x: int,
    start_y: int,
    step_x: int,
    crop_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if step_x <= 0:
        raise ValueError("step-x must be positive")
    crop_width, crop_height = crop_size
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop-size must be positive")

    boxes: list[tuple[int, int, int, int]] = []
    image_width, image_height = image_size
    for index in range(count):
        left = start_x + index * step_x
        upper = start_y
        right = left + crop_width
        lower = upper + crop_height
        if left < 0 or upper < 0 or right > image_width or lower > image_height:
            raise ValueError(
                f"Crop {index} exceeds ROI bounds: "
                f"box={(left, upper, right, lower)}, roi_size={image_size}"
            )
        boxes.append((left, upper, right, lower))
    return tuple(boxes)


def write_runtime_event(path: Path, event: RuntimeDecisionEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event.to_log_payload(), ensure_ascii=False) + "\n")


def format_runtime_event(event: RuntimeDecisionEvent) -> str:
    lines = [
        "Dou Dizhu Phase 3 Runtime",
        f"frame={event.frame_id} source={event.source} latency={event.latency_ms:.1f}ms",
        f"roi_box={event.roi_box}",
        f"识别手牌: {' '.join(event.recognized_cards) if event.recognized_cards else '(none)'}",
        "单牌置信度:",
    ]
    for observation in event.observations:
        lines.append(f"  {observation.index:02d} {observation.rank:>2} {observation.confidence:.3f}")
    if event.warnings:
        lines.append("WARNING:")
        lines.extend(f"  {warning}" for warning in event.warnings)
    lines.extend([
        f"上一手牌: {' '.join(event.last_play) if event.last_play else 'pass'}",
        f"候选动作数: {event.candidate_count}",
        f"推荐动作: {' '.join(event.recommended_action) if event.recommended_action else 'pass'}",
        f"推荐理由: {event.reason}",
    ])
    return "\n".join(lines)


def _build_model_predictor(
    model,
    classes: tuple[str, ...],
    image_size: tuple[int, int],
    device: torch.device,
) -> PredictionFn:
    def predict(crops: Sequence[Image.Image]) -> Sequence[CardPrediction]:
        if not crops:
            return []
        tensors = torch.stack([preprocess_image(crop, image_size=image_size) for crop in crops])
        return predict_tensors(model, tensors, classes=classes, device=device)

    return predict


def _parse_last_play_safely(last_play: str) -> list[str]:
    try:
        return CardSet.parse(last_play).to_list()
    except Exception:  # noqa: BLE001
        return [part for part in last_play.split() if part]
