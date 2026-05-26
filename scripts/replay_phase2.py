from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image

from scripts.crop_hand_roi_cards import crop_hand_roi_cards, parse_crop_size
from scripts.predict_card_crops import predict_crop_dir
from src.logic.decision import recommend_action
from src.logic.rules import Play, legal_actions
from src.state.game_state import GameStateSnapshot


WINDOW_MODE_ROI_BOX = (380, 1110, 2555, 1515)
WINDOW_MODE_CARD_COUNT = 15
WINDOW_MODE_START_X = 0
WINDOW_MODE_START_Y = 20
WINDOW_MODE_STEP_X = 135
WINDOW_MODE_CROP_SIZE = (126, 210)


def crop_roi_from_screenshot(screenshot: Path, output: Path, box: tuple[int, int, int, int]) -> Path:
    image = Image.open(screenshot).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(output)
    return output


def replay_phase2(
    model_path: Path,
    roi_path: Path | None,
    screenshot_path: Path | None,
    crop_dir: Path | None,
    last_play: str,
    device_name: str,
    roi_box: tuple[int, int, int, int],
    count: int,
    start_x: int,
    start_y: int,
    step_x: int,
    crop_size: tuple[int, int],
    confidence_threshold: float,
) -> int:
    with tempfile.TemporaryDirectory(prefix="doudizhu_phase2_") as temp:
        temp_dir = Path(temp)
        working_crop_dir = crop_dir or temp_dir / "crops"
        working_roi = roi_path
        if screenshot_path is not None:
            working_roi = crop_roi_from_screenshot(
                screenshot_path,
                temp_dir / "window_mode_hand_roi.png",
                roi_box,
            )
        if working_roi is not None and crop_dir is None:
            crop_hand_roi_cards(
                roi_path=working_roi,
                output_dir=working_crop_dir,
                count=count,
                start_x=start_x,
                start_y=start_y,
                step_x=step_x,
                crop_size=crop_size,
            )
        if not working_crop_dir.exists():
            raise FileNotFoundError("Provide --crop-dir, --roi, or --screenshot.")

        predictions = predict_crop_dir(model_path, working_crop_dir, device_name=device_name)
        ranks = [str(prediction["rank"]) for prediction in predictions]
        hand_text = " ".join(ranks)
        state = GameStateSnapshot.from_inputs(hand_text, last_play)
        previous = Play.parse(state.last_play.cards)
        actions = legal_actions(state.hand, previous)
        decision = recommend_action(state)

        print(f"识别手牌: {hand_text}")
        print("单牌置信度:")
        low_confidence: list[dict[str, object]] = []
        for prediction in predictions:
            print(f"  {prediction['index']:02d} {prediction['rank']:>2} {prediction['confidence']:.3f}")
            if float(prediction["confidence"]) < confidence_threshold:
                low_confidence.append(prediction)
        if low_confidence:
            indexes = ", ".join(f"{item['index']:02d}:{item['rank']}={float(item['confidence']):.3f}" for item in low_confidence)
            print(f"WARNING: low-confidence predictions below {confidence_threshold:.2f}: {indexes}")
        print(f"上一手牌: {state.last_play}")
        print(f"候选动作数: {len([action for action in actions if not action.is_pass])}")
        print(f"推荐动作: {decision.action}")
        print(f"推荐理由: {decision.reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Phase 2 vision output through the Phase 1 rule engine.")
    parser.add_argument("--model", default="models/card_cnn.pt", help="PyTorch checkpoint path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--roi", help="Hand ROI image path.")
    source.add_argument("--screenshot", help="Full screenshot path; uses the current window-mode ROI box.")
    source.add_argument("--crop-dir", help="Directory containing card_*.png crops.")
    parser.add_argument("--last-play", default="", help="Previous play, empty means lead play.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--roi-box", default=",".join(str(value) for value in WINDOW_MODE_ROI_BOX), help="Full screenshot ROI box: left,top,right,bottom.")
    parser.add_argument("--count", type=int, default=WINDOW_MODE_CARD_COUNT, help="Number of hand cards to crop from ROI.")
    parser.add_argument("--start-x", type=int, default=WINDOW_MODE_START_X)
    parser.add_argument("--start-y", type=int, default=WINDOW_MODE_START_Y)
    parser.add_argument("--step-x", type=int, default=WINDOW_MODE_STEP_X)
    parser.add_argument("--crop-size", type=parse_crop_size, default=WINDOW_MODE_CROP_SIZE)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    return parser


def parse_roi_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI box must be left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("ROI box must have right > left and bottom > top")
    return left, top, right, bottom


def main() -> int:
    args = build_parser().parse_args()
    return replay_phase2(
        model_path=Path(args.model),
        roi_path=Path(args.roi) if args.roi else None,
        screenshot_path=Path(args.screenshot) if args.screenshot else None,
        crop_dir=Path(args.crop_dir) if args.crop_dir else None,
        last_play=args.last_play,
        device_name=args.device,
        roi_box=parse_roi_box(args.roi_box),
        count=args.count,
        start_x=args.start_x,
        start_y=args.start_y,
        step_x=args.step_x,
        crop_size=args.crop_size,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
