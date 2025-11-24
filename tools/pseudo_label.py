"""
Pseudo-label screenshots using a trained YOLO model, producing YOLO txt labels.

Usage:
    python tools/pseudo_label.py --weights runs/train/doudizhu/weights/best.pt --images data/raw_screenshots --out data/labels_yolo_pseudo --conf 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_logic.cards import RANK_ORDER


def load_model(weights: Path) -> YOLO:
    return YOLO(str(weights))


def predict_image(model: YOLO, image_path: Path, conf: float) -> List[str]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    results = model.predict(img, conf=conf, verbose=False)
    if not results:
        return []
    res = results[0]
    lines: List[str] = []
    h, w = img.shape[:2]
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id < 0 or cls_id >= len(RANK_ORDER):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w_box = x2 - x1
        h_box = y2 - y1
        x_center = x1 + w_box / 2
        y_center = y1 + h_box / 2
        lines.append(
            f"{cls_id} {x_center / w:.6f} {y_center / h:.6f} {w_box / w:.6f} {h_box / h:.6f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo labels using a trained YOLO model.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained YOLO weights (e.g., best.pt).")
    parser.add_argument("--images", type=Path, default=Path("data/raw_screenshots"), help="Directory of images to label.")
    parser.add_argument("--out", type=Path, default=Path("data/labels_yolo_pseudo"), help="Output directory for labels.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    args = parser.parse_args()

    model = load_model(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(args.images.glob("*.png"))
    if not imgs:
        print(f"No images found in {args.images}")
        return

    for idx, img_path in enumerate(imgs):
        lines = predict_image(model, img_path, args.conf)
        out_path = args.out / f"{img_path.stem}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        if idx % 20 == 0:
            print(f"[{idx}/{len(imgs)}] pseudo-labeled {img_path.name}, {len(lines)} boxes")


if __name__ == "__main__":
    main()
