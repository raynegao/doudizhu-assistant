"""
Generate synthetic Dou Dizhu screenshots by compositing card templates.

Features:
- Uses `data/pic` card crops as templates (suffix char drives class id via mapping).
- Composites on a plain background image (default white.png) or solid color.
- Random card count per image, random positions/scale/rotation with light noise.
- Outputs images to `data/raw_screenshots_synth/` and YOLO labels to `data/labels_yolo_synth/`.

Usage:
    python tools/synthetic_generator.py --num-images 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_logic.cards import rank_to_id

TEMPLATE_DIR = Path("data/pic")
BG_DEFAULT = TEMPLATE_DIR / "white.png"
OUT_IMG_DIR = Path("data/raw_screenshots_synth")
OUT_LBL_DIR = Path("data/labels_yolo_synth")

RANK_CHAR_TO_RANK: Dict[str, str] = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "0": "10",
    "t": "10",
    "T": "10",
    "J": "J",
    "j": "J",
    "Q": "Q",
    "q": "Q",
    "K": "K",
    "k": "K",
    "A": "A",
    "a": "A",
    "X": "joker_small",
    "x": "joker_small",
    "Y": "joker_big",
    "y": "joker_big",
}


def load_templates() -> List[Tuple[str, np.ndarray]]:
    templates: List[Tuple[str, np.ndarray]] = []
    for png in sorted(TEMPLATE_DIR.glob("*.png")):
        key = png.stem[-1]
        rank = RANK_CHAR_TO_RANK.get(key)
        if rank is None:
            continue
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        templates.append((rank, img))
    if not templates:
        raise RuntimeError("No card templates found in data/pic")
    return templates


def load_background(size: Tuple[int, int]) -> np.ndarray:
    if BG_DEFAULT.exists():
        bg = cv2.imread(str(BG_DEFAULT))
        if bg is not None:
            return cv2.resize(bg, (size[0], size[1]))
    # fallback solid color
    return np.full((size[1], size[0], 3), 220, dtype=np.uint8)


def place_card(
    canvas: np.ndarray, card_img: np.ndarray, scale: float, angle: float, x: int, y: int
) -> Tuple[int, int, int, int]:
    h, w = card_img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(card_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # rotation
    center = (new_w // 2, new_h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(resized, mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    # split channels for alpha
    if rotated.shape[2] == 4:
        bgr = rotated[:, :, :3]
        alpha = rotated[:, :, 3] / 255.0
    else:
        bgr = rotated
        alpha = np.ones((new_h, new_w), dtype=float)

    h_canvas, w_canvas = canvas.shape[:2]
    x1 = max(0, min(x, w_canvas - 1))
    y1 = max(0, min(y, h_canvas - 1))
    x2 = min(x1 + new_w, w_canvas)
    y2 = min(y1 + new_h, h_canvas)

    roi = canvas[y1:y2, x1:x2]
    h_roi, w_roi = roi.shape[:2]
    bgr = bgr[:h_roi, :w_roi]
    alpha = alpha[:h_roi, :w_roi]
    alpha = alpha[..., None]

    blended = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
    canvas[y1:y2, x1:x2] = blended
    return x1, y1, x2, y2


def generate_image(
    templates: List[Tuple[str, np.ndarray]],
    out_idx: int,
    img_size: Tuple[int, int],
    min_cards: int,
    max_cards: int,
) -> Tuple[np.ndarray, List[str]]:
    canvas = load_background(img_size)
    lines: List[str] = []
    card_count = random.randint(min_cards, max_cards)
    for _ in range(card_count):
        rank, tpl = random.choice(templates)
        class_id = rank_to_id(rank)
        scale = random.uniform(0.9, 1.2)
        angle = random.uniform(-8, 8)
        x = random.randint(0, img_size[0] - int(tpl.shape[1] * scale))
        y = random.randint(img_size[1] // 2, img_size[1] - int(tpl.shape[0] * scale))
        x1, y1, x2, y2 = place_card(canvas, tpl, scale, angle, x, y)

        w_box = x2 - x1
        h_box = y2 - y1
        x_center = x1 + w_box / 2
        y_center = y1 + h_box / 2
        lines.append(
            f"{class_id} {x_center / img_size[0]:.6f} {y_center / img_size[1]:.6f} {w_box / img_size[0]:.6f} {h_box / img_size[1]:.6f}"
        )

    return canvas, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Dou Dizhu data.")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--min-cards", type=int, default=8)
    parser.add_argument("--max-cards", type=int, default=17)
    args = parser.parse_args()

    templates = load_templates()
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_images):
        img, lines = generate_image(
            templates,
            idx,
            (args.width, args.height),
            args.min_cards,
            args.max_cards,
        )
        stem = f"synth_{idx:05d}"
        img_path = OUT_IMG_DIR / f"{stem}.png"
        lbl_path = OUT_LBL_DIR / f"{stem}.txt"
        cv2.imwrite(str(img_path), img)
        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        if idx % 50 == 0:
            print(f"[{idx}/{args.num_images}] saved {img_path}")


if __name__ == "__main__":
    main()
