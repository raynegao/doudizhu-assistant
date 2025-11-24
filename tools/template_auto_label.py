"""
Use template matching to auto-generate YOLO labels from prepared card/states cutouts.

Default behavior:
- 扫描 `data/raw_screenshots/*.png`
- 使用 `data/pic/*.png` 模板进行匹配
- 只输出“牌面”类别（默认 class_id 映射见 rank_char_to_rank/rank_to_id）
- 将匹配结果写入 `data/labels_yolo/<screenshot>.txt` (YOLO: class x_center y_center w h)

CLI:
    python tools/template_auto_label.py ^
        --screenshots data/raw_screenshots ^
        --templates data/pic ^
        --labels data/labels_yolo ^
        --threshold 0.85
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from game_logic.cards import rank_to_id


@dataclass
class TemplateDef:
    name: str
    image: np.ndarray
    rank: str


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Template-based auto labeling for Dou Dizhu cards.")
    parser.add_argument("--screenshots", type=Path, default=Path("data/raw_screenshots"), help="Directory of screenshots.")
    parser.add_argument("--templates", type=Path, default=Path("data/pic"), help="Directory containing template PNGs.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels_yolo"), help="Output directory for YOLO txt labels.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Match score threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="NMS IoU threshold to suppress duplicates.")
    return parser.parse_args()


def load_templates(template_dir: Path) -> List[TemplateDef]:
    """
    Load template images and infer rank from filename suffix.
    """

    rank_char_to_rank: Dict[str, str] = {
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
    templates: List[TemplateDef] = []
    for png in sorted(template_dir.glob("*.png")):
        name = png.stem
        if not name:
            continue
        key = name[-1]
        rank = rank_char_to_rank.get(key)
        if rank is None:
            # Skip unknown templates (e.g., landlord/pass) for card labeling.
            continue
        image = cv2.imread(str(png))
        if image is None:
            continue
        templates.append(TemplateDef(name=name, image=image, rank=rank))
    if not templates:
        raise RuntimeError(f"模板目录 {template_dir} 未加载到任何牌面模板。")
    return templates


def match_templates(screen: np.ndarray, templates: Sequence[TemplateDef], threshold: float) -> List[Detection]:
    detections: List[Detection] = []
    for tpl in templates:
        res = cv2.matchTemplate(screen, tpl.image, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        h, w = tpl.image.shape[:2]
        class_id = rank_to_id(tpl.rank)
        for x, y in zip(xs, ys):
            score = float(res[y, x])
            detections.append(Detection(x, y, x + w, y + h, score, class_id))
    return detections


def nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    if not dets:
        return []
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2, d.score] for d in dets], dtype=float)
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return [dets[i] for i in keep_indices]


def detections_to_yolo_lines(dets: Sequence[Detection], image_shape: Tuple[int, int, int]) -> List[str]:
    h, w = image_shape[:2]
    lines: List[str] = []
    for det in dets:
        box_w = det.x2 - det.x1
        box_h = det.y2 - det.y1
        x_center = det.x1 + box_w / 2
        y_center = det.y1 + box_h / 2
        lines.append(
            f"{det.class_id} {x_center / w:.6f} {y_center / h:.6f} {box_w / w:.6f} {box_h / h:.6f}"
        )
    return lines


def process_screenshot(
    screenshot_path: Path, templates: Sequence[TemplateDef], threshold: float, nms_iou: float, label_dir: Path
) -> None:
    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"[跳过] 无法读取 {screenshot_path}")
        return
    raw_dets = match_templates(image, templates, threshold)
    filtered = nms(raw_dets, nms_iou)
    lines = detections_to_yolo_lines(filtered, image.shape)
    label_dir.mkdir(parents=True, exist_ok=True)
    out_txt = label_dir / f"{screenshot_path.stem}.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {screenshot_path.name}: 匹配到 {len(filtered)} 个框，已写入 {out_txt}")


def main() -> None:
    args = parse_args()
    templates = load_templates(args.templates)
    screenshots = sorted(args.screenshots.glob("*.png"))
    if not screenshots:
        print(f"目录 {args.screenshots} 下没有 PNG 截图。")
        return
    for shot in screenshots:
        process_screenshot(shot, templates, args.threshold, args.nms_iou, args.labels)


if __name__ == "__main__":
    main()
