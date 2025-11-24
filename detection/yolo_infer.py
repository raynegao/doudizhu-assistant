"""
YOLO 推理与可视化工具。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import cv2
from ultralytics import YOLO

Detection = Tuple[float, float, float, float, int, float]  # xc, yc, w, h, cls, conf


def load_detector(weights_path: Path) -> YOLO:
    """
    加载 YOLO 模型权重。
    """

    return YOLO(str(weights_path))


def run_inference(model: YOLO, image_source: Any, conf: float = 0.3) -> List[Detection]:
    """
    对单张图片执行推理，返回标准化的检测结果列表。
    image_source 可以是路径或已加载的图像（numpy 数组）。
    Detection: (x_center, y_center, width, height, class_id, confidence) in pixels.
    """

    source = str(image_source) if isinstance(image_source, Path) else image_source
    results = model.predict(source=source, conf=conf, verbose=False)
    if not results:
        return []
    res = results[0]
    dets: List[Detection] = []
    for box in res.boxes:
        x_center, y_center, w, h = box.xywh[0].tolist()
        cls_id = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        dets.append((x_center, y_center, w, h, cls_id, conf_score))
    return dets


def visualize_detections(image_path: Path, detections: Sequence[Detection], class_names: Iterable[str]) -> None:
    """
    将检测框绘制到图片上用于调试。
    """

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    names = list(class_names)
    for det in detections:
        xc, yc, w, h, cls_id, conf = det
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)
        label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a screenshot.")
    parser.add_argument("--weights", type=Path, required=True, help="YOLO 权重路径 (e.g., runs/train/doudizhu/weights/best.pt)")
    parser.add_argument("--image", type=Path, required=True, help="输入截图路径")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--show", action="store_true", help="是否弹窗展示检测结果")
    args = parser.parse_args()

    model = load_detector(args.weights)
    detections = run_inference(model, args.image, conf=args.conf)
    print(f"检测到 {len(detections)} 个框")
    for det in detections:
        print(det)

    if args.show:
        try:
            from game_logic.cards import RANK_ORDER
        except Exception:
            RANK_ORDER = []
        visualize_detections(args.image, detections, RANK_ORDER or range(100))


if __name__ == "__main__":
    main()
