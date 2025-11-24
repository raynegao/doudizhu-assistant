"""
Minimal labeling tool for annotating card bounding boxes.

Workflow description:
- 打开某张截图后，鼠标按下记录起点，拖动显示矩形，松开确认矩形。
- 立即按下对应牌面字符（3/4/.../A/X/Y）即可为该框设置标签。
- 工具将 YOLO `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>` 行保存到
  `data/labels_yolo/<screenshot_name>.txt`。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from game_logic.cards import RANK_ORDER, rank_to_id

RAW_DIR = Path("data/raw_screenshots")
LABEL_DIR = Path("data/labels_yolo")
WINDOW_NAME = "Doudizhu Labeling"

LABEL_KEY_TO_RANK: Dict[str, str] = {
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
    "j": "J",
    "J": "J",
    "q": "Q",
    "Q": "Q",
    "k": "K",
    "K": "K",
    "a": "A",
    "A": "A",
    "2": "2",
    "x": "joker_small",
    "X": "joker_small",
    "y": "joker_big",
    "Y": "joker_big",
}


@dataclass
class Annotation:
    """
    Represents a single bounding box and associated class id.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: Optional[int] = None


class AnnotationSession:
    """
    Holds state for labeling a single image.
    """

    def __init__(self, image_path: Path, label_path: Path):
        self.image_path = image_path
        self.label_path = label_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        self.height, self.width = self.image.shape[:2]
        self.annotations: List[Annotation] = []
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.current_point: Optional[Tuple[int, int]] = None
        self.quit_requested = False
        self._load_existing_annotations()

    def _load_existing_annotations(self) -> None:
        if not self.label_path.exists():
            return
        lines = self.label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
            box_width = width_norm * self.width
            box_height = height_norm * self.height
            x_center = x_center_norm * self.width
            y_center = y_center_norm * self.height
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            self.annotations.append(Annotation(x1, y1, x2, y2, class_id))

    def run(self) -> bool:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, handle_mouse_event, self)
        print(
            f"\n正在标注 {self.image_path.name}。"
            " 鼠标拖拽选框 → 输入牌面字符（3/4/.../A/X/Y）→ 自动保存到标签文件。"
            " 按 N 进入下一张，按 Esc 退出，U 撤销，C 清空。"
        )

        while True:
            canvas = self._render_canvas()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(16) & 0xFF
            if key in {27}:  # ESC only
                self.quit_requested = True
                break
            if key in {ord("n"), ord("N")}:
                if self._has_pending_boxes():
                    print("还有未标注的矩形，请先输入标签。")
                    continue
                break
            if key in {ord("u"), ord("U")}:
                self._undo_last()
            elif key in {ord("c"), ord("C")}:
                self.annotations.clear()
                self._save_annotations()
                print("已清空该图片标注。")
            elif key != 255:
                self._handle_label_key(chr(key))

        cv2.destroyWindow(WINDOW_NAME)
        if not self.quit_requested:
            self._save_annotations()
        return not self.quit_requested

    def _render_canvas(self) -> np.ndarray:
        canvas = self.image.copy()
        for ann in self.annotations:
            color = (0, 255, 0) if ann.class_id is not None else (0, 165, 255)
            cv2.rectangle(canvas, (ann.x1, ann.y1), (ann.x2, ann.y2), color, 2)
            if ann.class_id is not None:
                rank = RANK_ORDER[ann.class_id]
                cv2.putText(canvas, rank, (ann.x1, max(15, ann.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if self.drawing and self.start_point and self.current_point:
            cv2.rectangle(canvas, self.start_point, self.current_point, (255, 255, 0), 1)

        instruction = "拖拽鼠标→输入牌面(3-9/0/T/J/Q/K/A/2/X/Y)→N下一张→Esc退出→U撤销→C清空"
        cv2.putText(canvas, instruction, (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return canvas

    def _handle_label_key(self, key: str) -> None:
        rank = LABEL_KEY_TO_RANK.get(key)
        if not rank:
            return
        idx = self._find_pending_annotation()
        if idx is None:
            print("没有等待标签的矩形。请先用鼠标画框。")
            return
        class_id = rank_to_id(rank)
        self.annotations[idx].class_id = class_id
        self._save_annotations()
        print(f"已为矩形 #{idx + 1} 设置标签 {rank} (id={class_id})。")

    def _find_pending_annotation(self) -> Optional[int]:
        for idx in range(len(self.annotations) - 1, -1, -1):
            if self.annotations[idx].class_id is None:
                return idx
        return None

    def _has_pending_boxes(self) -> bool:
        return any(ann.class_id is None for ann in self.annotations)

    def _save_annotations(self) -> None:
        boxes = [(ann.x1, ann.y1, ann.x2, ann.y2) for ann in self.annotations]
        classes = [ann.class_id for ann in self.annotations]
        save_annotations(self.label_path, boxes, classes, (self.width, self.height))

    def _undo_last(self) -> None:
        if self.annotations:
            removed = self.annotations.pop()
            self._save_annotations()
            print(f"已撤销最后一个矩形（class_id={removed.class_id}）。")

    def on_mouse_event(self, event: int, x: int, y: int, _flags: int) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if not self.start_point:
                return
            new_box = Annotation(self.start_point[0], self.start_point[1], x, y)
            self.annotations.append(new_box)
            self.start_point = None
            self.current_point = None
            print("已创建矩形，请输入对应牌面标签。")


def launch_labeling_tool(image_dir: Path = RAW_DIR, label_dir: Path = LABEL_DIR) -> None:
    """
    Open the annotation UI for a batch of screenshots.
    """

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        print(f"目录 {image_dir} 下没有 PNG 截图。请先运行 screenshot_capture.py。")
        return

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        session = AnnotationSession(image_path, label_path)
        should_continue = session.run()
        if not should_continue:
            print("用户中断了标注流程。")
            break


def handle_mouse_event(event: int, x: int, y: int, flags: int, params: object) -> None:
    """
    Process mouse events for drawing bounding boxes.
    """

    if isinstance(params, AnnotationSession):
        params.on_mouse_event(event, x, y, flags)


def save_annotations(
    label_path: Path, boxes: Sequence[Tuple[int, int, int, int]], classes: Sequence[Optional[int]], image_size: Tuple[int, int]
) -> None:
    """
    Write YOLO-formatted labels collected from the UI.
    """

    width, height = image_size
    lines: List[str] = []
    for box, class_id in zip(boxes, classes):
        if class_id is None:
            continue
        x1, y1, x2, y2 = box
        left = max(0, min(x1, x2))
        right = min(width - 1, max(x1, x2))
        top = max(0, min(y1, y2))
        bottom = min(height - 1, max(y1, y2))

        box_width = max(1, right - left)
        box_height = max(1, bottom - top)
        x_center = (left + right) / 2
        y_center = (top + bottom) / 2
        line = f"{class_id} {x_center / width:.6f} {y_center / height:.6f} {box_width / width:.6f} {box_height / height:.6f}"
        lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """
    Command-line entry point for the labeling tool.
    """

    launch_labeling_tool()


if __name__ == "__main__":
    main()
