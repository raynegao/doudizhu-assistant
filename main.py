"""
入口：截图 -> YOLO 检测 -> 状态解析 -> 简单模拟，打印胜率。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import mss
import numpy as np

from detection.yolo_infer import load_detector, run_inference
from game_logic.state_parser import parse_game_state
from game_logic.simulator import estimate_win_rate

MODEL_PATH = Path("runs/train/doudizhu/weights/best.pt")
DEFAULT_REGION: Dict[str, int] = {
    "top": 42,
    "left": 152,
    "width": 1620,
    "height": 946,
}


def capture_screen(region: Optional[Dict[str, int]] = None) -> Any:
    """
    捕获屏幕区域，返回 BGR 图像（numpy 数组）。
    """

    with mss.mss() as sct:
        monitor = region or sct.monitors[1]
        shot = sct.grab(monitor)
        img = np.array(shot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def process_frame(frame: Any, model: Any, num_samples: int = 80) -> float:
    """
    执行检测 -> 解析 -> 模拟，返回胜率。
    """

    detections = run_inference(model, frame, conf=0.35)
    state = parse_game_state(detections)
    win_rate = estimate_win_rate(state, num_samples=num_samples)
    return win_rate


def main_loop() -> None:
    """
    连续截图并打印胜率估计。按 Ctrl+C 退出。
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型权重: {MODEL_PATH}，请先训练或配置路径。")

    model = load_detector(MODEL_PATH)
    print("Dou Dizhu Assistant 已启动，按 Ctrl+C 退出。")

    try:
        while True:
            frame = capture_screen(DEFAULT_REGION)
            win_rate = process_frame(frame, model)
            print(f"当前胜率估计: {win_rate:.2%}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("已退出。")


if __name__ == "__main__":
    main_loop()
