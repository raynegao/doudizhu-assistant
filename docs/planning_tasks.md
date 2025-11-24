一、整体目标和架构

目标：
在打斗地主的时候，程序自动截取你屏幕上的游戏界面 → 识别三家牌面和出牌历史 → 估计你当前手牌的获胜概率，并用一个简单 UI 实时显示。

逻辑数据流：

截图模块（Screenshot）

牌面检测模块（Card Detection, YOLO）

牌局解析模块（Game State Parser：手牌 + 已出牌 + 底牌）

斗地主规则引擎（Rule Engine：判断合法出牌、比较大小）

胜率计算模块（Monte-Carlo / 模拟器）

UI 显示模块（Probability UI）

二、技术选型（默认用 Python）

语言：Python 3.10+

截图：mss 或 PIL.ImageGrab

图像处理：opencv-python

检测模型：YOLOv8/YOLOv11（ultralytics 库）

深度学习框架：PyTorch（随 YOLO 安装）

UI：

桌面：PyQt5 / PySide6

或者快速版本：Streamlit / Gradio 做 Web UI

AI/Codex 帮忙点：

生成数据标注小工具（简单图形界面或键盘操作）

YOLO 训练脚本

斗地主规则引擎代码（很多 if-else，可以让 AI 帮你生成和重构）

Monte-Carlo 模拟代码骨架

三、推荐项目目录结构
doudizhu_assistant/
├── data/
│   ├── raw_screenshots/        # 原始截图
│   ├── labels_yolo/            # YOLO 标注（txt）
│   └── models/                 # 训练好的权重
├── detection/
│   ├── yolo_train.py           # 训练脚本
│   ├── yolo_infer.py           # 推理脚本
│   └── card_dataset_notes.md   # 标注规范说明
├── game_logic/
│   ├── cards.py                # 牌表示、花色、点数
│   ├── doudizhu_rules.py       # 斗地主规则引擎
│   ├── state_parser.py         # 从检测结果解析成手牌 + 出牌历史
│   └── simulator.py            # Monte-Carlo 模拟、胜率估计
├── ui/
│   ├── desktop_ui.py           # PyQt / PySide UI
│   └── web_ui.py               # 可选：Streamlit / Gradio
├── tools/
│   ├── screenshot_capture.py   # 自动截图脚本
│   └── labeling_tool.py        # 半自动标注小工具
├── main.py                     # 整体入口（截屏 → 检测 → 计算胜率 → UI）
└── requirements.txt

四、开发路线（6 阶段）

你可以按顺序做，每一阶段都能看到成果。

阶段 0：环境和基础脚手架

创建虚拟环境，安装依赖：

mss, opencv-python, ultralytics, PyQt5 / PySide6, numpy

建好上面的目录结构和空文件

用 Codex 让它帮你生成：

简单的 requirements.txt

每个模块的函数空壳（只写 pass 和 docstring）

阶段 1：屏幕截图采集 + 简单标注工具

写一个固定区域截图脚本（tools/screenshot_capture.py）

编写一个非常简易的标注工具：

用 OpenCV 显示图片，鼠标拖动矩形框标注一张牌

按键盘数字选择标签（比如 3, 4, 5, J, Q, K, A, 小王, 大王）

把标注保存成 YOLO 格式 txt

这一块非常适合丢给 Codex 先生成一个粗糙版本，然后你自己改。

阶段 2：YOLO 检测模型

把标注好的数据整理成 YOLO 数据集格式：

dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/


写 detection/yolo_train.py：

加载 ultralytics.YOLO

训练若干 epoch（先训练一个小模型看效果）

写 detection/yolo_infer.py：

读取一张截图

输出每张牌的位置（bbox）和类别（牌面）

阶段 3：牌局解析（最关键的桥梁）

目标：把 YOLO 的输出转换成斗地主游戏状态：

你的手牌（列表/多重集）

地主 / 农民牌

已经打出的牌（按顺序存）

在 game_logic/state_parser.py 里：

定义数据结构（可以用 dataclass）：

@dataclass
class Card:
    rank: str   # '3', '4', ..., 'A', '2', 'joker_small', 'joker_big'
    # 不需要花色

@dataclass
class GameState:
    my_hand: List[Card]
    left_opponent_count: int
    right_opponent_count: int
    last_play: List[Card]
    history: List[List[Card]]


解析流程：

根据 bbox 位置判断哪一行是「你手牌」，哪一行是「对手手牌」

转成 rank 列表

这里可以直接让 Codex 帮你写一版解析逻辑，你自己再根据实际界面改坐标阈值。

阶段 4：斗地主规则引擎（Rule Engine）

在 game_logic/doudizhu_rules.py 中实现：

基础功能：

判断一组牌属于哪种牌型：单张、对子、三带一、三带二、顺子、连对、飞机、炸弹、王炸

比较两个牌型的大小（前提是同一类型）

对外接口建议：

def classify_hand(cards: List[Card]) -> HandType:
    ...

def can_beat(prev: List[Card], current: List[Card]) -> bool:
    ...

def generate_all_legal_hands(hand: List[Card], prev: Optional[List[Card]]) -> List[List[Card]]:
    ...


这部分的 if-else 很多，非常适合用 Codex 帮你生成初始版本，然后你来测试和修 bug。

阶段 5：Monte-Carlo 胜率计算 / 模拟器

在 game_logic/simulator.py：

假设：

已知你的手牌

只知道对手剩余牌数，不知道具体牌

Monte-Carlo 思路：

随机生成很多种「对手可能持有的牌分布」，和真实剩余牌数一致

对每个分布，进行若干局模拟对局（使用规则引擎）：

三家轮流出牌

每一步用一个简单策略：

尝试用能压住的最小牌型

或者随机选一个合法牌型

统计你作为地主/农民的胜率

对外接口：

def estimate_win_rate(state: GameState, num_samples: int = 200) -> float:
    ...


Codex 可以帮你生成一整个「模拟器循环逻辑」，你主要负责设定策略和调参。

阶段 6：UI 集成

在 ui/desktop_ui.py 或 ui/web_ui.py：

流程：

每隔 X 毫秒截取屏幕特定区域

调用 YOLO 检测 → 得到牌

解析成 GameState

调用 estimate_win_rate → 得到一个 0–1 的概率

UI 显示：

当前获胜概率（进度条 / 圆形仪表）

简单提示：建议出牌类型（可以后期再做）

五、现在就能写的代码骨架示例

下面是一个非常简化的「截图 + 检测 + 打印结果」骨架，你可以放到 main.py，然后慢慢填充其它模块：

# main.py
import time
from pathlib import Path

import mss
import numpy as np
import cv2
from ultralytics import YOLO

from game_logic.state_parser import parse_game_state
from game_logic.simulator import estimate_win_rate

MODEL_PATH = Path("data/models/cards_yolov8.pt")

def capture_screen(region=None):
    """
    region: dict with keys {'top', 'left', 'width', 'height'}
    Returns: BGR image (numpy array)
    """
    with mss.mss() as sct:
        monitor = region or sct.monitors[1]  # full screen
        img = np.array(sct.grab(monitor))
        # mss gives BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

def main_loop():
    model = YOLO(str(MODEL_PATH))

    while True:
        frame = capture_screen()  # TODO: later only capture game window
        results = model(frame)[0]

        # TODO: convert YOLO results to your internal "detected cards" format
        # e.g. list of (x_center, y_center, w, h, class_id, confidence)
        detections = []
        for box in results.boxes:
            x_center, y_center, w, h = box.xywh[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append((x_center, y_center, w, h, cls_id, conf))

        # Parse into game state
        game_state = parse_game_state(detections)

        # Estimate win rate
        win_rate = estimate_win_rate(game_state, num_samples=200)

        print(f"Estimated win rate: {win_rate:.2%}")

        # show frame (optional)
        cv2.imshow("Doudizhu Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)  # avoid too frequent capture

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()


parse_game_state 和 estimate_win_rate 可以先写成「假实现」，只返回一些固定值，让主流程先跑起来，再逐步替换为真正逻辑。