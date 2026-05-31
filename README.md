# doudizhu-assistant
AI-based Dou Dizhu assistant: screen card recognition + win rate estimation using YOLO and Monte Carlo simulation.

## 环境
- Python 3.12.13（>=3.10 均可，当前本地开发使用 3.12）
- 建议使用虚拟环境隔离依赖

## 快速开始
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Phase 1：规则引擎 MVP
当前阶段先不依赖 CV/YOLO，使用手动输入跑通规则引擎闭环：

```bash
python -m src.ui.cli \
  --hand "3 3 4 4 5 5 6 6 7 8 9 SJ BJ" \
  --last-play "5 5" \
  --log-file logs/phase1.jsonl
```

牌面记法：
- 普通牌：`3 4 5 6 7 8 9 10 J Q K A 2`
- 小王：`SJ`
- 大王：`BJ`

运行测试：

```bash
python -m pytest -q
```

当前 Phase 1 已跑通的能力：

- 解析手牌和上一手牌，校验重复牌和大小王数量。
- 识别单张、对子、三张、三带一、三带二、顺子、连对、飞机、炸弹和火箭等基础牌型。
- 根据上一手牌生成可行动作集合。
- 输出一个确定性的基础推荐动作和中文理由。
- 写入 JSONL 日志，字段包括 `event`、`input_cards`、`last_play`、`candidate_count`、`recommended_action`、`reason`、`warnings`。

Phase 1 现在作为稳定基础保留；后续 CV 和实时系统继续复用 CLI、测试和日志字段。

## Phase 2：CNN 牌面识别闭环

Phase 2 采用“固定 ROI + 规则切牌 + PyTorch CNN 分类”路线。Mac 本地已完成数据准备、训练、ONNX 导出、crop 推理和 replay 演示，当前进入收尾状态，下一步准备进入 Phase 3 实时系统。

当前本地评估结果来自 `models/card_cnn.metrics.json` 和 `models/card_cnn.eval.json`：

- train：`1589/1589 = 100%`
- val：`422/422 = 100%`
- test：`99/99 = 100%`
- `error_count = 0`

这个准确率只代表当前小样本、本地 ROI/CNN 闭环验收，不代表真实游戏窗口泛化准确率。Phase 3 需要继续用更多真实截图验证。

```bash
python -m scripts.prepare_card_templates --dry-run
python -m scripts.prepare_card_templates
python -m scripts.rebuild_card_cls_dataset \
  --template-per-seed 20 \
  --real-per-crop 30
```

`rebuild_card_cls_dataset` 会清空并重建 `data/cards_cls/train`、`data/cards_cls/val`、`data/cards_cls/test`，并写出 `data/cards_cls/manifest.jsonl` 记录每张增强样本的来源。当前会跳过坏 crop 目录 `data/roi_samples/hand_roi_001_step135`。

训练小 CNN、导出 ONNX，并生成基础 metrics：

```bash
python -m scripts.train_card_cnn --epochs 30 --batch-size 32
```

单独评估训练集、验证集和测试集，输出混淆矩阵与错例报告：

```bash
python -m scripts.evaluate_card_cnn \
  --model models/card_cnn.pt \
  --dataset data/cards_cls \
  --output-dir models
```

输出目录：

- `data/cards_cls_seed/<rank>/`：从 DouZero fork 的 `pics/` 提取的模板 seed。
- `data/cards_cls/train/<rank>/`：增强后的训练集。
- `data/cards_cls/val/<rank>/`：增强后的验证集。
- `data/cards_cls/test/<rank>/`：从真实 crop 拆分出的最小测试集。
- `data/cards_cls/manifest.jsonl`：增强样本的数据来源记录。
- `models/card_cnn.pt`：PyTorch checkpoint。
- `models/card_cnn.onnx`：ONNX 推理模型。
- `models/card_cnn.metrics.json`：训练过程中的最佳 epoch 和 train/val/test 指标。
- `models/card_cnn.confusion_matrix.csv`：混淆矩阵。
- `models/card_cnn.error_report.jsonl`：错例报告。

这些数据和模型目录默认不提交 Git。Phase 2 默认在 Mac 本地完成，不依赖 Windows/WSL。

## Phase 2D：Mac 斗地主 ROI 切牌

当前 Mac 客户端可用固定 ROI + 固定步长切牌。先从完整截图裁出手牌 ROI，再把重叠手牌切成可分类的 rank+suit 小图：

```bash
python -m scripts.crop_hand_roi_cards \
  --roi data/raw_screenshots/window_mode_hand_roi_tight_001.png \
  --output-dir data/roi_samples/window_mode_hand_roi_tight_001 \
  --count 15 \
  --start-x 0 \
  --start-y 20 \
  --step-x 135 \
  --crop-size 126x210
```

把真实 crop 加入训练集用于当前客户端风格微调：

```bash
python -m scripts.add_labeled_crops_to_dataset \
  --crop-dir data/roi_samples/window_mode_hand_roi_tight_001 \
  --labels "A K Q J 10 10 9 8 7 7 6 5 4 3 3"
```

`data/raw_screenshots/` 和 `data/roi_samples/` 默认不提交 Git。当前窗口模式实测手牌 ROI 为 `(380, 1110, 2555, 1515)`，当前样本手牌数为 15，`step-x=135`，`start-y=20`；如果窗口缩放、移动到其他显示器或分辨率变化，需要重新标定。

补充 `2/BJ/SJ` 覆盖时，可使用包含大王和 `2` 的 17 张手牌截图。当前 joker 样本参数为 ROI `(360, 1110, 2600, 1515)`、`count=17`、`step-x=120`；`SJ` 暂用 `BJ` crop 的灰度版本生成，后续截到真实小王后再替换。

当前重建数据集使用的真实 crop：

- `data/roi_samples/window_mode_hand_roi_tight_001`：标签为 `A K Q J 10 10 9 8 7 7 6 5 4 3 3`。
- `data/roi_samples/window_mode_jokers_hand_roi_001`：标签为 `BJ 2 2 K Q J J 9 9 9 8 8 8 6 4 4 3`。
- `data/roi_samples/window_mode_jokers_sj_gray`：临时 `SJ` 灰度样本；后续应替换为真实小王截图。
- `data/roi_samples/hand_roi_001_step135`：存在空白/坏 crop，不参与当前数据集重建。

预测已切好的 crop：

```bash
python -m scripts.predict_card_crops \
  --model models/card_cnn.pt \
  --crop-dir data/roi_samples/window_mode_hand_roi_tight_001
```

端到端 replay：CNN 识别手牌后接入 Phase 1 规则引擎，输出候选动作和推荐理由。

```bash
python -m scripts.replay_phase2 \
  --model models/card_cnn.pt \
  --roi data/raw_screenshots/window_mode_hand_roi_tight_001.png
```

如果当前手牌数量或重叠间距不同，replay 可显式传入切牌参数，例如：

```bash
python -m scripts.replay_phase2 \
  --model models/card_cnn.pt \
  --roi data/raw_screenshots/window_mode_jokers_hand_roi_001.png \
  --count 17 \
  --step-x 120
```

Replay 默认会在单牌置信度低于 `0.70` 时输出 warning，但不中断规则引擎推荐。

当前 Phase 2 replay smoke 的期望识别结果：

```text
A K Q J 10 10 9 8 7 7 6 5 4 3 3
```

## Phase 3：实时系统准备

Phase 3 第一目标是把固定截图源或窗口源接成可刷新流水线：

```text
固定截图源/窗口源 -> ROI -> CNN 识别 -> 状态刷新 -> 推荐输出 -> JSONL 日志
```

实时主循环应放在后续新增的 `src/pipeline/` 或 `src/runtime/`，不要塞进 `vision`、`logic` 或 `ui` 内部。第一版只做稳定刷新、日志和简单展示，再考虑复杂 GUI、胜率估计和蒙特卡洛策略。

## 配置与日志
- 配置文件支持 YAML/JSON，示例见 `configs/app.example.yaml`。
- `src/config/settings.py` 提供 Pydantic 校验与热重载轮询。
- `src/config/logging_config.py` 提供文本/JSON 日志输出（控制台 + 可选文件）。
