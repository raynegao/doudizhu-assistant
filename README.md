# doudizhu-assistant
AI-based Dou Dizhu assistant with Mac screen capture, CNN card recognition, a rule engine, real-time recommendations, and multi-frame stabilization.

[![CI](https://github.com/raynegao/doudizhu-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/raynegao/doudizhu-assistant/actions/workflows/ci.yml)

当前已完成 Phase 1–5B：规则引擎、CNN 牌面识别 replay、Mac 固定 ROI 实时流水线、窗口标定、跨帧稳定投票、显式事件状态、蒙特卡洛 Top-K 推荐、可复现 Showcase/CI/Docker，以及最小只读 Web/API 展示、Demo GIF 生成与真实窗口 holdout 评测流程。

## 一分钟作品集 Demo

不需要模型权重、原始数据或游戏窗口即可生成可复现报告：

```bash
source .venv/bin/activate
make demo
open runs/showcase/index.html
```

输出包括机器可读 `report.json`、CI 摘要 `summary.md` 和完全离线的响应式 `index.html`。默认覆盖地主主动、地主响应和农民响应三个固定事件场景，并验证 sampled worlds 完成数、Top-K 排序与固定 seed 决策指纹。

- [精简架构图](docs/ARCHITECTURE.md)
- [Demo 与录屏说明](docs/SHOWCASE.md)
- [评测口径与基准](docs/EVALUATION.md)
- [中英文作品集/简历描述](docs/PORTFOLIO.md)

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

Phase 2 采用“固定 ROI + 规则切牌 + PyTorch CNN 分类”路线。Mac 本地已完成数据准备、训练、ONNX 导出、crop 推理和 replay 演示，并已被 Phase 3/3.5 实时系统复用。

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

## Phase 3：实时系统第一版

Phase 3 第一版已把 Mac 固定 ROI 截屏源接成可刷新流水线：

```text
固定截图源/窗口源 -> ROI -> CNN 识别 -> 状态刷新 -> 推荐输出 -> JSONL 日志
```

实时主循环位于 `src/pipeline/`，不塞进 `vision`、`logic` 或 `ui` 内部。第一版只做固定 ROI 截屏、内存切牌、CNN 推理、规则推荐、终端实时面板和 JSONL 日志；复杂 GUI、胜率估计和蒙特卡洛策略仍留到后续阶段。

运行一帧 smoke：

```bash
python -m scripts.run_phase3_runtime \
  --max-frames 1 \
  --device cpu \
  --no-clear
```

连续运行：

```bash
python -m scripts.run_phase3_runtime \
  --interval 1.0 \
  --device cpu
```

当前 Mac 斗地主窗口实测运行方式：

```bash
cd "/Users/rayne/Documents/university/junior_year_spring/doudizhu-assistant"
source .venv/bin/activate

python -m scripts.run_phase3_runtime \
  --max-frames 1 \
  --device cpu \
  --no-clear \
  --roi-box 180,555,1300,758 \
  --count 17 \
  --step-x 60 \
  --crop-size 63x105 \
  --start-y 10 \
  --last-play 3
```

如果当前是主动出牌，去掉 `--last-play 3`。如果上一手是对子、三张或其他牌型，按牌面写入，例如 `--last-play "5 5"`、`--last-play "Q Q Q"`。如果窗口位置、窗口大小、显示器或缩放发生变化，`--roi-box 180,555,1300,758` 需要重新标定。

## Phase 3.5：窗口标定与稳定识别

Phase 3.5 支持从当前 Mac 斗地主窗口生成本地 ROI 配置，并用最近 N 帧投票稳定识别结果。

先打开斗地主窗口并进入牌局，然后运行：

```bash
python -m scripts.calibrate_phase3_roi \
  --app-name 斗地主 \
  --save-config configs/phase3_runtime.local.json
```

该命令会读取 `斗地主` 窗口位置，生成 `configs/phase3_runtime.local.json`。这个本地配置已被 `.gitignore` 忽略，不会提交到 Git。

使用本地配置运行：

```bash
python -m scripts.run_phase3_runtime \
  --config configs/phase3_runtime.local.json \
  --max-frames 3 \
  --device cpu \
  --no-clear
```

也可以在运行时重新读取窗口位置：

```bash
python -m scripts.run_phase3_runtime \
  --config configs/phase3_runtime.local.json \
  --auto-window \
  --app-name 斗地主 \
  --max-frames 3 \
  --device cpu \
  --no-clear
```

默认稳定窗口为 `--stability-window 3`。设为 `1` 可关闭跨帧投票：

```bash
python -m scripts.run_phase3_runtime \
  --config configs/phase3_runtime.local.json \
  --stability-window 1 \
  --max-frames 1 \
  --device cpu \
  --no-clear
```

如果斗地主窗口移动、缩放或换显示器，重新运行 `scripts.calibrate_phase3_roi` 即可。

常用参数：

- `--roi-box`：屏幕 ROI，格式为 `left,top,right,bottom`，默认 `(380,1110,2555,1515)`。
- `--count`、`--start-x`、`--start-y`、`--step-x`、`--crop-size`：沿用 Phase 2 固定切牌参数。
- `--last-play`：上一手牌；留空表示主动出牌。
- `--config`：读取本地 ROI 配置，通常使用 `configs/phase3_runtime.local.json`。
- `--auto-window`：运行前重新读取斗地主窗口并换算 ROI。
- `--stability-window`：最近 N 帧投票稳定识别结果，默认 `3`。
- `--confidence-threshold`：低置信度 warning 阈值，默认 `0.70`。
- `--log-file`：默认写入 `logs/phase3_runtime.jsonl`。

macOS 首次运行可能需要给终端或 Codex app 授权“屏幕录制”。如果未授权，CLI 会输出截图权限错误；授权后重新运行即可。如果看到 `does not intersect any displays`，说明当前 `--roi-box` 不在活动显示器坐标范围内，需要重新标定 ROI；Retina 屏幕下 `screencapture` 使用的坐标可能和截图像素尺寸不同。

Phase 3 JSONL 每帧包含 `event`、`frame_id`、`timestamp`、`source`、`roi_box`、`raw_recognized_cards`、`recognized_cards`、`observations`、`last_play`、`candidate_count`、`recommended_action`、`reason`、`warnings`、`latency_ms`、`stabilized` 和 `stability_window`。

## Phase 4：可观测牌局状态与蒙特卡洛决策

Phase 4 已完成离线/显式事件闭环：

```text
game_started + play/pass 事件
  -> 54 张牌守恒与状态 reducer
  -> 未知牌池/对手剩余牌均匀采样
  -> 三人团队 rollout
  -> 策略评分、Top-K 推荐、理由和风险
  -> 终端输出与 JSONL 日志
```

运行仓库内置事件 replay：

```bash
python -m scripts.run_phase4_decision \
  --events-file examples/phase4_round.jsonl \
  --simulations 200 \
  --seed 20260721 \
  --top-k 3 \
  --log-file logs/phase4_decision.jsonl
```

`examples/phase4_round.jsonl` 第一行是 `game_started`，后续每行是带稳定 `event_id` 和递增 `sequence_no` 的 `play_observed` 或 `pass_observed`。状态层会拒绝乱序、越权、物理重复或无法压制的动作；重复事件保持幂等，未确认的低置信度事件会把派生状态标记为 `uncertain` 并阻断推荐，确认事件到达后才能恢复。

Phase 4 当前能力：

- 维护地主、三人顺序、当前行动者、剩余张数、上一手、连续过牌、历史和未知牌池。
- 两家连续过牌后清空当前牌型，由最后出牌者重新领出。
- 使用固定 seed 的局部随机数，对未知牌无放回分配，所有候选动作共用同一批 sampled worlds。
- 支持地主/农民团队胜负、最大深度、时间预算、候选裁剪和确定性回退。
- 输出估计胜率、真实终局率、终局胜率、策略分、风险字段、对手持牌概率和 Top-K 推荐；状态信息不足时回退到确定性策略，并将胜率显示为 `n/a`，不会伪装成 `0%`。
- `RolloutPolicy` 是后续替换更强启发式、MCTS 或 RL policy 的显式接口；Phase 4 本身不引入强化学习。
- 补充四带二单/四带二对；飞机单翅膀仍采用“不同点数单牌”的严格规则口径。

重要边界：当前 Phase 3 只能自动识别自己的手牌，尚未自动识别对手出牌、过牌和剩余张数。因此 Phase 4 的智能决策通过显式事件/JSONL replay 验证，默认不会拖慢 Phase 3 实时路径。均匀分配是可解释的概率基线，不是对手真实手牌预测；日志会保留 `uniform_opponent_model` 和 `rule_subset_only` 风险提示。

## Phase 5A：工程化展示

Phase 5A 已完成可复现证据链：

- `scripts/run_phase5_showcase.py` 一条命令运行 Phase 1 基线和三个 Phase 4 场景。
- 固定 seed 重复执行并生成决策指纹，区分算法一致性与机器相关延迟。
- 输出 JSON、Markdown 和完全自包含的 HTML 作品集报告。
- 核心/视觉依赖分层，CPU Docker 不携带权重、截图或私有数据。
- GitHub Actions 在 Python 3.10/3.12 验证核心路径，并在 Python 3.12 跑完整测试。
- 当前自动化测试基线以最新 CI 为准，Phase 5 证据见 `docs/evidence/`。

Phase 5B 已实现最小 Web API/UI、可重复生成的 Demo GIF 和真实窗口 holdout 工具链；不引入自动点击、强化学习或未经验证的真实窗口准确率宣传。

## Phase 5B：本地 Web 展示与真实窗口 Holdout

启动只读本地展示：

```bash
make web-demo
# 浏览器打开 http://127.0.0.1:8765
```

页面只接受仓库版本化的 Phase 4 replay 场景，展示当前状态、Top-K、风险字段和推荐理由；不会读取任意本地路径、截图或执行游戏操作。生成公开演示素材：

```bash
make demo-gif
# 输出 runs/phase5b/demo.gif
```

真实窗口独立 holdout 的采集与评测说明见 [`docs/REAL_WINDOW_HOLDOUT.md`](docs/REAL_WINDOW_HOLDOUT.md)。工具支持 ROI 自动切牌、单行/交互标注、contact sheet、crop/ROI SHA256、训练集泄漏阻断、逐类别指标与混淆矩阵；尚未采集的数据不会被虚构为泛化指标。

## 配置与日志
- 配置文件支持 YAML/JSON，示例见 `configs/app.example.yaml`。
- `src/config/settings.py` 提供 Pydantic 校验与热重载轮询。
- `src/config/logging_config.py` 提供文本/JSON 日志输出（控制台 + 可选文件）。
