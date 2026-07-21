# 斗地主助手阶段计划

## 当前进度

当前项目已完成 Phase 1、Phase 2、Phase 3 第一版、Phase 3.5 和 Phase 4，下一阶段是 Phase 5 工程化展示。

| 阶段 | 状态 | 已验证闭环 |
| --- | --- | --- |
| Phase 1 | 已完成 | 手动输入 -> 规则引擎 -> 合法动作 -> 推荐 -> CLI/JSONL |
| Phase 2 | 已完成 | ROI/crop -> CNN -> 结构化手牌 -> replay 推荐 |
| Phase 3 | 第一版已完成 | Mac 截屏 -> 实时识别 -> 状态刷新 -> 终端面板/JSONL |
| Phase 3.5 | 已完成 | 窗口定位/ROI 配置 -> 跨帧稳定投票 -> 推荐 |
| Phase 4 | 已完成 | 显式事件状态 -> 对手牌采样 -> 蒙特卡洛 -> Top-K 推荐/JSONL |
| Phase 5 | 未开始 | 作品集 Demo、指标报告、GUI/API、部署材料 |

当前验证基线为 `73 passed`；Phase 2 本地小样本 train/val/test 为 `1589/422/99`，三组准确率均为 `100%`，`error_count = 0`。这些数据只证明当前固定 ROI/CNN 小样本闭环，不代表真实游戏窗口泛化准确率。

当前主要缺口：真实窗口尚未自动识别上一手牌、过牌和对手剩余张数；GUI/API、Docker、性能报告和作品集级 Demo 尚未完成。Phase 4 已通过显式事件/JSONL replay 完成算法闭环，不把不完整视觉输入伪装成自动整局跟踪。

## Phase 1：规则引擎 MVP

状态：已完成，作为后续阶段的稳定规则基础维护。

完成项：

- `src/state/cards.py`：牌面解析、别名归一化、排序和重复牌校验。
- `src/state/game_state.py`：手动输入到结构化状态。
- `src/logic/rules.py`：牌型识别、压制判断和合法动作生成。
- `src/logic/decision.py`：基础确定性推荐策略。
- `src/ui/cli.py`：命令行输入、输出和 JSONL 日志。
- `tests/`：Phase 1 的最小测试集。

验收命令：

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m compileall src tests
python -m pytest -q
python -m src.ui.cli \
  --hand "3 3 4 4 5 5 6 6 7 8 9 SJ BJ" \
  --last-play "5 5" \
  --log-file logs/phase1.jsonl
```

## Phase 2：CV 检测接入

目标：把手动输入替换或补充为屏幕/截图识别输入。先完成手牌区域识别和牌面结构转换，不做完整实时系统。

状态：已完成。当前路线是固定 ROI + 规则切牌 + PyTorch CNN 分类，已跑通 Mac 本地 replay 闭环：ROI/crop -> CNN 识别手牌 -> Phase 1 规则引擎 -> 推荐动作。

### Phase 2A：模板资源整理

- 从 `tianqiraf/DouZero_For_HappyDouDiZhu`、`Vincentzyx/DouZero_For_HLDDZ_FullAuto`、`cyt0125/DouZero_For_Offline_Doudizhu` 的 `pics/` 中提取牌面模板。
- 统一映射类别：`3 4 5 6 7 8 9 10 J Q K A 2 SJ BJ`。
- 兼容 DouZero fork 的记法：`T -> 10`、`X -> SJ`、`D -> BJ`。
- 使用 `scripts/prepare_card_templates.py` 输出到 `data/cards_cls_seed/<rank>/`。

### Phase 2B：生成 CNN 分类数据集

- 使用 `scripts/generate_card_cls_dataset.py` 对 seed 模板做增强。
- 增强包括 resize、brightness/contrast、blur、JPEG compression、slight crop、background noise。
- 输出到 `data/cards_cls/train/<rank>/` 和 `data/cards_cls/val/<rank>/`。

### Phase 2C：训练小 CNN

- 使用 `src/vision/card_classifier.py` 定义轻量 CNN、类别映射、预处理、checkpoint 加载和批量预测。
- 使用 `scripts/train_card_cnn.py` 在 Mac 本地训练，优先 `mps`，否则 `cpu`。
- 输出 `models/card_cnn.pt` 和 `models/card_cnn.onnx`，二者默认不提交 Git。
- 使用 `scripts/predict_card_crops.py` 对已切 crop 做批量推理。

### Phase 2D：真实截图校验

- 优先使用 Mac 本地 replay 模式：读取固定截图、手牌 ROI 或预录样本，稳定演示识别和推荐链路。
- 当前窗口模式基准：ROI box `(380, 1110, 2555, 1515)`，当前样本 `count=15`，`start-y=20`，`step-x=135`，`crop-size=126x210`。
- 使用 `scripts.crop_hand_roi_cards` 裁出 rank+suit crop。
- 使用 `scripts.add_labeled_crops_to_dataset` 将少量真实 crop 加入训练集，缓解模板合成数据和 Mac 客户端牌面风格差异。
- 已补充真实 `2` 和 `BJ` 样本；`SJ` 暂用 `BJ` 灰度版本生成，后续截到真实小王后替换。
- 使用 `scripts.replay_phase2` 将 CNN 识别结果接入 `GameStateSnapshot`、合法动作生成和推荐策略。
- 当前合成数据只保证工程闭环；真实 crop 后续可作为 validation 或 fine-tune 数据。

建议范围：

- 定义 `Frame`、`Detection`、`CardObservation` 等数据对象。
- 实现截图/图片文件输入的最小读取接口。
- 明确 ROI 配置和坐标约定。
- 准备固定截图或 fixture。
- 接入检测结果到 `CardSet` / `GameStateSnapshot` 的转换。
- 为视觉后处理和状态转换写最小测试。

暂不做：

- 自动鼠标操作或自动代打。
- 强化学习。
- 高复杂 GUI。
- 云部署。
- 复杂实时 pipeline。

## Phase 3：实时系统

目标：把截图、识别、跟踪、状态、决策和展示串成可刷新系统。

状态：第一版已完成。当前已经支持 Mac 固定 ROI 截屏、内存切牌、CNN 推理、状态刷新、规则推荐、终端实时面板、延迟统计和 JSONL 日志。

已完成范围：

- 新增 `src/pipeline/runtime.py` 承接实时主循环。
- 接入屏幕截取、ROI、CNN 推理、规则推荐和错误边界。
- 输出延迟、候选动作和推荐日志。

## Phase 3.5：窗口定位与稳定识别

状态：已完成。

完成项：

- 使用 macOS `System Events` 查找斗地主窗口。
- 生成并加载本地 `configs/phase3_runtime.local.json`。
- 支持 `--auto-window` 根据当前窗口重新换算 ROI。
- 使用最近 N 帧多数票和平均置信度稳定单牌识别。
- 日志同时保留原始识别、稳定识别和稳定窗口字段。

当前限制：只稳定当前手牌识别，不自动跟踪上一手牌、轮次和上下家出牌事件。

## Phase 4：智能决策增强

目标：在规则引擎稳定后提升推荐质量。

状态：已完成显式状态/离线 replay 版本。

完成项：

- `src/state/events.py`、`observable_state.py`、`game_tracker.py`：显式事件、54 张牌守恒、幂等 reducer 和过牌重置。
- `src/logic/opponent_model.py`：固定 seed、无放回、均匀对手牌采样和概率摘要。
- `src/logic/monte_carlo.py`：三人团队 rollout、common random worlds、时间/深度预算和候选裁剪。
- 策略分、估计胜率、真实终局率、Top-K 推荐、理由和风险提示。
- `RolloutPolicy` 接口预留后续启发式、MCTS 或 RL policy；Phase 4 不实现强化学习。
- `scripts/run_phase4_decision.py`：手动状态和 JSONL 事件 replay CLI。
- 低置信度/乱序/重复/非法事件保护；未确认事件阻断推荐，信息不足时确定性回退且胜率标记为不可用。

验收命令：

```bash
python -m pytest -q
python -m scripts.run_phase4_decision \
  --events-file examples/phase4_round.jsonl \
  --simulations 200 \
  --seed 20260721 \
  --top-k 3
```

边界：真实窗口自动对手事件识别尚未完成；均匀对手模型和当前规则子集会作为风险字段写入结果，不宣称为真实对手牌或完美胜率。

## Phase 5：工程化展示

目标：让项目适合申请材料和实习简历展示。

建议范围：

- README 和系统架构图。
- Demo 截图或视频说明。
- 数据、模型和评测指标报告。
- 延迟/FPS/准确率统计。
- Docker 或本地一键运行脚本。
- 项目总结和简历描述。
