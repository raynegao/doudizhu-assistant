# 斗地主助手阶段计划

## 当前进度

当前项目已完成 Phase 1–5B，并完成 Phase 6 的代码、自动化测试、独立录制/replay 验收工具和工程质量门禁。Phase 6 的最终发布验收仍待 5–10 局真实完整对局和独立牌面 holdout 数据到位后执行。

| 阶段 | 状态 | 已验证闭环 |
| --- | --- | --- |
| Phase 1 | 已完成 | 手动输入 -> 规则引擎 -> 合法动作 -> 推荐 -> CLI/JSONL |
| Phase 2 | 已完成 | ROI/crop -> CNN -> 结构化手牌 -> replay 推荐 |
| Phase 3 | 第一版已完成 | Mac 截屏 -> 实时识别 -> 状态刷新 -> 终端面板/JSONL |
| Phase 3.5 | 已完成 | 窗口定位/ROI 配置 -> 跨帧稳定投票 -> 推荐 |
| Phase 4 | 已完成 | 显式事件状态 -> 对手牌采样 -> 蒙特卡洛 -> Top-K 推荐/JSONL |
| Phase 5A | 已完成 | 三场景 Showcase -> 固定指纹/基准 -> HTML/JSON -> CI/Docker |
| Phase 5B | 已完成（评测待数据） | 最小只读 Web/API、Demo GIF 生成、真实窗口独立 holdout 评测流程 |
| Phase 6 | 离线工程闭环已完成（真实验收待数据） | Retina 窗口 -> 场面观测 -> 视觉事件 -> 状态跟踪 -> 异步 Top-3 -> 置顶小窗；带录制指纹、replay 指标和聚合验收门禁 |

当前自动化测试基线以最新 CI 为准；Phase 2 本地小样本 train/val/test 为 `1589/422/99`，三组准确率均为 `100%`，`error_count = 0`。这些数据只证明当前固定 ROI/CNN 小样本闭环，不代表真实游戏窗口泛化准确率。

当前唯一不能在纯离线条件下补齐的核心缺口，是 5–10 局彼此独立的真实完整对局、逐局动作/余牌/终局标注，以及未参与模板修正的牌面 holdout。聚合验收会校验 session/frame SHA256、评测输入指纹、事件 F1、整组牌点、余牌、牌数守恒和整局成功率；门禁通过前不宣称真实准确率达标。

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

边界：Phase 4 本身仍是显式事件/离线决策层；Phase 6 负责把真实窗口观测转换为事件。均匀未知牌模型会剔除已观测牌和未知历史弃牌，并输出抽样标准误与 95% 置信区间，但仍不等同于真实对手手牌分布或精确胜率。

## Phase 5：工程化展示

目标：让项目适合申请材料和实习简历展示。

状态：Phase 5A/5B 已完成；真实窗口 holdout 工具链已完成，实际评测结果等待新数据采集。

Phase 5A 已完成：

- README 作品集入口、Mermaid 架构图、Demo 录屏脚本和中英文简历描述。
- 三个版本化 JSONL 场景与固定 seed 决策指纹。
- JSON/Markdown/HTML 一键 Showcase 和机器相关延迟基准。
- 历史 CNN 指标摘要、manifest 哈希和口径限制。
- Python 3.10/3.12 核心 CI、完整视觉测试任务和 CPU Docker Demo。
- `requirements-core.txt` / `requirements-vision.txt` 依赖分层。

Phase 5B 已实现范围：

- 最小只读 Web API/UI，展示当前状态、Top-K 和风险字段。
- 可重复生成的 Demo GIF（产物默认在被忽略的 `runs/`）。
- 真实窗口 ROI 自动切牌/标注、contact sheet、SHA256 防泄漏、逐类别指标、混淆矩阵和错误报告流程；实际指标等待新采集的独立标注数据。
- 如需公开发布，再规划 GitHub Release 模型资产和 SHA256 下载流程。

模型权重继续不直接进入 Git；自动点击、强化学习和云部署不进入当前范围。

## Phase 6：完整场面感知与实时胜率助手

状态：代码闭环已完成，真实对局指标等待独立录制数据。

完成项：

- WindowServer Window ID 窗口级截图和 Retina 逻辑/像素转换；窗口被遮挡时仍读取牌桌，移动后无需重写归一化 ROI。
- 归一化 `LiveLayoutConfig`、ROI 预览、contact sheet 和本地配置。
- `SceneObservation` / `SeatObservation`：三家出牌、过牌、余牌和置信度；角色徽标按局部字色跨座位识别，并用 17/20 张结构校验。
- `VisualEventTracker`：连续帧稳定、空白到动作门控、余牌交叉校验、54 张牌守恒和不确定状态阻断。
- `VisualEventTracker` 会把稳定地主位置变化作为新局边界，立即暂停旧状态的推荐。
- `LiveGameRuntime`：截图、视觉、状态、后台决策和 JSONL 编排；断点仅在同一 UI 运行会话内恢复。
- Top-1 改为估计胜率优先；Phase 6 默认 1.5 秒预算、至少 32 组 sampled worlds 和 Top-3，并展示 95% 置信区间和候选区间重叠风险。
- Tkinter 只读置顶助手窗与截图/Vision 分进程运行，识别子进程异常退出自动拉起；不读取鼠标、不点击游戏、不自动代打。
- 标定、模板采集、完整窗口录制和出牌 crop 标注脚本。
- 录制 session 保存不可变配置快照，每帧/每个 ROI 带 SHA256；中断安全收尾并由独立 finalize 确认完整对局。replay 报告绑定 manifest、配置、模型、模板、预测日志和两类人工标注的输入指纹。
- `phase6-acceptance` 聚合审计 5–10 局独立样本，`live-diagnostics` 流式汇总现有 JSONL 的故障、延迟和轮次覆盖。
- 错误截图按原因冷却并设每局/每次运行上限；PID 原子写入且仅由所属进程清理，识别进程和队列句柄在退出/重启时显式释放。
- Ruff、Mypy、分支覆盖率和完整 Pytest 已进入本地 `make quality` 与 CI。

运行入口：

```bash
python -m scripts.calibrate_live_game --save-config configs/live_game.local.json
python -m scripts.record_live_game \
  --config configs/live_game.local.json \
  --session game-001 \
  --until-interrupt
make live-finalize SESSION=game-001
python -m scripts.run_live_assistant --config configs/live_game.local.json
make live-diagnostics
make phase6-acceptance
```

验收边界：必须使用未参与模板/微调的完整对局 replay 验证出牌/过牌 F1、牌点整组准确率、余牌准确率和整局状态守恒。没有真实数据时，只能确认离线代码、状态保护、证据链和自动化质量门禁通过；详细流程见 `docs/PHASE6_ACCEPTANCE.md`。
