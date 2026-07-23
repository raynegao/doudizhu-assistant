# 斗地主 AI 辅助系统架构评审

## 1. 项目定位

本项目目标是构建一个实时斗地主 AI 辅助系统：从屏幕采集牌面，通过 CV 模型识别手牌与牌局信息，维护游戏状态，再用规则引擎、蒙特卡洛模拟或 Agent 工作流生成出牌建议、胜率估计和可解释理由。

从 AI 应用工程角度看，这不是单一模型项目，而是一个端到端 AI 系统项目。它适合作为港新 AI 硕士申请项目和 AI 应用开发实习项目，但需要持续补齐数据闭环、实时推理、评测指标、工程部署和可解释展示。

## 2. 当前结构评审

当前仓库已经完成 Phase 1 规则引擎 MVP、Phase 2 本地 CNN 牌面识别 replay、Phase 3/3.5 实时识别与稳定化、Phase 4 显式牌局状态/蒙特卡洛决策、Phase 5 工程展示，以及 Phase 6 完整场面感知代码闭环：

```text
configs/
  app.example.yaml
docs/
  ARCHITECTURE.md
  EVALUATION.md
  PORTFOLIO.md
  SHOWCASE.md
tests/
  test_card_classifier.py
  test_card_dataset_scripts.py
  test_cards.py
  test_decision_cli.py
  test_phase2_replay.py
  test_phase3_runtime.py
  test_game_tracker.py
  test_opponent_model.py
  test_phase4_decision.py
  test_phase4_cli.py
  test_event_replay.py
  test_phase5_showcase.py
  test_rules.py
scripts/
  calibrate_phase3_roi.py
  crop_hand_roi_cards.py
  predict_card_crops.py
  run_phase4_decision.py
  run_phase5_showcase.py
  run_phase3_runtime.py
  replay_phase2.py
  train_card_cnn.py
src/
  capture/
  config/
  logic/
    action_validation.py
    decision.py
    monte_carlo.py
    opponent_model.py
    rules.py
  pipeline/
    calibration.py
    runtime.py
    stabilizer.py
  reporting/
    showcase.py
  state/
    cards.py
    events.py
    game_tracker.py
    game_state.py
    observable_state.py
    replay.py
  tracking/
  ui/
    cli.py
  vision/
    card_classifier.py
README.md
pytest.ini
requirements-dev.txt
```

整体模块方向合理，已经把采集、视觉、跟踪、状态、决策和 UI 分成不同包。Phase 6 使用 Retina-aware 窗口帧和归一化 ROI 生成 `SceneObservation`，由 `VisualEventTracker` 产生既有 `ObservedAction`，再由 `LiveGameRuntime` 异步计算胜率并推送只读置顶窗。当前缺口转为真实完整对局数据和独立 replay 指标，而不是代码接口缺失。

## 3. 推荐分层

建议采用以下逻辑链路：

```text
capture -> vision -> tracking -> state -> logic -> ui
              pipeline/runtime 负责编排和事件传递
              agent 只消费状态、建议和日志做解释/复盘
```

核心原则：

- CV 模块只输出视觉观测，不做游戏决策。
- 游戏状态模块只维护牌局事实、置信度和未知信息，不读取图片。
- 决策模块只消费结构化状态，不关心屏幕坐标和 GUI。
- GUI 只展示状态、建议和解释，不承载模型推理主循环。
- Pipeline/Runtime 负责连接所有模块，是实时系统的编排层。
- Agent 层负责解释、复盘、工具调用和开发辅助，不替代底层确定性规则。

禁止依赖方向：

- `vision` 不调用 `logic`。
- `logic` 不读取截图、屏幕坐标或 GUI 控件。
- `ui` 不承载实时主循环。
- `agent` 不绕过 `logic` 直接给最终出牌结论。
- `pipeline/runtime` 可以编排模块，但不应把视觉细节、规则细节或 GUI 状态写进同一个大函数。

### 3.1 开发、训练与演示环境边界

本项目当前采用 Mac 单机闭环，而不是把训练或演示绑定到远程 GPU 机器：

- Mac：唯一默认环境，负责日常编码、规则引擎、状态建模、CLI、文档、单元测试、CV 数据准备、小 CNN 训练、推理验证和 Demo 演示。
- Apple Silicon 本地训练：Phase 2 采用“固定 ROI + 规则切牌 + 小 CNN 分类”路线，训练规模较小，适合在 MacBook Air M4/16GB 上完成。
- 演示方式：优先做 replay demo，读取固定截图、手牌 ROI 或预录样本，稳定展示识别结果、结构化手牌、合法动作和推荐理由。
- Windows/WSL：不再作为默认开发、训练或演示链路。只有后续明确需要 YOLO 大规模训练、CUDA 对比实验或性能基准时，才作为可选增强资源单独规划。
- 同步方式：默认不进行跨机器同步；如未来重新启用远程训练，必须先确认分支、未提交改动、数据目录和模型产物边界。
- Docker：不作为当前训练前置条件；后续如用于工程化展示，再单独纳入 DevOps 范围。

## 4. Codex 多 Agent 开发模式

本项目后续采用 Codex Subagents 模式开发，但不是让多个 Agent 同时无边界改代码。主 Agent 负责拆分任务、控制范围、合并结果和最终验证；子 Agent 按专业角色承担审查、设计和小范围实现。

### 4.1 Agent 角色

| Agent | 职责 | Phase 1 参与方式 |
| --- | --- | --- |
| Project Architect Agent | 架构、目录、模块边界、文档维护、范围控制 | 维护 MVP 边界，防止提前做 CV/RL/复杂 GUI |
| CV Detection Agent | 截屏、ROI、检测、YOLO 数据集、标注和后处理 | 只定义未来接口，不实现真实检测 |
| Game State Agent | 检测结果到牌局状态，手牌、上下家、轮次和可观测状态 | 定义手动输入到结构化状态 |
| Rule Engine Agent | 牌型识别、比较、合法出牌、可行动作生成 | Phase 1 核心实现者 |
| Decision Agent | 策略、蒙特卡洛、RL 预留、推荐和解释 | 只做简单可解释推荐 |
| GUI Agent | PyQt/Streamlit/Web UI，展示识别、状态、建议和置信度 | 只做 CLI 或最简单展示 |
| Testing & Debugging Agent | 单测、规则测试、端到端测试、bug 定位 | 为规则和状态建模建立测试基线 |
| DevOps Agent | 依赖、配置、Docker、README、运行说明、Demo 材料 | 维护本地可运行说明，不做云部署 |

### 4.2 协作原则

- 子 Agent 不直接跨模块重构。
- 每个开发任务先定义归属 Agent、输入、输出、测试方式和完成标准。
- 主 Agent 每次只合并小范围变更，并说明改了哪些文件、为什么这样设计、如何验证。
- Project Architect Agent 优先审查边界，不替代具体模块实现。
- Testing & Debugging Agent 与功能开发并行出现，避免最后才补测试。
- DevOps Agent 只在功能有最小闭环后推进 Docker、README 和 Demo 材料。

Phase 1 不实现 `src/agent/`。Agent 工作流要等 `GameStateSnapshot`、`DecisionResult` 和 JSONL 日志稳定后再接入，用于解释、复盘和实验辅助。

## 5. MVP 路线

### Phase 1：规则引擎 MVP

目标：在不依赖 CV 模型的情况下跑通“输入手牌 -> 结构化状态 -> 合法动作 -> 基础推荐 -> 展示”的最小闭环。

状态：已完成，作为后续阶段的稳定规则基础维护。后续只做缺陷修复、测试补强和日志字段稳定，不再扩大 Phase 1 范围。

更收敛的 Phase 1 数据流：

```text
模拟/手动输入 -> GameState -> 合法动作/基础建议 -> JSONL 日志 -> CLI 输出
```

范围：

- 手牌输入。
- 上一手牌输入。
- 牌面解析和规范化。
- 牌型识别。
- 合法出牌判断。
- 可行动作生成。
- 简单推荐策略。
- CLI 或最小 UI 展示。
- 单元测试覆盖基础牌型和动作生成。

测试边界：

- 测牌面解析。
- 测牌型识别。
- 测牌型比较。
- 测合法出牌生成。
- 测推荐策略确定性。
- 测 CLI smoke test。
- 不测 YOLO、截图、实时 pipeline、GUI、云部署、视觉 fixture、回放系统或性能压测。

验收标准：

- 能从手动输入生成 `GameStateSnapshot` 或等价结构化状态。
- 能生成合法动作列表。
- 能输出一个基础建议和理由。
- 能写入最小 JSONL 日志。
- 有规则引擎单元测试。
- 不依赖真实 CV、GUI、Agent、模型权重或 Docker。

Phase 1 最小日志字段：

- `event`
- `input_cards`
- `last_play`
- `candidate_count`
- `recommended_action`
- `reason`
- `warnings`

暂不做：

- 真实 YOLO 检测。
- 强化学习。
- 自动鼠标操作。
- 复杂对局记忆。
- 云部署。
- 高复杂 GUI。
- 多平台适配。

### Phase 2：CV 检测接入

目标：把手动输入替换或补充为屏幕/截图识别输入。当前 Mac 本地 replay 闭环已完成，已经能从手牌 ROI 或 crop 识别牌面，并接入 Phase 1 的规则引擎输出候选动作和推荐理由。

状态：已实现，当前进入收尾和交接阶段。当前小样本评估为 train `1589/1589 = 100%`、val `422/422 = 100%`、test `99/99 = 100%`、`error_count = 0`。该准确率只代表本地 ROI/CNN 闭环验收，不代表真实游戏窗口泛化准确率。

核心内容：

- 截屏和 ROI。
- 固定步长切牌。
- PyTorch 小 CNN 分类训练与推理。
- 导出 `models/card_cnn.pt` 和 `models/card_cnn.onnx`。
- 视觉结果到 `CardObservation`。
- replay 接入 `GameStateSnapshot`、合法动作生成和推荐输出。
- 固定截图/crop 测试。

环境策略：CV 代码、数据格式、推理接口、测试、小 CNN 训练、模型导出和 replay 演示都优先在 Mac 本地完成。Windows/WSL 只保留为后续可选的性能增强资源，不是 Phase 2 的默认依赖。

### Phase 3：实时系统

目标：把截图、识别、状态、决策、展示串成可刷新系统。

状态：第一版已实现。当前支持 Mac 固定 ROI 截屏、内存切牌、CNN 推理、`GameStateSnapshot` 构造、合法动作/推荐、终端实时面板和 JSONL 日志。第一版未包含的窗口定位和跨帧稳定识别已在 Phase 3.5 补齐；复杂 GUI、蒙特卡洛胜率估计和自动操作仍未实现。

核心内容：

- Pipeline/Runtime。
- 固定截图源或窗口源。
- ROI 到 CNN 识别的周期性刷新。
- 状态刷新。
- 终端实时面板。
- JSONL 日志。
- 单帧延迟统计。

Phase 3 第一版数据流：

```text
固定截图源/窗口源 -> ROI -> CNN 识别 -> 状态刷新 -> 推荐输出 -> JSONL 日志
```

### Phase 3.5：窗口定位与稳定识别

目标：把 Phase 3 的手动 ROI 参数升级为可保存的本地窗口配置，并降低单帧识别抖动。

状态：已实现。当前支持 macOS `System Events` 查找斗地主窗口，生成 `configs/phase3_runtime.local.json`，运行时读取 config 或通过 `--auto-window` 重新换算 ROI，并用最近 N 帧对每张牌做多数/置信度投票。暂不跟踪上一手牌、轮次或对手出牌事件。

Phase 3.5 数据流：

```text
窗口定位 -> ROI 配置 -> 截屏 -> 单帧识别 -> 稳定投票 -> 规则推荐 -> JSONL 日志
```

### Phase 4：智能决策增强

目标：在规则引擎稳定后提升推荐质量。

状态：已实现显式事件/离线 replay 版本。当前具备可观测牌局状态、54 张牌守恒、未知牌池、对手剩余牌均匀采样、三人团队 rollout、Top-K 策略评分、风险字段和 JSONL 日志。Phase 3 默认仍使用快速确定性策略，不被蒙特卡洛阻塞。

核心内容：

- 蒙特卡洛模拟。
- 对手剩余牌估计。
- 策略评分。
- 推荐理由解释。
- 预留 RL 接口。

Phase 4 数据流：

```text
显式 game/play/pass 事件 -> ObservableGameState -> 对手牌 sampled worlds
  -> 三人 rollout -> ActionEvaluation -> Top-K 推荐/理由/风险 -> JSONL
```

边界：当前实时视觉没有对手出牌 ROI、过牌信号和剩余张数 OCR，因此不能自动生成完整事件流。Phase 4 的概率模型是固定 seed 的均匀剩余牌基线，不等同于精确对手牌预测；未确认的低置信度事件会把状态标记为 `uncertain` 并阻断推荐，信息不足的确定性回退不输出伪胜率。

### Phase 5：工程化展示

目标：让项目适合申请材料和实习简历展示。

状态：Phase 5A 已完成；Phase 5B 的本地只读展示、GIF 生成和真实窗口 holdout 评测流程已完成，实际 holdout 指标等待独立数据采集。

Phase 5A 已完成：

- 三个固定 JSONL 场景和固定 seed 决策指纹。
- JSON、Markdown、HTML 一键 Showcase。
- 历史 CNN 指标摘要与 Phase 4 延迟基准证据。
- Python 3.10/3.12 CI、依赖分层和 CPU Docker Demo。
- 精简架构图、录屏说明和中英文作品集材料。

Phase 5B 已增加最小只读 Web API/UI、可重复生成的本地 Demo GIF，以及真实窗口独立 holdout 的 manifest/错误报告流程。真实窗口独立 holdout 的数值结果必须等待未参与训练的标注数据，不得用历史固定 ROI 指标替代。

### Phase 6：完整场面感知与实时胜率助手

状态：核心代码、CLI 和自动化测试已实现；真实对局指标等待 5–10 局独立录制数据。

```text
macOS WindowServer window capture + Retina geometry -> LiveLayoutConfig ROIs -> SceneObservation
  -> VisualEventTracker -> ObservableGameState
  -> async Monte Carlo -> LiveDecisionRecord -> read-only Tk overlay / JSONL
```

关键边界：

- 仅支持当前 Mac 经典玩法窗口；按 Window ID 抓取，不受普通窗口遮挡影响；从完整新局开始跟踪。
- 助手可先于游戏启动；窗口未创建时持续等待，最小化时暂停识别。活动牌局发生窗口丢失时进入不确定状态，避免恢复后沿用可能漏事件的旧状态。
- 视觉事件必须连续稳定并符合当前行动者、合法牌型和余牌变化。
- 任何漏事件、低置信度或牌数冲突都切换为 `uncertain`，不继续输出伪胜率。
- 推荐按估计团队胜率排序，策略分只作为解释字段。
- UI 只消费运行快照，不承载截图、状态更新或决策主循环。
- 不自动点击；均匀未知牌模型仍属于可解释概率基线。

## 6. 逐层模块职责

### `src/capture/`

职责：屏幕截图、窗口定位、ROI 裁剪、帧率控制和输入源抽象。

合理性：独立成包是正确的，因为截图逻辑与模型推理、状态推断、GUI 展示都应解耦。

建议：后续定义统一 `Frame` 对象，包含图像、时间戳、屏幕坐标、ROI 元数据和来源类型。不要让下游模块直接依赖具体截图库，例如 `mss`、`pyautogui` 或平台 API。

### `src/vision/`

职责：牌面检测、裁剪分类、视觉后处理、模型加载、ONNX/Torch 推理封装。

合理性：这是项目最重要的 AI 模块之一，当前单独放置合理。未来应拆成 `detector.py`、`classifier.py`、`preprocess.py`、`postprocess.py`、`schemas.py`。

风险：如果直接返回游戏语义，例如“我方手牌可出顺子”，会和 `state/logic` 产生高耦合。CV 层应只返回 `Detection`、`Classification`、`CardObservation`、置信度和坐标。

扩展建议：保留两阶段路线：YOLO 检测定位 + 分类器识别牌面。这样便于调试、数据标注和指标拆分。后续可以扩展为端到端检测类别模型，但不要过早绑定。

### `src/tracking/`

职责：多帧目标跟踪、去重、置信度融合、ID 稳定和时间窗口投票。

合理性：独立模块非常必要，实时 CV 项目中单帧识别抖动是常见问题。

建议：跟踪层输入视觉观测，输出稳定观测，不输出游戏状态。可先实现轻量规则融合，后续再接 ByteTrack 或 OC-SORT。评测指标应包括 ID 稳定性、漏检率、重复率和延迟。

### `src/state/`

职责：牌编码、bitboard/计数表示、手牌状态、已出牌历史、未知牌集合、合法性校验。

合理性：这是连接 CV 与决策的核心边界。单独成包正确。

风险：状态模块容易同时吸收 CV 坐标、规则判断和 UI 展示，必须保持纯数据和状态转换职责。

建议：定义 `CardRank`、`CardSet`、`GameState`、`ObservationState`。状态更新应支持置信度和不确定性，因为真实 CV 会有误检和漏检。

### `src/logic/`

职责：斗地主规则、牌型识别、合法动作生成、压制判断、启发式策略、蒙特卡洛模拟和建议排序。

合理性：当前命名清晰，适合承载核心决策能力。

风险：蒙特卡洛模拟如果直接依赖实时截图或 GUI，会难以测试。规则引擎必须可离线单测。

建议：先做确定性规则和动作生成，再做随机模拟。每个建议应输出动作、胜率、置信度、主要理由和风险提示，方便 GUI 与 Agent 解释。

### `src/ui/`

职责：CLI、TUI、Web API、桌面 GUI 或前端接口。

合理性：作为展示层合理，但不应放业务主逻辑。

建议：后续可以分为 `cli.py`、`api.py`、`desktop.py`。如果做 Web GUI，建议 FastAPI + WebSocket 提供实时状态流，前端只负责渲染和交互。

### `src/reporting/`

职责：把已完成的离线决策结果汇总为可追溯证据，不改变状态或算法输出。

当前实现：Phase 5A 使用纯 Python 生成 JSON、Markdown 和自包含 HTML，记录环境、固定 seed、决策指纹、Top-K、延迟和风险字段；输出默认位于被忽略的 `runs/`。

### `src/config/`

职责：配置模型、配置加载、热更新和日志参数。

合理性：已有 Pydantic 配置是好的开始。

当前实现已用 `LoggingConfig.json_output` 消除 Pydantic 属性遮蔽，并通过 alias 兼容配置文件中的 `json` 字段。`ConfigManager` 仍要求 `Path`，README/示例需保持一致。

建议：随着项目扩展，配置应拆分为 `CaptureConfig`、`VisionConfig`、`TrackingConfig`、`StateConfig`、`DecisionConfig`、`RuntimeConfig`、`UIConfig`、`AgentConfig`、`LoggingConfig`。

## 7. 建议新增模块

### `src/pipeline/` 或 `src/runtime/`

这是后续实时系统最应该补的模块。

职责：串联截图、推理、跟踪、状态更新、决策和 UI 推送；管理异步队列、线程池、背压、FPS 和错误恢复。

当前 Phase 3/3.5 已新增 `src/pipeline/runtime.py`、`calibration.py` 和 `stabilizer.py`，职责收敛为固定 ROI 截屏、窗口标定、内存切牌、CNN 分类、跨帧投票、规则推荐、终端事件和 JSONL 日志。后续再逐步扩展到对局事件跟踪、异步队列和 GUI 推送。

建议数据流：

```text
Frame
  -> DetectionBatch
  -> CardObservationBatch
  -> StableObservation
  -> GameStateSnapshot
  -> DecisionResult
  -> UIEvent / LogEvent
```

### `src/agent/`

职责：Agent 工作流、工具调用、自然语言解释、复盘总结、实验助手、多 Agent 协作边界。

建议不要让 Agent 直接控制核心出牌逻辑。更好的方式是让 Agent 消费 `GameStateSnapshot`、`DecisionResult` 和日志，负责解释、复盘、调参建议和开发辅助。

Phase 1 不创建该模块；等规则、状态和日志稳定后再引入。

### `scripts/`

职责：数据合成、训练、导出 ONNX、回放、评测、性能测试和 Showcase 构建。

当前 Phase 5A 入口是 `scripts/run_phase5_showcase.py`，使用三个已提交事件场景生成可复现决策证据。

建议脚本：

- `synth_generate.py`
- `train_detector.py`
- `export_onnx.py`
- `evaluate_vision.py`
- `replay_session.py`
- `benchmark_runtime.py`

### `tests/`

建议根目录使用 `tests/`，不要长期放在 `src/tests/`。核心测试包括：

- 牌编码和牌型判断单元测试
- 合法动作生成测试
- 固定截图视觉推理测试
- 回放日志集成测试
- 决策稳定性和性能预算测试

### `docs/`

用于申请和实习展示。当前已包含：

- 系统架构图
- 本地小样本 CNN 指标摘要与 manifest 哈希
- Phase 4 三场景延迟/指纹证据
- Demo 运行与录屏说明
- 中英文作品集文案、已知风险和诚实边界

## 8. 数据流设计建议

实时系统不要把每个模块直接互相调用成网状结构。建议使用明确事件对象和快照对象：

```text
ScreenFrame
VisionResult
TrackingResult
ObservationUpdate
GameStateSnapshot
DecisionRequest
DecisionResult
ExplanationEvent
```

每个对象都应带：

- `timestamp`
- `frame_id` 或 `session_id`
- `source`
- `confidence`
- `latency_ms`
- `warnings`

这会显著提升日志复盘、回放评测和简历展示质量。

## 9. 工程化建议

短期优先级：

1. 保持 Phase 1 的 `GameState -> 合法动作 -> 基础推荐 -> CLI 输出 -> JSONL 日志` 闭环稳定。
2. 保持 Phase 2 的固定 ROI、CNN 识别、replay 和评估脚本可复现。
3. 进入 Phase 3 前，先定义实时 `Frame`、`CardObservation`、状态刷新事件和 JSONL 日志字段。
4. 继续补充真实截图 fixture，验证当前 100% 小样本准确率的泛化边界。
5. `ruff`、`mypy` 或 `pyright` 放到后续工程化阶段。

中期优先级：

1. 建立合成数据和少量真实截图标注流程。
2. 先训练小 CNN 牌面分类器并导出 ONNX；YOLOv8n 或 YOLOv11n 检测器只作为后续扩展路线。
3. 建立视觉指标：mAP、precision、recall、分类 Top-1/Top-3、端到端牌面准确率。
4. 建立实时指标：平均延迟、P95 延迟、FPS、掉帧率。
5. 建立日志回放工具，让 Demo 可复现。

长期优先级：

1. Docker 化评测和 API 服务；训练容器化等 Phase 2/3 稳定后再评估。
2. 支持 WebSocket 实时前端或桌面 overlay。
3. 引入实验追踪，例如 MLflow、Weights & Biases 或本地 JSONL 指标。
4. 将 Agent 作为解释器和实验助手接入，而不是让 Agent 替代规则引擎。

## 10. 技术栈建议

### CV

- 初期：OpenCV + Ultralytics YOLO + PyTorch
- 部署：ONNX Runtime
- 性能优化：批量裁剪、ROI 缓存、模型 warmup、异步推理
- 可选：TensorRT 或 NCNN，等基础 pipeline 稳定后再做

### 实时推理

- 初期：Python `asyncio` 或 `queue.Queue` + worker threads
- API：FastAPI + WebSocket
- 数据对象：Pydantic 或 dataclass，核心热路径可转 dataclass 减少开销

### 决策

- 规则引擎：纯 Python，强单测
- 模拟：蒙特卡洛随机 rollout，后续可加入 MCTS
- 性能：关键枚举可用 NumPy、Numba 或 Cython 优化，但不要过早优化

### GUI

- 快速 Demo：CLI/TUI + WebSocket JSON
- 作品集展示：FastAPI + React/Vite
- 桌面 overlay：后续再考虑 PySide6、Qt 或系统级透明窗口

### Docker

Phase 5A 已提供 core-only CPU `Dockerfile`，通过 `.dockerignore` 排除 `.venv`、`data`、`models`、`logs` 和本地配置。容器只承诺 Phase 1/4/5 离线 Showcase，不承诺 macOS 截屏或模型推理。

模型权重不直接进 Git；如 Phase 5B 需要公开视觉 Demo，应使用带 SHA256 的 release artifact 或只读挂载。

## 11. 申请和实习竞争力判断

作为港新 AI 硕士申请项目，本项目有潜力，因为它覆盖 CV、实时系统、规则推理、可解释 AI 和工程部署。要增强说服力，需要展示：

- 自建或合成数据流程
- 训练和评测指标
- 端到端 Demo
- 失败案例分析
- 延迟和性能指标
- 清晰系统架构图

作为 AI 应用开发实习项目，本项目也合适，因为它体现：

- 真实输入源处理
- 模型推理服务化
- 配置、日志、回放和评测
- 前后端实时交互
- 工程可维护性和部署意识

Phase 5A/5B 已补齐可复现 Showcase、三类事件场景、指标证据、CI、CPU Docker、本地只读 Web/API、GIF 和真实窗口 holdout 工具链。Phase 6 已补齐对手出牌/过牌/剩余张数事件接口、严格状态门控和置顶展示。当前主要短板是尚未采集满足覆盖要求的 5–10 局完整数据，因此真实事件 F1 和整局跟踪成功率仍未知；模型 Release 也尚未公开。

## 12. 当前架构结论

当前目录划分已经形成 Phase 1–6 的可测试闭环：规则、CNN/replay、Retina 窗口、场面识别、视觉事件状态、异步蒙特卡洛、置顶 UI，以及可复现报告/CI/Docker。最需要避免的是在真实完整对局 holdout 尚未通过时，把代码闭环描述成已经验证的高准确率自动整局跟踪。

推荐下一阶段目标（真实窗口验证与事件感知）：

```text
录制 5–10 局 -> 模板/牌面微调 -> 完整 replay holdout -> 错例修正 -> 可选模型 Release
```

下一阶段先取得真实窗口证据，再决定是否调整 CNN；自动操作和强化学习仍不进入当前路线。
