# 斗地主 AI 辅助系统架构评审

## 1. 项目定位

本项目目标是构建一个实时斗地主 AI 辅助系统：从屏幕采集牌面，通过 CV 模型识别手牌与牌局信息，维护游戏状态，再用规则引擎、蒙特卡洛模拟或 Agent 工作流生成出牌建议、胜率估计和可解释理由。

从 AI 应用工程角度看，这不是单一模型项目，而是一个端到端 AI 系统项目。它适合作为港新 AI 硕士申请项目和 AI 应用开发实习项目，但需要持续补齐数据闭环、实时推理、评测指标、工程部署和可解释展示。

## 2. 当前结构评审

当前仓库已经完成 Phase 1 规则引擎 MVP 的最小闭环：

```text
configs/
  app.example.yaml
tests/
  test_cards.py
  test_decision_cli.py
  test_rules.py
src/
  capture/
  config/
  logic/
    decision.py
    rules.py
  state/
    cards.py
    game_state.py
  tracking/
  ui/
    cli.py
  vision/
README.md
PLAN.md
pytest.ini
requirements-dev.txt
```

整体模块方向合理，已经把采集、视觉、跟踪、状态、决策和 UI 分成不同包，符合 AI 工程项目的基本分层意识。当前已实现手动输入到 `GameStateSnapshot`、合法动作生成、基础推荐、CLI 输出和 JSONL 日志。主要缺口是 Phase 2 及之后的 CV 检测、实时推理编排层、数据集/训练/评测层、Agent 工作流层和部署层。后续如果直接在 `vision`、`logic` 或 `ui` 里堆主循环，耦合会快速上升。

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

状态：已实现，当前进入收尾维护阶段。后续只做缺陷修复、测试补强、日志字段稳定和文档同步，不再扩大 Phase 1 范围。

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

目标：把手动输入替换或补充为屏幕/截图识别输入。当前实现重点是 Mac 本地 replay 闭环，先做手牌区域识别和牌面结构转换，再考虑整局状态。

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

核心内容：

- Pipeline/Runtime。
- 状态刷新。
- 简单 GUI 展示。
- JSONL 日志。
- 回放机制。
- 延迟统计。

### Phase 4：智能决策增强

目标：在规则引擎稳定后提升推荐质量。

核心内容：

- 蒙特卡洛模拟。
- 对手剩余牌估计。
- 策略评分。
- 推荐理由解释。
- 预留 RL 接口。

### Phase 5：工程化展示

目标：让项目适合申请材料和实习简历展示。

核心内容：

- README。
- 架构图。
- Demo 视频或截图说明。
- Docker。
- 测试和指标报告。
- 项目总结和简历描述。

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

### `src/config/`

职责：配置模型、配置加载、热更新和日志参数。

合理性：已有 Pydantic 配置是好的开始。

问题：当前 `LoggingConfig.json` 字段名会遮蔽 Pydantic `BaseModel.json` 方法，建议后续改为 `json_format` 或 `structured`。另外 `ConfigManager` 当前要求 `Path`，README/示例要保持一致。

建议：随着项目扩展，配置应拆分为 `CaptureConfig`、`VisionConfig`、`TrackingConfig`、`StateConfig`、`DecisionConfig`、`RuntimeConfig`、`UIConfig`、`AgentConfig`、`LoggingConfig`。

## 7. 建议新增模块

### `src/pipeline/` 或 `src/runtime/`

这是后续实时系统最应该补的模块。

职责：串联截图、推理、跟踪、状态更新、决策和 UI 推送；管理异步队列、线程池、背压、FPS 和错误恢复。

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

职责：数据合成、训练、导出 ONNX、回放、评测、性能测试。

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

用于申请和实习展示。建议包含：

- 系统架构图
- 数据集与标注流程
- 模型训练与评测报告
- 实时推理延迟报告
- Demo 截图/视频说明
- 失败案例分析

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
2. 补齐 Phase 1 缺陷修复和边界测试，不扩大到真实 CV、GUI、Docker 或 RL。
3. 进入 Phase 2 前，先定义 `Frame`、`Detection`、`CardObservation` 等视觉输入对象边界。
4. 建立少量固定截图或手工 fixture，为后续 CV 检测接入准备验收样本。
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

Phase 1 不做 Docker。先规划 `.dockerignore` 排除 `.venv`、`data`、`models`、`logs`。

Phase 2/3 再考虑：

- `Dockerfile`：CPU API/评测环境
- `docker-compose.yml`：API + 前端 + 日志目录挂载

模型权重不要直接进 Git，可用下载脚本或 release artifact。

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

目前短板是还没有真实模型、数据闭环、实时 pipeline、Docker 和作品集级 Demo。Phase 1 已有 CLI 级可运行闭环，后续开发应优先把 CV 输入接到这个稳定闭环上，而不是先追求复杂模型。

## 12. 当前架构结论

当前目录划分是一个合格的起点，模块方向正确，Phase 1 已经具备手动输入规则引擎闭环，但还不是完整 AI 工程项目。最需要避免的是把实时主循环、CV 推理、状态更新、决策和 UI 混在同一个模块里。下一步应先补齐 CV 输入对象、固定截图验收样本和检测后处理边界，再进入实时 pipeline 和 GUI 展示。

推荐下一阶段目标：

```text
截图/ROI -> 视觉观测 -> CardObservation -> GameStateSnapshot -> 规则决策 -> JSONL 日志
```

这个阶段只把视觉输入接入结构化状态，继续复用 Phase 1 的规则、推荐和日志闭环，工程风险会低很多。
