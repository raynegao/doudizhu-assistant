# doudizhu-assistant

AI-based Dou Dizhu assistant with Mac screen capture, CNN card recognition, a rule engine, real-time recommendations, and multi-frame stabilization.

[![CI](https://github.com/raynegao/doudizhu-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/raynegao/doudizhu-assistant/actions/workflows/ci.yml)

一个**只读**的斗地主实时助手：抓取 macOS 上斗地主客户端的窗口画面，用纯 CV + 小 CNN 识别三家出牌、过牌、角色和余牌，把稳定的视觉事件送进带 54 张牌守恒约束的状态机，再用蒙特卡洛 rollout 给出带 95% 置信区间的 Top-3 出牌建议，显示在一个置顶小窗里。

它**不会**点牌、不会代打、不会修改游戏客户端；所有推荐只在最新画面手牌与状态 revision 完全一致时才展示。

## 现状一览

| 维度 | 状态 | 证据 |
|---|---|---|
| 规则引擎 / 合法动作 / 牌型比较 | 完成，CLI 可独立使用 | `tests/`，`python -m src.ui.cli` |
| CNN 牌面分类（固定 ROI 切牌） | 完成；本地 train/val/test 100%，**不是**真实窗口泛化准确率 | `models/card_cnn.metrics.json`（不入库） |
| Mac 实时流水线（ScreenCaptureKit 持久流 → 识别 → 状态 → 决策 → Tk 置顶窗） | 完成；热态采集约 9.8 ms，完整识别中位数约 128 ms | [PHASE6_LIVE_ASSISTANT.md](docs/PHASE6_LIVE_ASSISTANT.md) |
| 蒙特卡洛 Top-K + 置信区间 | 完成；对手模型为**均匀未知牌抽样**，是可解释基线而非行为模型 | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 无损录制 / 逐帧 SHA256 / 盲标封印 / replay 评测 | 完成 | [PHASE6_ACCEPTANCE.md](docs/PHASE6_ACCEPTANCE.md) |
| AI 参考标注回归（六局 11,025 帧） | 动作 188/188、余牌 557/564、严格完整局 6/6 | [AI_REFERENCE_FIXES_20260908_V11.md](docs/AI_REFERENCE_FIXES_20260908_V11.md) |
| **独立盲测验收**（5–10 局新录制 + 300 张牌面 holdout） | **未完成** | `make phase6-acceptance` 当前预期返回非零 |

关于最后两行要说清楚：2026-09-07 冻结的五局 AI 参考基线最初整局成功 **0/5**，之后据失败原帧做了 v8–v11 四轮修复，回归到 6/6。这批数据因此已参与调参，属于**回归证据**，不能当成泛化指标；真实准确率仍要等代码冻结后录制全新对局。详见 [AI_REFERENCE_ACCEPTANCE_20260907.md](docs/AI_REFERENCE_ACCEPTANCE_20260907.md)。

## 一分钟 Demo

不需要模型权重、原始数据或游戏窗口：

```bash
source .venv/bin/activate
make demo
open runs/showcase/index.html
```

生成机器可读 `report.json`、CI 摘要 `summary.md` 和完全离线的 `index.html`，覆盖地主主动、地主响应、农民响应三个固定事件场景，并验证固定 seed 的决策指纹。更多见 [SHOWCASE.md](docs/SHOWCASE.md)。

## 架构

```text
capture ──> vision ──> tracking ──> state ──> logic ──> ui / reporting
 窗口帧     SceneObservation  稳定事件   ObservableGameState  合法动作+MC   Tk/CLI/Web/JSONL
```

| 层 | 目录 | 职责 | 禁止 |
|---|---|---|---|
| capture | `src/capture/` | ScreenCaptureKit / WindowServer 窗口采集、Retina 几何、无损录制 | 识别牌、选动作 |
| config | `src/config/` | Pydantic 配置模型，含 `LiveLayoutConfig` 和各视觉阈值 | 引入运行时编排 |
| vision | `src/vision/` | 牌面、角色、过牌、回合控件、余牌的纯 CV / CNN 观测 + 置信度 | 导入 `pipeline`、调用规则 |
| tracking | `src/tracking/` | 多帧融合，只输出稳定、有序的视觉事件 | 模拟对手、渲染 UI |
| state | `src/state/` | 事件 reducer、54 张牌守恒、未知历史牌池、团队结果语义 | 读图像、依赖模型框架 |
| logic | `src/logic/` | 牌型、合法动作、对手采样、蒙特卡洛、推荐解释 | 读屏幕坐标、GUI 控件 |
| pipeline | `src/pipeline/` | 实时主循环、标定、异步决策、录制写入 | 把规则藏进循环 |
| ui | `src/ui/` | CLI、只读 Web、Tk 置顶窗 | 持有采集或状态变更 |
| reporting | `src/reporting/` | Showcase、诊断、证据会话、聚合验收 | 改游戏状态或模型输出 |

完整边界、运行时安全门和验收拒绝条件见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 快速开始

需要 Python ≥ 3.10（本地开发用 3.12），macOS 实时功能需要 Apple Silicon Mac 和屏幕录制权限。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
make quality
```

`make quality` = Ruff + 核心层 Mypy + 带分支覆盖率的完整 Pytest（门槛 65%，当前约 78%）。`make macos-smoke` 用真实 Tk / spawn / Swift 编译验证桌面集成但不读屏幕。本地存在 `.venv/bin/python` 时 Makefile 自动使用它。

### 离线使用：规则引擎 CLI

```bash
python -m src.ui.cli --hand "3 3 4 4 5 5 6 6 7 8 9 SJ BJ" --last-play "5 5"
```

牌面记法：`3 4 5 6 7 8 9 10 J Q K A 2`，小王 `SJ`，大王 `BJ`。

### 离线使用：事件 replay + 蒙特卡洛

```bash
python -m scripts.run_phase4_decision \
  --events-file examples/phase4_round.jsonl \
  --simulations 200 --seed 20260721 --top-k 3
```

只读 Web 展示：`make web-demo` 后打开 http://127.0.0.1:8765。

### 实时使用：macOS 置顶助手

1. 打开斗地主客户端并进入牌局，标定窗口与 ROI：
   ```bash
   make live-calibrate
   ```
2. 启动助手（默认处于"扫描已停止"，点击"开始扫描"后才创建截图/识别进程）：
   ```bash
   make live-assistant
   ```
3. 终端调试模式：
   ```bash
   python -m scripts.run_live_assistant --config configs/live_game.local.json --no-ui --start-scanning --no-clear
   ```

助手可以先于游戏启动，也可以在抢地主、加倍完成后再启动；窗口最小化会暂停推荐，恢复后在下一次自己回合自动重扫。模板采集、断点恢复、诊断证据和日志策略见 [PHASE6_LIVE_ASSISTANT.md](docs/PHASE6_LIVE_ASSISTANT.md)。

### 录制与验收

```bash
python -m scripts.record_live_game --config configs/live_game.local.json \
  --session acceptance-001 --evidence-split acceptance --until-interrupt
make live-finalize SESSION=acceptance-001
make live-annotate-prepare SESSION=acceptance-001   # 盲标，只看原始帧 contact sheet
make live-annotate-seal SESSION=acceptance-001
make phase6-acceptance                              # 数据不足时返回非零
```

正式录制会封印 Python / Swift 采集器 / FFmpeg / 模型 / 模板指纹；封印后任何实现改动都会让旧正式局失效，防止针对验收局调参。流程、阈值与拒绝条件见 [PHASE6_ACCEPTANCE.md](docs/PHASE6_ACCEPTANCE.md)。

## 常用命令

| 命令 | 用途 |
|---|---|
| `make quality` / `make macos-smoke` | 质量门禁 / 桌面集成冒烟 |
| `make demo` / `make benchmark` / `make docker-demo` | 可复现 Showcase、基准、CPU Docker |
| `make web-demo` / `make demo-gif` | 只读 Web 展示、Demo GIF |
| `make live-calibrate` / `make live-assistant` | 标定、启动实时助手 |
| `make live-finalize` / `live-annotate-prepare` / `live-annotate-seal` | 录制封存与盲标 |
| `make live-diagnostics` / `make live-storage` | 分析最近证据会话 / 只读盘点存储 |
| `make holdout-seal` / `make holdout-evaluate` | 独立牌面 holdout 封存与评测 |
| `make phase6-acceptance` | 聚合验收门禁 |

## 仓库结构

```text
src/           分层实现（见上表）
scripts/       训练、导出、录制、replay、标注、诊断、验收、Showcase 的 CLI
tests/         单元、集成、状态不变量、蒙特卡洛、生命周期、证据指纹、CLI 回放（约 2000 项）
native/        macOS ScreenCaptureKit 持久流 Swift 源码；二进制按源码哈希本地缓存
configs/       配置模板；*.local.json 为本机配置，不入库
docs/          架构、评测、验收、修复报告、作品集文案
examples/      内置事件 replay 样例
data/ models/ logs/ runs/   数据、模型、日志、产物，均不入库
```

## 文档索引

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 分层边界、运行时安全、决策语义、验收拒绝条件
- [PHASE6_LIVE_ASSISTANT.md](docs/PHASE6_LIVE_ASSISTANT.md) — 实时助手标定、模板、录制、运行、状态保护
- [PHASE6_ACCEPTANCE.md](docs/PHASE6_ACCEPTANCE.md) — 正式录制、盲标、replay、聚合门禁
- [EVALUATION.md](docs/EVALUATION.md) — 评测口径与基准
- [REAL_WINDOW_HOLDOUT.md](docs/REAL_WINDOW_HOLDOUT.md) — 独立牌面 holdout 采集与评测
- [SHOWCASE.md](docs/SHOWCASE.md) — Demo 与录屏说明
- [PORTFOLIO.md](docs/PORTFOLIO.md) — 中英文作品集 / 简历描述
- [RELIABILITY_REVIEW_20260905.md](docs/RELIABILITY_REVIEW_20260905.md) — 可靠性修复验收表
- AI 参考回归：[ACCEPTANCE_20260907](docs/AI_REFERENCE_ACCEPTANCE_20260907.md) · [ACCEPTANCE_20260907_003](docs/AI_REFERENCE_ACCEPTANCE_20260907_003.md) · 修复 [v8](docs/AI_REFERENCE_FIXES_20260907_V8.md) · [v9](docs/AI_REFERENCE_FIXES_20260907_V9.md) · [v10](docs/AI_REFERENCE_FIXES_20260908_V10.md) · [v11](docs/AI_REFERENCE_FIXES_20260908_V11.md)
- [PHASE_HISTORY.md](docs/PHASE_HISTORY.md) — Phase 1–6 的逐阶段命令、参数与实测记录（原 README 主体）

## 已知限制

- **单皮肤绑定**：几何、颜色阈值、Joker 墨色、回合控件实体均针对当前 macOS 斗地主客户端的经典布局标定；换皮肤或平台需要重新标定并重新验证，不宣称跨皮肤泛化。
- **对手模型是均匀基线**：未知牌均匀抽样，不建模对手行为；UI 会标出 `uniform_opponent_model` 风险，Top 候选置信区间重叠时标记排序不稳定。
- **中途启动是近似模型**：看不见的历史牌只记 `hidden_played_count` 并进入未知采样池，不伪造牌点。
- **真实准确率未独立验证**：所有现有数值均为本地评估或已参与调参的回归数据；`make phase6-acceptance` 在新独立数据齐全前预期返回非零。
- 不做自动点击、强化学习、云部署或多平台适配。

## 配置与日志

- 配置支持 YAML / JSON，示例见 `configs/app.example.yaml`、`configs/live_game.example.json`；`src/config/settings.py` 提供 Pydantic 校验与热重载轮询。
- `src/config/logging_config.py` 提供文本 / JSON 日志（控制台 + 可选文件）；实时 JSONL 按内容变化与每 20 帧心跳写入，单文件 64 MiB 后归档。
- 本机桌面快捷方式 `斗地主助手.command` 转交 `scripts/launch_live_assistant_macos.command`，原子锁防重复实例。
