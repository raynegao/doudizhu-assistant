# 斗地主助手阶段计划

## 当前进度

当前项目处于 Phase 1 收尾阶段。规则引擎 MVP 已经跑通：

- 手动输入手牌和上一手牌。
- 构造 `GameStateSnapshot`。
- 解析和校验牌面。
- 识别基础牌型并比较大小。
- 生成合法动作集合。
- 输出基础推荐动作和中文理由。
- 通过 CLI 展示结果。
- 写入最小 JSONL 日志。
- 使用根目录 `tests/` 覆盖牌面解析、规则、推荐和 CLI smoke。

Phase 1 收尾只做缺陷修复、测试补强、文档同步和日志字段稳定，不新增 CV、GUI、Docker、RL 或复杂对局记忆。

## Phase 1：规则引擎 MVP

状态：已实现，收尾维护中。

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

建议范围：

- 新增 `src/pipeline/` 或 `src/runtime/`。
- 管理异步队列、FPS、错误恢复和日志。
- 加入多帧融合和基础回放。
- 输出延迟、候选动作和推荐日志。

## Phase 4：智能决策增强

目标：在规则引擎稳定后提升推荐质量。

建议范围：

- 蒙特卡洛模拟。
- 对手剩余牌估计。
- 策略评分。
- Top-K 推荐和风险提示。
- 保留 RL 接口，但不把强化学习作为 MVP。

## Phase 5：工程化展示

目标：让项目适合申请材料和实习简历展示。

建议范围：

- README 和系统架构图。
- Demo 截图或视频说明。
- 数据、模型和评测指标报告。
- 延迟/FPS/准确率统计。
- Docker 或本地一键运行脚本。
- 项目总结和简历描述。
