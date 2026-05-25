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

当前 Phase 2 路线：固定 ROI + 规则切牌 + PyTorch CNN 分类。验收目标是 Mac 本地 replay 闭环：ROI/crop -> CNN 识别手牌 -> Phase 1 规则引擎 -> 推荐动作。

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
