# Phase 6 真实对局验收

Phase 6 只有在独立完整对局通过本页全部门槛后，才能声明真实窗口验收完成。代码测试、手动试玩、自动扫描成功或固定 ROI 小样本 `100%` 都不能替代这份证据。

## 目录与隔离规则

每局使用独立 session，不能把同一局拆到开发集和验收集：

```text
data/live_game/recordings/game-001/
  session.json
  config.snapshot.json
  manifest.jsonl
  video/
    segment-001.mkv
  annotation_sheets/
  annotation-workbook.json
  expected-events.jsonl
  expected-scenes.jsonl
  annotation.json

runs/live-replay/game-001/
  events.jsonl
  replay.json
  evaluation.json
```

录制目录会保存不可变配置快照、配置 SHA256、完整帧 RGB 像素 SHA256 和各 ROI 的解码像素 SHA256。正式 `acceptance` 局把 10 FPS RGB 帧写入 `libx264rgb` 无损 Matroska 分段；每次中断恢复新建一个 segment，manifest 仍逐帧记录时间戳、segment/index 和像素哈希。完整性检查先校验容器 SHA256，再顺序解码每帧复算完整图与 ROI，因此帧间压缩不会降低牌点质量或可追溯性。当前真实窗口 50 帧 smoke 为 3.4 MiB，逐帧 RGB 完全可逆；旧 PNG session 仍兼容。窗口像素优先来自持久化 ScreenCaptureKit 原生流，默认 0.1 秒周期；当前 Mac 实测中位数约 10.00 FPS、最大间隔约 0.105 秒，避免相邻帧间漏掉多个动作。replay 来源文件还会绑定 manifest、配置快照、视频容器、CNN 模型、模板目录、Python/Swift 实现源码哈希、Python/核心依赖版本和事件日志；单局评测再绑定动作标注和场景标注。正式审计拒绝跨 session 重复帧、被修改的容器/ROI/配置/模型/模板/实现环境、过期 replay 和评测后被替换的标注。

录制分为 `development` 和 `acceptance` 两类。默认是开发集，可用于排错，但聚合验收会明确排除。正式验收必须显式选择 `acceptance`；录制时会封印完整 `src/`、`scripts/` 评测工具链、模型和模板 SHA256，任何后续算法、权重、模板或验收程序改动都会让该 session 自动失效，必须重新录制。这保证最终 5 局是代码冻结后的盲测，不是针对已有牌局调参。

## 1. 录制完整对局

```bash
python -m scripts.record_live_game \
  --config configs/live_game.local.json \
  --session acceptance-001 \
  --evidence-split acceptance \
  --until-interrupt
```

对局结束后按 `Ctrl-C`。已写入的帧和 `session.json` 会安全收尾，随后由你确认这确实是一局完整对局，再执行：

```bash
make live-finalize SESSION=acceptance-001
```

同名 session 默认拒绝静默追加。录制意外中断但对局尚未结束时可改用固定帧数并加 `--resume` 继续，配置 SHA256 必须保持一致；已 finalize 的 session 不能继续追加。`--mark-complete` 只适合固定帧数恰好覆盖完整一局的自动录制，不建议日常使用。

至少准备 5 个完整 `acceptance` session，建议 5–10 局。用于定位问题、模板调整或模型微调的局必须保留为默认的 `development`；如果看过正式验收局后又修改代码、模型或模板，封印校验会失败，整批正式局需要在新版本上重录。完整 session 还必须通过采样间隔门禁，防止低帧率录屏被误当作算法漏事件。

## 2. 先盲标，再回放

正式局禁止先生成 `runs/live-replay/<session>/events.jsonl` 再补真值。先生成只包含原始录制帧和时间戳的 contact sheet：

```bash
make live-annotate-prepare SESSION=acceptance-001
```

人工只查看 `annotation_sheets/`，填写 `expected-events.jsonl` 和 `expected-scenes.jsonl`。此时不得运行 replay、不得查看助手预测日志，也不能把助手输出复制成真值。

`expected-events.jsonl` 使用运行日志中的 `play_observed`、`pass_observed` 格式，并为完整对局增加一条人工确认的结果：

```json
{"event":"round_result_detected","winner":"self","outcome":"victory"}
```

人工标注不需要抄写运行时内部 `round_id`；单局 replay 只有一个内部 round 时，评测器会把 session 别名或省略的 `round_id` 映射到该 round，避免 UUID 造成假失败。

`expected-scenes.jsonl` 必须为每个预期出牌/过牌动作至少选择一帧稳定画面，并标注三家余牌：

```json
{"frame_id":120,"after_sequence_no":4,"remaining":{"self":14,"right":13,"left":17}}
```

标注必须来自人工复核，不能直接把助手预测复制成真值。

标完后先封存：

```bash
make live-annotate-seal SESSION=acceptance-001
```

封存会校验动作顺序、牌型、逐动作余牌变化、终局赢家、帧号及 contact sheet/manifest/标注 SHA256，并生成 `annotation.json`。只要对应 replay 目录已经存在预测输出，工具就拒绝封存；正式 replay 也反向拒绝缺少有效盲标封印的 session。审计还要求 `annotation.completed_at <= replay.created_at`，并复核两边指纹，因此不能在看到预测后悄悄修改真值。

## 3. 回放和单局评测

```bash
python -m scripts.replay_live_game \
  --manifest data/live_game/recordings/game-001/manifest.jsonl \
  --output-dir runs/live-replay/game-001 \
  --quiet

python -m scripts.evaluate_live_replay \
  --predicted-log runs/live-replay/game-001/events.jsonl \
  --expected-events data/live_game/recordings/game-001/expected-events.jsonl \
  --expected-scenes data/live_game/recordings/game-001/expected-scenes.jsonl \
  --output runs/live-replay/game-001/evaluation.json \
  --require-complete-round \
  --require-thresholds
```

牌数守恒会计算 `三家余牌 + 已知已出牌 + hidden_played_count = 54`，因此中途安全扫描不会被错误判定为守恒失败。

replay 默认读取 session 内的 `config.snapshot.json`，不能在不知情的情况下用新 ROI 覆盖旧录制；如确需重跑已有输出，必须显式加 `--overwrite`。`replay.json` 会保存配置、模型、模板、manifest、实现源码、运行时版本和事件日志指纹。

## 4. 总体验收

先封存并评测独立牌面 holdout：

```bash
make holdout-seal
make holdout-evaluate
```

随后运行：

```bash
make phase6-acceptance
```

正式门槛：

- 代码冻结后新录制、带封印的独立完整 `acceptance` session ≥ 5；
- 录制目标周期 ≤0.10 秒、中位帧间隔 ≤0.15 秒、P95 ≤0.20 秒，且最大间隔不超过 session 声明门槛（默认 0.30 秒）；
- 正式局每帧均来自持久化 ScreenCaptureKit 窗口流；发生旧截图回退时立即终止该次 acceptance 录制；
- 跨 session 完整帧 SHA256 重合数为 0；
- 各 session 的原始帧时间范围互不重叠；
- 每帧完整图、由完整图重算的全部 ROI、配置快照、模型、模板和 replay 日志校验通过；
- 录制时的实现、模型、模板封印与 replay 完全一致；
- 每局人工真值在 replay 预测生成前完成盲标封存，且标注/contact sheet 指纹未变化；
- 出牌/过牌事件 F1 ≥ 95%；
- 出牌牌点整组准确率 ≥ 95%；
- 余牌准确率 ≥ 98%；
- 每个预期动作至少有一帧三家余牌人工标注；
- 54 张牌守恒检查 100% 通过；
- 整局事件和终局结果完全正确的 session 比例 ≥ 80%；
- 每个单局评测均通过；
- 每个评测报告的 schema、输入路径和三份输入 SHA256 均通过复核；
- 真实窗口牌面 holdout 在预测前完成人工标签盲封存，总体准确率 ≥ 95%、每类准确率 ≥ 90% 并达到 `publication_ready=true`，且封存模型与全部 replay 相同，训练集、采集 session、输入和输出工件 SHA256 均未变化。

审计无论通过与否都会输出 `runs/phase6-acceptance/report.json`。任一证据缺失都会明确列入 `remaining_work`，不会把“没有数据”当成通过。

## 5. 运行诊断

```bash
make live-diagnostics
```

该命令流式分析大型 JSONL，输出事件数、有效 round、动作/决策/终局覆盖、自动扫描、延迟分布和错误类别到 `runs/live-diagnostics/report.json`，用于决定下一轮应该修手牌、余牌、角色、回合控件还是窗口采集。
