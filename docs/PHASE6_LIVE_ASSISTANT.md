# Phase 6 完整场面感知与实时胜率助手

## 能力和边界

Phase 6 读取当前 Mac “斗地主”经典玩法窗口，识别三家的场上动作、角色和余牌，把稳定结果转换成既有 `ObservedAction`，再用 Phase 4 蒙特卡洛输出估计胜率最高的 Top-3。

它是只读助手，不点击游戏。胜率来自已观测牌、剩余张数和均匀未知牌采样，不是已知对手真实手牌后的精确概率。

助手必须在完整新局开始前运行。中途启动无法恢复此前已打出牌的点数，因此会等待下一局。

## 1. 标定窗口

启动游戏并确保牌桌窗口已创建（可以被其他窗口遮挡，但不要最小化或关闭）：

```bash
python -m scripts.calibrate_live_game \
  --app-name 斗地主 \
  --save-config configs/live_game.local.json
```

输出：

- `configs/live_game.local.json`：本地配置，已被 Git 忽略。
- `data/live_game/calibration/live_layout_preview.png`：窗口 ROI 叠加图。
- `data/live_game/calibration/live_layout_contact_sheet.png`：各 ROI 预览。

采集使用 WindowServer Window ID 和窗口级 `screencapture`，不会把覆盖在牌桌上的 Codex/终端窗口误截进来。先检查预览是否准确覆盖手牌、三家出牌区、左右余牌、三家角色和自己的出牌按钮。窗口移动无需重写归一化 ROI；窗口布局或缩放样式改变时需要重新标定。

## 2. 建立真实界面模板

模板目录结构：

```text
data/live_game/templates/
  pass/pass/
  pass/neutral/
  remaining/1/ ... remaining/20/
  role/landlord/
  role/farmer/
  turn/active/
  turn/inactive/
```

在画面出现对应状态时采集。例如右侧玩家显示“不出”：

```bash
python -m scripts.add_live_template \
  --config configs/live_game.local.json \
  --kind pass \
  --label pass \
  --roi right_pass
```

同一个出牌区为空时采集 `neutral`；地主/农民、左右余牌和自己的回合按钮同理：

```bash
python -m scripts.add_live_template --kind role --label landlord --roi right_role
python -m scripts.add_live_template --kind role --label farmer --roi left_role
python -m scripts.add_live_template --kind remaining --label 17 --roi left_remaining
python -m scripts.add_live_template --kind turn --label active --roi self_turn
python -m scripts.add_live_template --kind turn --label inactive --roi self_turn
```

建议每个标签从不同对局采集至少 3 个模板。模板、截图和模型都是本地数据，不提交 Git。

## 3. 录制和标注完整对局

每局单独录制：

```bash
python -m scripts.record_live_game \
  --config configs/live_game.local.json \
  --session game-001 \
  --frames 800
```

输出完整窗口、每个 ROI 和 `manifest.jsonl`。建议录制 5–10 局，按完整 session 划分模板/微调数据和 replay holdout，不能把同一局拆到两侧。

从录制帧中提取并标注场上出牌：

```bash
python -m scripts.label_live_play \
  --config configs/live_game.local.json \
  --image data/live_game/recordings/game-001/frames/000120.png \
  --seat right \
  --labels "3 3 3 4"
```

若分割出的牌数与标签数不同，脚本会拒绝写入。修正 ROI 或选择动画结束后的稳定帧再试。标注 crop 可通过现有 `scripts.add_labeled_crops_to_dataset` 加入训练数据，然后重新训练并用独立 session 评测。

对完整录制进行离线回放：

```bash
python -m scripts.replay_live_game \
  --config configs/live_game.local.json \
  --manifest data/live_game/recordings/game-005/manifest.jsonl \
  --output-dir runs/live-replay/game-005 \
  --quiet
```

将人工确认的动作保存为 `expected-events.jsonl`，格式与 `play_observed` / `pass_observed` 一致；可选的 `expected-scenes.jsonl` 每行保存 `frame_id` 和三家 `remaining`。生成验收报告：

```bash
python -m scripts.evaluate_live_replay \
  --predicted-log runs/live-replay/game-005/events.jsonl \
  --expected-events data/live_game/recordings/game-005/expected-events.jsonl \
  --expected-scenes data/live_game/recordings/game-005/expected-scenes.jsonl \
  --output runs/live-replay/game-005/evaluation.json \
  --require-thresholds
```

## 4. 运行助手

置顶小窗：

```bash
python -m scripts.run_live_assistant \
  --config configs/live_game.local.json
```

终端调试：

```bash
python -m scripts.run_live_assistant \
  --config configs/live_game.local.json \
  --no-ui \
  --no-clear
```

首次运行需要给终端或 Codex 屏幕录制权限。置顶窗默认位于屏幕左侧；不要移动到配置中的游戏 ROI 上方，否则屏幕截图会包含助手窗。

## 5. 状态保护和日志

系统只在以下条件满足时输出推荐：

- 地主身份唯一；
- 完整新局手牌和初始余牌稳定；
- 当前轮到自己；
- 视觉事件顺序和牌型合法；
- 余牌变化一致；
- 54 张牌守恒；
- 状态置信度不低于阈值。

低置信度、漏事件或冲突会切换到 `uncertain`，保存错误帧到 `data/live_game/errors/` 并等待下一局。日志默认写入 `logs/live_assistant.jsonl`，包含 `scene_observation`、`play/pass_observed`、`state_update`、`live_decision` 和运行延迟。

默认决策预算为 1.5 秒、至少 32 组 sampled worlds、Top-3，并按估计团队胜率优先排序。

## 6. 真实验收

在未参与模板和模型微调的完整 session 上统计：

- 出牌/过牌事件 F1 ≥ 95%；
- 出牌牌点整组准确率 ≥ 95%；
- 屏幕余牌准确率 ≥ 98%；
- 每局始终满足 54 张牌守恒；
- 无法确认的事件必须暂停推荐。

在完成真实数据评测前，只能声明 Phase 6 代码闭环和自动化测试通过，不能宣传上述真实指标已经达到。
