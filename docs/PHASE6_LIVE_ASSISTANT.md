# Phase 6 完整场面感知与实时胜率助手

## 能力和边界

Phase 6 读取当前 Mac “斗地主”经典玩法窗口，识别三家的场上动作、角色和余牌，把稳定结果转换成既有 `ObservedAction`，再用 Phase 4 蒙特卡洛输出估计胜率最高的 Top-3。

它是只读助手，不点击游戏。胜率来自已观测牌、剩余张数和均匀未知牌采样，不是已知对手真实手牌后的精确概率。

助手应在完整新局开始前运行。如果只漏掉了空场初始帧，但地主第一手仍显示在场上，系统可以在自己的完整 17 张农民手牌、三家角色、另一名农民 17 张、合法首手和 54 张牌守恒都稳定时，重建这一手并直接进入跟踪。更晚的中途启动仍无法恢复此前已打出牌的点数，因此会等待下一局。

助手可以先于游戏启动。未检测到斗地主窗口时，小窗会持续显示“等待斗地主窗口”；窗口打开后自动开始识别。斗地主窗口被最小化时，小窗立即显示“无法识别”并暂停推荐。若最小化发生在正在跟踪的牌局中，因为可能漏过事件，该局会进入不确定状态并等待下一局重新初始化。

## 1. 标定窗口

需要重新检查布局时，启动游戏并确保牌桌窗口已创建（可以被其他窗口遮挡，但不要最小化或关闭）：

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

若新局开始时尚无地主 `20` 模板，跟踪器只会在三家角色明确、两名农民均为 17、自己的完整初始手牌稳定且三个出牌区都为空时，按规则补全地主初始 20 张。唯一例外是地主第一手仍完整显示：系统会验证该牌型、首手后的推导余牌、下一行动者和 54 张守恒，再把它记录为第 1 个视觉事件。

macOS 实时入口会优先使用系统 Vision 文字识别读取左右余牌。整块余牌 ROI 的大部分像素是固定背景，未采集的数字可能与已有模板得到较高相似度，因此模板只作为兼容回退：`remaining_count` 会保留最佳匹配用于日志，但只有接近完全一致的模板才标记为 `remaining_verified` 并参与冲突阻断；否则状态机按已经确认的出牌张数扣减，不会把“13 误匹配为 16”当成可信事实。

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
make live-assistant
```

终端调试：

```bash
python -m scripts.run_live_assistant \
  --config configs/live_game.local.json \
  --no-ui \
  --no-clear
```

Makefile 会优先使用项目 `.venv/bin/python`，无需手动激活虚拟环境。首次运行需要给终端或 Codex 屏幕录制权限。置顶窗默认位于屏幕左侧；窗口级截图不会把覆盖在牌桌上的助手窗截入游戏画面。

## 5. 状态保护和日志

系统只在以下条件满足时输出推荐：

- 地主身份唯一；
- 完整新局手牌和初始余牌稳定，或地主第一手满足安全重建条件；
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
