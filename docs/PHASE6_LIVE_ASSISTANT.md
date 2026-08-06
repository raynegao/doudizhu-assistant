# Phase 6 完整场面感知与实时胜率助手

## 能力和边界

Phase 6 读取当前 Mac “斗地主”经典玩法窗口，识别三家的场上动作、角色和余牌，把稳定结果转换成既有 `ObservedAction`，再用 Phase 4 蒙特卡洛输出估计胜率最高的 Top-3。

它是只读助手，不点击游戏。胜率来自已观测牌、剩余张数和均匀未知牌采样，不是已知对手真实手牌后的精确概率。

助手不需要从发牌或抢地主阶段运行。可以先完成抢地主和加倍，在完整 17/20 张初始手牌已经显示、首手尚未打出时再启动；系统连续确认 2 帧后直接建立牌局。如果只漏掉了空场初始帧，但地主第一手仍显示在场上，系统可以在自己的完整 17 张农民手牌、三家角色、另一名农民 17 张、合法首手和 54 张牌守恒都稳定时，重建这一手并直接进入跟踪。更晚的中途启动默认暂停；轮到自己时可点击“扫描当前牌局”，显式建立带未知历史牌池的近似状态。

助手可以先于游戏启动，也可以在地主和加倍完成后启动。未检测到斗地主窗口时，小窗会持续显示“等待斗地主窗口”；从桌面、其他窗口或其他 Space 切回可见牌桌后自动开始识别。斗地主窗口被最小化时，小窗立即显示“无法识别”并暂停推荐。若最小化发生在正在跟踪的牌局中，因为可能漏过事件，该局会进入不确定状态；恢复后等待下一次稳定的自己回合自动重扫，重扫成功前不会沿用旧 revision 给出推荐。

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

采集优先使用 `native/macos_window_stream.swift` 提供的 ScreenCaptureKit 持久化窗口流；源码按哈希在本地忽略目录编译缓存，不提交二进制。它不会把覆盖在牌桌上的 Codex/终端窗口误截进来，也不再为每帧启动 `screencapture`。Swift 不可用或流异常时自动回退到 WindowServer Window ID 截图。Retina 比例直接由窗口逻辑尺寸和窗口图像尺寸计算，不会为了获取比例额外截取整个桌面。先检查预览是否准确覆盖手牌、三家出牌区、左右余牌、三家角色和自己的出牌按钮。窗口移动无需重写归一化 ROI；窗口布局或缩放样式改变时需要重新标定。

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

牌点识别另外内置 `src/vision/assets/rank_glyph_signatures.json`。它只包含从真实 crop 归一化得到的 64×64 二值字形位图，不包含牌局截图；因此全新检出即使没有 `data/live_game/templates/rank/`，也能修正横幅遮挡下的手牌 CNN 结果。已有新的独立标注集时可运行：

```bash
python -m scripts.export_rank_glyph_signatures \
  --source data/cards_cls/test
```

若新局开始时尚无地主 `20` 模板，跟踪器只会在三家角色明确、两名农民均为 17、自己的完整初始手牌稳定且三个出牌区都为空时，按规则补全地主初始 20 张。唯一例外是地主第一手仍完整显示：系统会验证该牌型、首手后的推导余牌、下一行动者和 54 张守恒，再把它记录为第 1 个视觉事件。

实时入口会直接分割牌背上的黄色数字，并用归一化字形轮廓读取左右余牌，不依赖系统 OCR 是否把单个描边数字判定为文本。匿名轮廓特征当前随代码分发 `0/1/2`，覆盖 `1/2/10/11/12…/20` 中最常见的组合读取；其他数字仍通过本机模板和轮廓分类器识别。整块余牌 ROI 的大部分像素是固定背景，未采集的数字可能与已有模板得到较高相似度，因此整块模板只作为兼容回退：`remaining_count` 会保留最佳匹配用于日志，但只有接近完全一致的模板才标记为 `remaining_verified` 并参与冲突阻断；否则状态机按已经确认的出牌张数扣减，不会把“3 误匹配为 16”当成可信事实。

## 3. 录制和标注完整对局

每局单独录制：

```bash
python -m scripts.record_live_game \
  --config configs/live_game.local.json \
  --session acceptance-001 \
  --evidence-split acceptance \
  --until-interrupt
```

对局结束后按 `Ctrl-C`，录制器会关闭并封印当前无损视频 segment，把已落盘帧数和中断状态原子写回。确认这一局完整后执行 `make live-finalize SESSION=acceptance-001`。输出还包含不可变 `config.snapshot.json`；正式局保存无损 Matroska 容器 SHA256、逐帧完整 RGB 像素 SHA256，以及可从解码帧无损重算的 ROI 像素 SHA256。这样既保留 10 FPS 快速动作，又不会用逐帧 PNG 写满磁盘；完整性检查和 replay 都会顺序解码并复算。同名 session 默认拒绝静默追加，中断后只能在配置和证据分区未变化时显式使用 `--resume`，恢复时创建新 segment 而不修改旧容器。

默认未写 `--evidence-split` 的录制属于 `development`，可用于模板调整、模型微调和排错，但不会进入正式聚合。正式 `acceptance` session 会在录制时封印实现、模型和模板 SHA256；录制后只要改过其中任何一项，该局就自动失效，必须重新录制。建议在代码冻结后一次性录制 5–10 局，不能把同一局拆到开发集和验收集。

正式局必须先盲标再 replay。运行 `make live-annotate-prepare SESSION=acceptance-001` 生成只含原始帧的 contact sheet，人工填写 `expected-events.jsonl` 与 `expected-scenes.jsonl` 后，运行 `make live-annotate-seal SESSION=acceptance-001` 封存。封存完成前禁止生成或查看 replay 预测；已有 replay 输出时工具会拒绝封存，正式 replay 也会拒绝未封存的标注。

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
  --manifest data/live_game/recordings/game-005/manifest.jsonl \
  --output-dir runs/live-replay/game-005 \
  --quiet
```

将人工确认的动作保存为 `expected-events.jsonl`，格式与 `play_observed` / `pass_observed` 一致；`expected-scenes.jsonl` 要为每个预期动作至少选一帧稳定画面，保存 `frame_id` 和三家 `remaining`。生成验收报告：

```bash
python -m scripts.evaluate_live_replay \
  --predicted-log runs/live-replay/game-005/events.jsonl \
  --expected-events data/live_game/recordings/game-005/expected-events.jsonl \
  --expected-scenes data/live_game/recordings/game-005/expected-scenes.jsonl \
  --output runs/live-replay/game-005/evaluation.json \
  --require-complete-round \
  --require-thresholds
```

replay 默认使用录制时的配置快照并生成 `replay.json`，其中包含 manifest、配置、模型、模板和事件日志指纹；已有输出默认拒绝覆盖。全部 session 完成后运行 `make phase6-acceptance`。聚合审计会逐帧核对完整图/ROI，检查独立 session、跨 session 泄漏、replay/标注指纹、动作/牌点/余牌指标、54 张守恒、终局结果、整局成功率和真实窗口牌面 holdout。完整说明见 [`docs/PHASE6_ACCEPTANCE.md`](PHASE6_ACCEPTANCE.md)。

## 4. 运行助手

置顶小窗：

```bash
make live-assistant
```

桌面一键启动：

```text
/Users/rayne/Desktop/斗地主助手.command
```

桌面文件只负责转交给版本化的 `scripts/launch_live_assistant_macos.command`。启动器成功退出后会按自己的 TTY 精确关闭对应 Terminal 标签；若助手已经运行，重复双击也会提示后自动关闭。它还会清理旧版本遗留、已经空闲且历史中包含助手专属启动提示的标签；不会按“所有空闲终端”这种宽泛条件删除窗口。原子启动锁会阻止快速连续双击产生多个助手进程或多个遗留启动窗口。正在运行其他任务的 Terminal 标签不会被清理。

终端调试：

```bash
python -m scripts.run_live_assistant \
  --config configs/live_game.local.json \
  --no-ui \
  --no-clear
```

Makefile 会优先使用项目 `.venv/bin/python`，无需手动激活虚拟环境。首次运行需要给实际启动助手的应用授予“屏幕与系统音频录制”权限：命令行启动对应 Terminal，从 Codex 启动则对应 Codex。权限按应用隔离，授权后需重新启动助手。置顶窗默认位于屏幕左侧；窗口级截图不会把覆盖在牌桌上的助手窗截入游戏画面。Tk 置顶窗与截图/Vision 识别分进程运行；识别子进程异常退出会自动恢复，连续异常才在小窗显示日志位置。完整开局和中途安全扫描都会保存可恢复的参考手牌帧，因此同一次 UI 会话内的子进程重启不会因为参考帧少于 17 张而丢弃可信 checkpoint。

推荐启动顺序：

1. 打开斗地主，完成抢地主和加倍；
2. 保持完整初始手牌可见，先不要打第一手；
3. 运行 `make live-assistant`；
4. 小窗先显示“正在建立牌局 1/2”，随后显示地主、当前行动者和三家余牌；
5. 自己是地主时直接等待 Top-3；对手是地主时，助手先跟踪其首手，轮到自己后再显示 Top-3。

配置中的 `initial_stability_frames` 默认是 `2`，只控制开局建模；`stability_frames` 默认是 `3`，继续控制普通出牌/不出事件，避免为了更快启动而降低整局跟踪稳定性。

如果是在一局中途启动，等到自己的“出牌/不出”按钮显示后，助手会在连续 2 帧确认角色、手牌、左右余牌和场上待压牌后自动重建。也可以点击置顶窗中的“扫描当前牌局”立即请求重扫；按钮会在最多 8 个可用帧内自动重试，吸收短暂遮挡或动画帧，显示“扫描中”时无需重复点击。若识别进程已因连续异常停止，同一个按钮会先重新拉起进程并清除旧错误，再提交重扫。此前已出但画面中不可见的牌只按张数放入未知历史池，不会伪造牌点；对应胜率会显示 `estimated_win_rate_uses_uniform_unknown_history` 风险提示。若不是自己回合，或角色、余牌、待压牌在重试后仍不稳定，本次扫描会保持暂停并等待下一次可确认场面。

## 5. 状态保护和日志

系统只在以下条件满足时输出推荐：

- 地主身份唯一；
- 完整新局手牌和初始余牌稳定，或地主第一手满足安全重建条件；
- 当前轮到自己；
- 视觉事件顺序和牌型合法；
- 余牌变化一致；
- 54 张牌守恒；
- 状态置信度不低于阈值。

跟踪过程采用以下交叉验证：

- 自己的实时手牌不再固定切成 17/20 张，而是先检测当前可见牌数，再识别牌点；
- 重叠手牌按水平网格去除竖排 `JOKER` 内部伪边，高置信度王不会被普通 `J` 参考覆盖；
- 点击选牌造成的牌面抬高不会触发“新一局”重置，真正按下出牌后再用稳定手牌差集生成事件；
- 自己回合优先检测提示/出牌按钮的饱和黄色区域，固定蓝色背景占主导的整块 turn 模板只作回退；
- 角色徽标先按局部字色（金色“地主”、白色“农民”）跨座位识别，避免头像和固定蓝色背景主导整块模板；完整开局检测到自己 20 张时，再以“自己是地主、另外两家是农民”的规则结构交叉校验；
- 自己出牌由连续稳定的手牌差集生成，并由规则引擎验证牌型和压制关系；
- 自己的完整手牌保持不变、回合控件连续稳定消失时直接生成 `pass_observed`；控件变化会重置稳定计数，单帧闪烁不会推进回合；
- 残局牌扇缩窄到 1–3 张时，手牌框按实际可见牌边重新定位；固定 ROI 仍覆盖原区域，但不再要求短手牌横跨完整手牌宽度；
- “不出”除真实模板外还直接检测各座位 ROI 上半区的白色字形，并避开下半区可能重叠的场上牌；
- 自己出牌和两名对手连续“不出”若全部发生在相邻采样帧之间，会按手牌差集、两家 pass 和重新出现的自己回合依次补齐三个事件，再恢复自由出牌推荐；
- 自己打出最后一手合法组合时，稳定空手牌区与已消失的回合控件会生成终局事件；这既支持单张也支持对子、三张和顺子等，无法构成合法牌型的异常空 ROI 不会结束牌局；本局断点随即清除，下一副完整手牌自动创建新 `round_id`；
- 自己画面中的“不出”文字或按钮不能单独推进状态，只有完整手牌不变且回合控件连续稳定消失才确认自己的 pass；
- 最新画面手牌必须与决策 revision 的状态手牌完全一致，否则已完成的 Top-K 也会隐藏，避免推荐手中不存在的牌；画面恢复后自动重算；
- 异步决策返回后，实时层会用该 revision 的手牌和 trick 重新生成合法动作，拒绝任何不合法的最佳动作、Top-K 动作或“最佳动作不等于 Top-1”的结果，并记录 `live_decision_rejected`；
- 对手出牌可以由场上牌和“上一余牌数 − 出牌张数”共同确认，即使错过短暂空白帧也能推进；
- 可见出牌优先于回合提示的过牌推断；接近阈值的牌点可由已验证的精确余牌下降交叉确认；
- 跟踪器会为三家分别记住最近出现过的空白边缘，不要求该座位当时正好轮到行动；两名对手在一次采样间隔内连续出牌时，后一个座位的新牌仍能被确认；
- 余牌字形除轮廓相似度外还比较闭环拓扑，避免本机 `16` 模板把实时 `15` 读成 `16`；
- 瞬时错误手牌先暂停推荐并等待视觉恢复；已验证余牌没有减少的低置信度牌影只作为噪声忽略，不会立刻污染整局；
- 当前 trick 中，若对手余牌未减少且界面已明确回到自己的回合，状态机会按行动顺序补记“不出”；
- 两名对手连续不出后清空当前 trick，当前行动者回到最后出牌者；若是自己，立即触发自由出牌 Top-3 计算。

低置信度、漏事件或冲突会暂停推荐；不可恢复的持续冲突才切换到 `uncertain`。错误帧按“round + 稳定故障类型”跨短暂恢复继续去重，默认同类冷却 60 秒、每局最多 4 张、每次运行最多 25 张；文件名永不覆盖，保存记录写入 JSONL。下次稳定识别到自己回合时会自动重建，也可按“扫描当前牌局”立即请求重扫。活动跟踪状态若与桌面稳定漂移，会在不增加任何一家的余牌、当前手牌仍是已跟踪手牌子集，并且手牌或至少一家余牌确实减少的前提下自动重建；仅有场上旧牌残留不会反向污染正确状态。日志默认写入 `logs/live_assistant.jsonl`，包含 `scene_observation`、`play/pass_observed`、`state_update`、`live_decision` 和运行延迟；每条动作额外带 `round_id` 与来源 `frame_id`。等待窗口和稳定跟踪期间只记录内容变化与每 20 帧心跳，动作、重扫和新决策仍立即记录。文件达到 64 MiB 后自动带时间戳归档并保留历史。`make live-diagnostics` 可生成错误类型与延迟汇总。主界面收到 `SIGTERM`/`SIGINT` 时也会走正常关闭路径，停止识别子进程、关闭已结束进程句柄并回收多进程队列；PID 文件由主进程原子写入，只由所属进程删除，启动器会清理失效或 PID 复用的旧记录。

默认决策预算为 1.5 秒、至少 32 组 sampled worlds、Top-3，并按估计团队胜率优先排序。每个候选同时输出样本均值的标准误和 95% 置信区间；前两名区间重叠时记录 `top_action_confidence_intervals_overlap`，避免把接近的有限样本结果描述成确定差距。

## 6. 真实验收

在未参与模板和模型微调的完整 session 上统计：

- 出牌/过牌事件 F1 ≥ 95%；
- 出牌牌点整组准确率 ≥ 95%；
- 屏幕余牌准确率 ≥ 98%；
- 每局始终满足 54 张牌守恒；
- 无法确认的事件必须暂停推荐。

在完成真实数据评测前，只能声明 Phase 6 代码闭环和自动化测试通过，不能宣传上述真实指标已经达到。

正式聚合口径和一键审计命令以 [`docs/PHASE6_ACCEPTANCE.md`](PHASE6_ACCEPTANCE.md) 为准。
