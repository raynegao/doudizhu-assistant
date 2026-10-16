# 独立验收：冻结实现 `04f83e90`（2026-10-14 ~ 10-16）

这是 v8–v11 修复之后第一批**先冻结代码、再录新局**的独立验收。录制开始到评测结束期间，`src/`、`scripts/`、`native/`、模型和随代码分发的资产一个字节都没改。标注口径与 [2026-09-07 批次](AI_REFERENCE_ACCEPTANCE_20260907.md) 相同：由 AI 只看原始解码帧完成并封印，**之后**才运行项目 replay，来源记录为 `annotation_source: ai_visual`、`independent_human_review: false`，不是独立人工盲标。

**结论：正式门禁未通过。** 5 个有效局的录制完整性、封印、标注先于预测、无跨局帧泄漏、54 张守恒全部合格；没过的是识别本身：完整局成功率 3/5 = 60%（门槛 80%），003、006 两局的单局门槛没过，300 张独立牌面 holdout 还没做。

## 冻结

| 项 | 值 |
|---|---|
| Git 提交 | `a0e0ac1`（已在 `main`） |
| 实现指纹 `sha256_python_implementation` | `04f83e9000952264e737f0fe9926675d4819ed4ae5bcb14dcc97ab07c6dfcc0e` |
| 归档 | `work/independent-acceptance-20261014/frozen-04f83e90.tar.gz`，180 个成员逐字节核对，从归档字节重算实现指纹一致 |
| 前一轮回归冻结 | `0cf6c97a`（v11），本批不复用其任何结论 |

## 全部 acceptance 录制（不挑选）

| 会话 | 帧 | 采集 | 结果 |
|---|---:|---|---|
| 001 | 711 | 最大间隔 0.134 s | 有效 |
| 002 | 693 | **0.461 s** 卡顿（第 150 帧） | finalize 拒收：录制时标注进程在同机解码视频 |
| 003 | 684 | 0.223 s | 有效 |
| 004 | 719 | **0.392 s** 卡顿（第 665 帧，结算画面） | finalize 拒收 |
| 005 | 484 | 31 次 >0.2 s，最大 **1.022 s**，P95 0.225 s | finalize 拒收：主机负载约 49（无关应用占满 WindowServer） |
| 006 | 1005 | 0.122 s | 有效 |
| 007 | 772 | 0.121 s | 有效 |
| 008 | 767 | 0.126 s | 有效 |

被拒的 3 局保留原件，原样计入批次审计；门槛 `max(0.30 s, 3×interval)` 在录制时就已封印，没有为了收下它们而放宽。之后录制改为：录制期间本机不做解码或 replay，结算横幅一出现就停止录制，录制前检查主机负载。

## 单局结果（5 个有效局）

| 会话 | 角色 / 结果 | 动作 | 出牌牌组 | 余牌 | 终局赢家 | 整局全对 | 单局门槛 |
|---|---|---:|---:|---:|---|---|---|
| 001 | 自己地主 / 胜 | 32/34 | 21/22 | 101/102 | 对 | ✗ | ✓ |
| 003 | 左家地主 / 负 | 30/34 | 20/22 | 101/102 | 对 | ✗ | ✗ |
| 006 | 左家地主 / 胜 | 32/32 | 18/18 | 91/96 | 对 | ✓ | ✗ |
| 007 | 自己地主 / 胜 | 37/37 | 22/22 | 110/111 | 对 | ✓ | ✓ |
| 008 | 自己地主 / 胜 | 37/37 | 20/20 | 110/111 | 对 | ✓ | ✓ |

## 聚合（`audit-valid5.json`）

| 指标 | 值 | 门槛 | |
|---|---:|---:|---|
| 事件精确率 | 168/168 = 100% | — | |
| 事件召回率 | 168/174 = 96.55% | — | |
| 事件 F1 | 98.25% | ≥95% | ✓ |
| 出牌牌组整组准确率 | 101/104 = 97.12% | ≥95% | ✓ |
| 余牌准确率 | 513/522 = 98.28% | ≥98% | ✓ |
| 54 张守恒 | 641/641 | 100% | ✓ |
| 终局赢家与团队结果 | 5/5 | — | |
| 完整局成功率 | 3/5 = 60% | ≥80% | ✗ |
| 每局单局门槛 | 3/5 | 全部 | ✗ |
| 独立牌面 holdout | 未做 | 300+ 张 | ✗ |

精确率 100%：系统没有记错过任何一手，也没有凭空多记。所有失误都是漏记或读不出，并且都发生在系统已经主动暂停推荐之后，不会给出错误建议。

审计按三种范围各跑了一次，三份报告都保留：

- `work/independent-acceptance-20261014/audit-valid5.json`：5 个有效局，与正式门禁"≥5 个完整局"的口径一致
- `work/independent-acceptance-20261014/audit-batch.json`：本冻结下全部 8 个 acceptance 会话，完整局成功率 3/8；额外失败的完整性检查全部来自 002/004/005
- `runs/phase6-acceptance/report.json`：`make phase6-acceptance` 全目录，包含 9 月旧实现录制的 8 局，它们按设计因封印不符而失败

## 失败根因（只从冻结后的 replay 日志读出，没有改代码）

1. **自己末手炸弹漏记**（001 第 34 手 `7777`，以及第 33 手左家 pass）：终局手牌清空之后，`live_self_table_terminal` 门没有在结算前提交，表现与 v10 之前"末手需要专门路径"的问题同类。
2. **对手王炸低置信度**（003 第 31 手起共 4 手）：左家 `BJ SJ` 叠着 ×2 特效，F514–517 连续 3 帧低于置信门槛，跟踪器按设计标记为不确定，等下一次自己回合重扫；但之后自己再没有回合，于是直接进入结算。
3. **左家余牌个位数被读成两位数**（006）：`4→14`、`3→13` 各出现在相邻多个场景帧，另有 1 帧余牌未知；终局自己的 0 张返回"未知"，这是按设计不把视觉未知填成 0。

这批数据已经看过，之后只能做回归。修复 1–3 之后必须重新冻结，并在新实现下另录至少 5 局。

## 复现

```bash
python -m scripts.replay_live_game --manifest data/live_game/recordings/acceptance-20261014-007/manifest.jsonl \
  --output-dir runs/live-replay/acceptance-20261014-007 --quiet
python -m scripts.evaluate_live_replay \
  --predicted-log runs/live-replay/acceptance-20261014-007/events.jsonl \
  --expected-events data/live_game/recordings/acceptance-20261014-007/expected-events.jsonl \
  --expected-scenes data/live_game/recordings/acceptance-20261014-007/expected-scenes.jsonl \
  --output runs/live-replay/acceptance-20261014-007/evaluation.json \
  --require-complete-round --require-thresholds
python -m scripts.audit_phase6_acceptance --recordings-root <只含本批会话的目录> \
  --replays-root runs/live-replay --require-thresholds
```

录制、标注规格（`work/independent-acceptance-20261014/spec-*.json`）、带规则引擎校验的标注生成脚本 `build_annotation.py`、冻结归档和审计 JSON 只保存在本地，不入库（`data/`、`runs/`、`work/` 均被 `.gitignore` 忽略）。
