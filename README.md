# doudizhu-assistant
AI-based Dou Dizhu assistant: screen card recognition + win rate estimation using YOLO and Monte Carlo simulation.

## 环境
- Python 3.12.13（>=3.10 均可，当前本地开发使用 3.12）
- 建议使用虚拟环境隔离依赖

## 快速开始
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Phase 1：规则引擎 MVP
当前阶段先不依赖 CV/YOLO，使用手动输入跑通规则引擎闭环：

```bash
python -m src.ui.cli \
  --hand "3 3 4 4 5 5 6 6 7 8 9 SJ BJ" \
  --last-play "5 5" \
  --log-file logs/phase1.jsonl
```

牌面记法：
- 普通牌：`3 4 5 6 7 8 9 10 J Q K A 2`
- 小王：`SJ`
- 大王：`BJ`

运行测试：

```bash
python -m pytest -q
```

当前 Phase 1 已跑通的能力：

- 解析手牌和上一手牌，校验重复牌和大小王数量。
- 识别单张、对子、三张、三带一、三带二、顺子、连对、飞机、炸弹和火箭等基础牌型。
- 根据上一手牌生成可行动作集合。
- 输出一个确定性的基础推荐动作和中文理由。
- 写入 JSONL 日志，字段包括 `event`、`input_cards`、`last_play`、`candidate_count`、`recommended_action`、`reason`、`warnings`。

下一阶段是 Phase 2：CV 检测接入。进入 Phase 2 前，应保持 Phase 1 的 CLI、测试和日志字段稳定。

## 配置与日志
- 配置文件支持 YAML/JSON，示例见 `configs/app.example.yaml`。
- `src/config/settings.py` 提供 Pydantic 校验与热重载轮询。
- `src/config/logging_config.py` 提供文本/JSON 日志输出（控制台 + 可选文件）。
