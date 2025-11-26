# doudizhu-assistant
AI-based Dou Dizhu assistant: screen card recognition + win rate estimation using YOLO and Monte Carlo simulation.

## 环境
- Python 3.13（>=3.10 均可，当前开发使用 3.13）
- 建议使用虚拟环境隔离依赖

## 快速开始
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## 配置与日志
- 配置文件支持 YAML/JSON，示例见 `configs/app.example.yaml`。
- `src/config/settings.py` 提供 Pydantic 校验与热重载轮询。
- `src/config/logging_config.py` 提供文本/JSON 日志输出（控制台 + 可选文件）。
