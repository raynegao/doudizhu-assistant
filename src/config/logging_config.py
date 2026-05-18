"""
简单的日志配置：支持文本或 JSON 格式，console + 可选文件输出。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .settings import LoggingConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_formatter(use_json: bool) -> logging.Formatter:
    if use_json:
        return JsonFormatter()
    return logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging(cfg: LoggingConfig) -> None:
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    formatter = _build_formatter(cfg.json)

    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    handlers.append(console)

    if cfg.file:
        file_path = Path(cfg.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)


__all__ = ["configure_logging", "JsonFormatter"]
