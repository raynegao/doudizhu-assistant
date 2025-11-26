"""
配置加载与热重载占位实现：支持 YAML/JSON，基于 Pydantic 校验。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ModelPaths(BaseModel):
    detector: Path = Field(default=Path("models/detector.onnx"))
    classifier: Path = Field(default=Path("models/classifier.onnx"))


class InferenceConfig(BaseModel):
    device: str = "cpu"
    input_size: int = 640
    detector_conf_threshold: float = 0.25
    detector_nms_iou: float = 0.45
    max_detections: int = 50


class MonteCarloConfig(BaseModel):
    simulations: int = 200
    max_depth: int = 50
    time_budget_ms: int = 100


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json: bool = False
    file: Optional[Path] = None


class AppConfig(BaseModel):
    model_paths: ModelPaths = ModelPaths()
    inference: InferenceConfig = InferenceConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig()
    logging: LoggingConfig = LoggingConfig()


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object/dict: {path}")

    return data


@dataclass
class ConfigSnapshot:
    source: Optional[Path]
    config: AppConfig
    last_loaded_at: float


class ConfigManager:
    """
    负责加载/存储 AppConfig，并提供简易热重载（轮询 mtime）。
    热重载线程是守护线程，可通过 stop_hot_reload 停止。
    """

    def __init__(self, path: Optional[Path] = None, reload_interval: float = 2.0) -> None:
        self._path = path
        self._reload_interval = reload_interval
        self._lock = threading.Lock()
        self._snapshot = ConfigSnapshot(source=path, config=AppConfig(), last_loaded_at=time.time())
        self._watcher: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_mtime: Optional[float] = None

    def load(self) -> AppConfig:
        if self._path is None:
            logger.info("No config path provided, using defaults.")
            with self._lock:
                self._snapshot = ConfigSnapshot(source=None, config=AppConfig(), last_loaded_at=time.time())
            return self._snapshot.config

        try:
            data = _read_config_file(self._path)
            config = AppConfig.model_validate(data)
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            logger.warning("Config file not found (%s); using defaults.", self._path)
            config = AppConfig()
            mtime = None
        except (ValidationError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", self._path, exc)
            raise

        with self._lock:
            self._snapshot = ConfigSnapshot(source=self._path, config=config, last_loaded_at=time.time())
            self._last_mtime = mtime
        return config

    def get(self) -> AppConfig:
        with self._lock:
            return self._snapshot.config

    def start_hot_reload(self, callback: Optional[Callable[[AppConfig], None]] = None) -> None:
        """
        开启热重载轮询线程。回调在配置重新加载后被调用。
        """
        if self._path is None:
            logger.info("Hot reload skipped: no config path.")
            return
        if self._watcher and self._watcher.is_alive():
            return

        self._stop_event.clear()
        self._watcher = threading.Thread(
            target=self._watch_loop,
            kwargs={"callback": callback},
            daemon=True,
            name="config-hot-reload",
        )
        self._watcher.start()

    def stop_hot_reload(self) -> None:
        self._stop_event.set()
        if self._watcher:
            self._watcher.join(timeout=1.0)

    def _watch_loop(self, callback: Optional[Callable[[AppConfig], None]] = None) -> None:
        while not self._stop_event.is_set():
            try:
                if self._path is None or not self._path.exists():
                    time.sleep(self._reload_interval)
                    continue

                mtime = self._path.stat().st_mtime
                if self._last_mtime is None or mtime > self._last_mtime:
                    logger.info("Config change detected, reloading from %s", self._path)
                    new_config = self.load()
                    if callback:
                        callback(new_config)
                self._last_mtime = mtime
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hot reload loop encountered an error: %s", exc)
            time.sleep(self._reload_interval)


__all__ = [
    "AppConfig",
    "ConfigManager",
    "ConfigSnapshot",
    "InferenceConfig",
    "LoggingConfig",
    "ModelPaths",
    "MonteCarloConfig",
]
