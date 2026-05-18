# 斗地主助手实现计划（屏幕截取识别 + 两阶段模型 + Web/CLI）

## 里程碑 0：环境与基础
- 初始化仓库/虚拟环境，配置 `requirements-dev.txt`，选择 Python 版本（≥3.10）。
- 建立基础目录：`data/`、`models/`、`src/`（含 `capture/`、`vision/`、`tracking/`、`state/`、`logic/`、`ui/`、`tests/`）、`configs/`、`scripts/`、`logs/`。
- 配置日志、配置管理（YAML/JSON + pydantic/dataclasses），实现热加载占位接口。


Python 环境：使用 3.13 建立 .venv，新增 requirements-dev.txt（pydantic、PyYAML）。
目录脚手架：创建 data/, models/, src/（含 capture/vision/tracking/state/logic/ui/tests/config）、configs/, scripts/, logs/，并用 .gitkeep 保留空目录。
配置管理：src/config/settings.py 提供 Pydantic 配置模型、YAML/JSON 加载、简易热重载轮询（基于 mtime）。
日志配置：src/config/logging_config.py 支持文本/JSON 输出，控制台 + 可选文件。
示例配置：configs/app.example.yaml 展示可调参数（模型路径、推理阈值、蒙特卡洛参数、日志设置）。
文档：README.md 更新为 Python 3.13，安装步骤改为使用 requirements-dev.txt；.gitignore 放行 data/.gitkeep、logs/.gitkeep 并忽略日志目录文件。
后续可选操作：

进入 .venv 后执行 pip install -r requirements-dev.txt 安装依赖。
复制 configs/app.example.yaml 为实际配置文件（如 configs/app.yaml），再用 ConfigManager 加载并调用 configure_logging 应用日志设置。
在 src/ 内补充各子模块的实际逻辑（截屏、检测、分类、跟踪、状态、决策、UI）。

## 里程碑 1：数据与合成
- 收集高清牌面模板（含大小王、不同花色），存放 `data/templates/`。
- 编写合成脚本（`scripts/synth_generate.py`）：随机背景/缩放/旋转/光照/噪声，生成检测数据（bbox+类别）与裁剪分类数据；支持 COCO/YOLO 导出。
- 采集少量真实截屏，使用 labelimg/labelme 校正；合并分布，划分 train/val/test。
- 准备 `configs/data.yaml` 定义路径、分辨率、数据增强参数。

## 里程碑 2：检测模型（YOLOv8n → ONNX）
- 训练 YOLOv8n（ultralytics）：设定输入尺寸、置信度阈值、NMS；记录 mAP/F1。
- 导出 ONNX（动态 batch/shape），存放 `models/detector.onnx`。
- 编写推理包装（`src/vision/detector.py`）：ONNX Runtime/NCNN 接口，支持设备选择、阈值配置。
- 单元测试：固定样例截图，验证输出类别与 bbox 数量。

## 里程碑 3：分类模型（小 CNN/ViT → ONNX）
- 准备裁剪牌面数据集；训练轻量分类模型（MobileNet/ShuffleNet 或小 ViT）。
- 导出 ONNX（`models/classifier.onnx`）；推理包装 `src/vision/classifier.py`，支持批量裁剪输入。
- 单元测试：给定裁剪样本，验证 Top-1/Top-3 准确率。

## 里程碑 4：跟踪与多帧融合
- 集成 ByteTrack/OC-SORT（`src/tracking/tracker.py`），输入检测结果输出稳定轨迹。
- 时间窗口去重：同类目标在近 N 帧内合并；多帧投票降低抖动。
- 基准测试：在录屏序列上统计 ID 稳定性与漏检率。

## 里程碑 5：状态建模（bitboard）
- 设计牌编码与位掩码；实现计数 ↔ 位掩码互转（`src/state/encoding.py`）。
- 合法性校验：校验输入牌集合法，避免重复/缺失。
- 单元测试：覆盖所有牌型映射、互转一致性。

## 里程碑 6：规则出牌与蒙特卡洛
- 规则引擎：牌型判定、可行出牌生成、压制逻辑；提供 Top-K 基线建议（`src/logic/rules.py`）。
- 启发式对手模型（保炸/控王/最小拆牌）；随机基线用于对比。
- 蒙特卡洛/随机模拟（`src/logic/mcts.py`）：对未知手牌抽样，对局模拟评估胜率；可配置模拟次数、时间预算。
- 输出解释：炸弹计数、顺子拆分代价、王控制等理由文本。
- 测试：规则正确性、模拟稳定性、性能基准。

## 里程碑 7：截屏与异步管线
- 截屏模块（`src/capture/screen.py`）：mss/pywin32/gdigrab，固定 FPS，ROI 裁剪。
- 异步队列：截屏 → 检测 → 分类 → 跟踪 → 状态聚合 → 决策；使用线程/进程池，分离渲染。
- 性能优化：批处理裁剪分类、缓存上一帧轨迹，目标 <100ms/步；支持 ONNX/NCNN，GPU 时切 TensorRT。

## 里程碑 8：接口与 UI
- CLI/TUI（`src/ui/cli.py`）：输入截图/流，打印 Top-3 出牌、胜率、理由；支持回放日志。
- Web 后端（FastAPI/Flask，`src/ui/api.py`）：REST/WebSocket 提供实时状态、建议、日志。
- 前端（React/Vue/Svelte，`ui/`）：展示实时牌面/检测框、Top-3、胜率、理由，查看日志/回放。
- 配置热更新：模型路径、阈值、输入尺寸、线程数、模拟次数可热切换。

## 里程碑 9：日志、回放与评测
- 日志格式：每帧检测/分类结果、轨迹、状态、建议、胜率、理由；可存 JSONL。
- 回放工具（`scripts/replay.py`）：读取日志，逐帧重现识别与决策。
- 评测脚本：检测/分类指标，端到端识别准确率，决策胜率，耗时统计。

## 里程碑 10：打包与发布
- 打包：PyInstaller（本地可执行），前端静态资源打包；模型文件独立。
- 配置模板与示例（`configs/*.yaml`），提供默认模型下载/路径说明。
- 文档更新：使用说明、性能指标、已知问题、TODO 列表。
