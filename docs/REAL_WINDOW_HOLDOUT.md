# 真实窗口独立 Holdout 评测

Phase 2 的历史指标只覆盖固定 ROI 的本地小样本。这个流程用于采集、登记并评测真正独立的斗地主窗口样本，不能与训练、验证或历史测试集混用。

## 验收口径

- 至少 300 张真实窗口 crop、3 个独立采集 session。
- `3`–`2`、`SJ`、`BJ` 共 15 类全部覆盖，每类至少 10 张。
- 训练 manifest 与 holdout 的 SHA256 重合数必须为 0。
- 报告必须保留逐类别准确率、完整预测、混淆矩阵、低置信度样本与错例。
- 重点检查 `10/J`、`6/9`、`SJ/BJ`、`J/Q/K/A` 混淆。

这些是“可发布数据集是否完整”的检查，不是预先承诺的准确率门槛。

## 1. 采集与标注一个 session

先从新的真实游戏窗口保存手牌 ROI。窗口、主题、分辨率或采集批次应与训练素材不同，然后运行：

```bash
python -m scripts.prepare_real_window_holdout \
  --roi data/raw_screenshots/new_session_hand_roi.png \
  --source-id window-theme-a-round-001 \
  --count 17 \
  --start-x 0 \
  --start-y 20 \
  --step-x 120 \
  --crop-size 126x210 \
  --labels "BJ 2 2 K Q J J 9 9 9 8 8 8 6 4 4 3"
```

如果省略 `--labels`，程序会在终端提示一次性输入从左到右的牌面。每个 session 会生成：

- `data/real_window_holdout/manifest.jsonl`：图片相对路径、标签、session、crop/ROI SHA256。
- `sessions/<source-id>/crops/`：原始 crop，不做数据增强。
- `sessions/<source-id>/contact_sheet.png`：带序号和标签的人工复核图。
- `sessions/<source-id>/session.json`：采集参数与来源摘要。

相同 `source-id`、相同 ROI 或相同 crop 会被拒绝。`data/` 默认被 Git 忽略，不提交私有截图。

## 2. 防泄漏评测

```bash
make holdout-evaluate
```

等价完整命令：

```bash
python -m scripts.evaluate_real_window_holdout \
  --model models/card_cnn.pt \
  --manifest data/real_window_holdout/manifest.jsonl \
  --training-manifest data/cards_cls/manifest.jsonl \
  --output-dir runs/real-window-holdout \
  --device cpu
```

评测会扫描训练 manifest 中的增强图和原始 source crop。发现任何 SHA256 重合会直接中止；找不到训练 manifest 时仍可评测，但 `publication_ready` 必须为 `false`。

输出：

- `report.json`：总体/逐类别指标、来源分布、防泄漏和发布准备度。
- `predictions.jsonl`：每张图片的预测、置信度与重点混淆标记。
- `errors.jsonl`：仅错例。
- `confusion_matrix.csv`：完整 15 类混淆矩阵。

样本数量或类别覆盖不足时，评测仍会输出结果，但不能把该数字作为真实窗口泛化结论宣传。
