# 真实窗口独立 Holdout 评测

Phase 2 的历史指标只覆盖固定 ROI 的本地小样本。这个流程专门用于真实斗地主窗口的独立泛化评测，不能与训练、验证或历史测试集混用。

## 采集规则

1. 使用与训练素材不同的游戏窗口、主题、分辨率或录制批次。
2. 先完整截图，再按实际手牌位置切出单张牌；不得从 `data/cards_cls` 或其派生文件复制样本。
3. 每张 crop 在 JSONL 中人工标注一次；保留 `source_id` 以便发现同源泄漏。
4. 将私有截图与 manifest 放在被 Git 忽略的 `data/real_window_holdout/`。

每行 manifest 格式如下，图片路径相对 manifest：

```json
{"image":"crops/session_a_0001.png","label":"10","source_id":"window-theme-a"}
```

## 运行

```bash
python -m scripts.evaluate_real_window_holdout \
  --model models/card_cnn.pt \
  --manifest data/real_window_holdout/manifest.jsonl \
  --output-dir runs/real-window-holdout \
  --device cpu
```

输出 `report.json`、`errors.jsonl`。报告会给出样本量、accuracy、类别分布与错误样本；样本不足或与训练数据同源时，不应把结果用于简历宣传。
