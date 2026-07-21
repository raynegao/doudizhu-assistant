# Evaluation and evidence

## Reproducible Phase 5 baseline

Source: [`docs/evidence/phase5_showcase_baseline.json`](evidence/phase5_showcase_baseline.json), generated from commit `e67035c` on Mac arm64 with Python 3.12.13.

```bash
python -m scripts.run_phase5_showcase \
  --output-dir runs/showcase-benchmark \
  --repeats 3 \
  --simulations 20 \
  --max-depth 20
```

| Scenario | Completed worlds | Median | P95 | Worlds/s | Fingerprint |
|---|---:|---:|---:|---:|---|
| Farmer response | 20/20 | 375.152 ms | 375.178 ms | 54.296 | `c07a71c931229e0e` |
| Landlord lead | 20/20 | 191.608 ms | 191.789 ms | 104.775 | `6f8a4d361d643bde` |
| Landlord response | 20/20 | 312.358 ms | 313.232 ms | 64.253 | `f1b00e9197f67003` |

These values describe one machine and configuration. CI verifies completeness and deterministic fingerprints but does not enforce these latency numbers across hardware.

## Historical Phase 2 evidence

[`card_cnn_eval.json`](evidence/card_cnn_eval.json) records the original local split sizes, accuracies, error count and manifest hash:

| Split | Samples | Local fixed-ROI accuracy |
|---|---:|---:|
| Train | 1589 | 100% |
| Validation | 422 | 100% |
| Test | 99 | 100% |

This is small-sample local evidence. Original images and weights remain ignored, and the result must not be described as real-window generalization accuracy.

## Automated gates and known risks

- Core CI runs on Python 3.10 and 3.12 without visual assets; full CI installs the vision stack.
- `PYTHONWARNINGS=error` prevents warning regressions.
- The showcase must complete every requested world and reproduce the same decision fingerprint.
- Uniform unknown-card sampling is a baseline, candidate pruning bounds work, and the current ruleset is a documented subset.
- Real opponent-event perception and a separate real-window holdout remain future work.
