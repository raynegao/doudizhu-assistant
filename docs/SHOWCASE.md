# Phase 5 showcase guide

Phase 5A turns the tested Phase 1–4 core into a reproducible portfolio artifact. It runs without model weights, private screenshots or the original dataset.

## One-command local demo

```bash
source .venv/bin/activate
make demo
open runs/showcase/index.html
```

The command generates:

- `report.json`: machine-readable environment, settings, checks, Top-K results and risk fields.
- `summary.md`: GitHub- and CI-friendly summary.
- `index.html`: self-contained responsive report with no CDN, JavaScript or network dependency.

For a repeated benchmark, run `make benchmark`. For the core-only CPU container, run `make docker-demo`.

## One-minute recording outline

1. Show the repository front page and architecture diagram.
2. Run `make demo` and show the successful three-scenario summary.
3. Open `runs/showcase/index.html` and point out landlord/farmer scenarios, Top-K actions and decision fingerprints.
4. Open `report.json` and show the fixed seed, completed worlds, latency and risk flags.
5. Close with the limitation panel: uniform opponent model, standard rules subset and no automatic opponent-event recognition.

## Scenario coverage

| Fixture | Role and turn | Evidence |
|---|---|---|
| `landlord_lead.jsonl` | Self is landlord and leads | Large action space and candidate pruning |
| `landlord_response.jsonl` | Self is landlord and responds | Active trick, pass history and legal response |
| `farmer_response.jsonl` | Self is farmer and responds | Farmer-team semantics and role-normalized rollout |

All scenarios use fixed input, seed and simulation count. Repeated decisions must produce the same fingerprint; latency is recorded but not used as a universal threshold.
