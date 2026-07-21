# Portfolio copy

## English project summary

Built a layered real-time Dou Dizhu assistant that connects Mac screen capture and fixed-ROI CNN card recognition to an observable game-state reducer, deterministic rule engine, and uncertainty-aware Monte Carlo decision module. Added reproducible JSONL replays, common-random-world candidate evaluation, landlord/farmer team semantics, structured Top-K explanations, CI, a core-only Docker demo, and self-contained HTML/JSON evidence reports. The system explicitly blocks low-confidence state transitions and never performs automatic game control.

## 中文项目简介

开发了一个分层斗地主实时 AI 助手：在 Mac 上完成固定 ROI 截屏与小型 CNN 牌面识别，通过不可变事件 reducer 维护可观测牌局状态，再由规则引擎和蒙特卡洛模块输出带胜率估计、理由和风险的 Top-K 出牌建议。项目提供固定 JSONL 回放、地主/农民团队模拟、CI、CPU Docker 演示以及可复现的 HTML/JSON 证据报告；低置信度状态会阻断推荐，系统不执行自动点击或代打。

## Resume bullets

- Engineered a layered Python real-time card-game assistant spanning Mac screen capture, CNN recognition, immutable game-state reduction, legal-action generation and Monte Carlo decision evaluation.
- Implemented fixed-seed common-random-world rollouts with landlord/farmer team semantics, bounded candidate evaluation, deterministic fingerprints and structured Top-K explanations.
- Built a reproducible engineering evidence pipeline with 79 automated tests, Python 3.10/3.12 CI, a core-only Docker demo, versioned replay fixtures and self-contained HTML/JSON reports.
- Preserved deployment honesty by blocking uncertain state transitions, separating historical small-sample CV metrics from live evidence, and excluding automatic game control.

## Interview talking points

- Why state, vision and decision layers must not call each other's internals.
- How common random worlds make candidate comparisons fairer.
- Why a continuous, role-normalized truncated rollout value is safer than a landlord/farmer card-count shortcut.
- How low-confidence events remain pending instead of corrupting the last trusted state.
- Why fixed replay evidence must remain separate from real-window generalization claims.
