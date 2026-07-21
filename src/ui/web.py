"""Phase 5B read-only local web/API showcase.

The server deliberately exposes only versioned replay fixtures.  It never reads
arbitrary paths from a request and never controls a game client.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from src.logic.action_validation import validate_observed_action
from src.logic.monte_carlo import MonteCarloSettings, recommend_phase4
from src.state.replay import load_event_replay


DEFAULT_SCENARIOS = {
    path.stem: path for path in sorted(Path("examples/phase5").glob("*.jsonl"))
}


class ShowcaseWebApp:
    def __init__(self, scenarios: dict[str, Path] | None = None) -> None:
        self.scenarios = scenarios or DEFAULT_SCENARIOS

    def __call__(self, environ: dict[str, object], start_response: Callable) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", "/"))
        query = parse_qs(str(environ.get("QUERY_STRING", "")))
        try:
            if path == "/":
                return self._respond(start_response, "200 OK", _html_page(), "text/html; charset=utf-8")
            if path == "/api/health":
                return self._json(start_response, "200 OK", {"status": "ok", "mode": "read_only"})
            if path == "/api/scenarios":
                return self._json(start_response, "200 OK", {"scenarios": self._scenario_list()})
            if path == "/api/decision":
                scenario_id = _single_query(query, "scenario")
                payload = self._decision(scenario_id)
                return self._json(start_response, "200 OK", payload)
            return self._json(start_response, "404 Not Found", {"error": "not_found"})
        except ValueError as exc:
            return self._json(start_response, "400 Bad Request", {"error": str(exc)})
        except OSError as exc:
            return self._json(start_response, "500 Internal Server Error", {"error": f"fixture_unavailable: {exc}"})

    def _scenario_list(self) -> list[dict[str, str]]:
        return [
            {"id": name, "fixture": path.as_posix(), "mode": "Phase 4 event replay"}
            for name, path in self.scenarios.items()
        ]

    def _decision(self, scenario_id: str) -> dict[str, object]:
        path = self.scenarios.get(scenario_id)
        if path is None:
            raise ValueError("unknown scenario; use /api/scenarios")
        replay = load_event_replay(path, validator=validate_observed_action)
        result = recommend_phase4(replay.state, MonteCarloSettings(
            simulations=8,
            max_depth=12,
            time_budget_ms=0,
            seed=20260721,
            top_k=3,
            max_candidates=8,
            min_rollouts_per_action=8,
        ))
        return {
            "scenario": scenario_id,
            "state": replay.state.to_log_payload(),
            "recommendation": result.to_log_payload(),
            "replay_warnings": list(replay.warnings),
            "limitations": [
                "仅回放版本化事件场景，不读取真实游戏窗口。",
                "对手牌采用均匀剩余牌模型，胜率不是精确对手牌预测。",
                "项目不执行自动点击或自动代打。",
            ],
        }

    @staticmethod
    def _respond(start_response: Callable, status: str, body: str, content_type: str) -> Iterable[bytes]:
        encoded = body.encode("utf-8")
        start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(encoded)))])
        return [encoded]

    def _json(self, start_response: Callable, status: str, payload: dict[str, object]) -> Iterable[bytes]:
        return self._respond(start_response, status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")


def _single_query(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key, [])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"{key} is required")
    return values[0]


def _html_page() -> str:
    scenario_options = "".join(f'<option value="{escape(name)}">{escape(name)}</option>' for name in DEFAULT_SCENARIOS)
    return f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>斗地主助手 Phase 5B</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:880px;margin:40px auto;padding:0 20px;background:#f7f8fa;color:#18212f}}button,select{{font:inherit;padding:8px}}pre{{white-space:pre-wrap;background:#101828;color:#d0d5dd;padding:18px;border-radius:8px}}.note{{color:#475467}}</style>
<h1>斗地主助手 · Phase 5B</h1><p class=\"note\">只读本地展示：版本化事件 replay → 蒙特卡洛 Top-K 推荐。不会读取屏幕或自动操作游戏。</p>
<label>场景 <select id=\"scenario\">{scenario_options}</select></label> <button id=\"run\">获取推荐</button><pre id=\"output\">请选择场景后点击“获取推荐”。</pre>
<script>document.querySelector('#run').onclick=async()=>{{let id=document.querySelector('#scenario').value;let r=await fetch('/api/decision?scenario='+encodeURIComponent(id));document.querySelector('#output').textContent=JSON.stringify(await r.json(),null,2)}};</script></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only Phase 5B local showcase server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    with make_server(args.host, args.port, ShowcaseWebApp()) as server:
        print(f"Phase 5B web showcase: http://{args.host}:{args.port}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
