from __future__ import annotations

import io
import json
from pathlib import Path

from scripts.evaluate_real_window_holdout import load_holdout_manifest
from src.ui.web import ShowcaseWebApp


def _request(app: ShowcaseWebApp, path: str, query: str = "") -> tuple[str, dict[str, str], bytes]:
    captured: dict[str, object] = {}
    body = b"".join(app({"PATH_INFO": path, "QUERY_STRING": query, "wsgi.input": io.BytesIO()}, lambda status, headers: captured.update(status=status, headers=dict(headers))))
    return str(captured["status"]), dict(captured["headers"]), body


def test_phase5b_web_exposes_only_known_replay_scenarios() -> None:
    app = ShowcaseWebApp({"fixture": Path("examples/phase5/farmer_response.jsonl")})
    status, _, body = _request(app, "/api/scenarios")
    assert status == "200 OK"
    assert json.loads(body)["scenarios"][0]["id"] == "fixture"

    status, _, body = _request(app, "/api/decision", "scenario=fixture")
    payload = json.loads(body)
    assert status == "200 OK"
    assert payload["recommendation"]["completed_simulations"] == 8
    assert len(payload["recommendation"]["top_k"]) == 3

    status, _, body = _request(app, "/api/decision", "scenario=../../private")
    assert status == "400 Bad Request"
    assert "unknown scenario" in json.loads(body)["error"]


def test_holdout_manifest_validates_labels_paths_and_duplicates(tmp_path: Path) -> None:
    crop = tmp_path / "card.png"
    crop.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image": "card.png", "label": "3", "source_id": "real-window-a"}) + "\n", encoding="utf-8")
    [record] = load_holdout_manifest(manifest)
    assert record["label"] == "3"
    assert record["source_id"] == "real-window-a"

    manifest.write_text(json.dumps({"image": "card.png", "label": "not-a-card"}) + "\n", encoding="utf-8")
    try:
        load_holdout_manifest(manifest)
    except ValueError as exc:
        assert "unsupported label" in str(exc)
    else:
        raise AssertionError("invalid holdout label must be rejected")
