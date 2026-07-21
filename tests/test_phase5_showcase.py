from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.reporting.showcase import (
    ShowcaseSettings,
    build_showcase_report,
    write_showcase_artifacts,
)


def _settings(**overrides: object) -> ShowcaseSettings:
    values: dict[str, object] = {
        "event_files": (Path("examples/phase5/farmer_response.jsonl"),),
        "simulations": 2,
        "max_depth": 4,
        "seed": 9,
        "repeats": 2,
        "top_k": 2,
        "max_candidates": 4,
        "vision_metrics_file": Path("docs/evidence/card_cnn_eval.json"),
    }
    values.update(overrides)
    return ShowcaseSettings(**values)


def test_showcase_report_is_reproducible_and_has_no_local_paths() -> None:
    report = build_showcase_report(_settings())
    [scenario] = report["phase4"]["scenarios"]

    assert report["overall_status"] == "passed"
    assert report["phase1"]["deterministic"] is True
    assert scenario["completed_simulations"] == 2
    assert scenario["deterministic_across_repeats"] is True
    assert len(scenario["decision_fingerprint"]) == 16
    assert scenario["top_k"]
    assert report["capabilities"][1]["status"] == "historical_evidence"
    serialized = json.dumps(report, ensure_ascii=False)
    assert "/Users/" not in serialized
    assert "rayne" not in serialized.lower()


def test_showcase_writes_json_markdown_and_self_contained_html(tmp_path: Path) -> None:
    report = build_showcase_report(_settings(repeats=1))
    paths = write_showcase_artifacts(report, tmp_path)

    assert set(paths) == {"json", "markdown", "html"}
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["overall_status"] == "passed"
    assert "Phase 4 replay scenarios" in paths["markdown"].read_text(encoding="utf-8")
    html = paths["html"].read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "Top-K Evidence" in html
    assert "http://" not in html
    assert "https://" not in html


def test_phase5_cli_builds_artifacts_without_models_or_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "showcase"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_phase5_showcase",
            "--events-file",
            "examples/phase5/landlord_response.jsonl",
            "--output-dir",
            str(output_dir),
            "--simulations",
            "1",
            "--max-depth",
            "2",
            "--repeats",
            "1",
            "--top-k",
            "1",
            "--max-candidates",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase 5 showcase: passed" in result.stdout
    assert (output_dir / "report.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "index.html").exists()


def test_showcase_marks_missing_vision_evidence_without_failing() -> None:
    report = build_showcase_report(
        _settings(vision_metrics_file=Path("missing-metrics.json"), repeats=1)
    )

    assert report["overall_status"] == "passed"
    assert report["capabilities"][1]["status"] == "not_available"
