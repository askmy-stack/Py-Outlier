"""Integration tests for fairness in pipeline and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from anomaly_detection.pipeline import run_detection

REPO_ROOT = Path(__file__).resolve().parents[2]
FAIRNESS_CONFIG = REPO_ROOT / "configs" / "examples" / "fairness.yaml"


def test_pipeline_includes_fairness_metrics(tmp_path: Path) -> None:
    from anomaly_detection.config.loader import load_config

    config = load_config(FAIRNESS_CONFIG)
    config["output"] = {"report_path": str(tmp_path / "report.json")}
    report = run_detection(config)

    assert "fairness_metrics" in report
    fairness = report["fairness_metrics"]
    assert "gender" in fairness["attributes"]
    assert "age_group" in fairness["attributes"]
    assert fairness["attributes"]["gender"]["demographic_parity_difference"] >= 0.0
    assert report["mitigation"]["strategy"] == "reweighing"


def test_detect_report_json_contains_fairness_block(tmp_path: Path) -> None:
    from anomaly_detection.cli.detect import main as detect_main

    report_path = tmp_path / "fairness_run.json"
    exit_code = detect_main(
        [
            "--config",
            str(FAIRNESS_CONFIG),
            "--output",
            str(report_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "fairness_metrics" in payload
    assert payload["fairness_metrics"]["attributes"]["gender"]["demographic_parity_difference"] >= 0.0
