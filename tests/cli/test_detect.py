"""Smoke tests for the anomaly detection CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from anomaly_detection.cli.detect import main

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def test_cli_main_writes_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(report_path),
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"] == "isolation_forest"
    assert report["n_samples"] > 0
    assert report["n_anomalies"] >= 1
    assert "metrics" in report
    assert "precision" in report["metrics"]
    assert "roc_auc" in report["metrics"]


def test_cli_module_invocation(tmp_path: Path) -> None:
    report_path = tmp_path / "module-report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "anomaly_detection.cli.detect",
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.exists()
