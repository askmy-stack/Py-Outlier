"""Tests for benchmark runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from anomaly_detection.cli.benchmark import main as benchmark_main
from anomaly_detection.evaluation.benchmark import run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_quick_single_dataset(tmp_path: Path) -> None:
    report_path = run_benchmark(
        dataset_ids=["credit_card_fraud"],
        detector_names=["zscore", "isolation_forest"],
        quick=True,
        output_dir=tmp_path,
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "credit_card_fraud" in content
    assert "zscore" in content
    assert "isolation_forest" in content
    assert "quick (fixtures)" in content


def test_benchmark_cli_quick(tmp_path: Path) -> None:
    exit_code = benchmark_main(
        [
            "--datasets",
            "credit_card_fraud",
            "--detectors",
            "iqr",
            "--quick",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    reports = list(tmp_path.glob("benchmark_*.md"))
    assert len(reports) == 1


def test_benchmark_quick_with_profile(tmp_path: Path) -> None:
    report_path = run_benchmark(
        dataset_ids=["credit_card_fraud"],
        detector_names=["zscore"],
        quick=True,
        profile=True,
        output_dir=tmp_path,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Performance profiling" in content
    assert "Wall time" in content


def test_benchmark_module_invocation(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "anomaly_detection.cli.benchmark",
            "--datasets",
            "nab",
            "--detectors",
            "zscore",
            "--quick",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("benchmark_*.md"))
