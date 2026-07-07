"""Benchmark runner across registry datasets and detectors."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anomaly_detection.config.loader import load_config
from anomaly_detection.data_ingestion.registry import load_registry
from anomaly_detection.models.registry import DETECTOR_REGISTRY
from anomaly_detection.pipeline import run_detection

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_CONFIG = REPO_ROOT / "configs" / "benchmark.yaml"

DEFAULT_DETECTORS = [
    "zscore",
    "iqr",
    "isolation_forest",
    "lof",
]


def _default_model_config(detector_name: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if detector_name in {"isolation_forest", "lof"}:
        params["contamination"] = 0.05
    if detector_name in {"isolation_forest", "one_class_svm"}:
        params["random_state"] = 42
    return {"name": detector_name, "params": params}


def _metric_value(metrics: dict[str, Any] | None, key: str) -> str:
    if not metrics or key not in metrics or metrics[key] is None:
        return "—"
    value = metrics[key]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _run_single(
    dataset_id: str,
    detector_name: str,
    *,
    quick: bool,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    config = {
        **base_config,
        "dataset": {
            "id": dataset_id,
        },
        "model": _default_model_config(detector_name),
        "preprocessing": base_config.get("preprocessing", {"scaler": "standard"}),
    }
    report = run_detection(config, quick=quick)
    return {
        "dataset": dataset_id,
        "detector": detector_name,
        "n_samples": report["n_samples"],
        "n_anomalies": report["n_anomalies"],
        "metrics": report.get("metrics", {}),
    }


def run_benchmark(
    dataset_ids: list[str] | None = None,
    detector_names: list[str] | None = None,
    *,
    quick: bool = False,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Run detectors across registry datasets and write a markdown report."""
    registry = load_registry()
    selected_datasets = dataset_ids or sorted(registry)
    for dataset_id in selected_datasets:
        if dataset_id not in registry:
            available = ", ".join(sorted(registry))
            raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {available}")

    selected_detectors = detector_names or DEFAULT_DETECTORS
    for detector_name in selected_detectors:
        if detector_name not in DETECTOR_REGISTRY:
            available = ", ".join(sorted(DETECTOR_REGISTRY))
            raise KeyError(f"Unknown detector '{detector_name}'. Available: {available}")

    config_file = Path(config_path) if config_path else DEFAULT_BENCHMARK_CONFIG
    base_config = load_config(config_file) if config_file.exists() else {}

    results: list[dict[str, Any]] = []
    for dataset_id in selected_datasets:
        for detector_name in selected_detectors:
            results.append(
                _run_single(
                    dataset_id,
                    detector_name,
                    quick=quick,
                    base_config=base_config,
                )
            )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(output_dir) if output_dir else REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"benchmark_{timestamp}.md"
    report_path.write_text(_format_markdown(results, quick=quick), encoding="utf-8")
    return report_path


def _format_markdown(results: list[dict[str, Any]], *, quick: bool) -> str:
    mode = "quick (fixtures)" if quick else "full"
    lines = [
        "# Anomaly Detection Benchmark",
        "",
        f"- Mode: `{mode}`",
        f"- Runs: `{len(results)}`",
        "",
        "| Dataset | Detector | Samples | Anomalies | Precision | Recall | F1 | ROC-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        metrics = row.get("metrics", {})
        lines.append(
            "| {dataset} | {detector} | {n_samples} | {n_anomalies} | "
            "{precision} | {recall} | {f1} | {roc_auc} |".format(
                dataset=row["dataset"],
                detector=row["detector"],
                n_samples=row["n_samples"],
                n_anomalies=row["n_anomalies"],
                precision=_metric_value(metrics, "precision"),
                recall=_metric_value(metrics, "recall"),
                f1=_metric_value(metrics, "f1"),
                roc_auc=_metric_value(metrics, "roc_auc"),
            )
        )
    lines.append("")
    return "\n".join(lines)
