"""Benchmark runner across registry datasets and detectors."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anomaly_detection.config.loader import load_config
from anomaly_detection.data_ingestion.loader import load_dataset_from_config
from anomaly_detection.data_ingestion.registry import load_registry
from anomaly_detection.evaluation.profiler import profile_detector
from anomaly_detection.models.registry import DETECTOR_REGISTRY, create_detector_from_config
from anomaly_detection.pipeline import run_detection
from anomaly_detection.preprocessing.scaler import apply_scaler

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


def _build_run_config(
    dataset_id: str,
    detector_name: str,
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
    if "fairness" in base_config:
        config["fairness"] = base_config["fairness"]
    return config


def _profile_detector_run(config: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    dataset_config = config.get("dataset", {})
    features, _, _ = load_dataset_from_config(dataset_config, quick=quick)
    preprocessing_config = config.get("preprocessing", {})
    scaled_features = apply_scaler(features, preprocessing_config.get("scaler"))
    model_config = config.get("model", {})
    detector_name = str(model_config.get("name", "unknown"))

    def fit_predict_fn(X: Any) -> None:
        detector = create_detector_from_config(config)
        detector.fit(X)
        detector.predict(X)
        detector.score(X)

    return profile_detector(detector_name, scaled_features, fit_predict_fn)


def _run_single(
    dataset_id: str,
    detector_name: str,
    *,
    quick: bool,
    base_config: dict[str, Any],
    profile: bool = False,
) -> dict[str, Any]:
    config = _build_run_config(dataset_id, detector_name, base_config)
    report = run_detection(config, quick=quick)
    result = {
        "dataset": dataset_id,
        "detector": detector_name,
        "n_samples": report["n_samples"],
        "n_anomalies": report["n_anomalies"],
        "metrics": report.get("metrics", {}),
    }
    if "fairness_metrics" in report:
        result["fairness_metrics"] = report["fairness_metrics"]
    if profile:
        result["profile"] = _profile_detector_run(config, quick=quick)
    return result


def run_benchmark(
    dataset_ids: list[str] | None = None,
    detector_names: list[str] | None = None,
    *,
    quick: bool = False,
    profile: bool = False,
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
                    profile=profile,
                )
            )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(output_dir) if output_dir else REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"benchmark_{timestamp}.md"
    json_path = reports_dir / f"benchmark_{timestamp}.json"
    report_path.write_text(
        _format_markdown(results, quick=quick, profile=profile),
        encoding="utf-8",
    )
    json_path.write_text(_format_json(results, quick=quick, profile=profile), encoding="utf-8")
    return report_path


def _profile_value(profile: dict[str, Any] | None, key: str) -> str:
    if not profile or key not in profile or profile[key] is None:
        return "—"
    value = profile[key]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_markdown(
    results: list[dict[str, Any]],
    *,
    quick: bool,
    profile: bool = False,
) -> str:
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
    fairness_rows = [row for row in results if row.get("fairness_metrics")]
    if fairness_rows:
        lines.extend(
            [
                "## Fairness metrics",
                "",
                "| Dataset | Detector | Attribute | Demographic parity | Equalized odds |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in fairness_rows:
            fairness = row["fairness_metrics"]
            for attribute, metrics in fairness.get("attributes", {}).items():
                lines.append(
                    "| {dataset} | {detector} | {attribute} | {dpd} | {eod} |".format(
                        dataset=row["dataset"],
                        detector=row["detector"],
                        attribute=attribute,
                        dpd=_metric_value(metrics, "demographic_parity_difference"),
                        eod=_metric_value(metrics, "equalized_odds_difference"),
                    )
                )
        lines.append("")
    profile_rows = [row for row in results if row.get("profile")]
    if profile or profile_rows:
        lines.extend(
            [
                "## Performance profiling",
                "",
                "| Dataset | Detector | Samples | Wall time (s) | Peak memory (MB) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in profile_rows:
            prof = row["profile"]
            lines.append(
                "| {dataset} | {detector} | {n_samples} | {wall_time} | {peak_memory} |".format(
                    dataset=row["dataset"],
                    detector=row["detector"],
                    n_samples=prof.get("n_samples", row["n_samples"]),
                    wall_time=_profile_value(prof, "wall_time_sec"),
                    peak_memory=_profile_value(prof, "peak_memory_mb"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _format_json(
    results: list[dict[str, Any]],
    *,
    quick: bool,
    profile: bool = False,
) -> str:
    payload = {
        "mode": "quick" if quick else "full",
        "profiled": profile or any(row.get("profile") for row in results),
        "runs": results,
    }
    return json.dumps(payload, indent=2) + "\n"
