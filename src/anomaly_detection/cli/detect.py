"""CLI entrypoint for anomaly detection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from anomaly_detection.config.loader import load_config
from anomaly_detection.pipeline import run_detection

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run anomaly detection from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        help="Override report output path from config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Append a plain-language anomaly explanation to the report (mock when llm.enabled=false).",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def _write_plot(
    plot_path: Path,
    scores: list[float],
    predictions: list[int],
    labels: list[int] | None = None,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(10, 4))
    normal_mask = np.array(predictions) == 0
    anomaly_mask = np.array(predictions) == 1

    ax.scatter(indices[normal_mask], np.array(scores)[normal_mask], c="steelblue", label="normal", s=18)
    ax.scatter(
        indices[anomaly_mask],
        np.array(scores)[anomaly_mask],
        c="crimson",
        label="anomaly",
        s=24,
    )
    if labels is not None:
        true_anomalies = np.array(labels) == 1
        ax.scatter(
            indices[true_anomalies],
            np.array(scores)[true_anomalies],
            facecolors="none",
            edgecolors="black",
            s=40,
            label="true anomaly",
        )

    ax.set_xlabel("sample index")
    ax.set_ylabel("anomaly score")
    ax.set_title("Anomaly detection scores")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config_path = Path(args.config)
    config = load_config(config_path)
    report = run_detection(config)

    if args.explain:
        from anomaly_detection.llm.explainer import explain_anomaly

        report["explanation"] = explain_anomaly(
            scores=report["scores"],
            predictions=report["predictions"],
            config=config,
            feature_names=report.get("feature_names"),
            model_name=report.get("model"),
        )

    output_config = config.get("output", {})
    report_path = Path(args.output or output_config.get("report_path", "reports/run.json"))
    _write_report(report_path, report)
    logger.info("Wrote report to %s", report_path)

    plot_path_value = output_config.get("plot_path")
    if plot_path_value:
        labels = None
        dataset_config = config.get("dataset", {})
        target_column = dataset_config.get("target_column")
        if target_column and (dataset_config.get("path") or dataset_config.get("id")):
            from anomaly_detection.data_ingestion.loader import load_dataset_from_config

            _, labels_array, _ = load_dataset_from_config(dataset_config)
            labels = labels_array.tolist() if labels_array is not None else None

        _write_plot(
            Path(plot_path_value),
            report["scores"],
            report["predictions"],
            labels=labels,
        )
        logger.info("Wrote plot to %s", plot_path_value)

    if args.verbose and "metrics" in report:
        logger.debug("Metrics: %s", report["metrics"])

    print(json.dumps({"report_path": str(report_path), "n_anomalies": report["n_anomalies"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
