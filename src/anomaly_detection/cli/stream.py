"""CLI entrypoint for streaming anomaly detection."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.config.loader import load_config
from anomaly_detection.models.streaming.registry import create_streaming_detector_from_config
from anomaly_detection.monitoring.alerts import SlackAlerter
from anomaly_detection.monitoring.metrics import StreamMetrics

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream anomaly scores from stdin or CSV.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML streaming configuration file.",
    )
    parser.add_argument(
        "--input",
        help="Optional CSV input path; defaults to stdin when omitted.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _iter_points(
    config: dict[str, Any],
    input_path: str | None,
) -> Iterator[np.ndarray]:
    input_config = config.get("input", {})
    source = input_config.get("source", "stdin" if input_path is None else "csv")
    columns = input_config.get("columns")

    if source == "stdin":
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            values = [float(part) for part in stripped.split(",")]
            yield np.asarray(values, dtype=float)
        return

    if source == "csv":
        csv_path = Path(input_path or input_config.get("path", ""))
        if not csv_path:
            raise ValueError("CSV input requires --input or input.path in config")
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if columns:
                    values = [float(row[column]) for column in columns]
                else:
                    values = [float(value) for key, value in row.items() if key != "label"]
                yield np.asarray(values, dtype=float)
        return

    raise ValueError(f"Unsupported input source: {source}")


def run_stream(config: dict[str, Any], input_path: str | None = None) -> dict[str, Any]:
    """Process a stream and return summary metrics."""
    stream_config = config.get("stream", {})
    max_window = int(stream_config.get("max_window", 100))

    detector = create_streaming_detector_from_config(config)
    monitoring_config = config.get("monitoring", {})
    metrics = StreamMetrics(track_memory=bool(monitoring_config.get("track_memory", False)))

    alerts_config = config.get("alerts", {})
    alerter = SlackAlerter(
        threshold=float(alerts_config.get("threshold", 3.0)),
        webhook_url=alerts_config.get("slack_webhook_url"),
        enabled=bool(alerts_config.get("enabled", False)),
    )

    scores: list[float] = []
    for point in _iter_points(config, input_path):
        start = time.perf_counter()
        score = detector.score_one(point)
        detector.fit_partial(point)
        latency = time.perf_counter() - start
        metrics.record(latency)
        scores.append(score)
        alerter.check(score)
        print(json.dumps({"score": score}))

        window_size = getattr(detector, "window_size", None)
        if window_size is not None and window_size > max_window:
            raise RuntimeError(
                f"Window size {window_size} exceeds max_window={max_window}; "
                "memory bound violated."
            )

    summary = metrics.summary()
    summary["n_points"] = len(scores)
    if scores:
        summary["max_score"] = float(max(scores))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config = load_config(args.config)
    summary = run_stream(config, input_path=args.input)
    logger.info("Stream complete: %s", summary)
    print(json.dumps({"summary": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
