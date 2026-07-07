"""CLI entrypoint for benchmark runs."""

from __future__ import annotations

import argparse
import logging
import sys

from anomaly_detection.evaluation.benchmark import run_benchmark

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark anomaly detectors across registry datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Registry dataset ids to benchmark (default: all).",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        help="Detector names to run (default: zscore, iqr, isolation_forest, lof).",
    )
    parser.add_argument(
        "--config",
        help="Optional YAML config for preprocessing defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for benchmark markdown reports.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small local fixtures instead of downloading datasets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    report_path = run_benchmark(
        dataset_ids=args.datasets,
        detector_names=args.detectors,
        quick=args.quick,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    logger.info("Wrote benchmark report to %s", report_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
