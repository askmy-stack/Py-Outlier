"""Smoke tests for package skeleton."""

import pytest

import anomaly_detection
from anomaly_detection.cli.detect import main


def test_version() -> None:
    assert anomaly_detection.__version__ == "1.0.0"


def test_detect_cli_requires_config() -> None:
    with pytest.raises(SystemExit):
        main()
