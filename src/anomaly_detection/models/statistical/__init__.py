"""Statistical anomaly detectors."""

from anomaly_detection.models.statistical.iqr import IQRDetector
from anomaly_detection.models.statistical.zscore import ZScoreDetector

__all__ = ["IQRDetector", "ZScoreDetector"]
