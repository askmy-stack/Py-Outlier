"""Fairness metrics for anomaly detection predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

AIF360_AVAILABLE = False

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

    AIF360_AVAILABLE = True
except ImportError:
    BinaryLabelDataset = None  # type: ignore[misc, assignment]
    BinaryLabelDatasetMetric = None  # type: ignore[misc, assignment]
    ClassificationMetric = None  # type: ignore[misc, assignment]


def _as_binary(array: np.ndarray) -> np.ndarray:
    return (np.asarray(array).astype(int).ravel() > 0).astype(int)


def _group_rate(y_pred: np.ndarray, mask: np.ndarray) -> float:
    subset = y_pred[mask]
    if subset.size == 0:
        return 0.0
    return float(subset.mean())


def _tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    true = y_true[mask]
    pred = y_pred[mask]
    if true.size == 0:
        return 0.0, 0.0
    positives = true == 1
    negatives = true == 0
    tpr = float(pred[positives].mean()) if positives.any() else 0.0
    fpr = float(pred[negatives].mean()) if negatives.any() else 0.0
    return tpr, fpr


def _numpy_demographic_parity_difference(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    y_pred = _as_binary(y_pred)
    groups = np.unique(protected_attribute)
    if groups.size < 2:
        return 0.0
    rates = [_group_rate(y_pred, protected_attribute == group) for group in groups]
    return float(max(rates) - min(rates))


def _numpy_equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    y_true = _as_binary(y_true)
    y_pred = _as_binary(y_pred)
    groups = np.unique(protected_attribute)
    if groups.size < 2:
        return 0.0

    overall_tpr, overall_fpr = _tpr_fpr(y_true, y_pred, np.ones(len(y_true), dtype=bool))
    max_diff = 0.0
    for group in groups:
        mask = protected_attribute == group
        tpr, fpr = _tpr_fpr(y_true, y_pred, mask)
        max_diff = max(max_diff, abs(tpr - overall_tpr) + abs(fpr - overall_fpr))
    return float(max_diff)


def _aif360_demographic_parity_difference(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    assert BinaryLabelDataset is not None
    assert ClassificationMetric is not None

    frame = {
        "label": np.zeros(len(y_pred)),
        "prediction": _as_binary(y_pred),
        "protected": protected_attribute.astype(str),
    }
    import pandas as pd

    dataset = BinaryLabelDataset(
        df=pd.DataFrame(frame),
        label_names=["label"],
        protected_attribute_names=["protected"],
        favorable_label=1,
        unfavorable_label=0,
    )
    privileged = [{ "protected": str(g) } for g in np.unique(protected_attribute)[:1]]
    metric = ClassificationMetric(dataset, dataset, unprivileged_groups=[], privileged_groups=privileged)
    return float(metric.mean_difference(method="between"))


def _aif360_equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    assert BinaryLabelDataset is not None
    assert ClassificationMetric is not None

    import pandas as pd

    true_dataset = BinaryLabelDataset(
        df=pd.DataFrame(
            {
                "label": _as_binary(y_true),
                "protected": protected_attribute.astype(str),
            }
        ),
        label_names=["label"],
        protected_attribute_names=["protected"],
        favorable_label=1,
        unfavorable_label=0,
    )
    pred_dataset = true_dataset.copy()
    pred_dataset.labels = _as_binary(y_pred).reshape(-1, 1)
    privileged = [{ "protected": str(g) } for g in np.unique(protected_attribute)[:1]]
    metric = ClassificationMetric(
        true_dataset,
        pred_dataset,
        unprivileged_groups=[],
        privileged_groups=privileged,
    )
    return float(metric.equalized_odds_difference())


def demographic_parity_difference(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    """Difference in positive prediction rates across protected groups."""
    if AIF360_AVAILABLE:
        try:
            return _aif360_demographic_parity_difference(y_pred, protected_attribute)
        except Exception:
            pass
    return _numpy_demographic_parity_difference(y_pred, protected_attribute)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    """Combined TPR and FPR disparity across protected groups."""
    if AIF360_AVAILABLE:
        try:
            return _aif360_equalized_odds_difference(y_true, y_pred, protected_attribute)
        except Exception:
            pass
    return _numpy_equalized_odds_difference(y_true, y_pred, protected_attribute)


def compute_fairness_metrics(
    predictions: np.ndarray,
    protected_attributes: dict[str, np.ndarray],
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute fairness metrics for each configured protected attribute."""
    y_pred = _as_binary(predictions)
    per_attribute: dict[str, dict[str, float | None]] = {}
    for name, values in protected_attributes.items():
        attr_metrics: dict[str, float | None] = {
            "demographic_parity_difference": demographic_parity_difference(y_pred, values),
        }
        if labels is not None:
            attr_metrics["equalized_odds_difference"] = equalized_odds_difference(
                labels,
                y_pred,
                values,
            )
        else:
            attr_metrics["equalized_odds_difference"] = None
        per_attribute[name] = attr_metrics

    return {
        "backend": "aif360" if AIF360_AVAILABLE else "numpy",
        "attributes": per_attribute,
    }
