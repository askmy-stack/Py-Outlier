"""Causal graph construction from multivariate metric time series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

GraphDict = dict[str, Any]

_PYRCA_AVAILABLE = False

try:
    from pyrca.graphs.causal import CausalModel  # type: ignore[import-untyped]

    _PYRCA_AVAILABLE = True
except ImportError:
    CausalModel = None  # type: ignore[misc, assignment]


def _pyrca_available() -> bool:
    return _PYRCA_AVAILABLE


def _build_graph_pyrca(metrics: pd.DataFrame) -> GraphDict:
    """Build causal graph using PyRCA when installed."""
    assert CausalModel is not None
    model = CausalModel()
    model.train(metrics)
    graph = model.export_graph()
    nodes = [{"id": node, "label": node} for node in metrics.columns]
    edges = [{"source": src, "target": dst, "weight": 1.0} for src, dst in graph.edges()]
    return {
        "nodes": nodes,
        "edges": edges,
        "method": "pyrca",
    }


def _lagged_correlation_direction(series_a: pd.Series, series_b: pd.Series) -> str | None:
    """Infer directed edge using lag-1 cross-correlation (A→B vs B→A)."""
    a = series_a.to_numpy(dtype=float)
    b = series_b.to_numpy(dtype=float)
    if len(a) < 3:
        return None

    corr_a_leads = np.corrcoef(a[:-1], b[1:])[0, 1]
    corr_b_leads = np.corrcoef(b[:-1], a[1:])[0, 1]
    if not np.isfinite(corr_a_leads) or not np.isfinite(corr_b_leads):
        return None

    threshold = 0.3
    if corr_a_leads > threshold and corr_a_leads > corr_b_leads:
        return "forward"
    if corr_b_leads > threshold and corr_b_leads > corr_a_leads:
        return "reverse"
    return None


def _build_graph_correlation(metrics: pd.DataFrame, corr_threshold: float = 0.3) -> GraphDict:
    """Fallback: lagged-correlation heuristic graph (not true causation)."""
    columns = list(metrics.columns)
    edges: list[dict[str, Any]] = []

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            direction = _lagged_correlation_direction(metrics[col_a], metrics[col_b])
            if direction == "forward":
                weight = float(abs(np.corrcoef(metrics[col_a], metrics[col_b])[0, 1]))
                if weight >= corr_threshold:
                    edges.append({"source": col_a, "target": col_b, "weight": weight})
            elif direction == "reverse":
                weight = float(abs(np.corrcoef(metrics[col_a], metrics[col_b])[0, 1]))
                if weight >= corr_threshold:
                    edges.append({"source": col_b, "target": col_a, "weight": weight})

    nodes = [{"id": col, "label": col} for col in columns]
    return {
        "nodes": nodes,
        "edges": edges,
        "method": "correlation_fallback",
    }


def build_causal_graph(
    metrics: pd.DataFrame,
    *,
    use_pyrca: bool = True,
    corr_threshold: float = 0.3,
) -> GraphDict:
    """Build a causal graph from a metric DataFrame.

    Uses PyRCA when available and ``use_pyrca`` is True; otherwise falls back to
    lagged-correlation heuristics.
    """
    if metrics.empty or metrics.shape[1] < 2:
        raise ValueError("metrics must contain at least two columns and one row")

    frame = metrics.copy()
    frame.columns = [str(c) for c in frame.columns]

    if use_pyrca and _pyrca_available():
        try:
            return _build_graph_pyrca(frame)
        except Exception:
            pass

    return _build_graph_correlation(frame, corr_threshold=corr_threshold)


def graph_to_json(graph: GraphDict) -> GraphDict:
    """Return graph dict suitable for API / frontend export."""
    return {
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "method": graph.get("method", "unknown"),
    }
