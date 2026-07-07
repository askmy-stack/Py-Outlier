"""Tests for causal graph construction and root-cause ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_detection.rca.graph import build_causal_graph
from anomaly_detection.rca.scorer import RCA_DISCLAIMER, analyze_root_cause, rank_root_causes


def _synthetic_chain(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build A -> B -> C chain: B lags A, C lags B."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
    b = np.empty(n)
    c = np.empty(n)
    b[0] = a[0]
    c[0] = b[0]
    for t in range(1, n):
        b[t] = 0.7 * a[t - 1] + 0.3 * a[t] + rng.normal(0, 0.05)
        c[t] = 0.7 * b[t - 1] + 0.3 * b[t] + rng.normal(0, 0.05)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_build_causal_graph_chain_structure() -> None:
    metrics = _synthetic_chain()
    graph = build_causal_graph(metrics, use_pyrca=False)

    assert graph["method"] == "correlation_fallback"
    assert len(graph["nodes"]) == 3
    edge_pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("A", "B") in edge_pairs
    assert ("B", "C") in edge_pairs


def test_rank_root_causes_anomaly_at_source() -> None:
    metrics = _synthetic_chain()
    anomaly_idx = len(metrics) - 1
    metrics.loc[anomaly_idx, "A"] += 20.0

    graph = build_causal_graph(metrics, use_pyrca=False)
    causes = rank_root_causes(metrics, graph, timestamp=anomaly_idx, top_k=3, use_pyrca=False)

    assert len(causes) == 3
    assert causes[0]["metric"] == "A"
    assert causes[0]["rank"] == 1
    assert causes[0]["score"] > causes[1]["score"]


def test_analyze_root_cause_includes_disclaimer() -> None:
    metrics = _synthetic_chain()
    metrics.iloc[-1, metrics.columns.get_loc("A")] += 15.0

    result = analyze_root_cause(metrics, use_pyrca=False)

    assert "disclaimer" in result
    assert "causation" in result["disclaimer"].lower() or "causal" in result["disclaimer"].lower()
    assert RCA_DISCLAIMER.split(".")[0] in result["disclaimer"]
    assert result["causes"][0]["metric"] == "A"


def test_rank_defaults_to_last_row() -> None:
    metrics = _synthetic_chain(n=50)
    metrics.iloc[-1, 0] += 25.0

    causes = rank_root_causes(metrics, use_pyrca=False)
    assert causes[0]["metric"] == "A"
