"""Root-cause ranking for detected anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.rca.graph import GraphDict, build_causal_graph

RCA_DISCLAIMER = (
    "Root cause rankings reflect statistical associations, not proven causal relationships. "
    "When PyRCA is unavailable, results use correlation-based heuristics only. "
    "Validate findings with domain knowledge before taking action."
)

CORRELATION_FALLBACK_DISCLAIMER = (
    "PyRCA is not installed; rankings use lagged-correlation heuristics only. "
    "Correlation does not imply causation."
)


def _resolve_timestamp_index(metrics: pd.DataFrame, timestamp: Any | None) -> int:
    if timestamp is None:
        return len(metrics) - 1

    if isinstance(timestamp, int):
        if timestamp < 0:
            timestamp = len(metrics) + timestamp
        if timestamp < 0 or timestamp >= len(metrics):
            raise ValueError(f"timestamp index {timestamp} out of range [0, {len(metrics)})")
        return timestamp

    if isinstance(timestamp, str):
        if timestamp in metrics.index.astype(str):
            return int(metrics.index.get_loc(timestamp))
        try:
            timestamp = int(timestamp)
            return _resolve_timestamp_index(metrics, timestamp)
        except ValueError as exc:
            raise ValueError(f"timestamp '{timestamp}' not found in metrics index") from exc

    # pandas Timestamp or datetime-like
    loc = metrics.index.get_loc(timestamp)
    if isinstance(loc, slice):
        return loc.start or 0
    if isinstance(loc, np.ndarray):
        return int(loc[0])
    return int(loc)


def _z_scores_at_index(metrics: pd.DataFrame, idx: int) -> pd.Series:
    values = metrics.iloc[idx].astype(float)
    std = metrics.std(ddof=0).replace(0, np.nan)
    z = (values - metrics.mean()) / std
    return z.fillna(0.0)


def _upstream_weights(graph: GraphDict) -> dict[str, float]:
    """Higher weight for nodes with fewer incoming edges (candidate roots)."""
    nodes = [n["id"] for n in graph.get("nodes", [])]
    incoming: dict[str, int] = dict.fromkeys(nodes, 0)
    for edge in graph.get("edges", []):
        target = edge["target"]
        incoming[target] = incoming.get(target, 0) + 1

    weights: dict[str, float] = {}
    for node in nodes:
        weights[node] = 1.0 / (1.0 + incoming.get(node, 0))
    return weights


def _correlation_boost(metrics: pd.DataFrame, z_scores: pd.Series) -> dict[str, float]:
    """Boost metrics that co-deviate with others (propagation signal)."""
    corr = metrics.corr().fillna(0.0)
    boosts: dict[str, float] = {}
    for col in metrics.columns:
        others = [c for c in metrics.columns if c != col]
        if not others:
            boosts[col] = 1.0
            continue
        weighted = sum(abs(corr.loc[col, other]) * abs(z_scores[other]) for other in others)
        boosts[col] = 1.0 + weighted / len(others)
    return boosts


def rank_root_causes(
    metrics: pd.DataFrame,
    graph: GraphDict | None = None,
    *,
    timestamp: Any | None = None,
    top_k: int = 3,
    use_pyrca: bool = True,
) -> list[dict[str, Any]]:
    """Rank likely root causes for an anomaly at the given timestamp.

    Returns a list of ``{"metric": str, "score": float, "rank": int}`` dicts.
    """
    frame = metrics.copy()
    frame.columns = [str(c) for c in frame.columns]
    idx = _resolve_timestamp_index(frame, timestamp)

    if graph is None:
        graph = build_causal_graph(frame, use_pyrca=use_pyrca)

    z_scores = _z_scores_at_index(frame, idx)
    upstream = _upstream_weights(graph)
    boosts = _correlation_boost(frame, z_scores)

    scored: list[tuple[str, float]] = []
    for col in frame.columns:
        deviation = abs(float(z_scores[col]))
        score = deviation * upstream.get(col, 1.0) * boosts.get(col, 1.0)
        scored.append((col, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:top_k]

    return [
        {"metric": metric, "score": round(score, 6), "rank": rank}
        for rank, (metric, score) in enumerate(top, start=1)
    ]


def analyze_root_cause(
    metrics: pd.DataFrame,
    *,
    timestamp: Any | None = None,
    top_k: int = 3,
    use_pyrca: bool = True,
) -> dict[str, Any]:
    """Full RCA result including graph, causes, and disclaimer."""
    graph = build_causal_graph(metrics, use_pyrca=use_pyrca)
    causes = rank_root_causes(metrics, graph, timestamp=timestamp, top_k=top_k, use_pyrca=False)
    method = graph.get("method", "unknown")
    disclaimer = RCA_DISCLAIMER
    if method == "correlation_fallback":
        disclaimer = f"{RCA_DISCLAIMER} {CORRELATION_FALLBACK_DISCLAIMER}"

    return {
        "graph": graph,
        "causes": causes,
        "method": method,
        "disclaimer": disclaimer,
    }
