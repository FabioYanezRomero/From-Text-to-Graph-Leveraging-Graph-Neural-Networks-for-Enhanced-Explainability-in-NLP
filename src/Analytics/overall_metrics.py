from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .utils import InsightFrame

_DEFAULT_METRICS: tuple[str, ...] = (
    "prediction_confidence",
    "origin_confidence",
    "masked_confidence",
    "maskout_confidence",
    "fidelity_drop",
    "maskout_effect",
    "sparsity",
    "minimal_coalition_size",
    "minimal_coalition_confidence",
    "insertion_auc",
    "graph_density",
    "struct_density",
    "struct_cut_ratio",
    "struct_boundary_edges",
    "struct_components",
    "struct_avg_shortest_path",
    "semantic_density",
)


def _numeric_columns(frame: pd.DataFrame) -> Iterable[str]:
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    for candidate in _DEFAULT_METRICS:
        if candidate in frame.columns:
            numeric.add(candidate)
    return sorted(numeric)


def _series_stats(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def run_overall_metrics(insight: InsightFrame, group_key: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    frame = insight.data.copy()
    metrics = _numeric_columns(frame)
    overall: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        stats = _series_stats(frame[metric])
        if stats:
            overall[metric] = stats

    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    if group_key in frame.columns:
        for group_value, subset in frame.groupby(group_key, dropna=False):
            label = "None" if pd.isna(group_value) else str(group_value)
            bucket: Dict[str, Dict[str, float]] = {}
            for metric in metrics:
                stats = _series_stats(subset[metric])
                if stats:
                    bucket[metric] = stats
            if bucket:
                grouped[label] = bucket

    return {"overall": overall, "by_group": grouped}
