from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import InsightFrame

_CANDIDATE_COLUMNS: tuple[str, ...] = (
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
    "struct_density",
    "struct_cut_ratio",
    "struct_boundary_edges",
    "struct_components",
    "struct_avg_shortest_path",
    "semantic_density",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(values: pd.Series) -> pd.Series:
    clean = _numeric(values)
    mean = clean.mean()
    std = clean.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(index=clean.index, dtype=float)
    return (clean - mean) / std


def _prepare_metrics(frame: pd.DataFrame) -> List[str]:
    metrics = []
    for column in _CANDIDATE_COLUMNS:
        if column in frame.columns:
            metrics.append(column)
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column not in metrics:
            metrics.append(column)
    return metrics


def run_group_outlier_analysis(insight: InsightFrame, output_dir: Path, group_key: str, *, z_threshold: float = 2.5) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = insight.data.copy()

    metrics = _prepare_metrics(frame)
    if not metrics:
        return {"available": False, "reason": "no numeric metrics detected"}

    outliers: List[Dict[str, object]] = []
    overall_counts: Dict[str, int] = {metric: 0 for metric in metrics}

    def _record_outliers(df: pd.DataFrame, label: str | None) -> None:
        for metric in metrics:
            z = _zscore(df[metric])
            if z.empty:
                continue
            flags = z.abs() >= z_threshold
            if not flags.any():
                continue
            overall_counts[metric] += int(flags.sum())
            for idx in z[flags].index:
                row = df.loc[idx]
                record = {
                    "group": label,
                    "graph_index": int(row["graph_index"]) if "graph_index" in row and not pd.isna(row["graph_index"]) else None,
                    "metric": metric,
                    "value": float(pd.to_numeric(row[metric], errors="coerce")),
                    "zscore": float(z.loc[idx]),
                }
                outliers.append(record)

    _record_outliers(frame, None)

    if group_key in frame.columns:
        for group_value, subset in frame.groupby(group_key, dropna=False):
            label = "None" if pd.isna(group_value) else str(group_value)
            _record_outliers(subset, label)

    if outliers:
        pd.DataFrame(outliers).sort_values(by="zscore", key=lambda s: s.abs(), ascending=False).to_csv(
            output_dir / "group_outliers.csv",
            index=False,
        )

    summary = {
        "available": True,
        "metrics_considered": metrics,
        "outlier_counts": {metric: count for metric, count in overall_counts.items() if count > 0},
        "total_outliers": len(outliers),
    }
    return summary
