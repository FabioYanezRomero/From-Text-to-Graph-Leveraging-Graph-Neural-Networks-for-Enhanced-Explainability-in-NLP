from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import InsightFrame


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _stats(series: pd.Series) -> Dict[str, float]:
    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _plot_auc_hist(values: pd.Series, path: Path) -> None:
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = min(40, max(10, int(np.sqrt(values.size))))
    ax.hist(values, bins=bins, color="#DD8452", edgecolor="black")
    ax.set_xlabel("insertion_auc")
    ax.set_ylabel("Count")
    ax.set_title("Insertion AUC Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_insertion_auc(insight: InsightFrame, output_dir: Path, group_key: str) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = insight.data

    if "insertion_auc" not in frame.columns:
        return {"available": False, "reason": "insertion_auc column missing"}

    auc_values = _numeric(frame["insertion_auc"])
    if auc_values.empty:
        return {"available": False, "reason": "no numeric insertion_auc values"}

    _plot_auc_hist(auc_values, output_dir / "insertion_auc_hist.png")

    group_summary: Dict[str, Dict[str, float]] = {}
    if group_key in frame.columns:
        for group_value, subset in frame.groupby(group_key, dropna=False):
            values = _numeric(subset["insertion_auc"])
            if values.empty:
                continue
            label = "None" if pd.isna(group_value) else str(group_value)
            group_summary[label] = _stats(values)

    return {"available": True, "auc_stats": _stats(auc_values), "group_summary": group_summary}
