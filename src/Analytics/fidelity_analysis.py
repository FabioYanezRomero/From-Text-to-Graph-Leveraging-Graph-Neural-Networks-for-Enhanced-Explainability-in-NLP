from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import InsightFrame


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _stats(values: pd.Series) -> Dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _plot(values: pd.Series, path: Path, kind: str) -> None:
    if values.empty:
        return
    if kind == "hist":
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(values.size)))), color="#55A868", edgecolor="black")
        ax.set_xlabel("fidelity_drop")
        ax.set_ylabel("Count")
        ax.set_title("Fidelity Drop Distribution")
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.boxplot(x=values, ax=ax, color="#C44E52")
        ax.set_xlabel("fidelity_drop")
        ax.set_title("Fidelity Drop Spread")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_fidelity_analysis(insight: InsightFrame, output_dir: Path, group_key: str) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = insight.data
    if "fidelity_drop" not in frame.columns:
        return {"available": False, "reason": "fidelity_drop column missing"}

    fidelity = _numeric(frame["fidelity_drop"])
    if fidelity.empty:
        return {"available": False, "reason": "no numeric fidelity values"}

    _plot(fidelity, output_dir / "fidelity_drop_hist.png", "hist")
    _plot(fidelity, output_dir / "fidelity_drop_box.png", "box")

    summary = {"overall": _stats(fidelity), "by_group": {}, "extremes": {}}

    if group_key in frame.columns:
        for group_value, subset in frame.groupby(group_key, dropna=False):
            values = _numeric(subset["fidelity_drop"])
            if values.empty:
                continue
            label = "None" if pd.isna(group_value) else str(group_value)
            summary["by_group"][label] = {
                **_stats(values),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }

    if "graph_index" in frame.columns:
        ordered = frame.assign(fidelity_drop=pd.to_numeric(frame["fidelity_drop"], errors="coerce"))
        ordered = ordered.dropna(subset=["fidelity_drop"])
        if not ordered.empty:
            cols = ["graph_index", "fidelity_drop"]
            if group_key in ordered.columns and group_key not in cols:
                cols.append(group_key)
            head = ordered.nlargest(20, "fidelity_drop")[cols]
            tail = ordered.nsmallest(20, "fidelity_drop")[cols]
            head.to_csv(output_dir / "highest_fidelity_drop.csv", index=False)
            tail.to_csv(output_dir / "lowest_fidelity_drop.csv", index=False)
            summary["extremes"] = {
                "max_graph_index": int(head.iloc[0]["graph_index"]) if not head.empty else None,
                "min_graph_index": int(tail.iloc[0]["graph_index"]) if not tail.empty else None,
            }

    return summary
