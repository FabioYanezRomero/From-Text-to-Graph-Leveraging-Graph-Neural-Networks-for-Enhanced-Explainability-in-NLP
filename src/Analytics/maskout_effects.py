from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import InsightFrame

_CANDIDATE_FEATURES: tuple[str, ...] = (
    "struct_density",
    "struct_cut_ratio",
    "struct_boundary_edges",
    "struct_components",
    "struct_induced_num_nodes",
    "struct_induced_num_edges",
    "struct_avg_shortest_path",
    "semantic_density",
    "sparsity",
    "minimal_coalition_size",
    "prediction_confidence",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _correlations(frame: pd.DataFrame, target: str, features: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    target_values = _numeric(frame[target])
    if target_values.empty:
        return scores
    for feature in features:
        if feature not in frame.columns:
            continue
        aligned = pd.concat(
            [target_values, _numeric(frame[feature])],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.shape[0] < 3:
            continue
        corr = float(aligned.corr().iloc[0, 1])
        if not np.isnan(corr):
            scores[feature] = corr
    return scores


def _plot_scatter(frame: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    aligned = pd.concat(
        [_numeric(frame[x_col]), _numeric(frame[y_col])],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(aligned.iloc[:, 0], aligned.iloc[:, 1], alpha=0.5, s=20, color="#4C72B0")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(f"{x_col} vs {y_col}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_maskout_effects(insight: InsightFrame, output_dir: Path, group_key: str) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = insight.data
    target = "maskout_effect"
    if target not in frame.columns:
        return {"available": False, "reason": "maskout_effect column missing"}

    maskout = _numeric(frame[target])
    if maskout.empty:
        return {"available": False, "reason": "no numeric maskout values"}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(maskout, bins=min(40, max(10, int(np.sqrt(maskout.size)))), color="#8172B3", edgecolor="black")
    ax.set_xlabel("maskout_effect")
    ax.set_ylabel("Count")
    ax.set_title("Maskout Effect Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "maskout_effect_hist.png", dpi=200)
    plt.close(fig)

    features = [col for col in _CANDIDATE_FEATURES if col in frame.columns]
    correlations = _correlations(frame, target, features)
    if correlations:
        corr_frame = pd.DataFrame(
            [{"feature": feature, "correlation": score} for feature, score in correlations.items()]
        ).sort_values(by="correlation", key=lambda s: s.abs(), ascending=False)
        corr_frame.to_csv(output_dir / "maskout_effect_correlations.csv", index=False)
        top_feature = corr_frame.iloc[0]["feature"]
        _plot_scatter(frame, top_feature, target, output_dir / f"{top_feature}_vs_maskout_effect.png")

    group_summary: Dict[str, Dict[str, float]] = {}
    if group_key in frame.columns:
        for group_value, subset in frame.groupby(group_key, dropna=False):
            values = _numeric(subset[target])
            if values.empty:
                continue
            label = "None" if pd.isna(group_value) else str(group_value)
            group_summary[label] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(values.median()),
            }

    return {"available": True, "correlations": correlations, "group_summary": group_summary}
