"""Faithfulness aggregation based on insertion AUC metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .utils import InsightFrame, load_insights


@dataclass
class FaithfulnessConfig:
    baseline: float = 0.5
    sensitivity_threshold: float = 0.0  # threshold for flagging positive faithfulness


def _sanitize_folder_name(dataset: str, graph_type: str | None, method: str | None) -> str:
    parts = [dataset.replace("/", "_")]
    if graph_type:
        parts.append(graph_type)
    if method:
        parts.append(method.replace("/", "_"))
    return "_".join(part for part in parts if part)


def _coerce_series(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return numeric.to_numpy()


def _faithfulness_stats(values: pd.Series, cfg: FaithfulnessConfig) -> Dict[str, float]:
    numeric = _coerce_series(values)
    if numeric.size == 0:
        return {}

    diffs = numeric - cfg.baseline
    count = diffs.size
    mean = float(diffs.mean())
    std = float(diffs.std(ddof=1)) if count > 1 else 0.0
    var = float(diffs.var(ddof=1)) if count > 1 else 0.0
    median = float(np.median(diffs))
    q25 = float(np.percentile(diffs, 25))
    q75 = float(np.percentile(diffs, 75))
    mad = float(np.mean(np.abs(diffs - mean)))
    sensitivity = float(np.mean(diffs > cfg.sensitivity_threshold))
    stability = float(1.0 / (1.0 + std))

    if count > 1 and std > 0.0:
        t_stat, p_value = stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        "count": int(count),
        "mean_faithfulness": mean,
        "std_faithfulness": std,
        "var_faithfulness": var,
        "median_faithfulness": median,
        "q25_faithfulness": q25,
        "q75_faithfulness": q75,
        "mad_faithfulness": mad,
        "sensitivity": sensitivity,
        "stability": stability,
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "mean_insertion_auc": float(numeric.mean()),
        "std_insertion_auc": float(numeric.std(ddof=1)) if count > 1 else 0.0,
    }


def _aggregate_groups(
    frame: pd.DataFrame,
    cfg: FaithfulnessConfig,
    group_column: str,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if group_column not in frame.columns:
        return results

    for value, subset in frame.groupby(group_column, dropna=False):
        stats_dict = _faithfulness_stats(subset["insertion_auc"], cfg)
        if not stats_dict:
            continue
        label = "None" if pd.isna(value) else str(value)
        results[label] = stats_dict
    return results


def _aggregate_correctness(frame: pd.DataFrame, cfg: FaithfulnessConfig) -> Dict[str, Dict[str, float]]:
    column = None
    if "is_correct" in frame.columns:
        column = "is_correct"
    elif "accuracy" in frame.columns:
        column = "accuracy"
    if column is None:
        return {}

    results: Dict[str, Dict[str, float]] = {}
    for value, subset in frame.groupby(column, dropna=False):
        stats_dict = _faithfulness_stats(subset["insertion_auc"], cfg)
        if not stats_dict:
            continue
        label = (
            "unknown"
            if pd.isna(value)
            else ("correct" if bool(value) else "incorrect")
        )
        results[label] = stats_dict
    return results


def run_faithfulness_aggregate(
    insight_paths: Iterable[str],
    output_dir: Path,
    *,
    baseline: float = 0.5,
    group_key: str = "label",
) -> Dict[str, object]:
    cfg = FaithfulnessConfig(baseline=baseline)
    insight = load_insights(list(insight_paths))
    frame = insight.data.copy()
    frame = frame[pd.to_numeric(frame.get("insertion_auc"), errors="coerce").notna()]
    if frame.empty:
        raise ValueError("No valid insertion_auc values found in supplied insights.")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "baseline": baseline,
        "groups": {},
        "output_dir": str(output_dir),
    }

    required_columns = {"dataset", "graph_type", "method", "insertion_auc"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in insight data: {', '.join(sorted(missing_columns))}")

    grouped = frame.groupby(["dataset", "graph_type", "method"], dropna=False)
    for (dataset, graph_type, method), subset in grouped:
        stats_dict = _faithfulness_stats(subset["insertion_auc"], cfg)
        if not stats_dict:
            continue

        label_groups = _aggregate_groups(subset, cfg, group_key)
        correctness_groups = _aggregate_correctness(subset, cfg)

        folder = output_dir / _sanitize_folder_name(dataset or "unknown_dataset", graph_type, method)
        folder.mkdir(parents=True, exist_ok=True)

        result_payload = {
            "dataset": dataset,
            "graph_type": graph_type,
            "method": method,
            "overall": stats_dict,
            "by_label": label_groups,
            "by_correctness": correctness_groups,
        }

        (folder / "faithfulness_summary.json").write_text(
            json.dumps(result_payload, indent=2),
            encoding="utf-8",
        )
        summary["groups"][f"{dataset}:{graph_type}:{method}"] = {
            **stats_dict,
            "folder": str(folder),
        }

    overall_summary_path = output_dir / "faithfulness_overview.json"
    overall_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["overview_path"] = str(overall_summary_path)
    return summary
