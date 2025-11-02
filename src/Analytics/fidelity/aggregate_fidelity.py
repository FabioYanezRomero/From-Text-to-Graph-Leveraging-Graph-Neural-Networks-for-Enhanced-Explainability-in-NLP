#!/usr/bin/env python3
"""
Aggregate fidelity analytics across all methods/datasets/graphs.

Input layout (produced by the fidelity extractors):

    outputs/analytics/fidelity/<method>/<dataset>/<graph>.csv

Each row contains fidelity+/- values alongside metadata (label, correctness,
confidence, sparsity, etc.).  This script summarises those CSVs using the same
stratification scheme as the progression aggregators:

  • overall
  • correct / incorrect cohorts
  • per-class cohorts
  • per-class × correctness cohorts

For every graph CSV we emit a summary file at:

    <output_root>/<method>/<dataset>/<graph>/fidelity_summary.csv

and we write a field-level global summary at:

    <output_root>/fidelity_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BASE_INPUT_ROOT = Path("outputs/analytics/fidelity")
DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/fidelity")
SUMMARY_FILENAME = "fidelity_summary.csv"

REQUIRED_COLUMNS = {
    "method",
    "graph_type",
    "label",
    "is_correct",
    "prediction_confidence",
    "fidelity_plus",
    "fidelity_minus",
    "sparsity",
}


def normalize_label(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def discover_fidelity_csvs(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3:  # method/dataset/graph.csv
            candidates.append(path)
    return sorted(candidates)


def dataset_slug(df: pd.DataFrame) -> str:
    if "dataset_backbone" in df.columns and pd.notna(df["dataset_backbone"].iloc[0]):
        return str(df["dataset_backbone"].iloc[0]).replace("/", "_")
    if "dataset" in df.columns and pd.notna(df["dataset"].iloc[0]):
        return str(df["dataset"].iloc[0]).replace("/", "_")
    return "unknown_dataset"


def compute_diagnostics(group: pd.DataFrame) -> Dict[str, float]:
    diagnostics: Dict[str, float] = {}
    if group.empty:
        return diagnostics

    fid_plus_neg = (group["fidelity_plus"] < 0).sum()
    diagnostics["fidelity_plus_negative_count"] = int(fid_plus_neg)
    diagnostics["fidelity_plus_negative_pct"] = round(
        float(fid_plus_neg) * 100.0 / len(group), 2
    ) if len(group) else np.nan

    fid_plus_mean = group["fidelity_plus"].mean()
    fid_minus_mean = group["fidelity_minus"].mean()
    diagnostics["asymmetry_index"] = round(
        float((fid_plus_mean - fid_minus_mean) / (abs(fid_minus_mean) + 1e-6)), 4
    )
    diagnostics["fidelity_plus_minus_ratio"] = round(
        float(fid_plus_mean / (fid_minus_mean + 1e-6)), 4
    )

    diagnostics["corr_confidence_fidelity_plus"] = round(
        float(group["prediction_confidence"].corr(group["fidelity_plus"])), 4
    )
    diagnostics["corr_confidence_fidelity_minus"] = round(
        float(group["prediction_confidence"].corr(group["fidelity_minus"])), 4
    )
    diagnostics["corr_sparsity_fidelity_plus"] = round(
        float(group["sparsity"].corr(group["fidelity_plus"])), 4
    )
    diagnostics["corr_sparsity_fidelity_minus"] = round(
        float(group["sparsity"].corr(group["fidelity_minus"])), 4
    )
    return diagnostics


def summarise_group(
    group: pd.DataFrame,
    *,
    method: str,
    dataset: str,
    graph: str,
    group_name: str,
) -> Dict[str, object]:
    if group.empty:
        return {}

    summary: Dict[str, object] = {
        "method": method,
        "dataset": dataset,
        "graph": graph,
        "group": group_name,
        "sample_size": int(len(group)),
        "fidelity_plus_mean": round(float(group["fidelity_plus"].mean()), 4),
        "fidelity_plus_std": round(float(group["fidelity_plus"].std(ddof=0)), 4),
        "fidelity_plus_min": round(float(group["fidelity_plus"].min()), 4),
        "fidelity_plus_max": round(float(group["fidelity_plus"].max()), 4),
        "fidelity_plus_q25": round(float(group["fidelity_plus"].quantile(0.25)), 4),
        "fidelity_plus_q75": round(float(group["fidelity_plus"].quantile(0.75)), 4),
        "fidelity_minus_mean": round(float(group["fidelity_minus"].mean()), 4),
        "fidelity_minus_std": round(float(group["fidelity_minus"].std(ddof=0)), 4),
        "fidelity_minus_min": round(float(group["fidelity_minus"].min()), 4),
        "fidelity_minus_max": round(float(group["fidelity_minus"].max()), 4),
        "fidelity_minus_q25": round(float(group["fidelity_minus"].quantile(0.25)), 4),
        "fidelity_minus_q75": round(float(group["fidelity_minus"].quantile(0.75)), 4),
        "mean_prediction_confidence": round(float(group["prediction_confidence"].mean()), 4),
        "mean_sparsity": round(float(group["sparsity"].mean()), 4),
    }

    summary.update(compute_diagnostics(group))
    return summary


def process_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    method = str(df["method"].iloc[0])
    dataset = dataset_slug(df)
    graph = str(df["graph_type"].iloc[0])

    records: List[Dict[str, object]] = []

    records.append(
        summarise_group(df, method=method, dataset=dataset, graph=graph, group_name="overall")
    )

    for is_correct, subset in df.groupby("is_correct", dropna=False):
        label = f"correct_{str(is_correct).lower()}"
        records.append(
            summarise_group(subset, method=method, dataset=dataset, graph=graph, group_name=label)
        )

    for label_value, subset in df.groupby("label", dropna=False):
        label_str = normalize_label(label_value)
        group_label = f"class_{label_str}" if label_str is not None else "class_unknown"
        records.append(
            summarise_group(subset, method=method, dataset=dataset, graph=graph, group_name=group_label)
        )

    for (label_value, is_correct), subset in df.groupby(["label", "is_correct"], dropna=False):
        label_str = normalize_label(label_value)
        class_part = f"class_{label_str}" if label_str is not None else "class_unknown"
        corr_part = f"correct_{str(is_correct).lower()}"
        records.append(
            summarise_group(
                subset,
                method=method,
                dataset=dataset,
                graph=graph,
                group_name=f"{class_part}_{corr_part}",
            )
        )

    cleaned_records = [rec for rec in records if rec]
    if not cleaned_records:
        return pd.DataFrame()
    return pd.DataFrame(cleaned_records)


def aggregate_fidelity(csv_paths: List[Path], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        summary_df = process_csv(csv_path)
        if summary_df.empty:
            continue

        method = summary_df["method"].iloc[0]
        dataset = summary_df["dataset"].iloc[0]
        graph = summary_df["graph"].iloc[0]

        graph_dir = output_root / graph
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_df = summary_df.sort_values(
            ["method", "dataset", "graph", "group"],
            kind="mergesort",
        )
        graph_path = graph_dir / SUMMARY_FILENAME
        graph_df.to_csv(graph_path, index=False)
        print(f"  [fidelity] {graph_path} ({len(graph_df)} rows)")
        all_records.append(graph_df)

    if not all_records:
        print("No fidelity summaries produced.")
        return

    global_df = pd.concat(all_records, ignore_index=True)
    global_df.sort_values(
        ["method", "dataset", "graph", "group"],
        inplace=True,
        kind="mergesort",
    )
    global_path = output_root / SUMMARY_FILENAME
    global_df.to_csv(global_path, index=False)
    print(f"  [fidelity] global summary -> {global_path} ({len(global_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fidelity metrics across methods/datasets/graphs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_INPUT_ROOT,
        help="Root directory containing fidelity CSV files (method/dataset/graph.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where aggregated summaries will be written.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    csv_paths = discover_fidelity_csvs(base_dir)
    if not csv_paths:
        print("No fidelity CSV files found.")
        return

    print("=" * 120)
    print(f"[fidelity] base_dir={base_dir}")
    print(f"[fidelity] output_root={args.output_root.resolve()}")
    print("=" * 120)
    aggregate_fidelity(csv_paths, args.output_root.resolve())
    print("\nAggregation complete.")


if __name__ == "__main__":
    main()
