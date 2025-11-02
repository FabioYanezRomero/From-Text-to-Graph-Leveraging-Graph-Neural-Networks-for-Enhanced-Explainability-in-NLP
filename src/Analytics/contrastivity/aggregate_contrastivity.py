#!/usr/bin/env python3
"""
Aggregate contrastivity analytics for every explainer/dataset/graph CSV.

The input directories follow the structure produced by the contrastivity
extractors:

    outputs/analytics/contrastivity/<method>/<dataset>/<graph>.csv

Each row contains scalar contrastivity values (origin/masked/maskout) together
with metadata such as class labels and correctness flags.  This script mirrors
the progression aggregator by emitting per-graph summaries and a field-level
global summary for each contrastivity signal:

    outputs/analytics/contrastivity/<field>/<method>/<dataset>/<graph>/
        contrastivity_summary.csv
    outputs/analytics/contrastivity/<field>/contrastivity_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BASE_INPUT_ROOT = Path("outputs/analytics/contrastivity")
DEFAULT_OUTPUT_ROOT = BASE_INPUT_ROOT

CONTRASTIVITY_FIELDS = (
    "origin_contrastivity",
    "masked_contrastivity",
    "maskout_contrastivity",
)

SUMMARY_FILENAME = "contrastivity_summary.csv"
EPS = 1e-10


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


def coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except Exception:
                return None
    return None


def add_group_value(groups: Dict[str, List[float]], group: str, value: float) -> None:
    groups.setdefault(group, []).append(value)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def discover_contrastivity_csvs(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3:  # method/dataset/graph.csv
            candidates.append(path)
    return sorted(candidates)


def process_csv(csv_path: Path, field: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty or field not in df.columns:
        return pd.DataFrame()

    method = csv_path.parts[-3]
    dataset = csv_path.parts[-2]
    graph = csv_path.stem

    records: List[Dict[str, object]] = []
    metrics_accumulator: Dict[str, Dict[str, List[float]]] = {field: {"overall": []}}

    for _, row in df.iterrows():
        value = coerce_float(row.get(field))
        if value is None:
            continue

        label = normalize_label(row.get("label"))

        correctness: Optional[bool] = None
        is_correct_raw = row.get("is_correct")
        if isinstance(is_correct_raw, (bool, np.bool_)):
            correctness = bool(is_correct_raw)
        elif isinstance(is_correct_raw, str):
            text = is_correct_raw.strip().lower()
            if text in {"true", "1", "yes", "y"}:
                correctness = True
            elif text in {"false", "0", "no", "n"}:
                correctness = False

        bucket = metrics_accumulator[field]
        add_group_value(bucket, "overall", float(value))
        if correctness is not None:
            add_group_value(bucket, f"correct_{correctness}", float(value))
        if label is not None:
            add_group_value(bucket, f"class_{label}", float(value))
            if correctness is not None:
                add_group_value(bucket, f"class_{label}_correct_{correctness}", float(value))

    for metric_name, group_values in metrics_accumulator.items():
        for group, values in group_values.items():
            stats = summarize(values)
            if stats["count"] == 0:
                continue
            records.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "graph": graph,
                    "contrastivity_field": metric_name,
                    "group": group,
                    **stats,
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def aggregate_field(csv_paths: List[Path], field: str, output_root: Path) -> None:
    field_root = output_root / field
    field_root.mkdir(parents=True, exist_ok=True)

    all_records: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        summary_df = process_csv(csv_path, field)
        if summary_df.empty:
            continue

        method = summary_df["method"].iloc[0]
        dataset = summary_df["dataset"].iloc[0]
        graph = summary_df["graph"].iloc[0]

        graph_dir = field_root / method / dataset / graph
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_summary_path = graph_dir / SUMMARY_FILENAME
        summary_df.sort_values(
            ["method", "dataset", "graph", "contrastivity_field", "group"],
            inplace=True,
            kind="mergesort",
        )
        summary_df.to_csv(graph_summary_path, index=False)
        print(f"  [field:{field}] {graph_summary_path} ({len(summary_df)} rows)")
        all_records.append(summary_df)

    if not all_records:
        print(f"  ⚠️  No data produced for field '{field}'.")
        return

    global_df = pd.concat(all_records, ignore_index=True)
    global_summary_path = field_root / SUMMARY_FILENAME
    global_df.sort_values(
        ["method", "dataset", "graph", "contrastivity_field", "group"],
        inplace=True,
        kind="mergesort",
    )
    global_df.to_csv(global_summary_path, index=False)
    print(f"  [field:{field}] global summary -> {global_summary_path} ({len(global_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate contrastivity metrics across methods/datasets/graphs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_INPUT_ROOT,
        help="Root directory containing contrastivity CSV files (method/dataset/graph.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where aggregated summaries will be written.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(CONTRASTIVITY_FIELDS),
        help="Contrastivity fields to aggregate (default: origin/masked/maskout).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    fields = list(dict.fromkeys(args.fields))
    csv_paths = discover_contrastivity_csvs(base_dir)
    if not csv_paths:
        print("No contrastivity CSV files found.")
        return

    print("=" * 120)
    print(f"[contrastivity] base_dir={base_dir}")
    print(f"[contrastivity] output_root={args.output_root.resolve()}")
    print(f"[contrastivity] fields={fields}")
    print("=" * 120)

    for field in fields:
        print("\n" + "-" * 120)
        print(f"[contrastivity] processing field: {field}")
        print("-" * 120)
        aggregate_field(csv_paths, field, args.output_root.resolve())

    print("\n" + "=" * 120)
    print("Contrastivity aggregation complete.")
    print("=" * 120)


if __name__ == "__main__":
    main()
