#!/usr/bin/env python3
"""Fit logistic models on top-performing modules and persist their coefficients."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUMMARY_PATH = Path("outputs/use_case/module_datasets/error_signal_classification_summary.csv")
DATASET_ROOT = Path("outputs/use_case/module_datasets")
OUTPUT_DIR = DATASET_ROOT / "coefficients"


def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError("Summary CSV is empty.")
    return df


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, List[str], np.ndarray]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"global_graph_index", "graph_index", "label", "label_id", "prediction_class", "is_correct"}
    feature_cols = [col for col in numeric_cols if col not in drop_cols]
    if not feature_cols:
        raise ValueError("No numeric analytic features available.")
    X = df[feature_cols].to_numpy(copy=True)
    y = df["is_correct"].astype(int).to_numpy(copy=True)
    if len(np.unique(y)) < 2:
        raise ValueError("Target variable has a single class; cannot fit logistic regression.")
    return X, feature_cols, y


def fit_logistic(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    pipe.fit(X, y)
    return pipe


def extract_coefficients(pipe, feature_names: List[str]) -> pd.DataFrame:
    lr: LogisticRegression = pipe.named_steps["logisticregression"]
    coef = lr.coef_.ravel()
    intercept = lr.intercept_[0]
    coeff_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    coeff_df = coeff_df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    coeff_df.attrs["intercept"] = intercept
    return coeff_df


def save_coefficients(coeff_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coeff_df.to_csv(output_path, index=False)
    intercept_path = output_path.with_suffix(".intercept.txt")
    intercept_path.write_text(f"intercept,{coeff_df.attrs.get('intercept', 0.0):.10f}\n")


def process_modules(summary: pd.DataFrame, output_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    results: Dict[Tuple[str, str, str], Path] = {}
    seen: set[Tuple[str, str, str]] = set()
    for _, row in summary.iterrows():
        dataset = row.get("dataset")
        method = row.get("method")
        graph = row.get("graph")
        if not isinstance(dataset, str) or not isinstance(method, str) or not isinstance(graph, str):
            continue
        key = (dataset, method, graph)
        if key in seen:
            continue
        seen.add(key)
        csv_path = DATASET_ROOT / dataset / f"module_dataset_{dataset}_{method}_{graph}.csv"
        if not csv_path.exists():
            print(f"Missing dataset CSV for {dataset}-{method}-{graph}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Empty dataset for {dataset}-{method}-{graph}, skipping.")
            continue
        try:
            X, feature_names, y = feature_matrix(df)
            pipe = fit_logistic(X, y)
            coeff_df = extract_coefficients(pipe, feature_names)
            coeff_path = output_dir / dataset / f"logistic_coefficients_{dataset}_{method}_{graph}.csv"
            save_coefficients(coeff_df, coeff_path)
            results[key] = coeff_path
        except ValueError as exc:
            print(f"Skipping {dataset}-{method}-{graph}: {exc}")
            continue
    return results


def dimension_for_feature(feature: str) -> str:
    normalized = feature.lower()
    if normalized.startswith("auc") or normalized.startswith("prediction_confidence"):
        return "AUC"
    if normalized.startswith("consistency"):
        return "Consistency"
    if normalized.startswith("progression"):
        return "Progression"
    if normalized.startswith("fidelity") or normalized.startswith("abs_fidelity") or normalized.startswith("sparsity"):
        return "Fidelity"
    raise ValueError(f"Unrecognised analytic feature '{feature}' – expected prefixes for AUC/Fidelity/Consistency/Progression.")


def summarise_dimensions(coeff_paths: Dict[Tuple[str, str, str], Path], output_dir: Path) -> Path:
    records = []
    for (dataset, method, graph), path in coeff_paths.items():
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["dimension"] = df["feature"].map(dimension_for_feature)
        agg = (
            df.groupby("dimension")["coefficient"].apply(lambda s: s.abs().sum()).reset_index(name="abs_weight_sum")
        )
        total = agg["abs_weight_sum"].sum()
        if total == 0:
            agg["weight_pct"] = 0.0
        else:
            agg["weight_pct"] = 100 * agg["abs_weight_sum"] / total
        agg["dataset"] = dataset
        agg["method"] = method
        agg["graph"] = graph
        records.append(agg)
    if not records:
        raise RuntimeError("No coefficient weights available to summarise.")
    summary = pd.concat(records, ignore_index=True)
    summary_path = output_dir / "dimension_weight_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save logistic coefficients for best-performing modules per dataset.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to the summary CSV containing accuracy results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where coefficient files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_path)
    results = process_modules(summary, args.output_dir)
    if not results:
        raise SystemExit("No coefficient files were produced.")
    print("Saved coefficient tables:")
    for (dataset, method, graph), path in results.items():
        print(f"  - {dataset}:{method}:{graph} -> {path}")
    summary_path = summarise_dimensions(results, args.output_dir)
    print(f"Dimension weight summary -> {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
