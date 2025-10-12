"""Relate graph structural complexity to explainability quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)

from .utils import InsightFrame, default_argument_parser, load_insights

STRUCT_FEATURES = [
    "struct_density",
    "struct_components",
    "struct_boundary_edges",
    "struct_cut_ratio",
    "struct_induced_num_nodes",
    "struct_induced_num_edges",
    "struct_avg_shortest_path",
]

TARGETS = [
    "prediction_confidence",
    "fidelity_drop",
    "maskout_effect",
    "minimal_coalition_confidence",
    "minimal_coalition_size",
]


def _bin_series(series: pd.Series, bins: int = 3) -> pd.Series:
    """Quantile-bin a series, returning consistent categorical labels."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype="category")

    unique_values = valid.nunique()
    if unique_values < 2:
        result = pd.Series(["bin_1"] * len(valid), index=valid.index, dtype="object")
        return result.reindex(series.index).astype("category")

    q = min(bins, unique_values)
    binned, _ = pd.qcut(valid, q=q, labels=False, retbins=True, duplicates="drop")
    labels = [f"bin_{i + 1}" for i in range(int(binned.nunique()))]
    mapped = binned.map(lambda idx: labels[int(idx)] if pd.notna(idx) else None)
    out = pd.Series(index=series.index, dtype="object")
    out.loc[mapped.index] = mapped
    return out.astype("category")


def _flatten_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Convert a multi-index column dataframe into JSON-friendly structure."""
    if df.empty:
        return {}
    flat = df.copy()
    flat.columns = [f"{metric}_{stat}" for metric, stat in flat.columns]
    payload: Dict[str, Dict[str, float]] = {}
    for index, row in flat.to_dict(orient="index").items():
        payload[str(index)] = {
            column: float(value) if isinstance(value, (int, float)) and pd.notna(value) else value
            for column, value in row.items()
            if pd.notna(value)
        }
    return payload


def run_structural_complexity(insight: InsightFrame, output_dir: Path, group_key: str) -> dict:
    """Evaluate how structural features interact with fidelity and confidence."""
    frame = insight.data
    output_dir.mkdir(parents=True, exist_ok=True)

    available_struct = [col for col in STRUCT_FEATURES if col in frame.columns]
    available_targets = [col for col in TARGETS if col in frame.columns]

    correlations = {}
    corr_cols = [col for col in available_struct + available_targets if frame[col].notna().any()]
    if len(corr_cols) >= 2:
        corr_matrix = frame[corr_cols].corr()
        corr_matrix.to_csv(output_dir / "structural_complexity_correlations.csv")
        struct_cols = [col for col in available_struct if col in corr_matrix.index]
        target_cols = [col for col in available_targets if col in corr_matrix.columns]
        correlations = corr_matrix.loc[struct_cols, target_cols].to_dict()

    density_col = "struct_density" if "struct_density" in frame.columns else None
    complexity_results: Dict[str, dict] = {}
    if density_col:
        density_bins = _bin_series(frame[density_col], bins=4)
        temp = frame.copy()
        temp["density_bin"] = density_bins
        density_stats = (
            temp.dropna(subset=["density_bin"])
            .groupby("density_bin", observed=False)[["fidelity_drop", "prediction_confidence", "minimal_coalition_confidence", "minimal_coalition_size"]]
            .agg(["mean", "median", "std", "count"])
        )
        density_stats.to_csv(output_dir / "density_bins_summary.csv")
        complexity_results["density_bins"] = _flatten_stats(density_stats)

    coalition_col = "minimal_coalition_size"
    if coalition_col in frame.columns:
        coalition_bins = _bin_series(frame[coalition_col], bins=4)
        temp = frame.copy()
        temp["coalition_bin"] = coalition_bins
        coalition_stats = (
            temp.dropna(subset=["coalition_bin"])
            .groupby("coalition_bin", observed=False)[["prediction_confidence", "fidelity_drop", "maskout_effect"]]
            .agg(["mean", "median", "std", "count"])
        )
        coalition_stats.to_csv(output_dir / "coalition_bins_summary.csv")
        complexity_results["coalition_bins"] = _flatten_stats(coalition_stats)

    if density_col and coalition_col:
        valid = frame[[density_col, coalition_col, "fidelity_drop"]].dropna()
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(valid[density_col], valid[coalition_col], c=valid["fidelity_drop"], cmap="viridis", alpha=0.6)
            ax.set_xlabel("Graph Density")
            ax.set_ylabel("Minimal Coalition Size")
            ax.set_title("Density vs Coalition Size (colour=fidelity drop)")
            fig.colorbar(scatter, ax=ax, label="Fidelity Drop")
            fig.tight_layout()
            fig.savefig(output_dir / "density_coalition_heat.png", dpi=200)
            plt.close(fig)

    boundary_col = "struct_boundary_edges"
    if boundary_col in frame.columns:
        scatter_data = frame[[boundary_col, "fidelity_drop"]].dropna()
        if not scatter_data.empty:
            coefficients = np.polyfit(scatter_data[boundary_col], scatter_data["fidelity_drop"], 1)
            slope, intercept = coefficients
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(scatter_data[boundary_col], scatter_data["fidelity_drop"], alpha=0.5, color="#DD8452", s=18)
            ax.plot(scatter_data[boundary_col], slope * scatter_data[boundary_col] + intercept, color="#4C72B0")
            ax.set_xlabel("Boundary Edges")
            ax.set_ylabel("Fidelity Drop")
            ax.set_title("Boundary Edges vs Fidelity Drop")
            fig.tight_layout()
            fig.savefig(output_dir / "boundary_vs_fidelity.png", dpi=200)
            plt.close(fig)

    group_summary = {}
    if group_key in frame.columns:
        selection = frame[[group_key] + available_struct + available_targets].copy()
        group_stats = selection.groupby(group_key, observed=False).agg(["mean", "median", "std", "count"])
        group_stats.to_csv(output_dir / f"{group_key}_structural_complexity.csv")
        group_summary = _flatten_stats(group_stats)

    return {
        "structural_features": available_struct,
        "targets": available_targets,
        "correlations": correlations,
        "complexity_results": complexity_results,
        "group_key": group_key,
        "group_summary": group_summary,
    }


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for structural complexity analytics."""
    parser = default_argument_parser("Link structural complexity with explainability quality.")
    args = parser.parse_args(argv)

    insight = load_insights(args.insight_paths)
    summary = run_structural_complexity(insight, args.output_dir, args.group_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "structural_complexity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
