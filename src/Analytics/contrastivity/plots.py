#!/usr/bin/env python3
"""
Contrastivity analytics visualisations.

This script consumes the aggregated contrastivity summaries produced by
``aggregate_contrastivity.py`` and recreates the six dashboards from the
initial notebook/sketch:

  1. Feature identification panels (masked contrastivity per graph type)
  2. Confounding robustness comparison (maskout contrastivity, 3-panel bars)
  3. Strategy profile radar chart (masked divergence, maskout mean, origin mean)
  4. Graph-type effectiveness comparison (masked contrastivity grouped bars)
  5. Method robustness violin plot (synthetic distributions from summary stats)
  6. Summary statistics table

Outputs are stored under ``outputs/analytics/contrastivity/plots/<dataset>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_ROOT = Path("outputs/analytics/contrastivity")
SUMMARY_FILENAME = "contrastivity_summary.csv"
PLOTS_ROOT = "plots"

CONTRASTIVITY_FIELDS = {
    "origin": "origin_contrastivity",
    "masked": "masked_contrastivity",
    "maskout": "maskout_contrastivity",
}

METHOD_ORDER: Sequence[str] = ("graphsvx", "subgraphx", "token_shap_llm")
METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
    "token_shap_llm": "TokenSHAP (LLM)",
}
METHOD_COLORS: Dict[str, str] = {
    "graphsvx": "#3498db",
    "subgraphx": "#e74c3c",
    "token_shap_llm": "#2ecc71",
}
GRAPH_ORDER = ["skipgrams", "window", "constituency", "syntactic", "tokens"]

DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def load_summary(root: Path, field: str) -> pd.DataFrame:
    path = root / field / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Field summary missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Field summary is empty: {path}")
    return df


def save_figure(fig: go.Figure, stem: Path, width: int, height: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    html_path = stem.with_suffix(".html")
    try:
        fig.write_image(str(pdf_path), width=width, height=height)
        print(f"  ✓ {pdf_path}")
    except Exception as exc:
        print(f"  ! PDF failed ({pdf_path.name}): {exc}")
    try:
        fig.write_html(str(html_path))
        print(f"  ✓ {html_path}")
    except Exception as exc:
        print(f"  ! HTML failed ({html_path.name}): {exc}")


def create_feature_id_panels(masked_df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = masked_df[
        (masked_df["dataset"] == dataset)
        & (masked_df["group"] == "correct_True")
    ]
    if subset.empty:
        return

    panels: List[Tuple[str, List[Dict[str, object]]]] = []

    for graph in GRAPH_ORDER:
        graph_df = subset[subset["graph"] == graph]
        if graph_df.empty:
            continue

        entries: List[Dict[str, object]] = []
        for method in METHOD_ORDER:
            row = graph_df[graph_df["method"] == method]
            if row.empty:
                continue
            entries.append(
                {
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    "mean": float(row["mean"].iloc[0]),
                    "std": float(row["std"].iloc[0]),
                    "count": int(row["count"].iloc[0]),
                }
            )

        if entries:
            panels.append((graph.title(), entries))

    if not panels:
        return

    rows = len(panels)
    height = max(420, rows * 220)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"{graph_title} · Correct predictions" for graph_title, _ in panels],
    )

    global_max = 0.0

    for row_idx, (graph_title, entries) in enumerate(panels, start=1):
        for entry in entries:
            mean_val = entry["mean"]
            std_val_raw = entry["std"]
            std_display = float(std_val_raw) if np.isfinite(std_val_raw) else np.nan
            has_error = np.isfinite(std_display) and std_display > 0
            std_bar = std_display if has_error else 0.0
            std_text = f"{std_display:.4f}" if np.isfinite(std_display) else "n/a"

            global_max = max(global_max, mean_val + (std_display if np.isfinite(std_display) else 0.0))

            fig.add_trace(
                go.Bar(
                    x=[mean_val],
                    y=[entry["label"]],
                    orientation="h",
                    name=entry["label"],
                    marker=dict(color=METHOD_COLORS.get(entry["method"], "#7f8c8d")),
                    legendgroup=entry["method"],
                    showlegend=row_idx == 1,
                    error_x=dict(type="data", array=[std_bar], visible=has_error),
                    hovertemplate=(
                        f"<b>{entry['label']}</b><br>"
                        f"Graph: {graph_title}<br>"
                        f"Masked contrastivity: {{x:.4f}}<br>"
                        f"Std: {std_text}<br>"
                        f"Count: {entry['count']}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(
            title_text="Method" if row_idx == 1 else "",
            row=row_idx,
            col=1,
            automargin=True,
        )

    axis_max = max(global_max * 1.05, 1.0)
    for row_idx in range(1, rows + 1):
        fig.update_xaxes(
            title_text="Masked Contrastivity (higher is better)" if row_idx == rows else "",
            row=row_idx,
            col=1,
            range=[0, axis_max],
            showgrid=True,
            gridcolor="rgba(210,210,210,0.4)",
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Feature Identification (Masked Contrastivity)</b><br>"
                f"<sub>{dataset_label(dataset)} · per-graph method performance</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=height,
        width=920,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        barmode="group",
    )
    save_figure(
        fig,
        output_dir / f"viz1_feature_id_panels_{dataset}",
        width=920,
        height=height,
    )

def create_confounding_robustness(maskout_df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    correct = maskout_df[(maskout_df["dataset"] == dataset) & (maskout_df["group"] == "correct_True")]
    incorrect = maskout_df[(maskout_df["dataset"] == dataset) & (maskout_df["group"] == "correct_False")]
    if correct.empty or incorrect.empty:
        return

    merged = (
        correct[["method", "graph", "mean"]]
        .rename(columns={"mean": "correct_mean"})
        .merge(
            incorrect[["method", "graph", "mean"]],
            on=["method", "graph"],
            how="outer",
            suffixes=("_correct", "_incorrect"),
        )
    )
    merged["correct_mean"] = merged["correct_mean"].fillna(0.0)
    merged["incorrect_mean"] = merged["mean"].fillna(0.0)
    merged.drop(columns=["mean"], inplace=True)
    merged["divergence"] = merged["correct_mean"] - merged["incorrect_mean"]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Panel A · Correct",
            "Panel B · Incorrect",
            "Panel C · Divergence (Correct - Incorrect)",
        ),
    )

    for method in METHOD_ORDER:
        subset = merged[merged["method"] == method]
        if subset.empty:
            continue
        graphs = [g.title() for g in subset["graph"]]
        fig.add_trace(
            go.Bar(
                x=graphs,
                y=subset["correct_mean"],
                name=METHOD_LABELS.get(method, method),
                marker_color=METHOD_COLORS.get(method, "#7f8c8d"),
                legendgroup=method,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=graphs,
                y=subset["incorrect_mean"],
                marker_color=METHOD_COLORS.get(method, "#7f8c8d"),
                legendgroup=method,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in subset["divergence"]]
        fig.add_trace(
            go.Bar(
                x=graphs,
                y=subset["divergence"],
                marker_color=colors,
                marker_line=dict(color=METHOD_COLORS.get(method, "#7f8c8d"), width=2),
                legendgroup=method,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    fig.add_hline(y=0, row=1, col=3, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(
            text=(
                "<b>Confounding Robustness (Maskout Contrastivity)</b><br>"
                f"<sub>{dataset_label(dataset)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=1400,
        height=520,
        plot_bgcolor="rgba(245,245,245,0.7)",
        paper_bgcolor="white",
        font=dict(size=11),
    )
    fig.update_xaxes(title_text="Graph Type", row=1, col=1)
    fig.update_xaxes(title_text="Graph Type", row=1, col=2)
    fig.update_xaxes(title_text="Graph Type", row=1, col=3)
    fig.update_yaxes(title_text="Maskout Contrastivity", row=1, col=1)
    fig.update_yaxes(title_text="Maskout Contrastivity", row=1, col=2)
    fig.update_yaxes(title_text="Divergence", row=1, col=3)
    save_figure(fig, output_dir / f"viz2_confounding_robustness_{dataset}", width=1400, height=520)


def create_strategy_radar(origin_df: pd.DataFrame, masked_df: pd.DataFrame, maskout_df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    data = []
    for method in METHOD_ORDER:
        origin_subset = origin_df[
            (origin_df["dataset"] == dataset)
            & (origin_df["method"] == method)
            & (origin_df["group"] == "overall")
        ]
        maskout_subset = maskout_df[
            (maskout_df["dataset"] == dataset)
            & (maskout_df["method"] == method)
            & (maskout_df["group"] == "overall")
        ]
        masked_correct = masked_df[
            (masked_df["dataset"] == dataset)
            & (masked_df["method"] == method)
            & (masked_df["group"] == "correct_True")
        ]
        masked_incorrect = masked_df[
            (masked_df["dataset"] == dataset)
            & (masked_df["method"] == method)
            & (masked_df["group"] == "correct_False")
        ]

        if origin_subset.empty or maskout_subset.empty or masked_correct.empty or masked_incorrect.empty:
            continue

        masked_div = float(masked_correct["mean"].mean() - masked_incorrect["mean"].mean())
        maskout_mean = float(maskout_subset["mean"].mean())
        origin_mean = float(origin_subset["mean"].mean())

        data.append(
            {
                "method": method,
                "masked_div": masked_div,
                "maskout_mean": maskout_mean,
                "origin_mean": origin_mean,
            }
        )

    if not data:
        return

    radar_df = pd.DataFrame(data)
    for col in ["masked_div", "maskout_mean", "origin_mean"]:
        max_val = radar_df[col].max()
        min_val = radar_df[col].min()
        denom = max(max_val - min_val, 1e-9)
        radar_df[col + "_norm"] = (radar_df[col] - min_val) / denom

    categories = ["Feature ID", "Confounding Robust", "Baseline Confidence"]
    fig = go.Figure()
    for _, row in radar_df.iterrows():
        method = row["method"]
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    row["masked_div_norm"],
                    row["maskout_mean_norm"],
                    row["origin_mean_norm"],
                ],
                theta=categories,
                fill="toself",
                name=METHOD_LABELS.get(method, method),
                marker_color=METHOD_COLORS.get(method, "#7f8c8d"),
                opacity=0.6,
                line=dict(color=METHOD_COLORS.get(method, "#7f8c8d"), width=2),
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=dict(
            text=(
                "<b>Strategy Profile Radar</b><br>"
                f"<sub>{dataset_label(dataset)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=800,
        height=700,
        font=dict(size=11),
    )
    save_figure(fig, output_dir / f"viz3_strategy_radar_{dataset}", width=800, height=700)


def create_graph_effectiveness(masked_df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = masked_df[(masked_df["dataset"] == dataset) & (masked_df["group"] == "correct_True")]
    if subset.empty:
        return

    entries = []
    for method in METHOD_ORDER:
        method_df = subset[subset["method"] == method]
        for graph in GRAPH_ORDER:
            row = method_df[method_df["graph"] == graph]
            if row.empty:
                continue
            entries.append(
                {
                    "Graph Type": graph.title(),
                    "Masked Contrastivity": float(row["mean"].iloc[0]),
                    "Method": METHOD_LABELS.get(method, method),
                }
            )

    if not entries:
        return

    eff_df = pd.DataFrame(entries)
    fig = px.bar(
        eff_df,
        x="Graph Type",
        y="Masked Contrastivity",
        color="Method",
        barmode="group",
        color_discrete_map=METHOD_COLORS,
        category_orders={"Graph Type": [g.title() for g in GRAPH_ORDER]},
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Graph Type Effectiveness</b><br>"
                f"<sub>{dataset_label(dataset)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=1000,
        height=600,
        font=dict(size=11),
    )
    save_figure(fig, output_dir / f"viz4_graph_effectiveness_{dataset}", width=1000, height=600)


def create_method_robustness_violin(origin_df: pd.DataFrame, masked_df: pd.DataFrame, maskout_df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    rng = np.random.default_rng(seed=42)
    records: List[Dict[str, object]] = []

    def append_samples(field_df: pd.DataFrame, metric: str, clip: bool = False) -> None:
        for method in METHOD_ORDER:
            for group in ("correct_True", "correct_False"):
                subset = field_df[
                    (field_df["dataset"] == dataset)
                    & (field_df["method"] == method)
                    & (field_df["group"] == group)
                ]
                if subset.empty:
                    continue
                mean_val = float(subset["mean"].mean())
                std_val = float(subset["std"].mean())
                std_val = max(std_val, 1e-6)
                samples = rng.normal(loc=mean_val, scale=std_val, size=120)
                if clip:
                    samples = np.clip(samples, 0.0, 1.0)
                for sample in samples:
                    records.append(
                        {
                            "Method": METHOD_LABELS.get(method, method),
                            "Metric": metric,
                            "Correctness": "Correct" if group == "correct_True" else "Incorrect",
                            "Value": sample,
                        }
                    )

    append_samples(masked_df, "Masked", clip=True)
    append_samples(maskout_df, "Maskout", clip=False)
    append_samples(origin_df, "Origin", clip=True)

    if not records:
        return

    violin_df = pd.DataFrame(records)
    fig = px.violin(
        violin_df,
        x="Method",
        y="Value",
        color="Correctness",
        facet_col="Metric",
        color_discrete_map={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
        category_orders={"Metric": ["Masked", "Maskout", "Origin"]},
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Method Robustness Distributions</b><br>"
                f"<sub>{dataset_label(dataset)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=1200,
        height=600,
        font=dict(size=11),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title_text="Contrastivity Value")
    save_figure(fig, output_dir / f"viz5_method_robustness_{dataset}", width=1200, height=600)


def create_summary_table(origin_df: pd.DataFrame, masked_df: pd.DataFrame, maskout_df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    rows = []
    for method in METHOD_ORDER:
        for graph in GRAPH_ORDER:
            masked_row = masked_df[
                (masked_df["dataset"] == dataset)
                & (masked_df["method"] == method)
                & (masked_df["graph"] == graph)
                & (masked_df["group"] == "correct_True")
            ]
            maskout_row = maskout_df[
                (maskout_df["dataset"] == dataset)
                & (maskout_df["method"] == method)
                & (maskout_df["graph"] == graph)
                & (maskout_df["group"] == "overall")
            ]
            origin_row = origin_df[
                (origin_df["dataset"] == dataset)
                & (origin_df["method"] == method)
                & (origin_df["graph"] == graph)
                & (origin_df["group"] == "overall")
            ]
            if masked_row.empty or maskout_row.empty or origin_row.empty:
                continue
            rows.append(
                {
                    "Method": METHOD_LABELS.get(method, method),
                    "Graph": graph.title(),
                    "Masked (Feature ID)": f"{masked_row['mean'].iloc[0]:.3f}",
                    "Maskout (Confounding)": f"{maskout_row['mean'].iloc[0]:.3f}",
                    "Origin (Baseline)": f"{origin_row['mean'].iloc[0]:.3f}",
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Method</b>",
                        "<b>Graph Type</b>",
                        "<b>Masked Contrastivity</b>",
                        "<b>Maskout Contrastivity</b>",
                        "<b>Origin Contrastivity</b>",
                    ],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[
                        df["Method"],
                        df["Graph"],
                        df["Masked (Feature ID)"],
                        df["Maskout (Confounding)"],
                        df["Origin (Baseline)"],
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Contrastivity Summary Table</b><br>"
                f"<sub>{dataset_label(dataset)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=900,
        height=600,
        font=dict(size=11),
    )
    save_figure(fig, output_dir / f"viz6_summary_table_{dataset}", width=900, height=600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contrastivity dashboards from aggregated summaries.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing contrastivity summaries (default: outputs/analytics/contrastivity).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    origin_df = load_summary(root, CONTRASTIVITY_FIELDS["origin"])
    masked_df = load_summary(root, CONTRASTIVITY_FIELDS["masked"])
    maskout_df = load_summary(root, CONTRASTIVITY_FIELDS["maskout"])

    datasets = sorted(masked_df["dataset"].unique())
    plots_root = root / PLOTS_ROOT

    print("=" * 120)
    print("Contrastivity visualisations")
    print("=" * 120)
    print(f"Root: {root}")
    print(f"Datasets: {datasets}")

    for dataset in datasets:
        dataset_dir = plots_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDataset: {dataset_label(dataset)}")
        create_feature_id_panels(masked_df, dataset, dataset_dir)
        create_confounding_robustness(maskout_df, dataset, dataset_dir)
        create_strategy_radar(origin_df, masked_df, maskout_df, dataset, dataset_dir)
        create_graph_effectiveness(masked_df, dataset, dataset_dir)
        create_method_robustness_violin(origin_df, masked_df, maskout_df, dataset, dataset_dir)
        create_summary_table(origin_df, masked_df, maskout_df, dataset, dataset_dir)

    print("\n" + "=" * 120)
    print("All contrastivity plots generated.")
    print("=" * 120)


if __name__ == "__main__":
    main()
