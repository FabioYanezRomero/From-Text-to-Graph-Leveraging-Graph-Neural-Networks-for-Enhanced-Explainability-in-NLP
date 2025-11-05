#!/usr/bin/env python3
"""
Decision margin consistency visualisations.

This script consumes the aggregated consistency summaries produced by
``aggregate_consistency.py`` (now housed under the consistency analytics
namespace) and generates four dashboards that surface how well explanation
methods preserve or disrupt the decision margin:

  A. Margin Preservation Cascade (baseline vs. sufficiency vs. necessity)
  B. Sufficiency–Necessity Trade-off Scatter
  C. Ratio Comparison Heatmap
  D. Correctness Stratification Violins

Outputs are stored under ``outputs/analytics/consistency/plots/<dataset>/``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_ROOT = Path("outputs/analytics/consistency")
SUMMARY_FILENAME = "consistency_summary.csv"
PLOTS_ROOT = "plots"
RATIO_LIMIT = (-1.0, 1.0)

METHOD_ORDER: Sequence[str] = ("graphsvx", "subgraphx", "token_shap_llm")
METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
    "token_shap_llm": "TokenSHAP (LLM)",
}
METHOD_COLORS: Dict[str, str] = {
    "graphsvx": "#3498db",
    "subgraphx": "#9b59b6",
    "token_shap_llm": "#2ecc71",
}
GRAPH_ORDER = ["skipgrams", "window", "constituency", "syntactic", "tokens"]
GRAPH_COLORS: Dict[str, str] = {
    "skipgrams": "#1f77b4",
    "window": "#ff7f0e",
    "constituency": "#2ca02c",
    "syntactic": "#d62728",
    "tokens": "#9467bd",
}

DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def graph_title_text(graph: str) -> str:
    return graph.replace("_", " ").title()


def order_graphs(graphs: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for candidate in GRAPH_ORDER:
        if candidate in graphs and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    for graph in graphs:
        if graph not in seen:
            ordered.append(graph)
            seen.add(graph)
    return ordered


def with_alpha(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected hex colour #RRGGBB, received {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    alpha_clamped = max(0.0, min(1.0, alpha))
    return f"rgba({r},{g},{b},{alpha_clamped:.3f})"


def load_summary(root: Path, field: str) -> pd.DataFrame:
    path = root / field / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Field summary missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Field summary is empty: {path}")
    return df


def load_instance_records(root: Path, dataset: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for method in METHOD_ORDER:
        method_dir = root / method / dataset
        if not method_dir.exists():
            continue
        for csv_path in method_dir.glob("*.csv"):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df["method"] = df.get("method", method)
            df["dataset"] = df.get("dataset", dataset)
            df["graph_type"] = df.get("graph_type", csv_path.stem)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    denominator = data.get("baseline_margin", np.nan).replace(0, np.nan)
    if "sufficiency_ratio" not in data.columns and "preservation_sufficiency" in data.columns:
        data["sufficiency_ratio"] = data["preservation_sufficiency"] / denominator
    if "necessity_ratio" not in data.columns and "preservation_necessity" in data.columns:
        data["necessity_ratio"] = data["preservation_necessity"] / denominator
    data["sufficiency_ratio"] = data["sufficiency_ratio"].clip(*RATIO_LIMIT)
    data["necessity_ratio"] = data["necessity_ratio"].clip(*RATIO_LIMIT)
    data["dataset_slug"] = dataset
    return data


def save_figure(fig: go.Figure, stem: Path, width: int, height: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    html_path = stem.with_suffix(".html")
    try:
        fig.write_image(str(pdf_path), width=width, height=height)
        print(f"  ✓ {pdf_path}")
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"  ! PDF failed ({pdf_path.name}): {exc}")
    try:
        fig.write_html(str(html_path))
        print(f"  ✓ {html_path}")
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"  ! HTML failed ({html_path.name}): {exc}")


def _confidence_interval(std: float, count: float) -> float:
    if not np.isfinite(std) or not np.isfinite(count) or count <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(count)


def create_margin_preservation_cascade(
    baseline_df: pd.DataFrame,
    sufficiency_df: pd.DataFrame,
    necessity_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    metric_specs = [
        ("baseline_margin", baseline_df, "Baseline Margin"),
        ("preservation_sufficiency", sufficiency_df, "Preservation Sufficiency"),
        ("preservation_necessity", necessity_df, "Preservation Necessity"),
    ]

    fig = make_subplots(
        rows=1,
        cols=len(metric_specs),
        subplot_titles=[title for _, _, title in metric_specs],
        shared_yaxes=True,
    )

    has_any = False

    for col_idx, (metric_key, source_df, title) in enumerate(metric_specs, start=1):
        subset = source_df[
            (source_df["dataset"] == dataset)
            & (source_df["group"] == "overall")
        ].copy()
        if subset.empty:
            continue

        subset["graph_title"] = subset["graph"].apply(graph_title_text)
        graphs = order_graphs(subset["graph"].unique())
        if not graphs:
            continue

        has_any = True

        for method in METHOD_ORDER:
            method_subset = subset[subset["method"] == method]
            if method_subset.empty:
                continue

            prepared = method_subset.copy()
            prepared.set_index("graph", inplace=True)
            means = [float(prepared.loc[g, "mean"]) if g in prepared.index else np.nan for g in graphs]
            cis = [
                _confidence_interval(
                    float(prepared.loc[g, "std"]) if g in prepared.index else np.nan,
                    float(prepared.loc[g, "count"]) if g in prepared.index else np.nan,
                )
                for g in graphs
            ]
            upper = [m + c if np.isfinite(m) else np.nan for m, c in zip(means, cis)]
            lower = [m - c if np.isfinite(m) else np.nan for m, c in zip(means, cis)]

            color = METHOD_COLORS.get(method, "#7f8c8d")
            method_label = METHOD_LABELS.get(method, method)

            fig.add_trace(
                go.Scatter(
                    x=[graph_title_text(g) for g in graphs],
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=method_label,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=[graph_title_text(g) for g in graphs],
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=with_alpha(color, 0.25),
                    legendgroup=method_label,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=[graph_title_text(g) for g in graphs],
                    y=means,
                    mode="lines+markers",
                    name=method_label if col_idx == 1 else None,
                    legendgroup=method_label,
                    line=dict(color=color, width=3),
                    marker=dict(size=8, symbol="circle", line=dict(color="#333333", width=1)),
                    showlegend=col_idx == 1,
                ),
                row=1,
                col=col_idx,
            )

        fig.add_hline(y=0, line_dash="dot", line_color="#7f8c8d", row=1, col=col_idx)
        fig.update_xaxes(title_text="Graph Type", row=1, col=col_idx, tickangle=-30)

    if not has_any:
        return

    fig.update_yaxes(title_text="Decision Margin", range=[-1.05, 1.05])
    fig.update_layout(
        title=dict(
            text=(
                "<b>Margin Preservation Cascade</b><br>"
                f"<sub>{dataset_label(dataset)} · lines = methods, columns = margin view</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=520,
        width=1300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )

    save_figure(fig, output_dir / f"vizA_margin_preservation_cascade_{dataset}", width=1300, height=520)


GRAPH_SYMBOLS = {
    "skipgrams": "circle",
    "window": "square",
    "constituency": "triangle-up",
    "syntactic": "diamond",
    "tokens": "star",
}


def create_tradeoff_scatter(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    if instances.empty:
        return

    subset = instances[(instances.get("dataset_slug") == dataset)]
    if subset.empty:
        return

    fig = go.Figure()

    for method in METHOD_ORDER:
        method_subset = subset[subset["method"] == method]
        if method_subset.empty:
            continue
        symbols = [GRAPH_SYMBOLS.get(graph, "circle") for graph in method_subset["graph_type"]]
        fig.add_trace(
            go.Scattergl(
                x=method_subset["sufficiency_ratio"],
                y=method_subset["necessity_ratio"],
                mode="markers",
                name=METHOD_LABELS.get(method, method),
                legendgroup=method,
                marker=dict(
                    color=METHOD_COLORS.get(method, "#7f8c8d"),
                    size=7,
                    symbol=symbols,
                    line=dict(color="#333333", width=0.5),
                    opacity=0.8,
                ),
                text=[
                    f"Graph: {graph_title_text(graph)}<br>"
                    f"Correct: {bool(correct)}<br>"
                    f"Baseline margin: {baseline:.3f}"
                    for graph, correct, baseline in zip(
                        method_subset["graph_type"],
                        method_subset.get("is_correct", False),
                        method_subset.get("baseline_margin", np.nan),
                    )
                ],
                hovertemplate="Sufficiency ratio: %{x:.3f}<br>"
                "Necessity ratio: %{y:.3f}<br>%{text}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color="#7f8c8d")
    fig.add_vline(x=0, line_dash="dot", line_color="#7f8c8d")

    # Legend handles for graph symbols and correctness
    for graph, symbol in GRAPH_SYMBOLS.items():
        fig.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(symbol=symbol, size=8, color="#555555"),
                legendgroup="graph",
                name=f"Graph · {graph_title_text(graph)}",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="#555555", size=8, line=dict(color="#333333", width=0.5)),
            name="Correct prediction",
            legendgroup="correctness",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="x", color="#555555", size=8, line=dict(color="#333333", width=0.5)),
            name="Incorrect prediction",
            legendgroup="correctness",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Sufficiency–Necessity Trade-off Scatter</b><br>"
                f"<sub>{dataset_label(dataset)} · colour = method, symbol = graph type</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=560,
        width=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Preservation Sufficiency Ratio", range=RATIO_LIMIT)
    fig.update_yaxes(title_text="Preservation Necessity Ratio", range=RATIO_LIMIT)

    save_figure(fig, output_dir / f"vizB_tradeoff_scatter_{dataset}", width=1000, height=560)


def create_ratio_heatmap(
    sufficiency_ratio_df: pd.DataFrame,
    necessity_ratio_df: pd.DataFrame,
    margin_coherence_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        subset = df[
            (df["dataset"] == dataset)
            & (df["group"] == "overall")
        ][["method", "graph", "mean"]].copy()
        if subset.empty:
            return subset
        subset["graph_title"] = subset["graph"].apply(graph_title_text)
        return subset

    suff = prepare(sufficiency_ratio_df)
    nec = prepare(necessity_ratio_df)
    coh = prepare(margin_coherence_df)
    combined_methods = sorted(set(suff["method"]).union(nec["method"]).union(coh["method"]), key=METHOD_ORDER.index)
    combined_graphs = order_graphs(
        set(suff["graph"]).union(nec["graph"]).union(coh["graph"])
    )
    if not combined_methods or not combined_graphs:
        return

    row_labels: List[str] = []
    for method in combined_methods:
        for graph in combined_graphs:
            row_labels.append(f"{METHOD_LABELS.get(method, method)} · {graph_title_text(graph)}")

    def matrix_from(df: pd.DataFrame) -> np.ndarray:
        mapping = {(row["method"], row["graph"]): row["mean"] for _, row in df.iterrows()}
        values = [
            float(mapping.get((method, graph), np.nan))
            for method in combined_methods
            for graph in combined_graphs
        ]
        return np.array(values, dtype=float).reshape(-1, 1)

    suff_matrix = matrix_from(suff)
    nec_matrix = matrix_from(nec)
    coh_matrix = matrix_from(coh)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Sufficiency Ratio", "Necessity Ratio", "Margin Coherence"],
        shared_yaxes=True,
    )

    fig.add_trace(
        go.Heatmap(
            z=suff_matrix,
            x=["Ratio"],
            y=row_labels,
            colorscale=[(0.0, "#e74c3c"), (1.0, "#2ecc71")],
            zmin=0,
            zmax=1,
            colorbar=dict(title=""),
            showscale=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=nec_matrix,
            x=["Ratio"],
            y=row_labels,
            colorscale=[(0.0, "#2ecc71"), (0.5, "#f1c40f"), (1.0, "#e74c3c")],
            zmin=-1,
            zmax=1,
            colorbar=dict(title=""),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=coh_matrix,
            x=["Coherence"],
            y=row_labels,
            colorscale=[(0.0, "#e74c3c"), (1.0, "#2ecc71")],
            zmin=0,
            zmax=1,
            colorbar=dict(title=""),
            showscale=True,
        ),
        row=1,
        col=3,
    )

    for col_idx in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=col_idx)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Ratio Comparison Heatmap</b><br>"
                f"<sub>{dataset_label(dataset)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=40 * len(row_labels) + 200,
        width=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )

    save_figure(fig, output_dir / f"vizC_ratio_comparison_heatmap_{dataset}", width=900, height=40 * len(row_labels) + 200)


def create_correctness_violins(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    if instances.empty:
        return

    subset = instances[instances.get("dataset_slug") == dataset]
    if subset.empty:
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Correct predictions", "Incorrect predictions"],
        shared_yaxes=True,
    )

    has_data = False
    ratio_specs = [
        ("sufficiency_ratio", "Sufficiency Ratio", "#1890ff", "negative"),
        ("necessity_ratio", "Necessity Ratio", "#f39c12", "positive"),
    ]

    for row_idx, correctness in enumerate((True, False), start=1):
        row_subset = subset[subset.get("is_correct") == correctness]
        if row_subset.empty:
            continue
        has_data = True
        for method in METHOD_ORDER:
            method_subset = row_subset[row_subset["method"] == method]
            if method_subset.empty:
                continue
            x_label = METHOD_LABELS.get(method, method)
            for metric, name, color, side in ratio_specs:
                fig.add_trace(
                    go.Violin(
                        x=[x_label] * len(method_subset),
                        y=method_subset[metric],
                        legendgroup=name,
                        name=name if (row_idx == 1 and method == METHOD_ORDER[0]) else None,
                        line=dict(color=color, width=1.5),
                        fillcolor=with_alpha(color, 0.45),
                        meanline=dict(visible=True),
                        spanmode="hard",
                        span=RATIO_LIMIT,
                        side=side,
                        width=0.6,
                        points=False,
                        showlegend=(row_idx == 1 and method == METHOD_ORDER[0]),
                    ),
                    row=row_idx,
                    col=1,
                )

    if not has_data:
        return

    fig.update_yaxes(title_text="Ratio", range=RATIO_LIMIT, row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_layout(
        title=dict(
            text=(
                "<b>Correctness Stratification Violins</b><br>"
                f"<sub>{dataset_label(dataset)} · left = sufficiency, right = necessity</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=650,
        width=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )

    save_figure(fig, output_dir / f"vizD_correctness_violins_{dataset}", width=1000, height=650)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decision margin consistency dashboards from aggregated summaries.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing decision margin summaries (default: outputs/analytics/consistency).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_df = load_summary(root, "baseline_margin")
    sufficiency_df = load_summary(root, "preservation_sufficiency")
    necessity_df = load_summary(root, "preservation_necessity")
    suff_ratio_df = load_summary(root, "sufficiency_ratio")
    nec_ratio_df = load_summary(root, "necessity_ratio")
    coherence_df = load_summary(root, "margin_coherence")

    datasets = sorted(sufficiency_df["dataset"].unique())
    plots_root = root / PLOTS_ROOT

    print("=" * 120)
    print("Decision margin consistency visualisations")
    print("=" * 120)
    print(f"Root: {root}")
    print(f"Datasets: {datasets}")

    for dataset in datasets:
        dataset_dir = plots_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDataset: {dataset_label(dataset)}")

        instance_records = load_instance_records(root, dataset)

        create_margin_preservation_cascade(baseline_df, sufficiency_df, necessity_df, dataset, dataset_dir)
        create_tradeoff_scatter(instance_records, dataset, dataset_dir)
        create_ratio_heatmap(suff_ratio_df, nec_ratio_df, coherence_df, dataset, dataset_dir)
        create_correctness_violins(instance_records, dataset, dataset_dir)

    print("\n" + "=" * 120)
    print("All decision margin consistency plots generated.")
    print("=" * 120)


if __name__ == "__main__":
    main()
