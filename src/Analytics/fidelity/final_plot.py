#!/usr/bin/env python3
"""
Render Fidelity⁺/Fidelity⁻ dashboards for the fidelity dimension.

This module produces two interactive Plotly figures for each dataset:
  • Quadrant scatter per class showing where explanations fall in F⁺/F⁻ space.
  • Violin plots summarising fidelity asymmetry distributions per class.

Both figures mirror the consistency layout (one subplot per class, shared
legend placement) so the visuals are ready for inclusion in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

INSTANCE_ROOT = Path("outputs/analytics/fidelity")
OUTPUT_ROOT = Path("outputs/analytics/fidelity/plots")
SUMMARY_PATH = Path("outputs/analytics/fidelity/fidelity_summary.csv")
OUTPUT_FILENAME = "fidelity_quadrants_{dataset}.html"

DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}

METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
    "token_shap_llm": "TokenSHAP (LLM)",
}

GRAPH_LABELS: Dict[str, str] = {
    "skipgrams": "Skipgrams",
    "window": "Window",
    "constituency": "Constituency",
    "syntactic": "Syntactic",
    "tokens": "Tokens",
}

EXPERIMENT_SYMBOLS: Dict[tuple[str, str], str] = {
    ("graphsvx", "skipgrams"): "circle",
    ("graphsvx", "window"): "square",
    ("subgraphx", "constituency"): "diamond",
    ("subgraphx", "syntactic"): "triangle-up",
    ("token_shap_llm", "tokens"): "star",
}

CORRECT_COLOR = "#1a9850"
INCORRECT_COLOR = "#d73027"

GROUP_ORDER: Sequence[str] = ("Correct", "Incorrect")
EXPERIMENT_ORDER: Sequence[tuple[str, str]] = (
    ("graphsvx", "skipgrams"),
    ("graphsvx", "window"),
    ("subgraphx", "constituency"),
    ("subgraphx", "syntactic"),
    ("token_shap_llm", "tokens"),
)

EXPERIMENT_LABELS: Dict[tuple[str, str], str] = {
    ("graphsvx", "skipgrams"): "GraphSVX<br>Skipgrams",
    ("graphsvx", "window"): "GraphSVX<br>Window",
    ("subgraphx", "constituency"): "SubgraphX<br>Constituency",
    ("subgraphx", "syntactic"): "SubgraphX<br>Syntactic",
    ("token_shap_llm", "tokens"): "TokenSHAP<br>Tokens",
}

QUADRANT_COLORS: Dict[str, str] = {
    "++": "rgba(26, 152, 80, 0.18)",    # Sufficient & Necessary (ideal)
    "-+": "rgba(255, 193, 7, 0.18)",    # Necessary but insufficient (warning)
    "--": "rgba(215, 48, 39, 0.18)",    # Distributed/noisy (failure)
    "+-": "rgba(145, 191, 219, 0.18)",  # Sufficient but redundant (informational)
}

FULLSCREEN_STYLE = """
<style>
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    background-color: #ffffff;
}
body {
    display: flex;
    align-items: stretch;
    justify-content: stretch;
}
.plotly-graph-div {
    width: 100vw !important;
    height: 100vh !important;
}
</style>
"""

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def discover_instance_paths(root: Path, dataset: str) -> List[Path]:
    """Return all per-instance fidelity CSV paths for ``dataset`` across methods."""
    paths: List[Path] = []
    for method_dir in root.iterdir():
        if not method_dir.is_dir():
            continue
        dataset_dir = method_dir / dataset
        if not dataset_dir.is_dir():
            continue
        for csv_path in dataset_dir.glob("*.csv"):
            if csv_path.is_file():
                paths.append(csv_path)
    return sorted(paths)


def load_dataset_instances(root: Path, dataset: str) -> pd.DataFrame:
    """Load and concatenate all per-instance fidelity CSVs for ``dataset``."""
    csv_paths = discover_instance_paths(root, dataset)
    if not csv_paths:
        raise FileNotFoundError(f"No fidelity instance CSVs found for dataset '{dataset}' in {root}")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        if "graph_type" not in df.columns:
            df["graph_type"] = path.stem
        df["method"] = df["method"].astype(str)
        df["graph_type"] = df["graph_type"].astype(str)
        frames.append(df)

    if not frames:
        raise ValueError(f"All fidelity CSVs for dataset '{dataset}' were empty.")

    data = pd.concat(frames, ignore_index=True)
    data["correctness_label"] = data["is_correct"].map({True: "Correct", False: "Incorrect"}).fillna("Unknown")
    data["prediction_class"] = data["prediction_class"].astype(str)
    return data


def slugify_dataset(name: str) -> str:
    return name.replace("/", "_").replace("-", "_").lower()


def load_summary_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Summary CSV empty: {path}")
    df = df.copy()
    df["dataset_slug"] = df["dataset"].astype(str).apply(slugify_dataset)
    df["method"] = df["method"].astype(str)
    df["graph"] = df["graph"].astype(str)
    return df


def sorted_classes(data: pd.DataFrame) -> List[str]:
    """Return sorted prediction classes present in ``data`` (fallback to ['All'])."""
    classes = sorted({cls for cls in data["prediction_class"].unique() if cls not in {"nan", "None", "NaN"}})
    return classes or ["All"]


def compute_axis_limits(data: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute symmetric axis limits with padding for fidelity scatter plots."""
    x_min = float(np.nanmin(data["fidelity_plus"])) if not data["fidelity_plus"].isna().all() else -0.1
    x_max = float(np.nanmax(data["fidelity_plus"])) if not data["fidelity_plus"].isna().all() else 0.1
    y_min = float(np.nanmin(data["fidelity_minus"])) if not data["fidelity_minus"].isna().all() else -0.1
    y_max = float(np.nanmax(data["fidelity_minus"])) if not data["fidelity_minus"].isna().all() else 0.1
    pad_x = max(0.05, 0.05 * max(abs(x_min), abs(x_max)))
    pad_y = max(0.05, 0.05 * max(abs(y_min), abs(y_max)))
    x_limits = (min(x_min, -pad_x), max(x_max, pad_x))
    y_limits = (min(y_min, -pad_y), max(y_max, pad_y))
    return (*x_limits, *y_limits)


def add_quadrant_background(fig: go.Figure, col_idx: int, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    """Add shaded quadrant backgrounds to subplot ``col_idx`` based on limits."""
    x0, x1 = x_limits
    y0, y1 = y_limits
    # Quadrant definitions relative to origin
    quadrants = [
        ("++", max(0.0, x0), x1, max(0.0, y0), y1),
        ("+-", max(0.0, x0), x1, y0, min(0.0, y1)),
        ("-+", x0, min(0.0, x1), max(0.0, y0), y1),
        ("--", x0, min(0.0, x1), y0, min(0.0, y1)),
    ]
    for key, xa0, xa1, ya0, ya1 in quadrants:
        if xa0 >= xa1 or ya0 >= ya1:
            continue
        fig.add_shape(
            type="rect",
            x0=xa0,
            x1=xa1,
            y0=ya0,
            y1=ya1,
            fillcolor=QUADRANT_COLORS.get(key, "rgba(0,0,0,0.05)"),
            opacity=1.0,
            layer="below",
            line_width=0,
            row=1,
            col=col_idx,
        )


def write_fullscreen_html(fig: go.Figure, output_path: Path) -> None:
    """Persist ``fig`` as a fullscreen HTML artifact for easier inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
        default_width="100vw",
        default_height="100vh",
    )
    if "</head>" in html:
        html = html.replace("</head>", FULLSCREEN_STYLE + "\n</head>", 1)
    output_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Plot construction                                                           #
# --------------------------------------------------------------------------- #


def build_quadrant_plot(dataset: str, root: Path, output_root: Path) -> Path:
    """Render the per-class fidelity quadrant scatter plot for ``dataset``."""
    instances = load_dataset_instances(root, dataset)
    if instances.empty:
        raise ValueError(f"No fidelity instances available for dataset '{dataset}'.")

    classes = sorted_classes(instances)
    dataset_label = DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())

    subplot_titles = []
    for label in classes:
        title = f"<span style='font-size:16px; font-weight:bold'>Class {label}</span>"
        subplot_titles.append(title)

    fig = make_subplots(
        rows=1,
        cols=len(classes),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
    )

    x0, x1, y0, y1 = compute_axis_limits(instances)

    for col_idx, class_label in enumerate(classes, start=1):
        class_data = instances[instances["prediction_class"] == class_label]
        if class_data.empty:
            continue

        add_quadrant_background(fig, col_idx, (x0, x1), (y0, y1))

        for method, graph in EXPERIMENT_ORDER:
            exp_subset = class_data[(class_data["method"] == method) & (class_data["graph_type"] == graph)]
            if exp_subset.empty:
                continue
            method_name = METHOD_LABELS.get(method, method)
            graph_name = GRAPH_LABELS.get(graph, graph)
            symbol = EXPERIMENT_SYMBOLS.get((method, graph), "circle")

            for correctness, opacity in zip(GROUP_ORDER, (0.9, 0.75)):
                group_subset = exp_subset[exp_subset["correctness_label"] == correctness]
                if group_subset.empty:
                    continue
                colour = CORRECT_COLOR if correctness == "Correct" else INCORRECT_COLOR
                legend_name = f"{method_name} · {graph_name} · {correctness}"
                show_legend = col_idx == 1

                fig.add_trace(
                    go.Scattergl(
                        x=group_subset["fidelity_plus"],
                        y=group_subset["fidelity_minus"],
                        mode="markers",
                        name=legend_name if show_legend else None,
                        marker=dict(
                            color=colour,
                            opacity=opacity,
                            size=8,
                            line=dict(color="#2c3e50", width=0.9),
                            symbol=symbol,
                        ),
                        hovertemplate=(
                            "<b>Method:</b> %{text}<br>"
                            "Fidelity⁺: %{x:.4f}<br>"
                            "Fidelity⁻: %{y:.4f}<br>"
                            "Label: %{customdata[0]}<br>"
                            "Prediction: %{customdata[1]}<extra></extra>"
                        ),
                        text=[f"{method_name} · {graph_name} · {correctness}"] * len(group_subset),
                        customdata=np.stack(
                            [
                                group_subset.get("label", pd.Series(["?"] * len(group_subset))),
                                group_subset.get("prediction_class", pd.Series(["?"] * len(group_subset))),
                            ],
                            axis=1,
                        ),
                        legendgroup=legend_name,
                        showlegend=show_legend,
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.add_vline(x=0.0, line_dash="dot", line_color="#555", line_width=1, row=1, col=col_idx)
        fig.add_hline(y=0.0, line_dash="dot", line_color="#555", line_width=1, row=1, col=col_idx)

        y_title_text = "Fidelity⁻" if col_idx == 1 else ""

        fig.update_xaxes(
            range=[x0, x1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            showline=True,
            linecolor="#2c3e50",
            linewidth=1.5,
            title=dict(
                text="",
                font=dict(size=18, family="Arial, sans-serif", color="#2c3e50"),
                standoff=24,
            ),
            tickfont=dict(size=12, family="Arial, sans-serif", color="#2c3e50"),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            range=[y0, y1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            showline=True,
            linecolor="#2c3e50",
            linewidth=1.5,
            title=dict(
                text=f"<b>{y_title_text}</b>" if y_title_text else "",
                font=dict(size=18, family="Arial, sans-serif", color="#2c3e50"),
                standoff=28,
            ),
            tickfont=dict(size=12, family="Arial, sans-serif", color="#2c3e50"),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Fidelity⁺ vs Fidelity⁻ by Class</b><br>"
                f"<span style='font-size:14px'>{dataset_label}</span>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        legend=dict(
            title=dict(text="<b>Method · Graph · Correctness</b>"),
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1080,
        width=1920,
        margin=dict(t=200, b=160, l=140, r=100),
    )

    fig.add_annotation(
        dict(
            text="<b>Fidelity⁺</b>",
            x=0.5,
            y=-0.1,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18, family="Arial, sans-serif", color="#2c3e50"),
        )
    )

    output_path = output_root / dataset / OUTPUT_FILENAME.format(dataset=dataset)
    write_fullscreen_html(fig, output_path)
    print(f"✓ {dataset}: plot -> {output_path}")
    return output_path


def build_asymmetry_plot(dataset: str, root: Path, output_root: Path) -> Path:
    """Render violin plots of fidelity asymmetry per class for ``dataset``."""
    instances = load_dataset_instances(root, dataset)
    if instances.empty or "fidelity_asymmetry" not in instances:
        raise ValueError(f"No fidelity asymmetry data available for dataset '{dataset}'.")

    classes = sorted_classes(instances)
    dataset_label = DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())

    y_min = float(np.nanmin(instances["fidelity_asymmetry"]))
    y_max = float(np.nanmax(instances["fidelity_asymmetry"]))
    padding = max(0.05, 0.05 * max(abs(y_min), abs(y_max)))
    y_range = [min(-1.0, y_min - padding), max(1.0, y_max + padding)]

    fig = make_subplots(
        rows=1,
        cols=len(classes),
        subplot_titles=[f"<span style='font-size:16px; font-weight:bold'>Class {label}</span>" for label in classes],
        horizontal_spacing=0.06,
        shared_yaxes=True,
    )

    for col_idx, class_label in enumerate(classes, start=1):
        class_data = instances[instances["prediction_class"] == class_label]
        if class_data.empty:
            continue

        for method, graph in EXPERIMENT_ORDER:
            method_data = class_data[
                (class_data["method"] == method) & (class_data["graph_type"] == graph)
            ]
            if method_data.empty:
                continue
            experiment_label = EXPERIMENT_LABELS.get(
                (method, graph),
                f"{METHOD_LABELS.get(method, method)}<br>{GRAPH_LABELS.get(graph, graph)}",
            )
            experiment_label_clean = experiment_label.replace("<br>", " · ")

            for correctness in GROUP_ORDER:
                subset = method_data[method_data["correctness_label"] == correctness]
                if subset.empty:
                    continue

                colour = CORRECT_COLOR if correctness == "Correct" else INCORRECT_COLOR
                legend_name = f"{experiment_label_clean} · {correctness}"
                show_legend = col_idx == 1
                offset_group = f"{method}-{graph}-{correctness}"

                fig.add_trace(
                    go.Violin(
                        x=[experiment_label] * len(subset),
                        y=subset["fidelity_asymmetry"],
                        name=legend_name if show_legend else None,
                        legendgroup=legend_name,
                        showlegend=show_legend,
                        orientation="v",
                        line=dict(color=colour, width=1.0),
                        fillcolor=colour,
                        opacity=0.6,
                        meanline=dict(visible=True, color="#2c3e50"),
                        box=dict(visible=True),
                        points=False,
                        hovertemplate=(
                            f"<b>{experiment_label_clean}</b><br>"
                            f"Correctness: {correctness}<br>"
                            "Asymmetry: %{y:.4f}<extra></extra>"
                        ),
                        offsetgroup=offset_group,
                        scalegroup=str(col_idx),
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.add_hline(y=0.0, line_dash="dot", line_color="#555", line_width=1, row=1, col=col_idx)

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=[
                EXPERIMENT_LABELS.get(
                    exp,
                    f"{METHOD_LABELS.get(exp[0], exp[0])}<br>{GRAPH_LABELS.get(exp[1], exp[1])}",
                )
                for exp in EXPERIMENT_ORDER
            ],
            tickangle=-10,
            tickfont=dict(size=12, family="Arial, sans-serif", color="#2c3e50"),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            range=y_range,
            tickfont=dict(size=12, family="Arial, sans-serif", color="#2c3e50"),
            title=dict(
                text="<b>Fidelity Asymmetry (F⁻ − F⁺)</b>" if col_idx == 1 else "",
                font=dict(size=18, family="Arial, sans-serif", color="#2c3e50"),
                standoff=24,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Fidelity Asymmetry by Class</b><br>"
                f"<span style='font-size:14px'>{dataset_label}</span>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        legend=dict(
            title=dict(text="<b>Method · Correctness</b>"),
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1080,
        width=1920,
        margin=dict(t=200, b=140, l=140, r=100),
    )

    output_path = output_root / dataset / f"fidelity_asymmetry_{dataset}.html"
    write_fullscreen_html(fig, output_path)
    print(f"✓ {dataset}: asymmetry plot -> {output_path}")
    return output_path


def build_correctness_sensitivity_plot(
    dataset: str, summary_df: pd.DataFrame, output_root: Path
) -> Path:
    subset = summary_df[
        (summary_df["dataset_slug"] == dataset)
        & (summary_df["group"].isin(["correct_true", "correct_false"]))
    ]
    if subset.empty:
        raise ValueError(f"No correctness summary rows for dataset '{dataset}'.")

    dataset_readable = DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())
    subset = subset.copy()
    subset["method_label"] = subset["method"].map(METHOD_LABELS).fillna(subset["method"])
    subset["graph_label"] = subset["graph"].map(GRAPH_LABELS).fillna(subset["graph"])
    subset["category"] = subset["method_label"] + "<br>" + subset["graph_label"]
    subset["correctness_label"] = subset["group"].map(
        {"correct_true": "Correct", "correct_false": "Incorrect"}
    )

    fig = go.Figure()
    for label, color in (("Correct", CORRECT_COLOR), ("Incorrect", INCORRECT_COLOR)):
        group = subset[subset["correctness_label"] == label]
        if group.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=group["category"],
                y=group["fidelity_asymmetry_mean"],
                name=label,
                marker_color=color,
                opacity=0.85,
                error_y=dict(
                    type="data",
                    array=group["fidelity_asymmetry_std"].fillna(0.0),
                    visible=True,
                ),
            )
        )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#666", line_width=1)
    fig.update_layout(
        title=dict(
            text=(
                "<b>Correctness Sensitivity of Fidelity Asymmetry</b><br>"
                f"<span style='font-size:14px'>{dataset_readable} · Mean F⁻ − F⁺</span>"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Method · Graph",
        yaxis_title="Mean asymmetry",
        legend_title="Correctness",
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1920,
        height=1080,
        margin=dict(t=180, b=140, l=140, r=100),
    )

    output_path = output_root / dataset / f"fidelity_correctness_{dataset}.html"
    write_fullscreen_html(fig, output_path)
    print(f"✓ {dataset}: correctness sensitivity plot -> {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def render_all(
    datasets: Iterable[str],
    instance_root: Path,
    summary_df: pd.DataFrame,
    output_root: Path,
) -> None:
    for dataset in datasets:
        build_quadrant_plot(dataset, instance_root, output_root)
        build_asymmetry_plot(dataset, instance_root, output_root)
        build_correctness_sensitivity_plot(dataset, summary_df, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fidelity quadrant scatter visualisations.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_LABELS.keys()),
        help="Dataset slug(s) to render (default: all available).",
    )
    parser.add_argument(
        "--instance-root",
        type=Path,
        default=INSTANCE_ROOT,
        help="Root directory containing per-instance fidelity CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where HTML plots will be written.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to fidelity_summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset or sorted(DATASET_LABELS.keys())
    summary_df = load_summary_data(args.summary_path)
    render_all(datasets, args.instance_root, summary_df, args.output_root)


if __name__ == "__main__":  # pragma: no cover
    main()
