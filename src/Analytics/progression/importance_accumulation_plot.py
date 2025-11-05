#!/usr/bin/env python3
"""
Generate importance accumulation plots for MULTIPLE progression fields.

Works with the multi-field aggregated data structure and generates
plots for each field independently.

Output structure:
  outputs/analytics/progression/
    ├── maskout_progression_drop/
    │   ├── concentration_summary.csv
    │   └── plots/
    │       └── importance_accumulation_*.pdf/html
    ├── sufficiency_progression_drop/
    │   ├── concentration_summary.csv
    │   └── plots/
    │       └── importance_accumulation_*.pdf/html
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PROGRESSION_FIELDS_ROOT = Path("outputs/analytics/progression")
SUMMARY_FILENAME = "concentration_summary.csv"
PLOTS_DIRNAME = "plots"

DEFAULT_TOPK_METRICS: Sequence[str] = (
    "top1_concentration",
    "top3_concentration",
    "top5_concentration",
    "top10_concentration",
)

TOPK_LABELS: Dict[str, str] = {
    "top1_concentration": "Top-1",
    "top3_concentration": "Top-3",
    "top5_concentration": "Top-5",
    "top10_concentration": "Top-10",
}

FIELD_LABELS: Dict[str, str] = {
    "maskout_progression_drop": "Maskout Drop (Error Detection)",
    "sufficiency_progression_drop": "Sufficiency Drop (Recovery)",
    "maskout_progression_confidence": "Maskout Confidence",
    "sufficiency_progression_confidence": "Sufficiency Confidence",
}

DROP_FIELDS = [name for name in FIELD_LABELS if name.endswith("_drop")]

DEFAULT_METHOD_ORDER: Sequence[str] = ("token_shap_llm", "graphsvx", "subgraphx")

DEFAULT_METHOD_LABELS: Dict[str, str] = {
    "token_shap_llm": "TokenSHAP (LLM)",
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
}

DEFAULT_GRAPH_TYPE_ORDER: Dict[str, Sequence[str]] = {
    "token_shap_llm": ("tokens",),
    "graphsvx": ("skipgrams", "window"),
    "subgraphx": ("constituency", "syntactic"),
}

DEFAULT_GRAPH_LABELS: Dict[str, str] = {
    "tokens": "Tokens",
    "skipgrams": "Skip-gram",
    "window": "Window",
    "constituency": "Constituency",
    "syntactic": "Syntactic",
}

DEFAULT_METHOD_BASE_COLORS: Dict[str, str] = {
    "token_shap_llm": "#3498db",
    "graphsvx": "#2ecc71",
    "subgraphx": "#e74c3c",
}

METHOD_GRAPH_COLORS: Dict[tuple[str, str], str] = {
    ("token_shap_llm", "tokens"): "#3498db",
    ("graphsvx", "skipgrams"): "#27ae60",
    ("graphsvx", "window"): "#1e8449",
    ("subgraphx", "constituency"): "#e74c3c",
    ("subgraphx", "syntactic"): "#c0392b",
}

DEFAULT_DATASET_ORDER: Sequence[str] = ("setfit_ag_news", "stanfordnlp_sst2")

DEFAULT_DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News",
    "stanfordnlp_sst2": "SST-2",
}

BASELINE_METHOD = "token_shap_llm"
BASELINE_GRAPH_TYPE = "tokens"
EPS = 1e-12


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def _save_figure(fig: go.Figure, output_stem: Path, width: int, height: int) -> None:
    """Save figure to PDF and HTML."""
    
    pdf_path = output_stem.with_suffix(".pdf")
    html_path = output_stem.with_suffix(".html")
    
    try:
        fig.write_image(str(pdf_path), width=width, height=height)
        print(f"  ✓ {pdf_path.name}")
    except Exception as exc:
        print(f"  ! PDF failed: {exc}")
    
    try:
        fig.write_html(str(html_path))
        print(f"  ✓ {html_path.name}")
    except Exception as exc:
        print(f"  ! HTML failed: {exc}")


def generate_plots_for_field(
    csv_path: Path,
    output_dir: Path,
    field_name: str,
) -> None:
    """Generate importance accumulation plots for a single progression field."""
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print(f"  Warning: No data in {csv_path}")
        return
    
    # Filter: top-k metrics, overall group
    focus = df[df["metric"].isin(DEFAULT_TOPK_METRICS)]
    focus = focus[focus["group"] == "overall"]
    
    if focus.empty:
        print(f"  Warning: No top-k concentration data found")
        return
    
    datasets = sorted(focus["dataset"].unique())
    
    for dataset in datasets:
        ds_df = focus[focus["dataset"] == dataset]

        x_positions = list(range(len(DEFAULT_TOPK_METRICS)))
        x_ticklabels = [TOPK_LABELS[m] for m in DEFAULT_TOPK_METRICS]

        baseline_slice = ds_df[
            (ds_df["method"] == BASELINE_METHOD)
            & (ds_df["graph"] == BASELINE_GRAPH_TYPE)
        ]
        baseline_map: Dict[str, Optional[float]] = {}
        for metric in DEFAULT_TOPK_METRICS:
            metric_row = baseline_slice[baseline_slice["metric"] == metric]
            if metric_row.empty:
                baseline_map[metric] = None
                continue
            value = float(metric_row["mean"].iloc[0])
            baseline_map[metric] = value if np.isfinite(value) else None

        series_data: List[Dict[str, object]] = []

        for method in DEFAULT_METHOD_ORDER:
            method_slice = ds_df[ds_df["method"] == method]
            if method_slice.empty:
                continue

            graph_order = DEFAULT_GRAPH_TYPE_ORDER.get(
                method,
                tuple(sorted(method_slice["graph"].unique())),
            )

            for graph in graph_order:
                graph_slice = method_slice[method_slice["graph"] == graph]
                if graph_slice.empty:
                    continue

                metrics_details: List[Dict[str, Optional[float]]] = []

                for metric in DEFAULT_TOPK_METRICS:
                    metric_row = graph_slice[graph_slice["metric"] == metric]
                    if metric_row.empty:
                        metrics_details.append(
                            {
                                "label": TOPK_LABELS[metric],
                                "raw": float("nan"),
                                "clipped": float("nan"),
                                "relative": float("nan"),
                            }
                        )
                        continue

                    raw_mean = float(metric_row["mean"].iloc[0])
                    clipped_mean = float(np.clip(raw_mean, 0.0, 1.0))
                    base_value = baseline_map.get(metric)
                    relative_value = float("nan")
                    if base_value is not None and np.isfinite(base_value) and abs(base_value) > EPS:
                        relative_value = raw_mean / base_value

                    metrics_details.append(
                        {
                            "label": TOPK_LABELS[metric],
                            "raw": raw_mean,
                            "clipped": clipped_mean,
                            "relative": relative_value,
                        }
                    )

                method_label = DEFAULT_METHOD_LABELS.get(method, method)
                graph_label = DEFAULT_GRAPH_LABELS.get(
                    graph,
                    graph.replace("_", " ").title(),
                )

                series_data.append(
                    {
                        "name": f"{method_label} — {graph_label}",
                        "color": METHOD_GRAPH_COLORS.get(
                            (method, graph),
                            DEFAULT_METHOD_BASE_COLORS.get(method, "#636EFA"),
                        ),
                        "metrics": metrics_details,
                    }
                )

        if not series_data:
            print(f"  Warning: No data to plot for dataset {dataset}")
            continue

        field_label = FIELD_LABELS.get(field_name, field_name)
        dataset_label = DEFAULT_DATASET_LABELS.get(dataset, dataset)

        def _build_scatter_figure(values_key: str, y_label: str, subtitle: str) -> Optional[go.Figure]:
            fig = go.Figure()

            for series in series_data:
                metrics_list = series["metrics"]
                y_values = [info.get(values_key, float("nan")) for info in metrics_list]
                y_array = np.asarray(y_values, dtype=float)
                if not np.any(np.isfinite(y_array)):
                    continue

                hover = []
                for info in metrics_list:
                    label = info["label"]
                    raw_val = info.get("raw", float("nan"))
                    clipped_val = info.get("clipped", float("nan"))
                    relative_val = info.get("relative", float("nan"))
                    parts = [f"Top-k: {label}"]
                    parts.append(
                        "Raw mean: "
                        + ("N/A" if not np.isfinite(raw_val) else f"{raw_val:.4f}")
                    )
                    parts.append(
                        "Clipped mean: "
                        + ("N/A" if not np.isfinite(clipped_val) else f"{clipped_val:.4f}")
                    )
                    parts.append(
                        "Relative lift: "
                        + (
                            "N/A"
                            if not np.isfinite(relative_val)
                            else f"{relative_val:.3f}×"
                        )
                    )
                    hover.append("<br>".join(parts))

                fig.add_trace(
                    go.Scatter(
                        x=x_positions,
                        y=y_values,
                        mode="lines+markers",
                        name=series["name"],
                        line=dict(color=series["color"], width=2),
                        marker=dict(size=8),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover,
                    )
                )

            fig.update_xaxes(
                tickmode="array",
                tickvals=x_positions,
                ticktext=x_ticklabels,
                title="Top-k concentration",
            )
            fig.update_yaxes(
                title=y_label,
                rangemode="tozero",
            )
            if values_key == "relative":
                fig.add_hline(
                    y=1.0,
                    line=dict(color="#666666", dash="dash", width=1),
                )
            fig.update_layout(
                title=dict(
                    text=(
                        f"<b>Feature Importance Accumulation: {dataset_label}</b><br>"
                        f"<sub>{field_label} — {subtitle}</sub>"
                    ),
                    x=0.5,
                    xanchor="center",
                    font=dict(size=12, family="Times New Roman"),
                ),
                legend=dict(x=1.02, y=1.0, title="Method · Graph"),
                height=520,
                width=1100,
                plot_bgcolor="rgba(245,245,245,0.7)",
                paper_bgcolor="white",
                margin=dict(l=80, r=250, t=140, b=100),
                hovermode="closest",
                font=dict(size=11, family="Times New Roman"),
            )
            if not fig.data:
                return None
            return fig

        raw_fig = _build_scatter_figure(
            "raw",
            "Raw mean concentration",
            "Raw mean values across top-k",
        )
        clipped_fig = _build_scatter_figure(
            "clipped",
            "Clipped mean concentration",
            "Clipped to [0, 1]",
        )
        relative_fig = _build_scatter_figure(
            "relative",
            "Relative lift vs. TokenSHAP",
            "Ratio to baseline (TokenSHAP Tokens)",
        )

        plot_dir = output_dir / PLOTS_DIRNAME
        plot_dir.mkdir(parents=True, exist_ok=True)

        output_name = dataset_label.replace(" ", "_").lower()
        raw_stem = plot_dir / f"importance_accumulation_{output_name}_raw_scatter"
        clipped_stem = plot_dir / f"importance_accumulation_{output_name}_clipped_scatter"
        relative_stem = plot_dir / f"importance_accumulation_{output_name}_relative_scatter"

        download_config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{output_name}_importance_accumulation",
                "width": 1920,
                "height": 1080,
                "scale": 3,
            }
        }

        if raw_fig is not None:
            raw_fig.write_html(
                str(raw_stem.with_suffix(".html")),
                include_plotlyjs="cdn",
                config=download_config,
            )
            try:
                raw_fig.write_image(str(raw_stem.with_suffix(".pdf")), width=1100, height=520)
                print(f"  ✓ {raw_stem.with_suffix('.pdf').name}")
            except Exception as exc:
                print(f"  ! PDF failed: {exc}")
            print(f"  ✓ {raw_stem.with_suffix('.html').name}")
        if clipped_fig is not None:
            clipped_fig.write_html(
                str(clipped_stem.with_suffix(".html")),
                include_plotlyjs="cdn",
                config=download_config,
            )
            try:
                clipped_fig.write_image(str(clipped_stem.with_suffix(".pdf")), width=1100, height=520)
                print(f"  ✓ {clipped_stem.with_suffix('.pdf').name}")
            except Exception as exc:
                print(f"  ! PDF failed: {exc}")
            print(f"  ✓ {clipped_stem.with_suffix('.html').name}")
        if relative_fig is not None:
            relative_fig.write_html(
                str(relative_stem.with_suffix(".html")),
                include_plotlyjs="cdn",
                config=download_config,
            )
            try:
                relative_fig.write_image(str(relative_stem.with_suffix(".pdf")), width=1100, height=520)
                print(f"  ✓ {relative_stem.with_suffix('.pdf').name}")
            except Exception as exc:
                print(f"  ! PDF failed: {exc}")
            print(f"  ✓ {relative_stem.with_suffix('.html').name}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Main entry point: process all progression fields."""
    
    parser = argparse.ArgumentParser(
        description="Generate importance accumulation plots for all progression fields."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PROGRESSION_FIELDS_ROOT,
        help=f"Root directory containing progression field folders (default: {DEFAULT_PROGRESSION_FIELDS_ROOT})",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=DROP_FIELDS,
        help="Progression drop fields to process (default: drop-based signals only).",
    )
    
    args = parser.parse_args()
    root_dir = args.root.resolve()
    
    print("=" * 140)
    print("IMPORTANCE ACCUMULATION ANALYSIS: Multi-Field Progression Evaluation")
    print("=" * 140)
    print(f"\nInput root: {root_dir}")
    print(f"Processing fields: {', '.join(args.fields)}\n")
    
    for field_name in args.fields:
        field_dir = root_dir / field_name
        
        if not field_dir.exists():
            print(f"⚠️  Field directory not found: {field_dir}")
            continue
        
        csv_path = field_dir / SUMMARY_FILENAME
        
        if not csv_path.exists():
            print(f"⚠️  Summary CSV not found: {csv_path}")
            continue
        
        print(f"\nProcessing: {FIELD_LABELS.get(field_name, field_name)}")
        print("-" * 140)
        
        generate_plots_for_field(csv_path, field_dir, field_name)
    
    print("\n" + "=" * 140)
    print("COMPLETE")
    print("=" * 140)


if __name__ == "__main__":
    main()
