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
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

DEFAULT_METHOD_COLORS: Dict[str, str] = {
    "token_shap_llm": "#3498db",
    "graphsvx": "#2ecc71",
    "subgraphx": "#e74c3c",
}

DEFAULT_DATASET_ORDER: Sequence[str] = ("setfit_ag_news", "stanfordnlp_sst2")

DEFAULT_DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News",
    "stanfordnlp_sst2": "SST-2",
}


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
        
        # Create 4 subplots (one per top-k metric)
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=[TOPK_LABELS[m] for m in DEFAULT_TOPK_METRICS],
            specs=[[{"type": "bar"} for _ in range(4)]],
            horizontal_spacing=0.12,
        )
        
        # For each top-k metric
        for col_idx, metric in enumerate(DEFAULT_TOPK_METRICS, start=1):
            metric_data = ds_df[ds_df["metric"] == metric]
            
            # Collect values for each method
            x_labels = []
            y_values = []
            colors_list = []
            texts = []
            
            for method in DEFAULT_METHOD_ORDER:
                method_row = metric_data[metric_data["method"] == method]
                
                if method_row.empty:
                    continue
                
                raw_mean = float(method_row["mean"].iloc[0])
                clipped_mean = float(np.clip(raw_mean, 0.0, 1.0))
                
                x_labels.append(DEFAULT_METHOD_LABELS[method])
                y_values.append(clipped_mean)
                colors_list.append(DEFAULT_METHOD_COLORS[method])
                
                texts.append(
                    f"{clipped_mean:.3f}\n({raw_mean:.3f})" if raw_mean != clipped_mean
                    else f"{clipped_mean:.3f}"
                )
            
            # Add bar trace
            if x_labels:
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_values,
                        marker=dict(color=colors_list, line=dict(color="black", width=1.5)),
                        text=texts,
                        textposition="outside",
                        showlegend=False,
                        hovertemplate="<b>%{x}</b><br>Concentration: %{y:.4f}<extra></extra>",
                    ),
                    row=1, col=col_idx
                )
        
        # Update axes
        for col in range(1, 5):
            fig.update_xaxes(tickangle=45, row=1, col=col)
            fig.update_yaxes(range=[0, 1.1], row=1, col=col)
        
        fig.update_yaxes(title_text="Concentration [0-1]", row=1, col=1)
        
        # Update layout
        field_label = FIELD_LABELS.get(field_name, field_name)
        dataset_label = DEFAULT_DATASET_LABELS.get(dataset, dataset)
        
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Feature Importance Accumulation: {dataset_label}</b><br>"
                    f"<sub>{field_label} (Raw Clipped Values)</sub><br>"
                    f"<sub>Side-by-side comparison showing distribution patterns</sub>"
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=12, family="Times New Roman"),
            ),
            height=600,
            width=1400,
            plot_bgcolor="rgba(245,245,245,0.7)",
            paper_bgcolor="white",
            margin=dict(l=100, r=50, t=150, b=100),
            hovermode="closest",
            font=dict(size=11, family="Times New Roman"),
        )
        
        # Save
        plot_dir = output_dir / PLOTS_DIRNAME
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        output_name = dataset_label.replace(" ", "_").lower()
        output_stem = plot_dir / f"importance_accumulation_{output_name}"
        _save_figure(fig, output_stem, width=1400, height=600)


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
