from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go


DEFAULT_INPUT_CSV = Path(
    "outputs/analytics/auc/plots/final_metrics/detection_rates_by_dataset.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/analytics/auc/plots/final_metrics/connected_scatter"
)


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_detection_rates(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Detection rate data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    expected_columns = {
        "method",
        "dataset",
        "graph_type",
        "threshold",
        "mispredictions",
        "detections",
        "rate",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in detection CSV: {sorted(missing)}")
    return df
def build_connected_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    graph_types: Optional[Sequence[str]] = None,
) -> list[Path]:
    output_dir = _ensure_output_dir(output_dir)
    dataset_filter = set(datasets) if datasets else None
    method_filter = set(methods) if methods else None
    graph_filter = set(graph_types) if graph_types else None
    created_files: list[Path] = []

    for dataset, dataset_group in df.groupby("dataset", sort=True):
        if dataset_filter and dataset not in dataset_filter:
            continue

        fig = go.Figure()
        for (method, graph_type), method_group in dataset_group.groupby(
            ["method", "graph_type"], sort=True
        ):
            if method_filter and method not in method_filter:
                continue
            if graph_filter and graph_type not in graph_filter:
                continue
            ordered = method_group.sort_values("threshold")
            display_name = method
            if pd.notna(graph_type):
                display_name = f"{method} ({graph_type})"
            fig.add_trace(
                go.Scatter(
                    x=ordered["threshold"],
                    y=ordered["rate"],
                    mode="lines+markers",
                    name=display_name,
                    hovertemplate=(
                        "Method: %{customdata[0]}<br>"
                        "Graph type: %{customdata[1]}<br>"
                        "Threshold: %{x:.2f}<br>"
                        "Rate: %{y:.1%}<extra></extra>"
                    ),
                    customdata=ordered[["method", "graph_type"]].to_numpy(),
                )
            )

        fig.update_layout(
            title=f"Error Detection Rate vs. Threshold — {dataset}",
            xaxis_title="Detection threshold (AUC cutoff)",
            yaxis_title="Detection rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            legend_title="Method",
        )

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{dataset}_connected_scatter",
                "width": 1920,
                "height": 1080,
                "scale": 3,
            },
        }
        output_path = output_dir / f"{dataset}_connected_scatter.html"
        fig.write_html(output_path, include_plotlyjs="cdn", config=config)
        created_files.append(output_path)

    return created_files


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate connected scatter plots (threshold vs. detection rate) per dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to detection_rates_by_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Plotly HTML files will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional list of datasets to include.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional list of methods to include.",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        help="Optional list of graph types to include.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    options = parse_args(argv)
    detection_rates = load_detection_rates(options.input_csv)
    plots = build_connected_scatter(
        detection_rates,
        options.output_dir,
        datasets=options.datasets,
        methods=options.methods,
        graph_types=options.graph_types,
    )
    print("Generated connected scatter plots:")
    for path in plots:
        print(f" - {path}")


if __name__ == "__main__":
    main()
