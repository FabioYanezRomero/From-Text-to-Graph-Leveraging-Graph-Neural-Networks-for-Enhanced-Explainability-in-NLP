"""Unified CLI entry point for insight analytics routines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
from .utils import InsightFrame, load_insights

DEFAULT_OUTPUT_BASE = Path("outputs/analytics")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Determine and create the directory where artefacts will be stored."""
    if args.output_dir is not None:
        target = args.output_dir
    else:
        target = DEFAULT_OUTPUT_BASE / args.command
    target.mkdir(parents=True, exist_ok=True)
    return target


def _handle_overall(args: argparse.Namespace) -> None:
    """Compute descriptive metrics and persist JSON/CSV summaries."""
    from . import overall_metrics

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    results = overall_metrics.run_overall_metrics(insight, args.group_key)

    json_path = args.output_json or output_dir / "overall_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat_rows: List[Dict[str, object]] = []
    for scope, scope_data in results.items():
        if scope == "overall":
            for metric, stats in scope_data.items():
                row = {"scope": scope, "group": "all", "metric": metric}
                row.update(stats)
                flat_rows.append(row)
        elif scope == "by_group":
            for group, metrics in scope_data.items():
                for metric, stats in metrics.items():
                    row = {"scope": scope, "group": group, "metric": metric}
                    row.update(stats)
                    flat_rows.append(row)

    if flat_rows:
        csv_path = args.output_csv or output_dir / "overall_metrics.csv"
        pd.DataFrame(flat_rows).to_csv(csv_path, index=False)


def _handle_distribution(args: argparse.Namespace) -> None:
    """Generate histograms, boxplots, and token frequencies."""
    from . import distribution_analysis

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    distribution_analysis.run_distribution_analysis(insight, output_dir)
    summary = {
        "metrics_processed": [
            metric
            for metric in distribution_analysis.DISTRIBUTION_METRICS
            if metric in insight.data.columns
        ],
        "token_rows": int(len(insight.token_frame)),
    }
    summary_path = output_dir / "distribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _handle_structural(args: argparse.Namespace) -> None:
    """Run structural distribution analysis."""
    from . import structural_analysis

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = structural_analysis.run_structural_analysis(insight, output_dir)
    (output_dir / "structural_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _handle_structural_visualise(args: argparse.Namespace) -> None:
    """Compute correlation analytics and render structural heatmaps."""
    from . import structural_visualisation

    results = structural_visualisation.run_structural_visualisation(
        dataset=args.dataset,
        graph_type=args.graph,
        config_path=args.config,
        output_root=args.output_dir,
        significance_threshold=args.significance_threshold,
        create_clustered=not args.no_clustered,
    )

    summary = {
        "output_dir": str(results["output_dir"]),
        "analytics_csv": [str(path) for path in results.get("analytics_csv", [])],
        "heatmaps": [str(path) for path in results.get("heatmaps", [])],
        "difference_heatmaps": [str(path) for path in results.get("difference_heatmaps", [])],
        "difference_tables": [str(path) for path in results.get("difference_tables", [])],
        "significance_threshold": args.significance_threshold,
        "clustered": not args.no_clustered,
    }
    summary_path = Path(results["output_dir"]) / "structural_visualisation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _handle_fidelity(args: argparse.Namespace) -> None:
    """Analyse fidelity drop distributions."""
    from . import fidelity_analysis

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = fidelity_analysis.run_fidelity_analysis(insight, output_dir, args.group_key)
    (output_dir / "fidelity_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _handle_maskout(args: argparse.Namespace) -> None:
    """Measure maskout effects and correlations."""
    from . import maskout_effects

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = maskout_effects.run_maskout_effects(insight, output_dir, args.group_key)
    (output_dir / "maskout_effects_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _handle_confidence(args: argparse.Namespace) -> None:
    """Relate prediction confidence to explanation size metrics."""
    from . import confidence_vs_size

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = confidence_vs_size.run_confidence_size(insight, output_dir, args.group_key)
    (output_dir / "confidence_size_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _handle_insertion(args: argparse.Namespace) -> None:
    """Compare insertion AUCs and averaged curves."""
    from . import insertion_auc_analysis

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = insertion_auc_analysis.run_insertion_auc(insight, output_dir, args.group_key)
    (output_dir / "insertion_auc_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _handle_outliers(args: argparse.Namespace) -> None:
    """Detect outliers and produce cohort comparisons."""
    from . import group_outliers

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = group_outliers.run_group_outlier_analysis(insight, output_dir, args.group_key)
    (output_dir / "group_outliers_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _handle_complexity(args: argparse.Namespace) -> None:
    """Relate structural complexity to explanation quality."""
    from . import structural_complexity

    output_dir = _resolve_output_dir(args)
    insight = load_insights(args.insight_paths)
    summary = structural_complexity.run_structural_complexity(insight, output_dir, args.group_key)
    (output_dir / "structural_complexity_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the analytics CLI parser."""
    from . import structural_visualisation

    parser = argparse.ArgumentParser(
        description="Analytics toolkit for explanation insight exports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, include_group: bool = True) -> None:
        subparser.add_argument(
            "insight_paths",
            nargs="+",
            help="Insight JSON files generated by src.Insights.",
        )
        if include_group:
            subparser.add_argument(
                "--group-key",
                default="label",
                help="Column used for stratified analysis (default: label).",
            )
        subparser.add_argument(
            "--output-dir",
            type=Path,
            help="Directory where artefacts are written (default: outputs/analytics/<command>).",
        )

    # Overall metrics
    overall_parser = subparsers.add_parser(
        "overall",
        help="Compute descriptive metrics (mean/median/std) overall and per group.",
    )
    add_common_arguments(overall_parser)
    overall_parser.add_argument(
        "--output-json",
        type=Path,
        help="Override path for the metrics JSON report.",
    )
    overall_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Override path for the metrics CSV report.",
    )
    overall_parser.set_defaults(func=_handle_overall)

    # Distribution analysis
    distribution_parser = subparsers.add_parser(
        "distributions",
        help="Generate histograms, boxplots, and token frequency tables.",
    )
    add_common_arguments(distribution_parser, include_group=False)
    distribution_parser.set_defaults(func=_handle_distribution)

    # Structural analysis
    structural_parser = subparsers.add_parser(
        "structural",
        help="Investigate structural metrics and correlations.",
    )
    add_common_arguments(structural_parser, include_group=False)
    structural_parser.set_defaults(func=_handle_structural)

    # Structural visualisations
    visualise_parser = subparsers.add_parser(
        "structural_visualise",
        help="Run structural graph analytics and render heatmaps for a dataset/graph pair.",
    )
    visualise_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset identifier (e.g. ag_news or sst2).",
    )
    visualise_parser.add_argument(
        "--graph",
        required=True,
        help="Graph flavour (constituency, syntactic, skipgrams, window).",
    )
    visualise_parser.add_argument(
        "--config",
        type=Path,
        default=structural_visualisation.DEFAULT_CONFIG_PATH,
        help="Path to structural config describing available datasets (default: configs/structural_analysis_config.json).",
    )
    visualise_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Root directory for generated analytics (default: outputs/analytics/structural).",
    )
    visualise_parser.add_argument(
        "--significance-threshold",
        type=float,
        help="Absolute correlation difference threshold to highlight in difference heatmaps.",
    )
    visualise_parser.add_argument(
        "--no-clustered",
        action="store_true",
        help="Skip clustered difference heatmap generation.",
    )
    visualise_parser.set_defaults(func=_handle_structural_visualise)

    # Fidelity
    fidelity_parser = subparsers.add_parser(
        "fidelity",
        help="Study fidelity drops and identify extreme subsets.",
    )
    add_common_arguments(fidelity_parser)
    fidelity_parser.set_defaults(func=_handle_fidelity)

    # Maskout
    maskout_parser = subparsers.add_parser(
        "maskout",
        help="Quantify maskout robustness and correlations.",
    )
    add_common_arguments(maskout_parser)
    maskout_parser.set_defaults(func=_handle_maskout)

    # Confidence vs size
    confidence_parser = subparsers.add_parser(
        "confidence",
        help="Relate confidence to explanation sparsity and coalition size.",
    )
    add_common_arguments(confidence_parser)
    confidence_parser.set_defaults(func=_handle_confidence)

    # Insertion AUC
    insertion_parser = subparsers.add_parser(
        "insertion",
        help="Compare insertion AUCs and average insertion curves.",
    )
    add_common_arguments(insertion_parser)
    insertion_parser.set_defaults(func=_handle_insertion)

    # Group/outlier analysis
    outlier_parser = subparsers.add_parser(
        "outliers",
        help="Detect outliers and compare metrics across cohorts.",
    )
    add_common_arguments(outlier_parser)
    outlier_parser.set_defaults(func=_handle_outliers)

    # Structural complexity
    complexity_parser = subparsers.add_parser(
        "complexity",
        help="Analyse how structural complexity interacts with fidelity and confidence.",
    )
    add_common_arguments(complexity_parser)
    complexity_parser.set_defaults(func=_handle_complexity)

    return parser


def main(argv: List[str] | None = None) -> int:
    """Dispatch analytics subcommands."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
