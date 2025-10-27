"""Unified CLI entry point for insight analytics routines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .utils import InsightFrame, load_insights
from .loaders import load_insights_from_directory, load_insights_from_files

DEFAULT_OUTPUT_BASE = Path("outputs/analytics")

def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Determine and create the directory where artefacts will be stored."""
    if args.output_dir is not None:
        target = args.output_dir
    else:
        target = DEFAULT_OUTPUT_BASE / args.command
    target.mkdir(parents=True, exist_ok=True)
    return target

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
    raise NotImplementedError("Maskout analytics disabled.")

def _handle_insertion(args: argparse.Namespace) -> None:
    """Compare insertion AUCs."""
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
    raise NotImplementedError("Outlier analytics disabled.")

def _handle_faithfulness(args: argparse.Namespace) -> None:
    """Compute aggregated faithfulness metrics using insertion AUC."""
    from .faithfulness_metrics import run_faithfulness_aggregate

    output_dir = _resolve_output_dir(args)
    summary = run_faithfulness_aggregate(
        args.insight_paths,
        output_dir,
        baseline=args.baseline,
        group_key=args.group_key,
    )
    (output_dir / "faithfulness_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_llm(args: argparse.Namespace) -> None:
    """Generate token-focused analytics for LLM explanations."""
    from .llm_analysis import run_llm_token_analysis

    output_dir = _resolve_output_dir(args)
    summary = run_llm_token_analysis(args.insight_paths, output_dir, args.top_k, args.stopwords)
    (output_dir / "llm_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_stratified(args: argparse.Namespace) -> None:
    """Run stratified analysis by class and correctness."""
    from .stratified_metrics import run_stratified_analysis
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights = load_insights_from_directory(Path(args.insights_dir))
    else:
        insights = load_insights_from_files(args.insight_paths)
    
    summary = run_stratified_analysis(
        insights.data,
        output_dir,
        metrics=args.metrics if hasattr(args, 'metrics') and args.metrics else None,
        class_col=args.class_col if hasattr(args, 'class_col') else "label",
        correctness_col=args.correctness_col if hasattr(args, 'correctness_col') else "is_correct",
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "stratified_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_comparative(args: argparse.Namespace) -> None:
    """Run comparative analysis between LLM and GNN methods."""
    from .comparative_analysis import run_comparative_analysis
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights = load_insights_from_directory(Path(args.insights_dir))
    else:
        insights = load_insights_from_files(args.insight_paths)
    
    summary = run_comparative_analysis(
        insights.data,
        output_dir,
        metrics=args.metrics if hasattr(args, 'metrics') and args.metrics else None,
        method_col=args.method_col if hasattr(args, 'method_col') else "method",
        model_type_col=args.model_type_col if hasattr(args, 'model_type_col') else "model_type",
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "comparative_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_ranking_agreement(args: argparse.Namespace) -> None:
    """Analyze ranking agreement metrics (RBO, Spearman, etc.)."""
    from .ranking_agreement import run_ranking_agreement_analysis
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights = load_insights_from_directory(Path(args.insights_dir))
    else:
        insights = load_insights_from_files(args.insight_paths)
    
    if insights.agreement_frame is None or insights.agreement_frame.empty:
        print("Warning: No agreement data found in insights.")
        return
    
    summary = run_ranking_agreement_analysis(
        insights.agreement_frame,
        output_dir,
        class_col=args.class_col if hasattr(args, 'class_col') else "label",
        correctness_col=args.correctness_col if hasattr(args, 'correctness_col') else "is_correct",
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "ranking_agreement_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_contrastivity_compactness(args: argparse.Namespace) -> None:
    """Analyze contrastivity and compactness metrics."""
    from .contrastivity_compactness import run_contrastivity_compactness_analysis
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights = load_insights_from_directory(Path(args.insights_dir))
    else:
        insights = load_insights_from_files(args.insight_paths)
    
    summary = run_contrastivity_compactness_analysis(
        insights.data,
        output_dir,
        class_col=args.class_col if hasattr(args, 'class_col') else "label",
        correctness_col=args.correctness_col if hasattr(args, 'correctness_col') else "is_correct",
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "contrastivity_compactness_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_enhanced_auc(args: argparse.Namespace) -> None:
    """Run enhanced AUC analysis with curve visualization."""
    from .enhanced_auc_analysis import run_enhanced_auc_analysis
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights = load_insights_from_directory(Path(args.insights_dir))
    else:
        insights = load_insights_from_files(args.insight_paths)
    
    summary = run_enhanced_auc_analysis(
        insights.data,
        output_dir,
        method_col=args.method_col if hasattr(args, 'method_col') else "method",
        model_type_col=args.model_type_col if hasattr(args, 'model_type_col') else "model_type",
        class_col=args.class_col if hasattr(args, 'class_col') else "label",
        correctness_col=args.correctness_col if hasattr(args, 'correctness_col') else "is_correct",
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "enhanced_auc_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_comprehensive_faithfulness(args: argparse.Namespace) -> None:
    """Run comprehensive faithfulness analysis (fidelity+/-, faithfulness)."""
    from .comprehensive_faithfulness import run_comprehensive_faithfulness_analysis
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights = load_insights_from_directory(Path(args.insights_dir))
    else:
        insights = load_insights_from_files(args.insight_paths)
    
    summary = run_comprehensive_faithfulness_analysis(
        insights.data,
        output_dir,
        method_col=args.method_col if hasattr(args, 'method_col') else "method",
        model_type_col=args.model_type_col if hasattr(args, 'model_type_col') else "model_type",
        class_col=args.class_col if hasattr(args, 'class_col') else "label",
        correctness_col=args.correctness_col if hasattr(args, 'correctness_col') else "is_correct",
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "comprehensive_faithfulness_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

def _handle_complete_pipeline(args: argparse.Namespace) -> None:
    """Run the complete analytics pipeline with all modules."""
    from .unified_analytics import run_complete_analytics_pipeline
    
    output_dir = _resolve_output_dir(args)
    
    # Load data using new loaders
    if hasattr(args, 'insights_dir') and args.insights_dir:
        insights_dir = Path(args.insights_dir)
    else:
        print("Error: --insights-dir is required for the complete pipeline.")
        return
    
    summary = run_complete_analytics_pipeline(
        insights_dir,
        output_dir,
        create_plots=not args.no_plots if hasattr(args, 'no_plots') else True,
    )
    
    (output_dir / "complete_pipeline_summary.json").write_text(
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

    # Structural analysis
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

    # Insertion AUC
    insertion_parser = subparsers.add_parser(
        "insertion",
        help="Summarise insertion AUC distributions.",
    )
    add_common_arguments(insertion_parser)
    insertion_parser.set_defaults(func=_handle_insertion)

    # Faithfulness aggregation
    faithfulness_parser = subparsers.add_parser(
        "faithfulness",
        help="Aggregate faithfulness metrics (insertion AUC based).",
    )
    add_common_arguments(faithfulness_parser)
    faithfulness_parser.add_argument(
        "--baseline",
        type=float,
        default=0.5,
        help="Baseline AUC used when computing faithfulness differences (default: 0.5).",
    )
    faithfulness_parser.set_defaults(func=_handle_faithfulness)

    # LLM token analytics
    llm_parser = subparsers.add_parser(
        "llm",
        help="Run token-level analytics for LLM explanation exports.",
    )
    add_common_arguments(llm_parser, include_group=False)
    llm_parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of highest-scoring words retained per record (<=0 keeps all).",
    )
    llm_parser.add_argument(
        "--stopwords",
        type=Path,
        help="Optional newline-delimited stopword list applied before ranking tokens.",
    )
    llm_parser.set_defaults(func=_handle_llm)

    # Stratified analysis
    stratified_parser = subparsers.add_parser(
        "stratified",
        help="Run stratified analysis by class and correctness.",
    )
    stratified_parser.add_argument(
        "--insights-dir",
        type=Path,
        help="Directory containing insights (GNN/ and LLM/ subdirectories).",
    )
    stratified_parser.add_argument(
        "insight_paths",
        nargs="*",
        help="Alternatively, provide specific insight JSON files.",
    )
    stratified_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    stratified_parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to analyze (default: all available).",
    )
    stratified_parser.add_argument(
        "--class-col",
        default="label",
        help="Column name for class labels (default: label).",
    )
    stratified_parser.add_argument(
        "--correctness-col",
        default="is_correct",
        help="Column name for correctness indicator (default: is_correct).",
    )
    stratified_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    stratified_parser.set_defaults(func=_handle_stratified)

    # Comparative analysis
    comparative_parser = subparsers.add_parser(
        "comparative",
        help="Run comparative analysis between LLM and GNN methods.",
    )
    comparative_parser.add_argument(
        "--insights-dir",
        type=Path,
        help="Directory containing insights (GNN/ and LLM/ subdirectories).",
    )
    comparative_parser.add_argument(
        "insight_paths",
        nargs="*",
        help="Alternatively, provide specific insight JSON files.",
    )
    comparative_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    comparative_parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to compare (default: all available).",
    )
    comparative_parser.add_argument(
        "--method-col",
        default="method",
        help="Column name for method identifier (default: method).",
    )
    comparative_parser.add_argument(
        "--model-type-col",
        default="model_type",
        help="Column name for model type (default: model_type).",
    )
    comparative_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    comparative_parser.set_defaults(func=_handle_comparative)

    # Ranking agreement analysis
    agreement_parser = subparsers.add_parser(
        "ranking_agreement",
        help="Analyze ranking agreement metrics (RBO, Spearman, Kendall, etc.).",
    )
    agreement_parser.add_argument(
        "--insights-dir",
        type=Path,
        help="Directory containing insights with agreement metrics.",
    )
    agreement_parser.add_argument(
        "insight_paths",
        nargs="*",
        help="Alternatively, provide specific insight JSON files.",
    )
    agreement_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    agreement_parser.add_argument(
        "--class-col",
        default="label",
        help="Column name for class labels (default: label).",
    )
    agreement_parser.add_argument(
        "--correctness-col",
        default="is_correct",
        help="Column name for correctness indicator (default: is_correct).",
    )
    agreement_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    agreement_parser.set_defaults(func=_handle_ranking_agreement)

    # Contrastivity and compactness analysis
    contrast_parser = subparsers.add_parser(
        "contrastivity_compactness",
        help="Analyze contrastivity and compactness metrics.",
    )
    contrast_parser.add_argument(
        "--insights-dir",
        type=Path,
        help="Directory containing insights.",
    )
    contrast_parser.add_argument(
        "insight_paths",
        nargs="*",
        help="Alternatively, provide specific insight JSON files.",
    )
    contrast_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    contrast_parser.add_argument(
        "--class-col",
        default="label",
        help="Column name for class labels (default: label).",
    )
    contrast_parser.add_argument(
        "--correctness-col",
        default="is_correct",
        help="Column name for correctness indicator (default: is_correct).",
    )
    contrast_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    contrast_parser.set_defaults(func=_handle_contrastivity_compactness)

    # Enhanced AUC analysis
    auc_parser = subparsers.add_parser(
        "enhanced_auc",
        help="Run enhanced AUC analysis with curve visualization.",
    )
    auc_parser.add_argument(
        "--insights-dir",
        type=Path,
        help="Directory containing insights.",
    )
    auc_parser.add_argument(
        "insight_paths",
        nargs="*",
        help="Alternatively, provide specific insight JSON files.",
    )
    auc_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    auc_parser.add_argument(
        "--method-col",
        default="method",
        help="Column name for method identifier (default: method).",
    )
    auc_parser.add_argument(
        "--model-type-col",
        default="model_type",
        help="Column name for model type (default: model_type).",
    )
    auc_parser.add_argument(
        "--class-col",
        default="label",
        help="Column name for class labels (default: label).",
    )
    auc_parser.add_argument(
        "--correctness-col",
        default="is_correct",
        help="Column name for correctness indicator (default: is_correct).",
    )
    auc_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    auc_parser.set_defaults(func=_handle_enhanced_auc)

    # Comprehensive faithfulness analysis
    faith_parser = subparsers.add_parser(
        "comprehensive_faithfulness",
        help="Run comprehensive faithfulness analysis (fidelity+/-, faithfulness).",
    )
    faith_parser.add_argument(
        "--insights-dir",
        type=Path,
        help="Directory containing insights.",
    )
    faith_parser.add_argument(
        "insight_paths",
        nargs="*",
        help="Alternatively, provide specific insight JSON files.",
    )
    faith_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    faith_parser.add_argument(
        "--method-col",
        default="method",
        help="Column name for method identifier (default: method).",
    )
    faith_parser.add_argument(
        "--model-type-col",
        default="model_type",
        help="Column name for model type (default: model_type).",
    )
    faith_parser.add_argument(
        "--class-col",
        default="label",
        help="Column name for class labels (default: label).",
    )
    faith_parser.add_argument(
        "--correctness-col",
        default="is_correct",
        help="Column name for correctness indicator (default: is_correct).",
    )
    faith_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    faith_parser.set_defaults(func=_handle_comprehensive_faithfulness)

    # Complete pipeline
    pipeline_parser = subparsers.add_parser(
        "complete_pipeline",
        help="Run the complete analytics pipeline with all modules.",
    )
    pipeline_parser.add_argument(
        "--insights-dir",
        type=Path,
        required=True,
        help="Directory containing insights (GNN/ and LLM/ subdirectories).",
    )
    pipeline_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where artefacts are written.",
    )
    pipeline_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    pipeline_parser.set_defaults(func=_handle_complete_pipeline)

    return parser

def main(argv: List[str] | None = None) -> int:
    """Dispatch analytics subcommands."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
