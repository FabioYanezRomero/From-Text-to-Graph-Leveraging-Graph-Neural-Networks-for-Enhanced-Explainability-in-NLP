"""Comparative analysis between LLM and GNN explainability modules.

This module provides comprehensive comparative analysis capabilities:
- Direct metric comparisons between LLM and GNN methods
- Statistical tests for significant differences
- Visualization of comparative distributions
- Method ranking and performance benchmarking
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


@dataclass
class ComparisonResult:
    """Results of comparing two methods on a metric."""
    
    metric_name: str
    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    std_a: float
    std_b: float
    count_a: int
    count_b: int
    mean_diff: float
    median_diff: float
    statistical_test: Dict[str, any]
    effect_size: Dict[str, any]


def compare_methods_on_metric(
    data: pd.DataFrame,
    metric_col: str,
    method_col: str = "method",
    model_type_col: str = "model_type",
) -> List[ComparisonResult]:
    """Compare all pairs of methods on a given metric.
    
    Args:
        data: DataFrame containing the data
        metric_col: Column name of the metric to compare
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type (llm/gnn)
        
    Returns:
        List of ComparisonResult objects
    """
    if metric_col not in data.columns:
        return []
    
    results = []
    
    # Get unique methods
    methods = data[method_col].dropna().unique()
    
    # Compare all pairs
    for i, method_a in enumerate(methods):
        for method_b in methods[i+1:]:
            data_a = data[data[method_col] == method_a][metric_col]
            data_b = data[data[method_col] == method_b][metric_col]
            
            # Clean data
            values_a = pd.to_numeric(data_a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            values_b = pd.to_numeric(data_b, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(values_a) < 3 or len(values_b) < 3:
                continue
            
            # Compute statistics
            mean_a, mean_b = values_a.mean(), values_b.mean()
            median_a, median_b = values_a.median(), values_b.median()
            std_a, std_b = values_a.std(ddof=1), values_b.std(ddof=1)
            
            # Mann-Whitney U test
            try:
                u_stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
                statistical_test = {
                    "test": "mann_whitney_u",
                    "u_statistic": float(u_stat),
                    "p_value": float(p_value),
                    "significant_at_0.05": bool(p_value < 0.05),
                    "significant_at_0.01": bool(p_value < 0.01),
                }
            except Exception as e:
                statistical_test = {"error": str(e)}
            
            # Effect size (Cohen's d)
            try:
                n_a, n_b = len(values_a), len(values_b)
                pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
                cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
                
                effect_size = {
                    "cohens_d": float(cohens_d),
                    "interpretation": (
                        "negligible" if abs(cohens_d) < 0.2 else
                        "small" if abs(cohens_d) < 0.5 else
                        "medium" if abs(cohens_d) < 0.8 else
                        "large"
                    ),
                }
            except Exception as e:
                effect_size = {"error": str(e)}
            
            results.append(ComparisonResult(
                metric_name=metric_col,
                method_a=method_a,
                method_b=method_b,
                mean_a=float(mean_a),
                mean_b=float(mean_b),
                median_a=float(median_a),
                median_b=float(median_b),
                std_a=float(std_a),
                std_b=float(std_b),
                count_a=len(values_a),
                count_b=len(values_b),
                mean_diff=float(mean_a - mean_b),
                median_diff=float(median_a - median_b),
                statistical_test=statistical_test,
                effect_size=effect_size,
            ))
    
    return results


def compare_llm_vs_gnn(
    data: pd.DataFrame,
    metric_col: str,
    model_type_col: str = "model_type",
) -> Optional[ComparisonResult]:
    """Compare LLM vs GNN on a metric (aggregating across all methods of each type).
    
    Args:
        data: DataFrame containing the data
        metric_col: Column name of the metric to compare
        model_type_col: Column name identifying model type
        
    Returns:
        ComparisonResult or None if comparison not possible
    """
    if metric_col not in data.columns or model_type_col not in data.columns:
        return None
    
    # Separate LLM and GNN data
    llm_data = data[data[model_type_col] == "llm"][metric_col]
    gnn_data = data[data[model_type_col] == "gnn"][metric_col]
    
    # Clean data
    llm_values = pd.to_numeric(llm_data, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    gnn_values = pd.to_numeric(gnn_data, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(llm_values) < 3 or len(gnn_values) < 3:
        return None
    
    # Compute statistics
    mean_llm, mean_gnn = llm_values.mean(), gnn_values.mean()
    median_llm, median_gnn = llm_values.median(), gnn_values.median()
    std_llm, std_gnn = llm_values.std(ddof=1), gnn_values.std(ddof=1)
    
    # Mann-Whitney U test
    try:
        u_stat, p_value = stats.mannwhitneyu(llm_values, gnn_values, alternative="two-sided")
        statistical_test = {
            "test": "mann_whitney_u",
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
            "significant_at_0.01": bool(p_value < 0.01),
        }
    except Exception as e:
        statistical_test = {"error": str(e)}
    
    # Effect size (Cohen's d)
    try:
        n_llm, n_gnn = len(llm_values), len(gnn_values)
        pooled_std = np.sqrt(((n_llm - 1) * std_llm**2 + (n_gnn - 1) * std_gnn**2) / (n_llm + n_gnn - 2))
        cohens_d = (mean_llm - mean_gnn) / pooled_std if pooled_std > 0 else 0.0
        
        effect_size = {
            "cohens_d": float(cohens_d),
            "interpretation": (
                "negligible" if abs(cohens_d) < 0.2 else
                "small" if abs(cohens_d) < 0.5 else
                "medium" if abs(cohens_d) < 0.8 else
                "large"
            ),
        }
    except Exception as e:
        effect_size = {"error": str(e)}
    
    return ComparisonResult(
        metric_name=metric_col,
        method_a="llm",
        method_b="gnn",
        mean_a=float(mean_llm),
        mean_b=float(mean_gnn),
        median_a=float(median_llm),
        median_b=float(median_gnn),
        std_a=float(std_llm),
        std_b=float(std_gnn),
        count_a=len(llm_values),
        count_b=len(gnn_values),
        mean_diff=float(mean_llm - mean_gnn),
        median_diff=float(median_llm - median_gnn),
        statistical_test=statistical_test,
        effect_size=effect_size,
    )


def plot_comparative_distributions(
    data: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
    plot_type: str = "violin",
) -> None:
    """Create comparative distribution plots.
    
    Args:
        data: DataFrame containing the data
        metric_col: Column name of the metric to plot
        output_path: Path to save the plot
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type
        plot_type: Type of plot ("violin", "box", or "kde")
    """
    # Clean data
    plot_data = data[[metric_col, method_col, model_type_col]].copy()
    plot_data[metric_col] = pd.to_numeric(plot_data[metric_col], errors="coerce")
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if plot_data.empty:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if plot_type == "violin":
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create violin plot with model type as hue
        sns.violinplot(
            data=plot_data,
            x=method_col,
            y=metric_col,
            hue=model_type_col,
            ax=ax,
            palette={"llm": "#E74C3C", "gnn": "#3498DB"},
            split=False,
        )
        
        ax.set_title(f"Distribution of {metric_col} by Method and Model Type")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric_col)
        plt.xticks(rotation=45, ha="right")
        ax.legend(title="Model Type")
        
    elif plot_type == "box":
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(
            data=plot_data,
            x=method_col,
            y=metric_col,
            hue=model_type_col,
            ax=ax,
            palette={"llm": "#E74C3C", "gnn": "#3498DB"},
        )
        
        ax.set_title(f"Distribution of {metric_col} by Method and Model Type")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric_col)
        plt.xticks(rotation=45, ha="right")
        ax.legend(title="Model Type")
        
    elif plot_type == "kde":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_type in plot_data[model_type_col].unique():
            subset = plot_data[plot_data[model_type_col] == model_type]
            for method in subset[method_col].unique():
                method_data = subset[subset[method_col] == method][metric_col]
                if len(method_data) > 3:
                    method_data.plot(
                        kind="kde",
                        ax=ax,
                        label=f"{model_type}-{method}",
                        linewidth=2,
                    )
        
        ax.set_title(f"KDE of {metric_col} by Method and Model Type")
        ax.set_xlabel(metric_col)
        ax.set_ylabel("Density")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_llm_vs_gnn_comparison(
    data: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    model_type_col: str = "model_type",
) -> None:
    """Create a comparison plot showing LLM vs GNN across multiple metrics.
    
    Args:
        data: DataFrame containing the data
        metrics: List of metrics to compare
        output_path: Path to save the plot
        model_type_col: Column name identifying model type
    """
    # Prepare data for plotting
    results = []
    
    for metric in metrics:
        if metric not in data.columns:
            continue
        
        llm_data = data[data[model_type_col] == "llm"][metric]
        gnn_data = data[data[model_type_col] == "gnn"][metric]
        
        llm_values = pd.to_numeric(llm_data, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        gnn_values = pd.to_numeric(gnn_data, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(llm_values) > 0 and len(gnn_values) > 0:
            results.append({
                "Metric": metric,
                "LLM": llm_values.mean(),
                "GNN": gnn_values.mean(),
                "LLM_std": llm_values.std(ddof=1),
                "GNN_std": gnn_values.std(ddof=1),
            })
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df["LLM"], width, label="LLM", 
                   yerr=df["LLM_std"], capsize=5, color="#E74C3C", alpha=0.8)
    bars2 = ax.bar(x + width/2, df["GNN"], width, label="GNN",
                   yerr=df["GNN_std"], capsize=5, color="#3498DB", alpha=0.8)
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean Value")
    ax.set_title("LLM vs GNN: Comparative Performance Across Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_method_ranking(
    data: pd.DataFrame,
    metrics: List[str],
    *,
    method_col: str = "method",
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """Create a ranking table for all methods across multiple metrics.
    
    Args:
        data: DataFrame containing the data
        metrics: List of metrics to rank on
        method_col: Column name identifying the method
        higher_is_better: Dict mapping metric names to whether higher is better
        
    Returns:
        DataFrame with method rankings
    """
    if higher_is_better is None:
        # Default assumptions
        higher_is_better = {
            "insertion_auc": True,
            "deletion_auc": True,
            "faithfulness": True,
            "robustness_score": False,
            "sparsity": False,
            "fidelity_plus": False,
            "fidelity_minus": True,
        }
    
    ranking_data = []
    
    for method in data[method_col].dropna().unique():
        method_data = data[data[method_col] == method]
        
        row = {"method": method, "count": len(method_data)}
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            values = pd.to_numeric(method_data[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)
                row[f"{metric}_median"] = values.median()
        
        ranking_data.append(row)
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Add rank columns
    for metric in metrics:
        mean_col = f"{metric}_mean"
        rank_col = f"{metric}_rank"
        
        if mean_col in ranking_df.columns:
            ascending = not higher_is_better.get(metric, True)
            ranking_df[rank_col] = ranking_df[mean_col].rank(ascending=ascending, method="min")
    
    return ranking_df


def run_comparative_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: Optional[List[str]] = None,
    method_col: str = "method",
    model_type_col: str = "model_type",
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run comprehensive comparative analysis.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        metrics: List of metrics to analyze (if None, use defaults)
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary containing all analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default metrics
    if metrics is None:
        metrics = [
            "fidelity_plus",
            "fidelity_minus",
            "faithfulness",
            "insertion_auc",
            "deletion_auc",
            "origin_contrastivity",
            "sparsity",
            "robustness_score",
        ]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in data.columns]
    
    print(f"Running comparative analysis on {len(available_metrics)} metrics...")
    
    results = {
        "output_dir": str(output_dir),
        "metrics_analyzed": available_metrics,
        "method_comparisons": {},
        "llm_vs_gnn": {},
        "method_ranking": {},
    }
    
    # Compare all method pairs on each metric
    for metric in available_metrics:
        print(f"  Comparing methods on {metric}...")
        
        comparisons = compare_methods_on_metric(data, metric, method_col, model_type_col)
        
        results["method_comparisons"][metric] = [
            {
                "method_a": c.method_a,
                "method_b": c.method_b,
                "mean_a": c.mean_a,
                "mean_b": c.mean_b,
                "mean_diff": c.mean_diff,
                "statistical_test": c.statistical_test,
                "effect_size": c.effect_size,
            }
            for c in comparisons
        ]
    
    # LLM vs GNN comparison
    if model_type_col in data.columns:
        print("  Comparing LLM vs GNN...")
        
        for metric in available_metrics:
            comparison = compare_llm_vs_gnn(data, metric, model_type_col)
            
            if comparison:
                results["llm_vs_gnn"][metric] = {
                    "llm_mean": comparison.mean_a,
                    "gnn_mean": comparison.mean_b,
                    "llm_median": comparison.median_a,
                    "gnn_median": comparison.median_b,
                    "mean_diff": comparison.mean_diff,
                    "statistical_test": comparison.statistical_test,
                    "effect_size": comparison.effect_size,
                }
    
    # Method ranking
    ranking_df = create_method_ranking(data, available_metrics, method_col=method_col)
    ranking_path = output_dir / "method_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    results["method_ranking"]["table_path"] = str(ranking_path)
    results["method_ranking"]["data"] = ranking_df.to_dict(orient="records")
    
    # Create plots
    if create_plots:
        print("  Creating comparative plots...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        results["plots"] = {}
        
        # Individual metric plots
        for metric in available_metrics:
            try:
                for plot_type in ["violin", "box"]:
                    plot_path = plots_dir / f"{metric}_{plot_type}.png"
                    plot_comparative_distributions(
                        data, metric, plot_path,
                        method_col=method_col,
                        model_type_col=model_type_col,
                        plot_type=plot_type,
                    )
                    
                    if metric not in results["plots"]:
                        results["plots"][metric] = []
                    results["plots"][metric].append(str(plot_path))
            except Exception as e:
                print(f"Warning: Could not create plots for {metric}: {e}")
        
        # LLM vs GNN summary plot
        if model_type_col in data.columns:
            try:
                summary_plot = plots_dir / "llm_vs_gnn_summary.png"
                plot_llm_vs_gnn_comparison(data, available_metrics, summary_plot, model_type_col)
                results["plots"]["llm_vs_gnn_summary"] = str(summary_plot)
            except Exception as e:
                print(f"Warning: Could not create LLM vs GNN summary plot: {e}")
    
    # Save summary
    summary_path = output_dir / "comparative_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparative analysis complete. Results saved to {output_dir}")
    
    return results

