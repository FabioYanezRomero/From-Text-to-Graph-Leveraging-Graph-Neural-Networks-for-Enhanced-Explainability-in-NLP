"""Stratified metrics analysis by class and correctness.

This module provides comprehensive stratified analysis capabilities for:
- Analysis by predicted class
- Analysis by correctness (correct vs incorrect predictions)
- Cross-stratification (class × correctness)
- Statistical comparisons between strata
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
class StratumStats:
    """Statistics for a single stratum."""
    
    count: int
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    
    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
        }


@dataclass
class StratifiedMetrics:
    """Complete stratified analysis results."""
    
    metric_name: str
    overall: StratumStats
    by_class: Dict[str, StratumStats]
    by_correctness: Dict[str, StratumStats]
    by_class_and_correctness: Dict[Tuple[str, str], StratumStats]
    statistical_tests: Dict[str, dict]


def compute_stats(series: pd.Series) -> Optional[StratumStats]:
    """Compute descriptive statistics for a series."""
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    if numeric.empty or len(numeric) == 0:
        return None
    
    return StratumStats(
        count=len(numeric),
        mean=float(numeric.mean()),
        std=float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        median=float(numeric.median()),
        q25=float(numeric.quantile(0.25)),
        q75=float(numeric.quantile(0.75)),
        min=float(numeric.min()),
        max=float(numeric.max()),
    )


def perform_statistical_tests(
    data: pd.DataFrame,
    metric_col: str,
    group_col: str,
) -> Dict[str, dict]:
    """Perform statistical tests between groups.
    
    Args:
        data: DataFrame containing the data
        metric_col: Column name of the metric to test
        group_col: Column name of the grouping variable
        
    Returns:
        Dictionary of test results
    """
    results = {}
    
    # Get groups
    groups = []
    group_names = []
    for name, group in data.groupby(group_col, dropna=False):
        values = pd.to_numeric(group[metric_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 0:
            groups.append(values.values)
            group_names.append(str(name))
    
    if len(groups) < 2:
        return {"available": False, "reason": "Need at least 2 groups for comparison"}
    
    # Kruskal-Wallis H-test (non-parametric alternative to ANOVA)
    try:
        h_stat, p_value = stats.kruskal(*groups)
        results["kruskal_wallis"] = {
            "h_statistic": float(h_stat),
            "p_value": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
            "significant_at_0.01": bool(p_value < 0.01),
        }
    except Exception as e:
        results["kruskal_wallis"] = {"error": str(e)}
    
    # Pairwise Mann-Whitney U tests
    if len(groups) == 2:
        try:
            u_stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            results["mann_whitney_u"] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05),
                "significant_at_0.01": bool(p_value < 0.01),
                "group_a": group_names[0],
                "group_b": group_names[1],
            }
        except Exception as e:
            results["mann_whitney_u"] = {"error": str(e)}
    
    # Effect size (Cohen's d for two groups)
    if len(groups) == 2:
        try:
            mean1, mean2 = np.mean(groups[0]), np.mean(groups[1])
            std1, std2 = np.std(groups[0], ddof=1), np.std(groups[1], ddof=1)
            n1, n2 = len(groups[0]), len(groups[1])
            
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
            
            results["effect_size"] = {
                "cohens_d": float(cohens_d),
                "interpretation": (
                    "small" if abs(cohens_d) < 0.5 else
                    "medium" if abs(cohens_d) < 0.8 else
                    "large"
                ),
                "group_a": group_names[0],
                "group_b": group_names[1],
            }
        except Exception as e:
            results["effect_size"] = {"error": str(e)}
    
    return results


def analyze_metric_stratified(
    data: pd.DataFrame,
    metric_col: str,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> Optional[StratifiedMetrics]:
    """Analyze a metric with full stratification.
    
    Args:
        data: DataFrame containing the data
        metric_col: Column name of the metric to analyze
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        StratifiedMetrics object or None if metric not available
    """
    if metric_col not in data.columns:
        return None
    
    # Overall statistics
    overall_stats = compute_stats(data[metric_col])
    if overall_stats is None:
        return None
    
    # By class
    by_class = {}
    if class_col in data.columns:
        for class_value, group in data.groupby(class_col, dropna=False):
            stats_obj = compute_stats(group[metric_col])
            if stats_obj is not None:
                label = "None" if pd.isna(class_value) else str(class_value)
                by_class[label] = stats_obj
    
    # By correctness
    by_correctness = {}
    if correctness_col in data.columns:
        for correct_value, group in data.groupby(correctness_col, dropna=False):
            stats_obj = compute_stats(group[metric_col])
            if stats_obj is not None:
                if pd.isna(correct_value):
                    label = "unknown"
                else:
                    label = "correct" if correct_value else "incorrect"
                by_correctness[label] = stats_obj
    
    # By class and correctness
    by_class_and_correctness = {}
    if class_col in data.columns and correctness_col in data.columns:
        for (class_value, correct_value), group in data.groupby([class_col, correctness_col], dropna=False):
            stats_obj = compute_stats(group[metric_col])
            if stats_obj is not None:
                class_label = "None" if pd.isna(class_value) else str(class_value)
                if pd.isna(correct_value):
                    correct_label = "unknown"
                else:
                    correct_label = "correct" if correct_value else "incorrect"
                by_class_and_correctness[(class_label, correct_label)] = stats_obj
    
    # Statistical tests
    statistical_tests = {}
    
    if class_col in data.columns and len(by_class) >= 2:
        statistical_tests["by_class"] = perform_statistical_tests(data, metric_col, class_col)
    
    if correctness_col in data.columns and len(by_correctness) >= 2:
        statistical_tests["by_correctness"] = perform_statistical_tests(data, metric_col, correctness_col)
    
    return StratifiedMetrics(
        metric_name=metric_col,
        overall=overall_stats,
        by_class=by_class,
        by_correctness=by_correctness,
        by_class_and_correctness=by_class_and_correctness,
        statistical_tests=statistical_tests,
    )


def plot_stratified_boxplots(
    data: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> List[Path]:
    """Create boxplot visualizations for stratified analysis.
    
    Args:
        data: DataFrame containing the data
        metric_col: Column name of the metric to plot
        output_dir: Directory to save plots
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    # Clean metric data
    plot_data = data[[metric_col]].copy()
    plot_data[metric_col] = pd.to_numeric(plot_data[metric_col], errors="coerce")
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if plot_data.empty:
        return generated_plots
    
    # By class
    if class_col in data.columns:
        plot_data[class_col] = data[class_col]
        plot_data = plot_data.dropna(subset=[class_col])
        
        if len(plot_data[class_col].unique()) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=plot_data, x=class_col, y=metric_col, ax=ax, palette="Set2")
            ax.set_title(f"{metric_col} by Class")
            ax.set_xlabel("Class")
            ax.set_ylabel(metric_col)
            plt.xticks(rotation=45)
            fig.tight_layout()
            
            path = output_dir / f"{metric_col}_by_class_boxplot.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            generated_plots.append(path)
    
    # By correctness
    if correctness_col in data.columns:
        plot_data = data[[metric_col, correctness_col]].copy()
        plot_data[metric_col] = pd.to_numeric(plot_data[metric_col], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Convert to readable labels
        plot_data["Correctness"] = plot_data[correctness_col].apply(
            lambda x: "Correct" if x else "Incorrect" if pd.notna(x) else "Unknown"
        )
        
        if len(plot_data["Correctness"].unique()) > 1:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.boxplot(data=plot_data, x="Correctness", y=metric_col, ax=ax, palette="Set1")
            ax.set_title(f"{metric_col} by Correctness")
            ax.set_xlabel("Prediction Correctness")
            ax.set_ylabel(metric_col)
            fig.tight_layout()
            
            path = output_dir / f"{metric_col}_by_correctness_boxplot.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            generated_plots.append(path)
    
    # By class and correctness (heatmap)
    if class_col in data.columns and correctness_col in data.columns:
        plot_data = data[[metric_col, class_col, correctness_col]].copy()
        plot_data[metric_col] = pd.to_numeric(plot_data[metric_col], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Convert correctness to readable labels
        plot_data["Correctness"] = plot_data[correctness_col].apply(
            lambda x: "Correct" if x else "Incorrect" if pd.notna(x) else "Unknown"
        )
        
        # Create pivot table
        try:
            pivot = plot_data.pivot_table(
                values=metric_col,
                index=class_col,
                columns="Correctness",
                aggfunc="mean",
            )
            
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": metric_col})
                ax.set_title(f"Mean {metric_col} by Class and Correctness")
                ax.set_xlabel("Prediction Correctness")
                ax.set_ylabel("Class")
                fig.tight_layout()
                
                path = output_dir / f"{metric_col}_by_class_correctness_heatmap.png"
                fig.savefig(path, dpi=200)
                plt.close(fig)
                generated_plots.append(path)
        except Exception as e:
            print(f"Warning: Could not create heatmap for {metric_col}: {e}")
    
    return generated_plots


def run_stratified_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: Optional[List[str]] = None,
    class_col: str = "label",
    correctness_col: str = "is_correct",
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run comprehensive stratified analysis on multiple metrics.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        metrics: List of metric columns to analyze (if None, use defaults)
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary containing all analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default metrics to analyze
    if metrics is None:
        metrics = [
            "fidelity_plus",
            "fidelity_minus",
            "faithfulness",
            "insertion_auc",
            "deletion_auc",
            "origin_contrastivity",
            "masked_contrastivity",
            "sparsity",
            "robustness_score",
            "prediction_confidence",
        ]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in data.columns]
    
    print(f"Analyzing {len(available_metrics)} metrics with stratification...")
    
    results = {
        "output_dir": str(output_dir),
        "metrics_analyzed": available_metrics,
        "class_column": class_col,
        "correctness_column": correctness_col,
        "stratified_results": {},
        "plots": {},
    }
    
    for metric in available_metrics:
        print(f"  Analyzing {metric}...")
        
        # Compute stratified statistics
        stratified = analyze_metric_stratified(
            data,
            metric,
            class_col=class_col,
            correctness_col=correctness_col,
        )
        
        if stratified is None:
            continue
        
        # Convert to dictionary
        metric_results = {
            "overall": stratified.overall.to_dict(),
            "by_class": {k: v.to_dict() for k, v in stratified.by_class.items()},
            "by_correctness": {k: v.to_dict() for k, v in stratified.by_correctness.items()},
            "by_class_and_correctness": {
                f"{k[0]}_{k[1]}": v.to_dict()
                for k, v in stratified.by_class_and_correctness.items()
            },
            "statistical_tests": stratified.statistical_tests,
        }
        
        results["stratified_results"][metric] = metric_results
        
        # Save individual metric results
        metric_output = output_dir / f"{metric}_stratified.json"
        with metric_output.open("w", encoding="utf-8") as f:
            json.dump(metric_results, f, indent=2)
        
        # Create plots
        if create_plots:
            try:
                plots = plot_stratified_boxplots(
                    data,
                    metric,
                    output_dir / "plots",
                    class_col=class_col,
                    correctness_col=correctness_col,
                )
                results["plots"][metric] = [str(p) for p in plots]
            except Exception as e:
                print(f"Warning: Could not create plots for {metric}: {e}")
    
    # Save summary
    summary_path = output_dir / "stratified_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nStratified analysis complete. Results saved to {output_dir}")
    
    return results

