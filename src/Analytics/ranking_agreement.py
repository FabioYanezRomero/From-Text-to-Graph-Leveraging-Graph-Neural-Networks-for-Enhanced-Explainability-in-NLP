"""Ranking agreement analytics for comparing explainer outputs.

This module analyzes agreement metrics between different explainers:
- Rank-Biased Overlap (RBO)
- Spearman and Kendall rank correlations
- KL Divergence
- Feature Overlap Ratio
- Stability metrics (Jaccard)
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
class AgreementSummary:
    """Summary statistics for agreement metrics."""
    
    metric_name: str
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
            "metric": self.metric_name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
        }


def compute_agreement_stats(series: pd.Series, metric_name: str) -> Optional[AgreementSummary]:
    """Compute statistics for an agreement metric."""
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    if numeric.empty or len(numeric) == 0:
        return None
    
    return AgreementSummary(
        metric_name=metric_name,
        count=len(numeric),
        mean=float(numeric.mean()),
        std=float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        median=float(numeric.median()),
        q25=float(numeric.quantile(0.25)),
        q75=float(numeric.quantile(0.75)),
        min=float(numeric.min()),
        max=float(numeric.max()),
    )


def analyze_agreement_by_methods(
    agreement_data: pd.DataFrame,
) -> Dict[Tuple[str, str], Dict[str, AgreementSummary]]:
    """Analyze agreement metrics grouped by method pairs.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        
    Returns:
        Dictionary mapping (method_a, method_b) to agreement statistics
    """
    results = {}
    
    if "method_a" not in agreement_data.columns or "method_b" not in agreement_data.columns:
        return results
    
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio", "stability_jaccard", "kl_divergence"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    for (method_a, method_b), group in agreement_data.groupby(["method_a", "method_b"], dropna=False):
        method_pair = (str(method_a), str(method_b))
        results[method_pair] = {}
        
        for metric in available_metrics:
            stats_obj = compute_agreement_stats(group[metric], metric)
            if stats_obj:
                results[method_pair][metric] = stats_obj
    
    return results


def analyze_agreement_stratified(
    agreement_data: pd.DataFrame,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> Dict[str, any]:
    """Analyze agreement metrics with stratification.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        Dictionary containing stratified agreement statistics
    """
    results = {
        "overall": {},
        "by_class": {},
        "by_correctness": {},
    }
    
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio", "stability_jaccard", "kl_divergence"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    # Overall statistics
    for metric in available_metrics:
        stats_obj = compute_agreement_stats(agreement_data[metric], metric)
        if stats_obj:
            results["overall"][metric] = stats_obj.to_dict()
    
    # By class
    if class_col in agreement_data.columns:
        for class_value, group in agreement_data.groupby(class_col, dropna=False):
            class_label = "None" if pd.isna(class_value) else str(class_value)
            results["by_class"][class_label] = {}
            
            for metric in available_metrics:
                stats_obj = compute_agreement_stats(group[metric], metric)
                if stats_obj:
                    results["by_class"][class_label][metric] = stats_obj.to_dict()
    
    # By correctness
    if correctness_col in agreement_data.columns:
        for correct_value, group in agreement_data.groupby(correctness_col, dropna=False):
            if pd.isna(correct_value):
                correct_label = "unknown"
            else:
                correct_label = "correct" if correct_value else "incorrect"
            
            results["by_correctness"][correct_label] = {}
            
            for metric in available_metrics:
                stats_obj = compute_agreement_stats(group[metric], metric)
                if stats_obj:
                    results["by_correctness"][correct_label][metric] = stats_obj.to_dict()
    
    return results


def plot_agreement_distributions(
    agreement_data: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create distribution plots for agreement metrics.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio", "stability_jaccard", "kl_divergence"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    for metric in available_metrics:
        # Clean data
        values = pd.to_numeric(agreement_data[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        # Create histogram with KDE
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(values, bins=30, color="#8E44AD", edgecolor="black", alpha=0.7)
        axes[0].set_xlabel(metric)
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{metric} Distribution")
        axes[0].axvline(values.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {values.mean():.3f}")
        axes[0].axvline(values.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {values.median():.3f}")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        
        # Box plot
        axes[1].boxplot([values], labels=[metric], vert=False)
        axes[1].set_xlabel(metric)
        axes[1].set_title(f"{metric} Box Plot")
        axes[1].grid(axis="x", alpha=0.3)
        
        fig.tight_layout()
        
        plot_path = output_dir / f"{metric}_distribution.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        generated_plots.append(plot_path)
    
    return generated_plots


def plot_agreement_by_methods(
    agreement_data: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Create plots showing agreement metrics grouped by method pairs.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    if "method_a" not in agreement_data.columns or "method_b" not in agreement_data.columns:
        return generated_plots
    
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    # Create method pair labels
    agreement_data = agreement_data.copy()
    agreement_data["method_pair"] = (
        agreement_data["method_a"].astype(str) + " vs " + agreement_data["method_b"].astype(str)
    )
    
    for metric in available_metrics:
        # Clean data
        plot_data = agreement_data[["method_pair", metric]].copy()
        plot_data[metric] = pd.to_numeric(plot_data[metric], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if plot_data.empty or len(plot_data["method_pair"].unique()) < 2:
            continue
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by median for better visualization
        method_pairs = plot_data.groupby("method_pair")[metric].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=plot_data,
            x="method_pair",
            y=metric,
            order=method_pairs,
            ax=ax,
            palette="viridis",
        )
        
        ax.set_title(f"{metric} by Method Pair")
        ax.set_xlabel("Method Pair")
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        
        fig.tight_layout()
        
        plot_path = output_dir / f"{metric}_by_methods.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated_plots.append(plot_path)
    
    return generated_plots


def plot_agreement_correlation_matrix(
    agreement_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a correlation matrix of agreement metrics.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        output_path: Path to save the plot
    """
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio", "stability_jaccard"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    if len(available_metrics) < 2:
        return
    
    # Extract and clean data
    metric_data = agreement_data[available_metrics].copy()
    for col in available_metrics:
        metric_data[col] = pd.to_numeric(metric_data[col], errors="coerce")
    metric_data = metric_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if metric_data.empty:
        return
    
    # Compute correlation matrix
    corr_matrix = metric_data.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    
    ax.set_title("Correlation Matrix of Agreement Metrics")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_agreement_by_correctness(
    agreement_data: pd.DataFrame,
    output_dir: Path,
    *,
    correctness_col: str = "is_correct",
) -> List[Path]:
    """Create plots showing agreement metrics by prediction correctness.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        output_dir: Directory to save plots
        correctness_col: Column name for correctness indicator
        
    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    if correctness_col not in agreement_data.columns:
        return generated_plots
    
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    # Prepare data
    plot_data = agreement_data.copy()
    plot_data["Correctness"] = plot_data[correctness_col].apply(
        lambda x: "Correct" if x else "Incorrect" if pd.notna(x) else "Unknown"
    )
    
    for metric in available_metrics:
        # Clean data
        metric_data = plot_data[["Correctness", metric]].copy()
        metric_data[metric] = pd.to_numeric(metric_data[metric], errors="coerce")
        metric_data = metric_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if metric_data.empty or len(metric_data["Correctness"].unique()) < 2:
            continue
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.violinplot(
            data=metric_data,
            x="Correctness",
            y=metric,
            ax=ax,
            palette="Set2",
        )
        
        ax.set_title(f"{metric} by Prediction Correctness")
        ax.set_xlabel("Prediction Correctness")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        
        fig.tight_layout()
        
        plot_path = output_dir / f"{metric}_by_correctness.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        generated_plots.append(plot_path)
    
    return generated_plots


def run_ranking_agreement_analysis(
    agreement_data: pd.DataFrame,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run comprehensive ranking agreement analysis.
    
    Args:
        agreement_data: DataFrame containing agreement metrics
        output_dir: Directory to save results
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary containing all analysis results
    """
    if agreement_data is None or agreement_data.empty:
        return {
            "available": False,
            "reason": "No agreement data provided",
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running ranking agreement analysis...")
    
    results = {
        "output_dir": str(output_dir),
        "total_comparisons": len(agreement_data),
        "agreement_metrics": {},
        "by_methods": {},
        "stratified": {},
        "plots": {},
    }
    
    # Overall agreement statistics
    agreement_metrics = ["rbo", "spearman", "kendall", "feature_overlap_ratio", "stability_jaccard", "kl_divergence"]
    available_metrics = [m for m in agreement_metrics if m in agreement_data.columns]
    
    for metric in available_metrics:
        stats_obj = compute_agreement_stats(agreement_data[metric], metric)
        if stats_obj:
            results["agreement_metrics"][metric] = stats_obj.to_dict()
    
    # By method pairs
    print("  Analyzing agreement by method pairs...")
    by_methods = analyze_agreement_by_methods(agreement_data)
    
    for method_pair, stats_dict in by_methods.items():
        key = f"{method_pair[0]}_vs_{method_pair[1]}"
        results["by_methods"][key] = {
            metric: stats_obj.to_dict()
            for metric, stats_obj in stats_dict.items()
        }
    
    # Stratified analysis
    print("  Analyzing agreement with stratification...")
    stratified = analyze_agreement_stratified(
        agreement_data,
        class_col=class_col,
        correctness_col=correctness_col,
    )
    results["stratified"] = stratified
    
    # Create plots
    if create_plots:
        print("  Creating agreement plots...")
        
        try:
            plots = plot_agreement_distributions(agreement_data, output_dir / "plots")
            results["plots"]["distributions"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create distribution plots: {e}")
        
        try:
            plots = plot_agreement_by_methods(agreement_data, output_dir / "plots")
            results["plots"]["by_methods"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create method plots: {e}")
        
        try:
            corr_plot = output_dir / "plots" / "agreement_correlation_matrix.png"
            plot_agreement_correlation_matrix(agreement_data, corr_plot)
            results["plots"]["correlation_matrix"] = str(corr_plot)
        except Exception as e:
            print(f"Warning: Could not create correlation matrix: {e}")
        
        try:
            plots = plot_agreement_by_correctness(
                agreement_data,
                output_dir / "plots",
                correctness_col=correctness_col,
            )
            results["plots"]["by_correctness"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create correctness plots: {e}")
    
    # Save summary
    summary_path = output_dir / "ranking_agreement_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Save detailed agreement table
    table_path = output_dir / "agreement_metrics_table.csv"
    agreement_data.to_csv(table_path, index=False)
    results["agreement_table"] = str(table_path)
    
    print(f"\nRanking agreement analysis complete. Results saved to {output_dir}")
    
    return results

