"""Comprehensive faithfulness metrics analysis.

This module provides analysis for:
- Fidelity+ (sufficiency): confidence drop when using only important features
- Fidelity- (necessity): confidence drop when removing important features
- General faithfulness: overall explanation quality
- Local faithfulness: instance-level explanation quality
- Relationships between faithfulness metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def analyze_fidelity_metrics(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> Dict[str, any]:
    """Analyze fidelity+ and fidelity- metrics.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        Dictionary containing analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fidelity_cols = ["fidelity_plus", "fidelity_minus"]
    available_cols = [c for c in fidelity_cols if c in data.columns]
    
    if not available_cols:
        return {
            "available": False,
            "reason": "No fidelity columns found",
        }
    
    results = {
        "output_dir": str(output_dir),
        "overall": {},
        "by_class": {},
        "by_correctness": {},
        "relationships": {},
    }
    
    # Overall statistics
    for col in available_cols:
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        results["overall"][col] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    
    # By class
    if class_col in data.columns:
        for col in available_cols:
            results["by_class"][col] = {}
            
            for class_value, group in data.groupby(class_col, dropna=False):
                values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                
                if values.empty:
                    continue
                
                class_label = "None" if pd.isna(class_value) else str(class_value)
                results["by_class"][col][class_label] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                }
    
    # By correctness
    if correctness_col in data.columns:
        for col in available_cols:
            results["by_correctness"][col] = {}
            
            for correct_value, group in data.groupby(correctness_col, dropna=False):
                values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                
                if values.empty:
                    continue
                
                if pd.isna(correct_value):
                    correct_label = "unknown"
                else:
                    correct_label = "correct" if correct_value else "incorrect"
                
                results["by_correctness"][col][correct_label] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                }
    
    # Relationship between fidelity+ and fidelity-
    if "fidelity_plus" in data.columns and "fidelity_minus" in data.columns:
        plot_data = data[["fidelity_plus", "fidelity_minus"]].copy()
        plot_data["fidelity_plus"] = pd.to_numeric(plot_data["fidelity_plus"], errors="coerce")
        plot_data["fidelity_minus"] = pd.to_numeric(plot_data["fidelity_minus"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(plot_data) >= 3:
            try:
                pearson_r, pearson_p = stats.pearsonr(plot_data["fidelity_plus"], plot_data["fidelity_minus"])
                spearman_r, spearman_p = stats.spearmanr(plot_data["fidelity_plus"], plot_data["fidelity_minus"])
                
                results["relationships"]["fidelity_plus_vs_fidelity_minus"] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            except Exception as e:
                results["relationships"]["fidelity_plus_vs_fidelity_minus"] = {"error": str(e)}
    
    return results


def analyze_faithfulness_metrics(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> Dict[str, any]:
    """Analyze general faithfulness metrics.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        Dictionary containing analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    faithfulness_cols = ["faithfulness", "faithfulness_monotonicity"]
    available_cols = [c for c in faithfulness_cols if c in data.columns]
    
    if not available_cols:
        return {
            "available": False,
            "reason": "No faithfulness columns found",
        }
    
    results = {
        "output_dir": str(output_dir),
        "overall": {},
        "by_class": {},
        "by_correctness": {},
    }
    
    # Overall statistics
    for col in available_cols:
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        results["overall"][col] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    
    # By class
    if class_col in data.columns:
        for col in available_cols:
            results["by_class"][col] = {}
            
            for class_value, group in data.groupby(class_col, dropna=False):
                values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                
                if values.empty:
                    continue
                
                class_label = "None" if pd.isna(class_value) else str(class_value)
                results["by_class"][col][class_label] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                }
    
    # By correctness
    if correctness_col in data.columns:
        for col in available_cols:
            results["by_correctness"][col] = {}
            
            for correct_value, group in data.groupby(correctness_col, dropna=False):
                values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                
                if values.empty:
                    continue
                
                if pd.isna(correct_value):
                    correct_label = "unknown"
                else:
                    correct_label = "correct" if correct_value else "incorrect"
                
                results["by_correctness"][col][correct_label] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                }
    
    return results


def plot_fidelity_distributions(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
) -> List[Path]:
    """Create distribution plots for fidelity metrics.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save plots
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type
        
    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    # Fidelity+ and Fidelity- comparison
    if "fidelity_plus" in data.columns and "fidelity_minus" in data.columns:
        plot_data = data[["fidelity_plus", "fidelity_minus", method_col, model_type_col]].copy()
        plot_data["fidelity_plus"] = pd.to_numeric(plot_data["fidelity_plus"], errors="coerce")
        plot_data["fidelity_minus"] = pd.to_numeric(plot_data["fidelity_minus"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_data.empty:
            # Violin plot by method
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            sns.violinplot(
                data=plot_data,
                x=method_col,
                y="fidelity_plus",
                hue=model_type_col,
                ax=axes[0],
                palette={"llm": "#E74C3C", "gnn": "#3498DB"},
            )
            axes[0].set_title("Fidelity+ (Sufficiency) by Method")
            axes[0].set_xlabel("Method")
            axes[0].set_ylabel("Fidelity+")
            plt.sca(axes[0])
            plt.xticks(rotation=45, ha="right")
            axes[0].grid(axis="y", alpha=0.3)
            
            sns.violinplot(
                data=plot_data,
                x=method_col,
                y="fidelity_minus",
                hue=model_type_col,
                ax=axes[1],
                palette={"llm": "#E74C3C", "gnn": "#3498DB"},
            )
            axes[1].set_title("Fidelity- (Necessity) by Method")
            axes[1].set_xlabel("Method")
            axes[1].set_ylabel("Fidelity-")
            plt.sca(axes[1])
            plt.xticks(rotation=45, ha="right")
            axes[1].grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "fidelity_plus_minus_comparison.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(plot_path)
            
            # Scatter plot: Fidelity+ vs Fidelity-
            fig, ax = plt.subplots(figsize=(8, 8))
            
            for model_type in plot_data[model_type_col].unique():
                subset = plot_data[plot_data[model_type_col] == model_type]
                color = "#E74C3C" if model_type == "llm" else "#3498DB"
                
                ax.scatter(
                    subset["fidelity_plus"],
                    subset["fidelity_minus"],
                    label=model_type,
                    alpha=0.5,
                    s=30,
                    color=color,
                )
            
            ax.set_xlabel("Fidelity+ (Sufficiency)")
            ax.set_ylabel("Fidelity- (Necessity)")
            ax.set_title("Fidelity+ vs Fidelity-")
            ax.legend()
            ax.grid(alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "fidelity_plus_vs_minus_scatter.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            generated_plots.append(plot_path)
    
    return generated_plots


def plot_faithfulness_distributions(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
    correctness_col: str = "is_correct",
) -> List[Path]:
    """Create distribution plots for faithfulness metrics.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save plots
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type
        correctness_col: Column name for correctness indicator
        
    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    # Faithfulness by method
    if "faithfulness" in data.columns and method_col in data.columns:
        plot_data = data[["faithfulness", method_col, model_type_col]].copy()
        plot_data["faithfulness"] = pd.to_numeric(plot_data["faithfulness"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.violinplot(
                data=plot_data,
                x=method_col,
                y="faithfulness",
                hue=model_type_col,
                ax=ax,
                palette={"llm": "#E74C3C", "gnn": "#3498DB"},
            )
            
            ax.set_title("Faithfulness by Method and Model Type")
            ax.set_xlabel("Method")
            ax.set_ylabel("Faithfulness")
            plt.xticks(rotation=45, ha="right")
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "faithfulness_by_method.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(plot_path)
    
    # Faithfulness by correctness
    if "faithfulness" in data.columns and correctness_col in data.columns:
        plot_data = data[["faithfulness", correctness_col]].copy()
        plot_data["faithfulness"] = pd.to_numeric(plot_data["faithfulness"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_data.empty:
            plot_data["Correctness"] = plot_data[correctness_col].apply(
                lambda x: "Correct" if x else "Incorrect" if pd.notna(x) else "Unknown"
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.boxplot(
                data=plot_data,
                x="Correctness",
                y="faithfulness",
                ax=ax,
                palette="Set2",
            )
            
            ax.set_title("Faithfulness by Prediction Correctness")
            ax.set_xlabel("Prediction Correctness")
            ax.set_ylabel("Faithfulness")
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "faithfulness_by_correctness.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            generated_plots.append(plot_path)
    
    return generated_plots


def plot_faithfulness_relationships(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create plots showing relationships between faithfulness metrics.
    
    Args:
        data: DataFrame containing the data
        output_path: Path to save the plot
    """
    # Metrics to compare
    metrics = ["faithfulness", "fidelity_plus", "fidelity_minus", "insertion_auc", "deletion_auc"]
    available_metrics = [m for m in metrics if m in data.columns]
    
    if len(available_metrics) < 2:
        return
    
    # Create correlation matrix
    metric_data = data[available_metrics].copy()
    for col in available_metrics:
        metric_data[col] = pd.to_numeric(metric_data[col], errors="coerce")
    metric_data = metric_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if metric_data.empty:
        return
    
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
    
    ax.set_title("Correlation Matrix of Faithfulness Metrics")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_comprehensive_faithfulness_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
    class_col: str = "label",
    correctness_col: str = "is_correct",
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run comprehensive faithfulness analysis.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary containing all analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running comprehensive faithfulness analysis...")
    
    results = {
        "output_dir": str(output_dir),
        "fidelity": {},
        "faithfulness": {},
        "plots": {},
    }
    
    # Analyze fidelity metrics
    print("  Analyzing fidelity metrics...")
    fidelity_results = analyze_fidelity_metrics(
        data,
        output_dir / "fidelity",
        class_col=class_col,
        correctness_col=correctness_col,
    )
    results["fidelity"] = fidelity_results
    
    # Analyze faithfulness metrics
    print("  Analyzing faithfulness metrics...")
    faithfulness_results = analyze_faithfulness_metrics(
        data,
        output_dir / "faithfulness",
        class_col=class_col,
        correctness_col=correctness_col,
    )
    results["faithfulness"] = faithfulness_results
    
    # Create plots
    if create_plots:
        print("  Creating faithfulness plots...")
        
        try:
            plots = plot_fidelity_distributions(
                data,
                output_dir / "plots",
                method_col=method_col,
                model_type_col=model_type_col,
            )
            results["plots"]["fidelity"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create fidelity plots: {e}")
        
        try:
            plots = plot_faithfulness_distributions(
                data,
                output_dir / "plots",
                method_col=method_col,
                model_type_col=model_type_col,
                correctness_col=correctness_col,
            )
            results["plots"]["faithfulness"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create faithfulness plots: {e}")
        
        try:
            corr_plot = output_dir / "plots" / "faithfulness_correlation_matrix.png"
            plot_faithfulness_relationships(data, corr_plot)
            results["plots"]["correlation_matrix"] = str(corr_plot)
        except Exception as e:
            print(f"Warning: Could not create correlation matrix: {e}")
    
    # Save summary
    summary_path = output_dir / "comprehensive_faithfulness_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive faithfulness analysis complete. Results saved to {output_dir}")
    
    return results

