"""Enhanced deletion and insertion AUC analysis with curve visualization.

This module extends the basic AUC analysis with:
- Visualization of insertion/deletion curves
- Curve shape analysis
- Comparison of curves across methods
- Stratified curve analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_curves_from_data(data: pd.DataFrame) -> Dict[str, List[List[Tuple[int, float]]]]:
    """Extract insertion and deletion curves from the data.
    
    Args:
        data: DataFrame containing the data
        
    Returns:
        Dictionary mapping curve type to list of curves
    """
    curves = {
        "insertion": [],
        "deletion": [],
    }
    
    for _, row in data.iterrows():
        if "insertion_curve" in row and row["insertion_curve"] is not None:
            if isinstance(row["insertion_curve"], list):
                curves["insertion"].append(row["insertion_curve"])
        
        if "deletion_curve" in row and row["deletion_curve"] is not None:
            if isinstance(row["deletion_curve"], list):
                curves["deletion"].append(row["deletion_curve"])
    
    return curves


def plot_average_curves(
    data: pd.DataFrame,
    output_path: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
) -> None:
    """Plot average insertion and deletion curves by method.
    
    Args:
        data: DataFrame containing the data
        output_path: Path to save the plot
        method_col: Column name identifying the method
        model_type_col: Column name identifying model type
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by method
    for method, group in data.groupby(method_col, dropna=False):
        method_str = str(method)
        
        # Get model type for color
        model_type = group[model_type_col].iloc[0] if model_type_col in group.columns else "unknown"
        color = "#E74C3C" if model_type == "llm" else "#3498DB" if model_type == "gnn" else "#95A5A6"
        
        # Insertion curves
        insertion_curves = []
        for _, row in group.iterrows():
            if "insertion_curve" in row and row["insertion_curve"] is not None:
                if isinstance(row["insertion_curve"], list) and len(row["insertion_curve"]) > 0:
                    insertion_curves.append(row["insertion_curve"])
        
        if insertion_curves:
            # Average the curves
            max_len = max(len(curve) for curve in insertion_curves)
            
            # Create interpolated curves
            avg_x = []
            avg_y = []
            
            for i in range(max_len):
                x_vals = []
                y_vals = []
                
                for curve in insertion_curves:
                    if i < len(curve):
                        x_vals.append(curve[i][0])
                        y_vals.append(curve[i][1])
                
                if x_vals:
                    avg_x.append(np.mean(x_vals))
                    avg_y.append(np.mean(y_vals))
            
            axes[0].plot(avg_x, avg_y, label=method_str, linewidth=2, color=color, alpha=0.8)
        
        # Deletion curves
        deletion_curves = []
        for _, row in group.iterrows():
            if "deletion_curve" in row and row["deletion_curve"] is not None:
                if isinstance(row["deletion_curve"], list) and len(row["deletion_curve"]) > 0:
                    deletion_curves.append(row["deletion_curve"])
        
        if deletion_curves:
            # Average the curves
            max_len = max(len(curve) for curve in deletion_curves)
            
            avg_x = []
            avg_y = []
            
            for i in range(max_len):
                x_vals = []
                y_vals = []
                
                for curve in deletion_curves:
                    if i < len(curve):
                        x_vals.append(curve[i][0])
                        y_vals.append(curve[i][1])
                
                if x_vals:
                    avg_x.append(np.mean(x_vals))
                    avg_y.append(np.mean(y_vals))
            
            axes[1].plot(avg_x, avg_y, label=method_str, linewidth=2, color=color, alpha=0.8)
    
    # Configure insertion plot
    axes[0].set_xlabel("Number of Features Added")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("Average Insertion Curves by Method")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(alpha=0.3)
    
    # Configure deletion plot
    axes[1].set_xlabel("Number of Features Removed")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Average Deletion Curves by Method")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(alpha=0.3)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_auc_comparison(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
) -> List[Path]:
    """Create comparative AUC plots.
    
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
    
    # Insertion AUC comparison
    if "insertion_auc" in data.columns and method_col in data.columns:
        plot_data = data[[method_col, model_type_col, "insertion_auc"]].copy()
        plot_data["insertion_auc"] = pd.to_numeric(plot_data["insertion_auc"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.violinplot(
                data=plot_data,
                x=method_col,
                y="insertion_auc",
                hue=model_type_col,
                ax=ax,
                palette={"llm": "#E74C3C", "gnn": "#3498DB"},
            )
            
            ax.set_title("Insertion AUC by Method and Model Type")
            ax.set_xlabel("Method")
            ax.set_ylabel("Insertion AUC")
            plt.xticks(rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "insertion_auc_comparison.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(plot_path)
    
    # Deletion AUC comparison
    if "deletion_auc" in data.columns and method_col in data.columns:
        plot_data = data[[method_col, model_type_col, "deletion_auc"]].copy()
        plot_data["deletion_auc"] = pd.to_numeric(plot_data["deletion_auc"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.violinplot(
                data=plot_data,
                x=method_col,
                y="deletion_auc",
                hue=model_type_col,
                ax=ax,
                palette={"llm": "#E74C3C", "gnn": "#3498DB"},
            )
            
            ax.set_title("Deletion AUC by Method and Model Type")
            ax.set_xlabel("Method")
            ax.set_ylabel("Deletion AUC")
            plt.xticks(rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "deletion_auc_comparison.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_plots.append(plot_path)
    
    # Scatter plot: Insertion vs Deletion AUC
    if "insertion_auc" in data.columns and "deletion_auc" in data.columns:
        plot_data = data[["insertion_auc", "deletion_auc", method_col, model_type_col]].copy()
        plot_data["insertion_auc"] = pd.to_numeric(plot_data["insertion_auc"], errors="coerce")
        plot_data["deletion_auc"] = pd.to_numeric(plot_data["deletion_auc"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            for model_type in plot_data[model_type_col].unique():
                subset = plot_data[plot_data[model_type_col] == model_type]
                color = "#E74C3C" if model_type == "llm" else "#3498DB"
                
                ax.scatter(
                    subset["insertion_auc"],
                    subset["deletion_auc"],
                    label=model_type,
                    alpha=0.5,
                    s=30,
                    color=color,
                )
            
            # Add diagonal line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label="y=x")
            
            ax.set_xlabel("Insertion AUC")
            ax.set_ylabel("Deletion AUC")
            ax.set_title("Insertion AUC vs Deletion AUC")
            ax.legend()
            ax.grid(alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / "insertion_vs_deletion_auc.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            generated_plots.append(plot_path)
    
    return generated_plots


def analyze_auc_by_correctness(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    correctness_col: str = "is_correct",
) -> Dict[str, any]:
    """Analyze AUC metrics stratified by correctness.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        correctness_col: Column name for correctness indicator
        
    Returns:
        Dictionary containing analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "insertion_auc": {},
        "deletion_auc": {},
    }
    
    if correctness_col not in data.columns:
        return results
    
    # Insertion AUC by correctness
    if "insertion_auc" in data.columns:
        for correct_value, group in data.groupby(correctness_col, dropna=False):
            values = pd.to_numeric(group["insertion_auc"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            
            if values.empty:
                continue
            
            if pd.isna(correct_value):
                label = "unknown"
            else:
                label = "correct" if correct_value else "incorrect"
            
            results["insertion_auc"][label] = {
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
    
    # Deletion AUC by correctness
    if "deletion_auc" in data.columns:
        for correct_value, group in data.groupby(correctness_col, dropna=False):
            values = pd.to_numeric(group["deletion_auc"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            
            if values.empty:
                continue
            
            if pd.isna(correct_value):
                label = "unknown"
            else:
                label = "correct" if correct_value else "incorrect"
            
            results["deletion_auc"][label] = {
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
    
    return results


def run_enhanced_auc_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    method_col: str = "method",
    model_type_col: str = "model_type",
    class_col: str = "label",
    correctness_col: str = "is_correct",
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run comprehensive AUC analysis with curve visualization.
    
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
    
    print("Running enhanced AUC analysis...")
    
    results = {
        "output_dir": str(output_dir),
        "by_correctness": {},
        "plots": {},
    }
    
    # Analyze by correctness
    print("  Analyzing AUC by correctness...")
    by_correctness = analyze_auc_by_correctness(
        data,
        output_dir,
        correctness_col=correctness_col,
    )
    results["by_correctness"] = by_correctness
    
    # Create plots
    if create_plots:
        print("  Creating AUC plots...")
        
        try:
            curve_plot = output_dir / "plots" / "average_curves.png"
            plot_average_curves(
                data,
                curve_plot,
                method_col=method_col,
                model_type_col=model_type_col,
            )
            results["plots"]["average_curves"] = str(curve_plot)
        except Exception as e:
            print(f"Warning: Could not create average curves plot: {e}")
        
        try:
            comparison_plots = plot_auc_comparison(
                data,
                output_dir / "plots",
                method_col=method_col,
                model_type_col=model_type_col,
            )
            results["plots"]["comparisons"] = [str(p) for p in comparison_plots]
        except Exception as e:
            print(f"Warning: Could not create AUC comparison plots: {e}")
    
    # Save summary
    summary_path = output_dir / "enhanced_auc_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEnhanced AUC analysis complete. Results saved to {output_dir}")
    
    return results

