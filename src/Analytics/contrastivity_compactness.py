"""Contrastivity and compactness analysis for explanations.

This module analyzes:
- Contrastivity: How well explanations distinguish between classes
- Compactness: How sparse/concise explanations are
- Relationships between contrastivity, compactness, and faithfulness
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


def analyze_contrastivity(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> Dict[str, any]:
    """Analyze contrastivity metrics.
    
    Contrastivity measures how well explanations distinguish the predicted
    class from other classes.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        Dictionary containing analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    contrastivity_cols = [
        "origin_contrastivity",
        "masked_contrastivity",
        "maskout_contrastivity",
    ]
    
    available_cols = [c for c in contrastivity_cols if c in data.columns]
    
    if not available_cols:
        return {
            "available": False,
            "reason": "No contrastivity columns found",
        }
    
    results = {
        "output_dir": str(output_dir),
        "metrics": {},
        "by_class": {},
        "by_correctness": {},
        "correlations": {},
    }
    
    # Overall statistics
    for col in available_cols:
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        results["metrics"][col] = {
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
    
    # Correlations with other metrics
    other_metrics = ["faithfulness", "insertion_auc", "deletion_auc", "fidelity_plus", "fidelity_minus"]
    
    for contrast_col in available_cols:
        results["correlations"][contrast_col] = {}
        
        for other_col in other_metrics:
            if other_col not in data.columns:
                continue
            
            # Compute correlation
            df_clean = data[[contrast_col, other_col]].copy()
            df_clean[contrast_col] = pd.to_numeric(df_clean[contrast_col], errors="coerce")
            df_clean[other_col] = pd.to_numeric(df_clean[other_col], errors="coerce")
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_clean) < 3:
                continue
            
            try:
                pearson_r, pearson_p = stats.pearsonr(df_clean[contrast_col], df_clean[other_col])
                spearman_r, spearman_p = stats.spearmanr(df_clean[contrast_col], df_clean[other_col])
                
                results["correlations"][contrast_col][other_col] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            except Exception as e:
                results["correlations"][contrast_col][other_col] = {"error": str(e)}
    
    return results


def analyze_compactness(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
) -> Dict[str, any]:
    """Analyze compactness/sparsity metrics.
    
    Compactness measures how concise explanations are (i.e., how few
    features are needed to explain the prediction).
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        
    Returns:
        Dictionary containing analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compactness_cols = [
        "sparsity",
        "minimal_coalition_size",
        "compactness",
    ]
    
    available_cols = [c for c in compactness_cols if c in data.columns]
    
    if not available_cols:
        return {
            "available": False,
            "reason": "No compactness columns found",
        }
    
    results = {
        "output_dir": str(output_dir),
        "metrics": {},
        "by_class": {},
        "by_correctness": {},
        "correlations": {},
    }
    
    # Overall statistics
    for col in available_cols:
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        results["metrics"][col] = {
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
    
    # Correlations with other metrics
    other_metrics = ["faithfulness", "insertion_auc", "deletion_auc", "origin_contrastivity"]
    
    for compact_col in available_cols:
        results["correlations"][compact_col] = {}
        
        for other_col in other_metrics:
            if other_col not in data.columns:
                continue
            
            # Compute correlation
            df_clean = data[[compact_col, other_col]].copy()
            df_clean[compact_col] = pd.to_numeric(df_clean[compact_col], errors="coerce")
            df_clean[other_col] = pd.to_numeric(df_clean[other_col], errors="coerce")
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_clean) < 3:
                continue
            
            try:
                pearson_r, pearson_p = stats.pearsonr(df_clean[compact_col], df_clean[other_col])
                spearman_r, spearman_p = stats.spearmanr(df_clean[compact_col], df_clean[other_col])
                
                results["correlations"][compact_col][other_col] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            except Exception as e:
                results["correlations"][compact_col][other_col] = {"error": str(e)}
    
    return results


def plot_contrastivity_distributions(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    correctness_col: str = "is_correct",
) -> List[Path]:
    """Create plots for contrastivity distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    contrastivity_cols = [
        "origin_contrastivity",
        "masked_contrastivity",
        "maskout_contrastivity",
    ]
    
    available_cols = [c for c in contrastivity_cols if c in data.columns]
    
    if not available_cols:
        return generated_plots
    
    # Combined distribution plot
    fig, axes = plt.subplots(1, len(available_cols), figsize=(6 * len(available_cols), 5))
    
    if len(available_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(available_cols):
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        axes[idx].hist(values, bins=30, color="#E67E22", edgecolor="black", alpha=0.7)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{col} Distribution")
        axes[idx].axvline(values.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {values.mean():.3f}")
        axes[idx].legend()
        axes[idx].grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    
    plot_path = output_dir / "contrastivity_distributions.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    generated_plots.append(plot_path)
    
    # By correctness
    if correctness_col in data.columns:
        for col in available_cols:
            plot_data = data[[col, correctness_col]].copy()
            plot_data[col] = pd.to_numeric(plot_data[col], errors="coerce")
            plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            if plot_data.empty:
                continue
            
            plot_data["Correctness"] = plot_data[correctness_col].apply(
                lambda x: "Correct" if x else "Incorrect" if pd.notna(x) else "Unknown"
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.violinplot(
                data=plot_data,
                x="Correctness",
                y=col,
                ax=ax,
                palette="Set2",
            )
            
            ax.set_title(f"{col} by Prediction Correctness")
            ax.set_xlabel("Prediction Correctness")
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / f"{col}_by_correctness.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            generated_plots.append(plot_path)
    
    return generated_plots


def plot_compactness_distributions(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    correctness_col: str = "is_correct",
) -> List[Path]:
    """Create plots for compactness distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    compactness_cols = [
        "sparsity",
        "minimal_coalition_size",
    ]
    
    available_cols = [c for c in compactness_cols if c in data.columns]
    
    if not available_cols:
        return generated_plots
    
    # Combined distribution plot
    fig, axes = plt.subplots(1, len(available_cols), figsize=(6 * len(available_cols), 5))
    
    if len(available_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(available_cols):
        values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        if values.empty:
            continue
        
        axes[idx].hist(values, bins=30, color="#3498DB", edgecolor="black", alpha=0.7)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{col} Distribution")
        axes[idx].axvline(values.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {values.mean():.3f}")
        axes[idx].legend()
        axes[idx].grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    
    plot_path = output_dir / "compactness_distributions.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    generated_plots.append(plot_path)
    
    # By correctness
    if correctness_col in data.columns:
        for col in available_cols:
            plot_data = data[[col, correctness_col]].copy()
            plot_data[col] = pd.to_numeric(plot_data[col], errors="coerce")
            plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            if plot_data.empty:
                continue
            
            plot_data["Correctness"] = plot_data[correctness_col].apply(
                lambda x: "Correct" if x else "Incorrect" if pd.notna(x) else "Unknown"
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.boxplot(
                data=plot_data,
                x="Correctness",
                y=col,
                ax=ax,
                palette="Set1",
            )
            
            ax.set_title(f"{col} by Prediction Correctness")
            ax.set_xlabel("Prediction Correctness")
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.3)
            
            fig.tight_layout()
            
            plot_path = output_dir / f"{col}_by_correctness.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            generated_plots.append(plot_path)
    
    return generated_plots


def plot_contrastivity_vs_compactness(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create scatter plot of contrastivity vs compactness."""
    # Use origin contrastivity and sparsity
    if "origin_contrastivity" not in data.columns or "sparsity" not in data.columns:
        return
    
    plot_data = data[["origin_contrastivity", "sparsity"]].copy()
    plot_data["origin_contrastivity"] = pd.to_numeric(plot_data["origin_contrastivity"], errors="coerce")
    plot_data["sparsity"] = pd.to_numeric(plot_data["sparsity"], errors="coerce")
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if plot_data.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(
        plot_data["sparsity"],
        plot_data["origin_contrastivity"],
        alpha=0.5,
        s=20,
        color="#9B59B6",
    )
    
    # Add trend line
    try:
        z = np.polyfit(plot_data["sparsity"], plot_data["origin_contrastivity"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data["sparsity"].min(), plot_data["sparsity"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label="Trend")
    except:
        pass
    
    ax.set_xlabel("Sparsity (Compactness)")
    ax.set_ylabel("Origin Contrastivity")
    ax.set_title("Contrastivity vs Compactness")
    ax.grid(alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_contrastivity_compactness_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    class_col: str = "label",
    correctness_col: str = "is_correct",
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run comprehensive contrastivity and compactness analysis.
    
    Args:
        data: DataFrame containing the data
        output_dir: Directory to save results
        class_col: Column name for class labels
        correctness_col: Column name for correctness indicator
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary containing all analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running contrastivity and compactness analysis...")
    
    results = {
        "output_dir": str(output_dir),
        "contrastivity": {},
        "compactness": {},
        "plots": {},
    }
    
    # Analyze contrastivity
    print("  Analyzing contrastivity...")
    contrastivity_results = analyze_contrastivity(
        data,
        output_dir / "contrastivity",
        class_col=class_col,
        correctness_col=correctness_col,
    )
    results["contrastivity"] = contrastivity_results
    
    # Analyze compactness
    print("  Analyzing compactness...")
    compactness_results = analyze_compactness(
        data,
        output_dir / "compactness",
        class_col=class_col,
        correctness_col=correctness_col,
    )
    results["compactness"] = compactness_results
    
    # Create plots
    if create_plots:
        print("  Creating plots...")
        
        try:
            plots = plot_contrastivity_distributions(
                data,
                output_dir / "plots",
                correctness_col=correctness_col,
            )
            results["plots"]["contrastivity"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create contrastivity plots: {e}")
        
        try:
            plots = plot_compactness_distributions(
                data,
                output_dir / "plots",
                correctness_col=correctness_col,
            )
            results["plots"]["compactness"] = [str(p) for p in plots]
        except Exception as e:
            print(f"Warning: Could not create compactness plots: {e}")
        
        try:
            plot_path = output_dir / "plots" / "contrastivity_vs_compactness.png"
            plot_contrastivity_vs_compactness(data, plot_path)
            results["plots"]["contrastivity_vs_compactness"] = str(plot_path)
        except Exception as e:
            print(f"Warning: Could not create contrastivity vs compactness plot: {e}")
    
    # Save summary
    summary_path = output_dir / "contrastivity_compactness_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nContrastivity and compactness analysis complete. Results saved to {output_dir}")
    
    return results

