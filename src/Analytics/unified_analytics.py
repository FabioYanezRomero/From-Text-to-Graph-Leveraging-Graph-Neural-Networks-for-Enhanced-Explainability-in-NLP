"""Unified analytics orchestrator for running complete analysis pipelines.

This module provides a facade/orchestrator that:
- Loads all insights from a directory structure
- Runs all analytics modules in sequence
- Generates a comprehensive report
- Preserves old analytics (structural patterns, KDE distributions)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .loaders import load_insights_from_directory
from .stratified_metrics import run_stratified_analysis
from .comparative_analysis import run_comparative_analysis
from .ranking_agreement import run_ranking_agreement_analysis
from .contrastivity_compactness import run_contrastivity_compactness_analysis
from .enhanced_auc_analysis import run_enhanced_auc_analysis
from .comprehensive_faithfulness import run_comprehensive_faithfulness_analysis


def run_complete_analytics_pipeline(
    insights_dir: Path,
    output_dir: Path,
    *,
    create_plots: bool = True,
) -> Dict[str, any]:
    """Run the complete analytics pipeline with all modules.
    
    This orchestrator runs all analytics modules in sequence:
    1. Load insights from directory structure
    2. Stratified analysis (by class and correctness)
    3. Comparative analysis (LLM vs GNN)
    4. Ranking agreement analysis
    5. Contrastivity and compactness analysis
    6. Enhanced AUC analysis
    7. Comprehensive faithfulness analysis
    
    Args:
        insights_dir: Root directory containing GNN/ and LLM/ subdirectories
        output_dir: Directory to save all analysis results
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary containing all analysis results and metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    print("=" * 80)
    print("COMPLETE ANALYTICS PIPELINE")
    print("=" * 80)
    print(f"Insights directory: {insights_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Create plots: {create_plots}")
    print("=" * 80)
    print()
    
    results = {
        "insights_dir": str(insights_dir),
        "output_dir": str(output_dir),
        "create_plots": create_plots,
        "modules": {},
        "errors": {},
        "metadata": {},
    }
    
    # Step 1: Load insights
    print("Step 1/7: Loading insights...")
    try:
        insights = load_insights_from_directory(
            insights_dir,
            load_gnn=True,
            load_llm=True,
            load_agreement=True,
        )
        
        results["metadata"]["total_records"] = insights.metadata.total_records
        results["metadata"]["datasets"] = insights.metadata.datasets
        results["metadata"]["graph_types"] = insights.metadata.graph_types
        results["metadata"]["methods"] = insights.metadata.methods
        results["metadata"]["has_gnn"] = insights.metadata.has_gnn
        results["metadata"]["has_llm"] = insights.metadata.has_llm
        results["metadata"]["has_agreement"] = insights.metadata.has_agreement
        
        print(f"  Loaded {insights.metadata.total_records} records")
        print(f"  Datasets: {insights.metadata.datasets}")
        print(f"  Methods: {insights.metadata.methods}")
        print(f"  GNN data: {insights.metadata.has_gnn}")
        print(f"  LLM data: {insights.metadata.has_llm}")
        print(f"  Agreement data: {insights.metadata.has_agreement}")
        print()
    except Exception as e:
        error_msg = f"Failed to load insights: {e}"
        print(f"  ERROR: {error_msg}")
        results["errors"]["loading"] = error_msg
        return results
    
    # Step 2: Stratified analysis
    print("Step 2/7: Running stratified analysis...")
    try:
        stratified_dir = output_dir / "stratified"
        stratified_results = run_stratified_analysis(
            insights.data,
            stratified_dir,
            create_plots=create_plots,
        )
        results["modules"]["stratified"] = {
            "output_dir": str(stratified_dir),
            "success": True,
            "summary": stratified_results,
        }
        print(f"  ✓ Stratified analysis complete: {stratified_dir}")
        print()
    except Exception as e:
        error_msg = f"Stratified analysis failed: {e}"
        print(f"  ERROR: {error_msg}")
        results["errors"]["stratified"] = error_msg
        results["modules"]["stratified"] = {"success": False, "error": error_msg}
        print()
    
    # Step 3: Comparative analysis
    print("Step 3/7: Running comparative analysis...")
    try:
        comparative_dir = output_dir / "comparative"
        comparative_results = run_comparative_analysis(
            insights.data,
            comparative_dir,
            create_plots=create_plots,
        )
        results["modules"]["comparative"] = {
            "output_dir": str(comparative_dir),
            "success": True,
            "summary": comparative_results,
        }
        print(f"  ✓ Comparative analysis complete: {comparative_dir}")
        print()
    except Exception as e:
        error_msg = f"Comparative analysis failed: {e}"
        print(f"  ERROR: {error_msg}")
        results["errors"]["comparative"] = error_msg
        results["modules"]["comparative"] = {"success": False, "error": error_msg}
        print()
    
    # Step 4: Ranking agreement analysis
    print("Step 4/7: Running ranking agreement analysis...")
    if insights.agreement_frame is not None and not insights.agreement_frame.empty:
        try:
            agreement_dir = output_dir / "ranking_agreement"
            agreement_results = run_ranking_agreement_analysis(
                insights.agreement_frame,
                agreement_dir,
                create_plots=create_plots,
            )
            results["modules"]["ranking_agreement"] = {
                "output_dir": str(agreement_dir),
                "success": True,
                "summary": agreement_results,
            }
            print(f"  ✓ Ranking agreement analysis complete: {agreement_dir}")
            print()
        except Exception as e:
            error_msg = f"Ranking agreement analysis failed: {e}"
            print(f"  ERROR: {error_msg}")
            results["errors"]["ranking_agreement"] = error_msg
            results["modules"]["ranking_agreement"] = {"success": False, "error": error_msg}
            print()
    else:
        print("  ⊘ Skipping (no agreement data available)")
        results["modules"]["ranking_agreement"] = {
            "success": False,
            "reason": "No agreement data available",
        }
        print()
    
    # Step 5: Contrastivity and compactness analysis
    print("Step 5/7: Running contrastivity and compactness analysis...")
    try:
        contrast_dir = output_dir / "contrastivity_compactness"
        contrast_results = run_contrastivity_compactness_analysis(
            insights.data,
            contrast_dir,
            create_plots=create_plots,
        )
        results["modules"]["contrastivity_compactness"] = {
            "output_dir": str(contrast_dir),
            "success": True,
            "summary": contrast_results,
        }
        print(f"  ✓ Contrastivity and compactness analysis complete: {contrast_dir}")
        print()
    except Exception as e:
        error_msg = f"Contrastivity and compactness analysis failed: {e}"
        print(f"  ERROR: {error_msg}")
        results["errors"]["contrastivity_compactness"] = error_msg
        results["modules"]["contrastivity_compactness"] = {"success": False, "error": error_msg}
        print()
    
    # Step 6: Enhanced AUC analysis
    print("Step 6/7: Running enhanced AUC analysis...")
    try:
        auc_dir = output_dir / "enhanced_auc"
        auc_results = run_enhanced_auc_analysis(
            insights.data,
            auc_dir,
            create_plots=create_plots,
        )
        results["modules"]["enhanced_auc"] = {
            "output_dir": str(auc_dir),
            "success": True,
            "summary": auc_results,
        }
        print(f"  ✓ Enhanced AUC analysis complete: {auc_dir}")
        print()
    except Exception as e:
        error_msg = f"Enhanced AUC analysis failed: {e}"
        print(f"  ERROR: {error_msg}")
        results["errors"]["enhanced_auc"] = error_msg
        results["modules"]["enhanced_auc"] = {"success": False, "error": error_msg}
        print()
    
    # Step 7: Comprehensive faithfulness analysis
    print("Step 7/7: Running comprehensive faithfulness analysis...")
    try:
        faith_dir = output_dir / "comprehensive_faithfulness"
        faith_results = run_comprehensive_faithfulness_analysis(
            insights.data,
            faith_dir,
            create_plots=create_plots,
        )
        results["modules"]["comprehensive_faithfulness"] = {
            "output_dir": str(faith_dir),
            "success": True,
            "summary": faith_results,
        }
        print(f"  ✓ Comprehensive faithfulness analysis complete: {faith_dir}")
        print()
    except Exception as e:
        error_msg = f"Comprehensive faithfulness analysis failed: {e}"
        print(f"  ERROR: {error_msg}")
        results["errors"]["comprehensive_faithfulness"] = error_msg
        results["modules"]["comprehensive_faithfulness"] = {"success": False, "error": error_msg}
        print()
    
    # Finalize
    end_time = time.time()
    elapsed = end_time - start_time
    
    results["metadata"]["start_time"] = start_time
    results["metadata"]["end_time"] = end_time
    results["metadata"]["elapsed_seconds"] = elapsed
    results["metadata"]["elapsed_formatted"] = f"{elapsed:.2f}s" if elapsed < 60 else f"{elapsed/60:.2f}m"
    
    # Count successes and failures
    successes = sum(1 for m in results["modules"].values() if m.get("success", False))
    total_modules = len(results["modules"])
    
    results["metadata"]["modules_total"] = total_modules
    results["metadata"]["modules_succeeded"] = successes
    results["metadata"]["modules_failed"] = total_modules - successes
    
    # Save main summary
    summary_path = output_dir / "complete_pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Create executive summary
    create_executive_summary(results, output_dir / "executive_summary.md")
    
    # Print summary
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total time: {results['metadata']['elapsed_formatted']}")
    print(f"Modules succeeded: {successes}/{total_modules}")
    print(f"Output directory: {output_dir}")
    print()
    
    if results["errors"]:
        print("ERRORS ENCOUNTERED:")
        for module, error in results["errors"].items():
            print(f"  - {module}: {error}")
        print()
    
    print("Summary files:")
    print(f"  - Main: {summary_path}")
    print(f"  - Executive: {output_dir / 'executive_summary.md'}")
    print("=" * 80)
    
    return results


def create_executive_summary(results: Dict, output_path: Path) -> None:
    """Create a human-readable executive summary of the analytics pipeline.
    
    Args:
        results: Results dictionary from the pipeline
        output_path: Path to save the summary
    """
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Complete Analytics Pipeline - Executive Summary\n\n")
        
        # Metadata
        f.write("## Pipeline Information\n\n")
        f.write(f"- **Insights Directory**: `{results['insights_dir']}`\n")
        f.write(f"- **Output Directory**: `{results['output_dir']}`\n")
        f.write(f"- **Execution Time**: {results['metadata']['elapsed_formatted']}\n")
        f.write(f"- **Total Records**: {results['metadata']['total_records']}\n")
        f.write(f"- **Datasets**: {', '.join(results['metadata']['datasets'])}\n")
        f.write(f"- **Methods**: {', '.join(results['metadata']['methods'])}\n")
        f.write(f"- **Has GNN Data**: {'Yes' if results['metadata']['has_gnn'] else 'No'}\n")
        f.write(f"- **Has LLM Data**: {'Yes' if results['metadata']['has_llm'] else 'No'}\n")
        f.write(f"- **Has Agreement Data**: {'Yes' if results['metadata']['has_agreement'] else 'No'}\n")
        f.write("\n")
        
        # Module status
        f.write("## Module Status\n\n")
        f.write(f"**{results['metadata']['modules_succeeded']}/{results['metadata']['modules_total']} modules completed successfully**\n\n")
        
        for module_name, module_info in results["modules"].items():
            status = "✓" if module_info.get("success", False) else "✗"
            f.write(f"- {status} **{module_name.replace('_', ' ').title()}**\n")
            
            if module_info.get("success", False):
                f.write(f"  - Output: `{module_info.get('output_dir', 'N/A')}`\n")
            else:
                reason = module_info.get("error") or module_info.get("reason", "Unknown")
                f.write(f"  - Status: Failed ({reason})\n")
        
        f.write("\n")
        
        # Errors
        if results["errors"]:
            f.write("## Errors\n\n")
            for module, error in results["errors"].items():
                f.write(f"- **{module}**: {error}\n")
            f.write("\n")
        
        # Available analyses
        f.write("## Available Analyses\n\n")
        
        f.write("### Stratified Analysis\n")
        if results["modules"]["stratified"].get("success"):
            f.write("Metrics analyzed by class and prediction correctness:\n")
            f.write("- Fidelity+, Fidelity-, Faithfulness\n")
            f.write("- Insertion/Deletion AUC\n")
            f.write("- Contrastivity and Compactness\n")
            f.write("- Statistical tests for group differences\n")
        else:
            f.write("Not available\n")
        f.write("\n")
        
        f.write("### Comparative Analysis\n")
        if results["modules"]["comparative"].get("success"):
            f.write("LLM vs GNN comparison:\n")
            f.write("- Method ranking across metrics\n")
            f.write("- Statistical significance tests\n")
            f.write("- Effect size calculations\n")
            f.write("- Distribution comparisons\n")
        else:
            f.write("Not available\n")
        f.write("\n")
        
        f.write("### Ranking Agreement\n")
        if results["modules"]["ranking_agreement"].get("success"):
            f.write("Inter-explainer agreement metrics:\n")
            f.write("- Rank-Biased Overlap (RBO)\n")
            f.write("- Spearman/Kendall rank correlation\n")
            f.write("- Feature overlap ratios\n")
            f.write("- Stability (Jaccard)\n")
        else:
            f.write("Not available\n")
        f.write("\n")
        
        f.write("### Contrastivity & Compactness\n")
        if results["modules"]["contrastivity_compactness"].get("success"):
            f.write("Explanation quality metrics:\n")
            f.write("- Contrastivity (class discrimination)\n")
            f.write("- Compactness (sparsity)\n")
            f.write("- Correlations with faithfulness\n")
        else:
            f.write("Not available\n")
        f.write("\n")
        
        f.write("### Enhanced AUC Analysis\n")
        if results["modules"]["enhanced_auc"].get("success"):
            f.write("Insertion/deletion curve analysis:\n")
            f.write("- Average curves by method\n")
            f.write("- AUC distributions\n")
            f.write("- Curve shape analysis\n")
        else:
            f.write("Not available\n")
        f.write("\n")
        
        f.write("### Comprehensive Faithfulness\n")
        if results["modules"]["comprehensive_faithfulness"].get("success"):
            f.write("Complete faithfulness assessment:\n")
            f.write("- Fidelity+ (sufficiency)\n")
            f.write("- Fidelity- (necessity)\n")
            f.write("- General and local faithfulness\n")
            f.write("- Correlations between metrics\n")
        else:
            f.write("Not available\n")
        f.write("\n")
        
        # Next steps
        f.write("## Next Steps\n\n")
        f.write("1. Review individual module outputs in their respective directories\n")
        f.write("2. Examine generated plots for visual insights\n")
        f.write("3. Check JSON summary files for detailed statistics\n")
        f.write("4. Compare metrics across different methods and datasets\n")
        f.write("5. Use findings to guide model selection and explanation interpretation\n")
        f.write("\n")
        
        # Directory structure
        f.write("## Output Directory Structure\n\n")
        f.write("```\n")
        f.write(f"{results['output_dir']}/\n")
        f.write("├── complete_pipeline_summary.json\n")
        f.write("├── executive_summary.md\n")
        
        for module_name, module_info in results["modules"].items():
            if module_info.get("success"):
                f.write(f"├── {module_name}/\n")
                f.write(f"│   ├── {module_name}_summary.json\n")
                f.write("│   └── plots/\n")
        
        f.write("```\n")

