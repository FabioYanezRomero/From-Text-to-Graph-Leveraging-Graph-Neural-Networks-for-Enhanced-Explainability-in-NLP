#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run comprehensive analytics on LLM and GNN explainability results
# ==============================================================================
# This script runs all analytics modules:
# - Complete pipeline on full datasets (LLM + all GNN graph types)
# - Stratified analysis by dataset and model
# - Comparative analysis (LLM vs GNN, graph type comparisons)
# - Ranking agreement analysis
# - Individual metric analyses (faithfulness, AUC, contrastivity)
# 
# Agreement analysis covers:
# - LLM vs each GNN graph type (syntactic, constituency, skipgrams, window)
# - GNN graph types vs each other
# - Within-method comparisons
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSIGHTS_BASE="${INSIGHTS_BASE:-outputs/insights/news}"
ANALYTICS_BASE="${ANALYTICS_BASE:-outputs/analytics}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set to "--no-plots" to skip plot generation for faster execution
PLOT_FLAG=${PLOT_FLAG:-""}

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_section() {
    echo ""
    echo -e "${BLUE}====================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}====================================================================${NC}"
}

log_subsection() {
    echo ""
    echo -e "${GREEN}>>> $1${NC}"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

run_analytics() {
    local cmd=$1
    local description=$2
    
    log_info "Running: ${description}"
    log_info "Command: ${cmd}"
    
    if eval "${cmd}"; then
        log_success "${description} completed"
        return 0
    else
        log_error "${description} failed"
        return 1
    fi
}

# ==============================================================================
# Complete Pipeline: Run all analytics on full datasets
# ==============================================================================
run_complete_pipeline() {
    log_section "COMPLETE ANALYTICS PIPELINE"
    
    # SST-2 (stanfordnlp)
    log_subsection "SST-2 Dataset (stanfordnlp)"
    
    local output_dir="${ANALYTICS_BASE}/complete/stanfordnlp_sst2"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics complete_pipeline \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Complete pipeline for stanfordnlp/sst2"
    
    # AG News (SetFit)
    log_subsection "AG News Dataset (SetFit)"
    
    output_dir="${ANALYTICS_BASE}/complete/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics complete_pipeline \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Complete pipeline for SetFit/ag_news"
}

# ==============================================================================
# Stratified Analysis: By class and correctness for each dataset
# ==============================================================================
run_stratified_analysis() {
    log_section "STRATIFIED ANALYSIS (By Class and Correctness)"
    
    # SST-2
    log_subsection "SST-2 Stratified Analysis"
    
    local output_dir="${ANALYTICS_BASE}/stratified/stanfordnlp_sst2"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics stratified \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            --class-col label \
            --correctness-col is_correct \
            ${PLOT_FLAG}" \
        "Stratified analysis for stanfordnlp/sst2"
    
    # AG News
    log_subsection "AG News Stratified Analysis"
    
    output_dir="${ANALYTICS_BASE}/stratified/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics stratified \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            --class-col label \
            --correctness-col is_correct \
            ${PLOT_FLAG}" \
        "Stratified analysis for SetFit/ag_news"
}

# ==============================================================================
# Comparative Analysis: LLM vs GNN and cross-method comparisons
# ==============================================================================
run_comparative_analysis() {
    log_section "COMPARATIVE ANALYSIS (LLM vs GNN)"
    
    # Overall LLM vs GNN comparison
    log_subsection "Overall LLM vs GNN Comparison"
    
    local output_dir="${ANALYTICS_BASE}/comparative/llm_vs_gnn_overall"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics comparative \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            --method-col method \
            --model-type-col model_type \
            ${PLOT_FLAG}" \
        "Overall LLM vs GNN comparison"
    
    # SST-2 specific comparison
    log_subsection "SST-2 LLM vs GNN"
    
    output_dir="${ANALYTICS_BASE}/comparative/stanfordnlp_sst2"
    
    run_analytics \
        "python -m src.Analytics comparative \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "LLM vs GNN comparison for stanfordnlp/sst2"
    
    # AG News specific comparison
    log_subsection "AG News LLM vs GNN"
    
    output_dir="${ANALYTICS_BASE}/comparative/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics comparative \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "LLM vs GNN comparison for SetFit/ag_news"
    
    # Graph type comparisons
    log_subsection "GNN Graph Type Comparisons"
    
    output_dir="${ANALYTICS_BASE}/comparative/gnn_graph_types"
    
    run_analytics \
        "python -m src.Analytics comparative \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            --method-col graph_type \
            ${PLOT_FLAG}" \
        "GNN graph type comparison (skipgrams vs constituency vs syntactic vs window)"
}

# ==============================================================================
# Ranking Agreement Analysis: Inter-explainer agreement
# ==============================================================================
run_ranking_agreement_analysis() {
    log_section "RANKING AGREEMENT ANALYSIS"
    
    log_info "Agreement metrics cover:"
    log_info "  - LLM vs GNN graph types"
    log_info "  - GNN graph types vs each other"
    log_info "  - Within-method stability"
    
    # Overall agreement analysis
    log_subsection "Overall Agreement Analysis"
    
    local output_dir="${ANALYTICS_BASE}/ranking_agreement/overall"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics ranking_agreement \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            --class-col label \
            --correctness-col is_correct \
            ${PLOT_FLAG}" \
        "Overall ranking agreement analysis"
    
    # SST-2 agreement
    log_subsection "SST-2 Agreement Analysis"
    
    output_dir="${ANALYTICS_BASE}/ranking_agreement/stanfordnlp_sst2"
    
    run_analytics \
        "python -m src.Analytics ranking_agreement \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Ranking agreement for stanfordnlp/sst2"
    
    # AG News agreement
    log_subsection "AG News Agreement Analysis"
    
    output_dir="${ANALYTICS_BASE}/ranking_agreement/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics ranking_agreement \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Ranking agreement for SetFit/ag_news"
}

# ==============================================================================
# Contrastivity & Compactness Analysis
# ==============================================================================
run_contrastivity_compactness_analysis() {
    log_section "CONTRASTIVITY & COMPACTNESS ANALYSIS"
    
    # Overall analysis
    log_subsection "Overall Quality Metrics"
    
    local output_dir="${ANALYTICS_BASE}/contrastivity_compactness/overall"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics contrastivity_compactness \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Overall contrastivity and compactness analysis"
    
    # Per dataset analysis
    log_subsection "SST-2 Quality Metrics"
    
    output_dir="${ANALYTICS_BASE}/contrastivity_compactness/stanfordnlp_sst2"
    
    run_analytics \
        "python -m src.Analytics contrastivity_compactness \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Contrastivity and compactness for stanfordnlp/sst2"
    
    log_subsection "AG News Quality Metrics"
    
    output_dir="${ANALYTICS_BASE}/contrastivity_compactness/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics contrastivity_compactness \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Contrastivity and compactness for SetFit/ag_news"
}

# ==============================================================================
# Enhanced AUC Analysis: Insertion/deletion curves
# ==============================================================================
run_enhanced_auc_analysis() {
    log_section "ENHANCED AUC ANALYSIS (Insertion/Deletion Curves)"
    
    # Overall AUC analysis
    log_subsection "Overall AUC Analysis"
    
    local output_dir="${ANALYTICS_BASE}/enhanced_auc/overall"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics enhanced_auc \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Overall AUC analysis with curves"
    
    # Per dataset
    log_subsection "SST-2 AUC Analysis"
    
    output_dir="${ANALYTICS_BASE}/enhanced_auc/stanfordnlp_sst2"
    
    run_analytics \
        "python -m src.Analytics enhanced_auc \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "AUC analysis for stanfordnlp/sst2"
    
    log_subsection "AG News AUC Analysis"
    
    output_dir="${ANALYTICS_BASE}/enhanced_auc/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics enhanced_auc \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "AUC analysis for SetFit/ag_news"
}

# ==============================================================================
# Comprehensive Faithfulness Analysis: Fidelity+/-, faithfulness
# ==============================================================================
run_comprehensive_faithfulness_analysis() {
    log_section "COMPREHENSIVE FAITHFULNESS ANALYSIS"
    
    # Overall faithfulness
    log_subsection "Overall Faithfulness Analysis"
    
    local output_dir="${ANALYTICS_BASE}/comprehensive_faithfulness/overall"
    local insights_dir="${INSIGHTS_BASE}"
    
    run_analytics \
        "python -m src.Analytics comprehensive_faithfulness \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Overall faithfulness analysis"
    
    # Per dataset
    log_subsection "SST-2 Faithfulness Analysis"
    
    output_dir="${ANALYTICS_BASE}/comprehensive_faithfulness/stanfordnlp_sst2"
    
    run_analytics \
        "python -m src.Analytics comprehensive_faithfulness \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Faithfulness analysis for stanfordnlp/sst2"
    
    log_subsection "AG News Faithfulness Analysis"
    
    output_dir="${ANALYTICS_BASE}/comprehensive_faithfulness/setfit_ag_news"
    
    run_analytics \
        "python -m src.Analytics comprehensive_faithfulness \
            --insights-dir '${insights_dir}' \
            --output-dir '${output_dir}' \
            ${PLOT_FLAG}" \
        "Faithfulness analysis for SetFit/ag_news"
}

# ==============================================================================
# Legacy Analytics: Preserve old analysis compatibility
# ==============================================================================
run_legacy_analytics() {
    log_section "LEGACY ANALYTICS (Structural Patterns for GNN)"
    
    # Only run structural analysis on GNN data
    log_subsection "GNN Structural Patterns Analysis"
    
    # Skip structural analysis for now as it requires different input format
    log_info "Skipping structural visualization (requires graph-specific config)"
    log_info "To run manually: python -m src.Analytics structural_visualise --dataset sst2 --graph skipgrams"
}

# ==============================================================================
# Generate Summary Reports
# ==============================================================================
generate_summary_reports() {
    log_section "GENERATING SUMMARY REPORTS"
    
    local summary_dir="${ANALYTICS_BASE}/summary_${TIMESTAMP}"
    mkdir -p "${summary_dir}"
    
    log_info "Creating consolidated summary report..."
    
    # Create summary markdown
    cat > "${summary_dir}/ANALYTICS_SUMMARY.md" << 'EOF'
# Analytics Summary Report

## Overview

This directory contains comprehensive analytics for LLM and GNN explainability comparisons.

## Directory Structure

```
outputs/analytics/
├── complete/                           # Complete pipeline results
│   ├── stanfordnlp_sst2/
│   └── setfit_ag_news/
├── stratified/                         # Stratified by class and correctness
│   ├── stanfordnlp_sst2/
│   └── setfit_ag_news/
├── comparative/                        # LLM vs GNN comparisons
│   ├── llm_vs_gnn_overall/
│   ├── stanfordnlp_sst2/
│   ├── setfit_ag_news/
│   └── gnn_graph_types/
├── ranking_agreement/                  # Inter-explainer agreement
│   ├── overall/
│   ├── stanfordnlp_sst2/
│   └── setfit_ag_news/
├── contrastivity_compactness/          # Quality metrics
│   ├── overall/
│   ├── stanfordnlp_sst2/
│   └── setfit_ag_news/
├── enhanced_auc/                       # AUC curve analysis
│   ├── overall/
│   ├── stanfordnlp_sst2/
│   └── setfit_ag_news/
└── comprehensive_faithfulness/         # Faithfulness metrics
    ├── overall/
    ├── stanfordnlp_sst2/
    └── setfit_ag_news/
```

## Key Files

### Executive Summaries
- `complete/*/executive_summary.md` - Human-readable overview
- `complete/*/complete_pipeline_summary.json` - Machine-readable results

### Method Rankings
- `comparative/*/method_ranking.csv` - Performance comparison across methods

### Agreement Metrics
- `ranking_agreement/*/agreement_metrics_table.csv` - Detailed agreement data

### Visualizations
- All modules: `*/plots/` subdirectories contain PNG visualizations

## Quick Analysis

### Best Overall Method
Check: `comparative/llm_vs_gnn_overall/method_ranking.csv`

### Metric-Specific Performance
Check: `stratified/*/` for per-class and correctness breakdowns

### Explainer Agreement
Check: `ranking_agreement/overall/ranking_agreement_summary.json`

## Metrics Analyzed

- **Faithfulness**: Fidelity+, Fidelity-, General Faithfulness
- **AUC**: Insertion AUC, Deletion AUC
- **Agreement**: RBO, Spearman, Kendall, Feature Overlap
- **Quality**: Contrastivity, Compactness
- **Statistical**: Significance tests, effect sizes

## Next Steps

1. Review executive summaries in `complete/*/executive_summary.md`
2. Compare methods using `comparative/*/method_ranking.csv`
3. Examine visualizations in `*/plots/` directories
4. Check detailed metrics in JSON summary files

EOF

    log_success "Summary report created: ${summary_dir}/ANALYTICS_SUMMARY.md"
    
    # Copy key executive summaries
    log_info "Copying executive summaries..."
    
    for exec_summary in "${ANALYTICS_BASE}"/complete/*/executive_summary.md; do
        if [[ -f "${exec_summary}" ]]; then
            dataset=$(basename "$(dirname "${exec_summary}")")
            cp "${exec_summary}" "${summary_dir}/executive_${dataset}.md"
            log_info "  - ${dataset}"
        fi
    done
    
    # List all method ranking files
    log_info "Method ranking tables:"
    find "${ANALYTICS_BASE}/comparative" -name "method_ranking.csv" -type f | while read -r ranking_file; do
        log_info "  - ${ranking_file}"
    done
    
    log_success "Summary reports generated in: ${summary_dir}"
}

# ==============================================================================
# Main Execution
# ==============================================================================
main() {
    cd "${ROOT_DIR}" || exit 1
    
    log_section "COMPREHENSIVE ANALYTICS PIPELINE"
    
    log_info "Configuration:"
    log_info "  Insights directory: ${INSIGHTS_BASE}"
    log_info "  Analytics output: ${ANALYTICS_BASE}"
    log_info "  Timestamp: ${TIMESTAMP}"
    log_info "  Plot generation: $([ -z "${PLOT_FLAG}" ] && echo "ENABLED" || echo "DISABLED")"
    
    # Track execution time
    START_TIME=$(date +%s)
    
    # Run all analytics modules
    # Comment out sections you don't want to run
    
    run_complete_pipeline
    run_stratified_analysis
    run_comparative_analysis
    run_ranking_agreement_analysis
    run_contrastivity_compactness_analysis
    run_enhanced_auc_analysis
    run_comprehensive_faithfulness_analysis
    # run_legacy_analytics  # Commented out by default
    
    # Generate summary
    generate_summary_reports
    
    # Calculate execution time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    log_section "ANALYTICS PIPELINE COMPLETE"
    
    log_success "All analytics modules completed!"
    log_info "Total execution time: ${MINUTES}m ${SECONDS}s"
    log_info "Results available in: ${ANALYTICS_BASE}"
    
    echo ""
    echo -e "${YELLOW}Quick Access:${NC}"
    echo "  Summary reports:  ${ANALYTICS_BASE}/summary_${TIMESTAMP}/"
    echo "  Complete results: ${ANALYTICS_BASE}/complete/"
    echo "  Comparisons:      ${ANALYTICS_BASE}/comparative/"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Review executive summaries: cat ${ANALYTICS_BASE}/complete/*/executive_summary.md"
    echo "  2. Compare methods: cat ${ANALYTICS_BASE}/comparative/llm_vs_gnn_overall/method_ranking.csv"
    echo "  3. Check visualizations: ls ${ANALYTICS_BASE}/*/plots/"
    echo ""
}

# Parse command line arguments
SKIP_MODULES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-plots)
            PLOT_FLAG="--no-plots"
            shift
            ;;
        --insights-dir)
            INSIGHTS_BASE="$2"
            shift 2
            ;;
        --output-dir)
            ANALYTICS_BASE="$2"
            shift 2
            ;;
        --skip-complete)
            SKIP_MODULES="${SKIP_MODULES} complete"
            shift
            ;;
        --skip-stratified)
            SKIP_MODULES="${SKIP_MODULES} stratified"
            shift
            ;;
        --skip-comparative)
            SKIP_MODULES="${SKIP_MODULES} comparative"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --no-plots              Skip plot generation (faster execution)"
            echo "  --insights-dir DIR      Set insights directory (default: outputs/insights/news)"
            echo "  --output-dir DIR        Set analytics output directory (default: outputs/analytics)"
            echo "  --skip-complete         Skip complete pipeline"
            echo "  --skip-stratified       Skip stratified analysis"
            echo "  --skip-comparative      Skip comparative analysis"
            echo "  --help                  Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  INSIGHTS_BASE           Override insights directory"
            echo "  ANALYTICS_BASE          Override analytics output directory"
            echo "  PLOT_FLAG               Set to '--no-plots' to skip plots"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all analytics with plots"
            echo "  $0 --no-plots                        # Run all analytics without plots"
            echo "  $0 --skip-complete                   # Skip complete pipeline"
            echo "  PLOT_FLAG='--no-plots' $0            # Use environment variable"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

main "$@"

