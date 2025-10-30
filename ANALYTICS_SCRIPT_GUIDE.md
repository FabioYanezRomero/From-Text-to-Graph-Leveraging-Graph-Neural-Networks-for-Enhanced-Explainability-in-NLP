# Analytics Script Guide

## Overview

The `run_all_analytics.sh` script runs comprehensive analytics on all LLM and GNN explainability results, with special focus on cross-method agreement analysis.

## Quick Start

```bash
# Run everything with plots (takes ~10-15 minutes)
./run_all_analytics.sh

# Run without plots (faster, ~3-5 minutes)
./run_all_analytics.sh --no-plots

# Run with custom directories
./run_all_analytics.sh \
    --insights-dir /path/to/insights \
    --output-dir /path/to/analytics
```

## What It Does

### 1. Complete Pipeline Analysis
Runs all analytics modules on the full datasets:
- **SST-2** (stanfordnlp/sst2)
- **AG News** (SetFit/ag_news)

**Output:** `outputs/analytics/complete/{dataset}/`

### 2. Stratified Analysis
Analyzes metrics stratified by:
- **Class label** (0, 1, 2, 3 for AG News; 0, 1 for SST-2)
- **Prediction correctness** (correct vs incorrect)

**Output:** `outputs/analytics/stratified/{dataset}/`

### 3. Comparative Analysis
Compares performance between:
- **LLM vs GNN** (overall and per-dataset)
- **Graph types** (skipgrams vs constituency vs syntactic vs window)
- Includes method ranking tables

**Output:** `outputs/analytics/comparative/`

### 4. Ranking Agreement Analysis
Analyzes inter-explainer agreement including:
- **LLM vs Syntactic graphs**
- **LLM vs Constituency graphs**
- **LLM vs Skipgrams graphs**
- **LLM vs Window graphs**
- **Constituency vs Syntactic**
- **Skipgrams vs Window**
- All other graph type combinations

Metrics computed:
- Rank-Biased Overlap (RBO)
- Spearman rank correlation
- Kendall rank correlation
- KL divergence (when available)
- Feature overlap ratio
- Stability (Jaccard)

**Output:** `outputs/analytics/ranking_agreement/`

### 5. Quality Metrics Analysis
Analyzes:
- **Contrastivity** (how well explanations distinguish classes)
- **Compactness** (explanation sparsity)
- Correlations with faithfulness

**Output:** `outputs/analytics/contrastivity_compactness/`

### 6. Enhanced AUC Analysis
Analyzes insertion/deletion curves:
- Average curves by method
- AUC distributions
- Curve comparisons

**Output:** `outputs/analytics/enhanced_auc/`

### 7. Comprehensive Faithfulness
Analyzes all faithfulness metrics:
- Fidelity+ (sufficiency)
- Fidelity- (necessity)
- General faithfulness
- Local faithfulness

**Output:** `outputs/analytics/comprehensive_faithfulness/`

## Command-Line Options

### Basic Options

```bash
# Skip plot generation (much faster)
./run_all_analytics.sh --no-plots

# Custom insights directory
./run_all_analytics.sh --insights-dir /custom/path/insights

# Custom output directory
./run_all_analytics.sh --output-dir /custom/path/analytics

# Show help
./run_all_analytics.sh --help
```

### Skip Specific Modules

```bash
# Skip complete pipeline (if you only want specific analyses)
./run_all_analytics.sh --skip-complete

# Skip stratified analysis
./run_all_analytics.sh --skip-stratified

# Skip comparative analysis
./run_all_analytics.sh --skip-comparative

# Combine multiple skips
./run_all_analytics.sh --skip-complete --skip-stratified --no-plots
```

### Environment Variables

```bash
# Set insights directory
INSIGHTS_BASE=/path/to/insights ./run_all_analytics.sh

# Set analytics output directory
ANALYTICS_BASE=/path/to/output ./run_all_analytics.sh

# Disable plots via environment
PLOT_FLAG="--no-plots" ./run_all_analytics.sh
```

## Output Structure

After running, your analytics directory will look like:

```
outputs/analytics/
├── complete/
│   ├── stanfordnlp_sst2/
│   │   ├── executive_summary.md
│   │   ├── complete_pipeline_summary.json
│   │   ├── stratified/
│   │   ├── comparative/
│   │   ├── ranking_agreement/
│   │   ├── contrastivity_compactness/
│   │   ├── enhanced_auc/
│   │   └── comprehensive_faithfulness/
│   └── setfit_ag_news/
│       └── ... (same structure)
├── stratified/
│   ├── stanfordnlp_sst2/
│   │   ├── stratified_analysis_summary.json
│   │   ├── fidelity_plus_stratified.json
│   │   └── plots/
│   └── setfit_ag_news/
├── comparative/
│   ├── llm_vs_gnn_overall/
│   │   ├── comparative_analysis_summary.json
│   │   ├── method_ranking.csv          ← KEY FILE
│   │   └── plots/
│   ├── stanfordnlp_sst2/
│   ├── setfit_ag_news/
│   └── gnn_graph_types/
├── ranking_agreement/
│   ├── overall/
│   │   ├── ranking_agreement_summary.json
│   │   ├── agreement_metrics_table.csv  ← KEY FILE
│   │   └── plots/
│   ├── stanfordnlp_sst2/
│   └── setfit_ag_news/
├── contrastivity_compactness/
│   └── ... (similar structure)
├── enhanced_auc/
│   └── ... (similar structure)
├── comprehensive_faithfulness/
│   └── ... (similar structure)
└── summary_<timestamp>/
    ├── ANALYTICS_SUMMARY.md
    ├── executive_stanfordnlp_sst2.md
    └── executive_setfit_ag_news.md
```

## Key Output Files

### Executive Summaries (Human-Readable)
```bash
cat outputs/analytics/complete/stanfordnlp_sst2/executive_summary.md
cat outputs/analytics/complete/setfit_ag_news/executive_summary.md
```

### Method Rankings (Best Methods)
```bash
cat outputs/analytics/comparative/llm_vs_gnn_overall/method_ranking.csv
```

### Agreement Analysis (How Explainers Agree)
```bash
cat outputs/analytics/ranking_agreement/overall/agreement_metrics_table.csv
cat outputs/analytics/ranking_agreement/overall/ranking_agreement_summary.json
```

### Summary Report (Generated at End)
```bash
cat outputs/analytics/summary_<timestamp>/ANALYTICS_SUMMARY.md
```

## Understanding Agreement Analysis

The ranking agreement module analyzes how different explainers agree on feature importance:

### LLM vs GNN Comparisons
For each instance, compares:
- **Token SHAP (LLM)** vs **Syntactic GNN**
- **Token SHAP (LLM)** vs **Constituency GNN**
- **Token SHAP (LLM)** vs **Skipgrams GNN**
- **Token SHAP (LLM)** vs **Window GNN**

### GNN vs GNN Comparisons
- **Constituency** vs **Syntactic** (tree-based methods)
- **Skipgrams** vs **Window** (n-gram methods)
- All other graph type pairs

### Metrics Reported
- **RBO** (0-1): Top-weighted agreement, higher = better
- **Spearman** (-1 to 1): Rank correlation
- **Kendall** (-1 to 1): Rank correlation (different method)
- **Feature Overlap** (0-1): Proportion of shared important features
- **Stability (Jaccard)** (0-1): Set overlap

### Interpretation
- **High RBO (>0.7)**: Explainers agree on what's important
- **Low RBO (<0.3)**: Explainers disagree significantly
- **Positive Spearman/Kendall**: Consistent ranking
- **Negative Spearman/Kendall**: Inverse ranking

## Execution Time

Approximate times (may vary by hardware):

| Configuration | Time | Outputs |
|--------------|------|---------|
| Full (with plots) | ~10-15 min | ~500+ files, ~200+ plots |
| Full (no plots) | ~3-5 min | ~300+ files, JSON only |
| Skip complete | ~5-8 min | Specific modules only |

## Troubleshooting

### Out of Memory
```bash
# Run without plots
./run_all_analytics.sh --no-plots

# Or skip heavy modules
./run_all_analytics.sh --skip-complete
```

### Missing Data
If you see "No agreement data found":
- Check that `*_agreement.json` files exist in insights
- These are generated by the Insights module
- Run the Insights module first to generate agreement metrics

### Partial Execution
The script continues even if one module fails. Check:
```bash
cat outputs/analytics/complete/*/complete_pipeline_summary.json | grep -A5 "errors"
```

## Examples

### Example 1: Quick Analysis (No Plots)
```bash
./run_all_analytics.sh --no-plots
```

**Result:** JSON summaries for all metrics, no visualizations
**Time:** ~3-5 minutes

### Example 2: Full Analysis with Everything
```bash
./run_all_analytics.sh
```

**Result:** Complete analysis with all plots
**Time:** ~10-15 minutes

### Example 3: Just Comparative and Agreement
```bash
./run_all_analytics.sh \
    --skip-complete \
    --skip-stratified \
    --no-plots
```

**Result:** Only LLM vs GNN comparisons and agreement analysis
**Time:** ~2-3 minutes

### Example 4: Custom Directories
```bash
./run_all_analytics.sh \
    --insights-dir /data/my_insights \
    --output-dir /data/my_analytics
```

**Result:** Uses custom paths for input and output

## Viewing Results

### Quick Overview
```bash
# Main summary
cat outputs/analytics/summary_*/ANALYTICS_SUMMARY.md

# Executive summaries
ls outputs/analytics/complete/*/executive_summary.md
```

### Method Comparison
```bash
# See which method performs best
cat outputs/analytics/comparative/llm_vs_gnn_overall/method_ranking.csv

# Visual comparison
ls outputs/analytics/comparative/*/plots/*.png
```

### Agreement Analysis
```bash
# Detailed agreement data
cat outputs/analytics/ranking_agreement/overall/agreement_metrics_table.csv

# Summary statistics
cat outputs/analytics/ranking_agreement/overall/ranking_agreement_summary.json

# Visual analysis
ls outputs/analytics/ranking_agreement/overall/plots/*.png
```

### Stratified Results
```bash
# By class and correctness
cat outputs/analytics/stratified/*/stratified_analysis_summary.json

# Visual breakdowns
ls outputs/analytics/stratified/*/plots/*by_class*.png
ls outputs/analytics/stratified/*/plots/*by_correctness*.png
```

## Integration with Papers/Reports

The outputs are designed for research papers:

### LaTeX Tables
Method ranking CSVs can be directly imported into LaTeX tables.

### Figures
All plots are 200+ DPI PNG files suitable for publication.

### Statistics
JSON files contain all statistical tests (p-values, effect sizes) for reporting.

### Executive Summaries
Markdown summaries provide narrative structure for results sections.

## Next Steps After Running

1. **Review Executive Summaries**
   ```bash
   cat outputs/analytics/complete/*/executive_summary.md
   ```

2. **Identify Best Methods**
   ```bash
   cat outputs/analytics/comparative/llm_vs_gnn_overall/method_ranking.csv
   ```

3. **Examine Agreement**
   ```bash
   cat outputs/analytics/ranking_agreement/overall/agreement_metrics_table.csv
   ```

4. **Check Visualizations**
   ```bash
   ls outputs/analytics/*/plots/
   ```

5. **Extract Statistics for Paper**
   ```bash
   cat outputs/analytics/*/summary.json | jq '.statistical_tests'
   ```

## Customization

To modify what gets run, edit `/app/run_all_analytics.sh`:

```bash
# Comment out modules you don't want:
main() {
    # run_complete_pipeline        # Comment to skip
    run_stratified_analysis
    run_comparative_analysis
    run_ranking_agreement_analysis
    # run_contrastivity_compactness_analysis  # Comment to skip
    run_enhanced_auc_analysis
    run_comprehensive_faithfulness_analysis
}
```

## Support

For issues or questions:
1. Check `/app/src/Analytics/NEW_ANALYTICS_README.md` for detailed documentation
2. Run with `--help` to see all options
3. Check the logs for specific error messages

## Summary

**The script analyzes:**
- ✅ LLM vs GNN performance
- ✅ Graph type comparisons
- ✅ Cross-method agreement (RBO, Spearman, Kendall, etc.)
- ✅ Stratified metrics (by class and correctness)
- ✅ Faithfulness and quality metrics
- ✅ Statistical significance tests

**Run it:**
```bash
./run_all_analytics.sh
```

**Check results:**
```bash
cat outputs/analytics/summary_*/ANALYTICS_SUMMARY.md
```

