# Quick Start: New Analytics System

## TL;DR

Run the complete analytics pipeline with one command:

```bash
python -m src.Analytics complete_pipeline \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/my_analysis
```

Then check the results:

```bash
# Read the executive summary
cat /app/outputs/analytics/my_analysis/executive_summary.md

# Or open the JSON summary
cat /app/outputs/analytics/my_analysis/complete_pipeline_summary.json
```

## What's New?

The Analytics module now supports:

### ✨ New Metrics
- **Fidelity+/Fidelity-** (sufficiency/necessity)
- **RBO, Spearman, Kendall** (ranking agreement)
- **Contrastivity** (class discrimination)
- **Compactness** (explanation sparsity)
- **Enhanced AUC** with curve visualization

### 📊 New Analysis Types
- **Stratified by class** and **correctness**
- **LLM vs GNN** direct comparisons
- **Inter-explainer agreement** analysis
- **Statistical significance** testing

## Quick Examples

### 1. Run Everything (Recommended First Time)

```bash
python -m src.Analytics complete_pipeline \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/full_run
```

**What you get:**
- 7 modules executed in sequence
- ~100+ plots generated
- JSON summaries for each module
- Executive summary in Markdown
- Method rankings and comparisons

**Time:** ~5-10 minutes (depending on data size)

### 2. Quick LLM vs GNN Comparison

```bash
python -m src.Analytics comparative \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/llm_vs_gnn
```

**What you get:**
- Statistical comparison of all metrics
- Violin plots for each metric
- Method ranking table
- Effect sizes and p-values

**Time:** ~1-2 minutes

### 3. Stratified Analysis (By Class and Correctness)

```bash
python -m src.Analytics stratified \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/stratified
```

**What you get:**
- Metrics broken down by class
- Correct vs incorrect predictions
- Statistical tests for differences
- Boxplots and heatmaps

**Time:** ~2-3 minutes

### 4. Fast Run (No Plots)

```bash
python -m src.Analytics complete_pipeline \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/fast \
    --no-plots
```

**What you get:**
- All statistics in JSON
- No plot generation
- Much faster execution

**Time:** ~1-2 minutes

## Understanding Your Results

### Executive Summary

The `executive_summary.md` tells you:
- How many records were analyzed
- Which modules succeeded/failed
- Quick overview of each analysis
- Directory structure of outputs

**Example:**
```markdown
# Complete Analytics Pipeline - Executive Summary

## Pipeline Information
- **Total Records**: 872
- **Datasets**: sst2
- **Methods**: graphsvx, token_shap_llm
- **Execution Time**: 8.45m

## Module Status
✓ Stratified Analysis
✓ Comparative Analysis
✓ Ranking Agreement
...
```

### Key Output Files

**Main Summary:**
- `complete_pipeline_summary.json` - All results in JSON

**Per-Module:**
- `stratified/stratified_analysis_summary.json`
- `comparative/comparative_analysis_summary.json`
- `ranking_agreement/ranking_agreement_summary.json`
- etc.

**Tables:**
- `comparative/method_ranking.csv` - Method performance ranking
- `ranking_agreement/agreement_metrics_table.csv` - Agreement data

**Plots:** All in `plots/` subdirectories

## Common Use Cases

### Use Case 1: "Which explainer is best?"

```bash
python -m src.Analytics comparative \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/compare

# Then check:
cat /app/outputs/analytics/compare/method_ranking.csv
```

### Use Case 2: "How do explainers perform on correct vs incorrect predictions?"

```bash
python -m src.Analytics stratified \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/by_correctness

# Then check:
ls /app/outputs/analytics/by_correctness/plots/*by_correctness*
```

### Use Case 3: "Do explainers agree on important features?"

```bash
python -m src.Analytics ranking_agreement \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/agreement

# Then check:
cat /app/outputs/analytics/agreement/ranking_agreement_summary.json
```

### Use Case 4: "Analyze specific metrics only"

```bash
python -m src.Analytics stratified \
    --insights-dir /app/outputs/insights/news \
    --metrics fidelity_plus fidelity_minus faithfulness \
    --output-dir /app/outputs/analytics/fidelity_only
```

## Interpreting Key Metrics

### Fidelity+
- **What**: Confidence drop when using only important features
- **Lower is better** (close to 0)
- **Good value**: < 0.05

### Fidelity-
- **What**: Confidence drop when removing important features
- **Lower is better** (close to 0)
- **Good value**: < 0.05

### Faithfulness
- **What**: `fidelity- - fidelity+`
- **Higher is better** (positive values)
- **Good value**: > 0

### Insertion AUC
- **What**: How quickly confidence rises when adding features
- **Higher is better** (closer to 1)
- **Good value**: > 0.7

### RBO (Rank-Biased Overlap)
- **What**: Agreement between explainers (top-weighted)
- **Higher is better** (0 to 1)
- **Good value**: > 0.5

### Contrastivity
- **What**: Difference between predicted and second-best class confidence
- **Higher is better**
- **Good value**: > 0.8

## Troubleshooting

### "No agreement data found"
Your insights don't have agreement metrics. This is normal if you only ran individual explainers.
**Solution**: Skip ranking_agreement module or regenerate insights with agreement metrics.

### "No records found"
Your insights directory structure might be wrong.
**Expected structure:**
```
/app/outputs/insights/news/
├── GNN/
│   └── <model>/<dataset>/
└── LLM/
    └── <model>/<dataset>/
```

### "Module failed"
The pipeline will continue even if one module fails.
**Solution**: Check the error in `complete_pipeline_summary.json` under `"errors"`.

### Out of memory
Try running without plots or process datasets separately:
```bash
python -m src.Analytics stratified \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/part1 \
    --no-plots
```

## What's Preserved?

All old analytics still work:

```bash
# Old commands (still supported)
python -m src.Analytics fidelity outputs/insights/file.json
python -m src.Analytics insertion outputs/insights/file.json
python -m src.Analytics structural_visualise --dataset sst2 --graph skipgrams
```

## Next Steps

1. **Run the complete pipeline** (see TL;DR)
2. **Read the executive summary**
3. **Examine the plots** in each module's directory
4. **Check JSON files** for detailed statistics
5. **Use method_ranking.csv** to compare approaches

## Getting Help

```bash
# List all commands
python -m src.Analytics --help

# Help for specific command
python -m src.Analytics stratified --help
python -m src.Analytics comparative --help
python -m src.Analytics complete_pipeline --help
```

## Full Documentation

For comprehensive documentation, see:
- `/app/src/Analytics/NEW_ANALYTICS_README.md` - Complete guide
- `/app/ANALYTICS_UPDATE_SUMMARY.md` - What changed
- `/app/src/Analytics/README.md` - Original documentation

---

**Ready to start?**

```bash
python -m src.Analytics complete_pipeline \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/my_first_run
```

Then check: `cat /app/outputs/analytics/my_first_run/executive_summary.md`

