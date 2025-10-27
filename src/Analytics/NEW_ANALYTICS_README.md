# Enhanced Analytics System for LLM and GNN Explainability

This document describes the updated analytics system that provides comprehensive analysis of explainability metrics for both LLM and GNN models.

## Overview

The enhanced analytics system supports:

- **Stratified Analysis**: Analysis by class label and prediction correctness
- **Comparative Analysis**: Direct comparison between LLM and GNN explainers
- **Ranking Agreement**: Analysis of inter-explainer agreement (RBO, Spearman, Kendall, KL divergence, etc.)
- **Faithfulness Metrics**: Comprehensive fidelity+/-, general/local faithfulness analysis
- **Contrastivity & Compactness**: Explanation quality metrics
- **Enhanced AUC Analysis**: Insertion/deletion curves with visualization
- **Complete Pipeline**: Orchestrated execution of all analytics modules

## New Metrics Supported

### Faithfulness Metrics
- **Fidelity+** (Sufficiency): How much confidence drops when using only important features
- **Fidelity-** (Necessity): How much confidence drops when removing important features
- **General Faithfulness**: Overall explanation quality
- **Local Faithfulness**: Instance-level explanation quality
- **Faithfulness Monotonicity**: Consistency of faithfulness curves

### Ranking Agreement Metrics
- **Rank-Biased Overlap (RBO)**: Top-weighted rank comparison
- **Spearman/Kendall Correlation**: Rank correlation coefficients
- **KL Divergence**: Distribution difference between explainers
- **Feature Overlap Ratio**: Proportion of shared important features
- **Stability (Jaccard)**: Feature set overlap

### Quality Metrics
- **Contrastivity**: How well explanations distinguish between classes
- **Compactness/Sparsity**: How concise explanations are
- **Insertion/Deletion AUC**: Cumulative confidence curves
- **Robustness Score**: Explanation stability

## Data Structure

The new analytics system expects insights in this directory structure:

```
/app/outputs/insights/news/
├── GNN/
│   ├── <model>/
│   │   └── <dataset>/
│   │       ├── skipgrams_summaries.json
│   │       ├── skipgrams_summaries.part0001.json
│   │       ├── skipgrams_summaries.part0002.json
│   │       ├── constituency_summaries.json
│   │       ├── syntactic_summaries.json
│   │       ├── window_summaries.json
│   │       ├── skipgrams_agreement.json
│   │       ├── constituency_agreement.json
│   │       ├── syntactic_agreement.json
│   │       └── window_agreement.json
└── LLM/
    ├── <model>/
    │   └── <dataset>/
    │       ├── token_shap.json
    │       ├── token_shap_shard00of03.json
    │       ├── token_shap_shard01of03.json
    │       └── token_shap_shard02of03.json
```

## Usage

### Command-Line Interface

All analytics are accessible through a unified CLI:

```bash
python -m src.Analytics <command> [options]
```

### Available Commands

#### 1. Stratified Analysis

Analyze metrics stratified by class and correctness:

```bash
python -m src.Analytics stratified \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/stratified \
    --class-col label \
    --correctness-col is_correct
```

**Options:**
- `--insights-dir`: Directory containing GNN/ and LLM/ subdirectories
- `--metrics`: Specific metrics to analyze (default: all)
- `--class-col`: Column name for class labels (default: label)
- `--correctness-col`: Column name for correctness (default: is_correct)
- `--no-plots`: Skip plot generation

**Outputs:**
- `stratified_analysis_summary.json`: Complete statistical summary
- `<metric>_stratified.json`: Per-metric results
- `plots/`: Boxplots, heatmaps for each metric

#### 2. Comparative Analysis

Compare LLM vs GNN explainers:

```bash
python -m src.Analytics comparative \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/comparative
```

**Options:**
- `--insights-dir`: Directory containing insights
- `--metrics`: Specific metrics to compare
- `--method-col`: Column identifying method (default: method)
- `--model-type-col`: Column identifying LLM/GNN (default: model_type)
- `--no-plots`: Skip plot generation

**Outputs:**
- `comparative_analysis_summary.json`: Comparison results
- `method_ranking.csv`: Method ranking table
- `plots/`: Violin plots, KDE plots, summary comparisons

#### 3. Ranking Agreement Analysis

Analyze inter-explainer agreement:

```bash
python -m src.Analytics ranking_agreement \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/agreement
```

**Options:**
- `--insights-dir`: Directory containing agreement metrics
- `--class-col`: Column for class labels
- `--correctness-col`: Column for correctness
- `--no-plots`: Skip plot generation

**Outputs:**
- `ranking_agreement_summary.json`: Agreement statistics
- `agreement_metrics_table.csv`: Detailed agreement data
- `plots/`: Distribution plots, correlation matrices

#### 4. Contrastivity & Compactness Analysis

Analyze explanation quality metrics:

```bash
python -m src.Analytics contrastivity_compactness \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/contrast
```

**Outputs:**
- `contrastivity_compactness_summary.json`: Quality metrics
- `plots/`: Distribution plots, correlation plots

#### 5. Enhanced AUC Analysis

Analyze insertion/deletion curves:

```bash
python -m src.Analytics enhanced_auc \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/auc
```

**Outputs:**
- `enhanced_auc_summary.json`: AUC statistics
- `plots/`: Average curves, AUC comparisons, scatter plots

#### 6. Comprehensive Faithfulness Analysis

Analyze all faithfulness metrics:

```bash
python -m src.Analytics comprehensive_faithfulness \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/faithfulness
```

**Outputs:**
- `comprehensive_faithfulness_summary.json`: Faithfulness results
- `plots/`: Fidelity distributions, faithfulness plots, correlation matrices

#### 7. Complete Pipeline

Run all analytics modules in sequence:

```bash
python -m src.Analytics complete_pipeline \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/complete
```

**Outputs:**
- `complete_pipeline_summary.json`: Full pipeline results
- `executive_summary.md`: Human-readable summary
- Subdirectories for each module with their respective outputs

### Legacy Commands (Still Supported)

The following commands from the old analytics system are still available:

```bash
# Structural visualization
python -m src.Analytics structural_visualise \
    --dataset sst2 \
    --graph skipgrams

# Basic fidelity analysis
python -m src.Analytics fidelity <insight_files>

# Basic insertion AUC
python -m src.Analytics insertion <insight_files>

# Basic faithfulness aggregation
python -m src.Analytics faithfulness <insight_files>

# LLM token analysis
python -m src.Analytics llm <insight_files>
```

## Module Architecture

### Design Patterns

The new analytics system follows these design patterns:

1. **Facade Pattern**: `unified_analytics.py` provides a simple interface to complex subsystems
2. **Strategy Pattern**: Each analytics module can be run independently or as part of the pipeline
3. **Data Transfer Objects**: `loaders.py` provides structured data containers (`EnhancedInsightFrame`)
4. **Separation of Concerns**: Each module focuses on a specific aspect of analysis

### Module Overview

```
src/Analytics/
├── loaders.py                      # Enhanced data loading with sharding support
├── stratified_metrics.py           # Stratified analysis by class/correctness
├── comparative_analysis.py         # LLM vs GNN comparison
├── ranking_agreement.py            # Inter-explainer agreement metrics
├── contrastivity_compactness.py    # Explanation quality metrics
├── enhanced_auc_analysis.py        # AUC curve analysis
├── comprehensive_faithfulness.py   # Complete faithfulness assessment
├── unified_analytics.py            # Orchestrator for complete pipeline
├── cli.py                          # Command-line interface (updated)
└── utils.py                        # Shared utilities (preserved)
```

### Preserved Functionality

The following old analytics are preserved:

- **Word/Node Rank Distribution**: KDE analysis of feature importance distributions
- **GNN Structural Patterns**: Graph topology analysis and correlations
- **Semantic Analytics**: Token-level analysis pipeline
- **Basic Metrics**: Fidelity drop, insertion AUC, minimal coalition analysis

## Output Structure

When running the complete pipeline, outputs are organized as follows:

```
outputs/analytics/complete/
├── complete_pipeline_summary.json
├── executive_summary.md
├── stratified/
│   ├── stratified_analysis_summary.json
│   ├── fidelity_plus_stratified.json
│   ├── fidelity_minus_stratified.json
│   └── plots/
│       ├── fidelity_plus_by_class_boxplot.png
│       ├── fidelity_plus_by_correctness_boxplot.png
│       └── ...
├── comparative/
│   ├── comparative_analysis_summary.json
│   ├── method_ranking.csv
│   └── plots/
│       ├── fidelity_plus_violin.png
│       ├── llm_vs_gnn_summary.png
│       └── ...
├── ranking_agreement/
│   ├── ranking_agreement_summary.json
│   ├── agreement_metrics_table.csv
│   └── plots/
│       ├── rbo_distribution.png
│       ├── agreement_correlation_matrix.png
│       └── ...
├── contrastivity_compactness/
│   ├── contrastivity_compactness_summary.json
│   └── plots/
│       ├── contrastivity_distributions.png
│       ├── compactness_distributions.png
│       └── contrastivity_vs_compactness.png
├── enhanced_auc/
│   ├── enhanced_auc_summary.json
│   └── plots/
│       ├── average_curves.png
│       ├── insertion_auc_comparison.png
│       └── insertion_vs_deletion_auc.png
└── comprehensive_faithfulness/
    ├── comprehensive_faithfulness_summary.json
    └── plots/
        ├── fidelity_plus_minus_comparison.png
        ├── faithfulness_by_method.png
        └── faithfulness_correlation_matrix.png
```

## Examples

### Example 1: Quick Comparative Analysis

```bash
# Run just the comparative analysis
python -m src.Analytics comparative \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/llm_vs_gnn
```

### Example 2: Stratified Analysis for Specific Metrics

```bash
# Analyze only faithfulness metrics, stratified by class
python -m src.Analytics stratified \
    --insights-dir /app/outputs/insights/news \
    --metrics fidelity_plus fidelity_minus faithfulness \
    --output-dir /app/outputs/analytics/faith_stratified
```

### Example 3: Complete Pipeline Without Plots

```bash
# Run full pipeline, skip plots for faster execution
python -m src.Analytics complete_pipeline \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/quick_run \
    --no-plots
```

### Example 4: Agreement Analysis Only

```bash
# Analyze ranking agreement between explainers
python -m src.Analytics ranking_agreement \
    --insights-dir /app/outputs/insights/news \
    --output-dir /app/outputs/analytics/agreement
```

## Statistical Tests

The analytics system performs the following statistical tests:

### Stratified Analysis
- **Kruskal-Wallis H-test**: Non-parametric test for differences across multiple groups
- **Mann-Whitney U test**: Pairwise comparison between two groups
- **Cohen's d**: Effect size calculation

### Comparative Analysis
- **Mann-Whitney U test**: Compare LLM vs GNN distributions
- **Spearman/Pearson correlation**: Assess relationships between metrics

### Significance Levels
- α = 0.05 (standard significance)
- α = 0.01 (high significance)

## Visualization

The analytics system generates various plots:

### Distribution Plots
- Histograms with KDE overlays
- Box plots
- Violin plots

### Comparison Plots
- Grouped box/violin plots by method
- Scatter plots (e.g., fidelity+ vs fidelity-)
- Bar charts with error bars

### Heatmaps
- Correlation matrices
- Class × Correctness metrics
- Agreement metrics across methods

### Curve Plots
- Insertion/deletion curves
- Average curves by method
- Sufficiency progression

## Metrics Interpretation

### Fidelity+ (Lower is Better)
- Measures sufficiency: how much confidence drops when using only important features
- Lower values indicate the explanation captures the most important information

### Fidelity- (Lower is Better)
- Measures necessity: how much confidence drops when removing important features
- Lower values indicate the removed features were not critical

### Faithfulness (Higher is Better)
- Combined measure: `fidelity- - fidelity+`
- Positive values indicate good faithfulness
- Negative values suggest the explanation may not be capturing the right features

### Insertion/Deletion AUC (Higher is Better)
- Area under the curve when adding/removing features
- Higher AUC indicates more effective feature importance ranking

### Contrastivity (Higher is Better)
- Measures how well the explanation distinguishes the predicted class
- Calculated as: `confidence_predicted - confidence_second_best`

### Compactness/Sparsity (Lower is Better)
- Proportion of features used in the explanation
- Lower values indicate more concise explanations

### RBO (Higher is Better, 0-1 range)
- Top-weighted rank agreement
- Higher values indicate better agreement between explainers

## Troubleshooting

### Issue: "No agreement data found"
**Solution**: Ensure your insights directory contains `*_agreement.json` files generated by the Insights module.

### Issue: "No records found"
**Solution**: Check that your insights directory follows the expected structure (GNN/ and LLM/ subdirectories).

### Issue: Missing plots
**Solution**: Ensure matplotlib backend is properly configured. If running in a headless environment, the 'Agg' backend is set automatically.

### Issue: Memory errors with large datasets
**Solution**: The sharded loading should handle large datasets automatically. If issues persist, process datasets separately using individual commands instead of the complete pipeline.

## Future Enhancements

Planned improvements:
- Interactive dashboards (Plotly/Dash)
- LaTeX report generation
- Time-series analysis for multiple runs
- Automated hypothesis testing
- Confidence intervals via bootstrap

## References

The analytics framework is based on:
- **GraphFramEx**: Fair comparison protocol for graph explainability
- **M4 Benchmark**: Multi-modal faithfulness metrics
- **XAI Literature**: Standard explainability evaluation metrics

## Contact

For questions or issues, please consult the main project documentation or open an issue.

