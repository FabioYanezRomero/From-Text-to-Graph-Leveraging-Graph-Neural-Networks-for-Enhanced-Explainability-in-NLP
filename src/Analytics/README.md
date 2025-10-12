# Analytics Toolkit

Utilities in this directory analyse the JSON insight exports produced by `src.Insights`. All scripts are now exposed through a single CLI so you can stay in one workflow when computing descriptive statistics, generating plots, or running cohort comparisons.

## Quick Start

```bash
# Aggregate descriptive metrics (overall + per class) and write them under outputs/analytics/overall
python -m src.Analytics overall outputs/insights/SetFit/ag_news/constituency.json

# Produce histograms, boxplots, and token-frequency tables
python -m src.Analytics distributions outputs/insights/SetFit/ag_news/constituency.json --output-dir outputs/analytics/ag_news_dist

# Inspect fidelity drops stratified by class label
python -m src.Analytics fidelity outputs/insights/SetFit/ag_news/constituency.json --group-key prediction_class
```

Every subcommand accepts one or more insight JSON files. If you omit `--output-dir`, results are written to `outputs/analytics/<command>` by default. Use `--help` with any subcommand for the full list of options.
