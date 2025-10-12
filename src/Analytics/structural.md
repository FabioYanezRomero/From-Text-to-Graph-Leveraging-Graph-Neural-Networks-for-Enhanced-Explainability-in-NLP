# Structural Visualisation Workflow

Use the analytics CLI to recompute structural correlations and render the full
set of heatmaps (standard, difference, and clustered) for a given dataset and
graph construction.

```bash
python -m src.Analytics.cli structural_visualise \
  --dataset ag_news \
  --graph constituency \
  --significance-threshold 0.1
```

Arguments:
- `--dataset`: dataset key in the structural config (e.g. `ag_news`, `sst2`).
- `--graph`: graph flavour (`constituency`, `syntactic`, `skipgrams`, `window`).
- `--config`: optional path to a structural configuration file mapping dataset
  and graph pairs to the underlying explanation artefacts. Defaults to
  `configs/structural_analysis_config.json`.
- `--output-dir`: optional override that points to the root directory where
  analytics will be written. By default artefacts land under
  `outputs/analytics/structural/<dataset>_<graph>`.
- `--significance-threshold`: absolute correlation difference required for a
  cell to remain coloured in the difference heatmaps.
- `--no-clustered`: skip the hierarchical clustering step when rendering
  difference heatmaps.

The command will:
1. Run the structural graph analytics pipeline for the requested
   dataset/graph pair (matching entries in the config file).
2. Persist `structural_graph_analytics.csv` alongside per-cohort correlation
   tables.
3. Render the original correlation heatmaps, difference heatmaps
   (`correct - incorrect`), and (optionally) clustered difference heatmaps.
4. Summarise generated artefacts in
   `structural_visualisation_summary.json` within the target output folder.
