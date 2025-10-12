# Semantic Analysis Workflow

Use the semantic analytics CLI to regenerate the summary, token, and aggregate
tables for every dataset/graph pair defined in the configuration file. The
command also emits partitioned CSVs (overall, correct vs. incorrect, per-class,
and class-by-correctness) that mirror the structural analytics layout.

```bash
python -m src.Analytics.semantic_analysis \
  --config configs/semantic_analysis_config.json \
  --output-dir outputs/analytics/semantic
```

Options:

- `--config`: Path to the semantic analysis configuration (defaults to
  `configs/semantic_analysis_config.json`). This file lists the GraphSVX and
  SubgraphX artefacts to process, along with dataset, graph type, backbone, and
  optional thresholds.
- `--output-dir`: Directory where all CSV artefacts will be written. Each
  dataset/graph pair produces:
  - `<dataset>_<graph>_summary.csv` – per-graph semantic metrics.
  - `<dataset>_<graph>_tokens.csv` – top-k token attributions per graph.
  - `<dataset>_<graph>_aggregate.csv` – dataset-level token statistics.
  - Partitioned views for correctness (`…_correct_*.csv`, `…_incorrect_*.csv`),
    per-class (`…_class_*`), and class-by-correctness combinations.

Run the command whenever new explanation artefacts are generated to keep the
semantic analytics up to date. The outputs are organised by dataset and graph
type, matching the structure used for the structural analytics module.
