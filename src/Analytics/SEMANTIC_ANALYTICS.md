# Semantic Analytics Toolkit

The semantic analytics workflow has been modularised to reflect the different
visualisations and artefacts we generate (token inspection, semantic density,
and context statistics).  Each module focuses on one responsibility and exposes
an explicit CLI entry point.

## Directory Layout

```
src/Analytics/semantic/
├── __init__.py                # Public export (`SemanticPipeline`)
├── cli_pipeline.py            # Run the full pipeline (tokens + summaries + aggregates)
├── pipeline.py                # High-level orchestration used by the CLI wrappers
└── common/                    # Semantic-specific configuration, models, IO helpers
    ├── __init__.py
    ├── config.py              # Dataclasses + configuration loader
    ├── data_loader.py         # GraphSVX/SubgraphX readers and provider wrapper
    ├── models.py              # Lightweight dataclasses for token-level results
    └── outputs.py             # Helpers for converting results into CSVs

Sibling analytics packages now live alongside `semantic/`:

src/Analytics/token/           # Token-level selection and aggregation logic
src/Analytics/score/           # Score/importance extensions
src/Analytics/sparsity/        # Sparsity-focused extensions
src/Analytics/confidence/      # Confidence/threshold extensions
src/Analytics/embeddings/      # Embedding-based analytics
```

Thin compatibility modules remain at `src/Analytics/semantic/` (`config.py`,
`tokens.py`, etc.) so existing imports continue to work while new code can
depend on the dedicated sibling packages directly.

## Commands

Run the full semantic analytics pipeline (mirrors the legacy behaviour of
`semantic_analysis.py`):

```bash
python3 -m src.Analytics.semantic.cli_pipeline \
  --config configs/semantic_analysis_config.json \
  --output-dir outputs/analytics/general
```

The command writes, for each dataset/graph-type combination:

* `semantic_tokens.csv` – token-level attributions per graph.
* `semantic_summary.csv` – per-graph semantic density and metadata.
* `semantic_aggregate.csv` – aggregated token statistics for heatmaps/word clouds.

The deprecated `semantic_analysis.py` module now delegates to the same pipeline
so existing scripts continue to work, but new code should prefer the CLI above.

## Extending the Pipeline

* Add new datasets/graph types by editing `configs/semantic_analysis_config.json`.
* Custom stopwords can be appended directly in the config or via `stopwords_files`.
* Shared helpers (loading, token selection, aggregation) live in
  `src.Analytics.semantic.common` and `src.Analytics.token` and can be imported
  from other analytics scripts as needed.
* Post-processing utilities export the CSV artefacts into category-specific
  folders alongside the other analytics modules (`outputs/analytics/token/`,
  `outputs/analytics/sparsity/`, `outputs/analytics/score/`, `outputs/analytics/confidence/`, …).
