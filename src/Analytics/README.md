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

# Summarise LLM explanation exports without structural analytics
python -m src.Analytics llm outputs/insights/LLM/SetFit/ag_news/token_shap.json --output-dir outputs/analytics/llm/ag_news

# Aggregate faithfulness metrics (insertion AUC) across methods
python -m src.Analytics faithfulness outputs/insights/SetFit/ag_news/constituency.json --output-dir outputs/analytics/faithfulness/ag_news_constituency
```

Every subcommand accepts one or more insight JSON files. If you omit `--output-dir`, results are written to `outputs/analytics/<command>` by default. Use `--help` with any subcommand for the full list of options.

The `llm` mode mirrors the semantic exports generated for the GNN pipeline: general CSVs live under `outputs/analytics/llm/general/<dataset_graph>/`, with derived splits in sibling directories (`token/`, `sparsity/`, `confidence/`, `score/`). Fidelity reports and embedding availability markers are also produced automatically within the same `outputs/analytics/llm` namespace so existing GNN artefacts remain untouched.

## Fair Multimodal Faithfulness Protocol

Our faithfulness analytics follow the experimental design introduced in **GraphFramEx** and subsequently expanded in the **M4 Benchmark**. When `fair_comparison=True` is enabled on the explainability runners, every explainer operates under the same evaluation budget and explanation sparsity so that cross-module comparisons remain meaningful.

- **Compute budget:** each explanation receives a budget of **2000 forward passes**. SubgraphX consumes this through the `rollout × sample_num` product, GraphSVX sets `num_samples_override = 2000`, and TokenSHAP derives its sampling ratio such that it evaluates ≈2000 coalitions within the exponential subset space.
- **Sampling ratio:** the fair advisor converts the fixed budget into a method-specific sampling ratio that reflects the size of each explainer's combinatorial domain: `2000/(2^n-1)` for TokenSHAP, `0.25` for GraphSVX, and the implicit Monte-Carlo coverage induced by the SubgraphX rollout schedule (≈0.20–0.25).
- **Explanation sparsity:** all explainers target a **20 % word-level sparsity**, aligning tokens in LLMs with nodes in the text-graph GNNs.
- **Contextual scope:** SubgraphX uses `num_hops = local_radius = 2` to match the GNN's message-passing depth, while GraphSVX's sampling ratio covers the same two-hop neighbourhood. TokenSHAP naturally considers the full attention context.
- **Exploration granularity:** SubgraphX explores with `expand_atoms = 1`, and both GraphSVX and TokenSHAP sample atomic feature coalitions, keeping perturbations consistent across methods.
- **Energy monitoring:** All explainers run inside a [CodeCarbon](https://mlco2.github.io/codecarbon/) tracker, logging per-run energy/emissions statistics under each explanation directory (`energy_metrics.json`).

### Reported Metrics

The analytics stack now tracks both **local** (per-instance) and **global** (dataset-level) faithfulness metrics:

- **Fidelity + (sufficiency)** and **Fidelity − (necessity)**
- **Faithfulness contrast** (difference between masked and mask-out confidences)
- **Insertion/Deletion AUCs**
- **Stability** (pairwise Jaccard overlap of top‑k features)

These values are derived directly from model predictions, providing the functional faithfulness perspective advocated by GraphFramEx and M4.
