# TokenSHAP Fair Comparison Mode

## Overview

The TokenSHAP explainer now supports a **fair comparison mode** that ensures consistent computational budgets across sequences of different lengths. This enables valid comparisons between LLM (TokenSHAP) and GNN (GraphSVX/SubgraphX) explainability methods.

## Default Configuration

- **Target Forward Passes**: 2000 (default)
- **Sampling Strategy**: Formula-based, token-count adaptive
- **Formula**: `sampling_ratio = target_samples / 2^num_tokens`

## Usage

### Command Line

```bash
# Use fair comparison with default 2000 forward passes
python -m src.explain.llm.main explain 'setfit/ag_news' --fair

# Override target forward passes
python -m src.explain.llm.main explain 'stanfordnlp/sst2' --fair --target-forward-passes 5000

# Collect hyperparameters in fair mode
python -m src.explain.llm.main collect-hyperparams 'setfit/ag_news' --fair --target-forward-passes 2000
```

### Docker (via run_all_explainers.sh)

The script automatically passes `--fair` flag to TokenSHAP:

```bash
bash run_all_explainers.sh
```

## How It Works

### Without Fair Mode
- Uses adaptive sampling based on token count
- Shorter sequences: higher sampling ratio (up to 0.5)
- Longer sequences: lower sampling ratio (down to 0.001)
- **Problem**: Inconsistent computational budgets

### With Fair Mode (--fair)
- Fixed computational budget: 2000 forward passes per sample
- Sampling ratio computed as: `2000 / 2^num_tokens`
- **Example**:
  - 5 tokens: ratio = 2000/32 = 62.5 → capped at 1.0 (all combos)
  - 10 tokens: ratio = 2000/1024 ≈ 1.95 → capped at 1.0
  - 15 tokens: ratio = 2000/32768 ≈ 0.061
  - 20 tokens: ratio = 2000/1048576 ≈ 0.0019

### Clamping
- Minimum ratio: 0.001
- Maximum ratio: 1.0
- If `2^num_tokens < target_samples`, all combinations are sampled

## Integration with GNN Explainers

Both GraphSVX and SubgraphX can also use the `--fair` flag to ensure:
1. **Consistent computational budgets** across samples
2. **Cross-method comparability** (GNN vs LLM)
3. **Token/node position alignment** in insights

## Output Compatibility

TokenSHAP outputs are fully compatible with GNN insights:
- Same schema: `top_nodes`, `top_tokens`, `node_importance`, `insertion_auc`
- Same faithfulness metrics: `masked_confidence`, `maskout_confidence`, `sparsity`
- Same directory structure: `outputs/insights/LLM/<backbone>/<dataset>/`

## Configuration Files

### src/explain/llm/config.py
```python
TOKEN_SHAP_DEFAULTS: Dict[str, float] = {
    "sampling_ratio": 0.1,
    "min_samples": 50,
    "max_samples": 2048,
    "target_forward_passes": 2000,  # Default for fair mode
}
```

### src/explain/llm/fair_sampling.py
Contains the formula implementation:
```python
def compute_fair_sampling_ratio(
    num_tokens: int,
    *,
    target_samples: int = 2000,  # Configurable
    min_ratio: float = 0.001,
    max_ratio: float = 1.0,
) -> float:
    max_combinations = 2**num_tokens
    if max_combinations <= target_samples:
        return max_ratio
    ratio = target_samples / max_combinations
    return max(min_ratio, min(ratio, max_ratio))
```

## Comparison with GNN Methods

| Method | Fair Mode | Computational Budget | Sampling Strategy |
|--------|-----------|---------------------|-------------------|
| **TokenSHAP** | `--fair` | 2000 forward passes | Formula: `2000 / 2^n` |
| **GraphSVX** | `--fair` | 2000 samples | Adaptive SHAP sampling |
| **SubgraphX** | `--fair` | Controlled rollouts | MCTS with budget |

## Analytics Compatibility

The fair comparison mode ensures outputs are directly comparable in:

### 1. Faithfulness Analytics
- `insertion_auc`: Area under insertion curve
- `masked_confidence`: Confidence with top-k tokens only
- `maskout_confidence`: Confidence without top-k tokens
- `fidelity_plus/minus`: Perturbation-based faithfulness

### 2. Embedding Analytics
- Token-level embedding geometry
- Class-wise centroids
- Correct vs incorrect separability

### 3. Predictive-Behavior Analytics
- Confidence correlation matrices
- Score distributions per class
- Label-wise performance

## Notes

1. **Memory Considerations**: TokenSHAP generates all combinations before sampling. For sequences >16 tokens, this can be memory-intensive even with low sampling ratios.

2. **Recommended Token Limits**: Set `max_tokens=21` (default) to avoid OOM errors. Sequences longer than 21 tokens are skipped.

3. **Energy Monitoring**: Energy metrics are automatically captured and saved to `energy_metrics.json`.

4. **Sharding Support**: Use `--num-shards` and `--shard-index` for parallel processing of large datasets.

## References

- GraphSVX fair comparison: `src/explain/gnn/graphsvx/main.py`
- SubgraphX fair comparison: `src/explain/gnn/subgraphx/main.py`
- Fair multimodal advisor: `src/explain/common/fairness.py`

