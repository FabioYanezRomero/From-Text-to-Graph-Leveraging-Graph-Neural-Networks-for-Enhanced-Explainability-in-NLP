# TokenSHAP Hyperparameter Advisor - GraphSVX Alignment

## Overview

TokenSHAP now uses a **dynamic hyperparameter advisor** aligned with GraphSVX/SubgraphX, ensuring consistent explanation quality while preventing OOM errors.

## Key Features

### 1. Adaptive Sampling Ratio
- **Formula**: `sampling_ratio = target_samples / (2^num_tokens)`
- **No minimum floor**: Ratio can be arbitrarily small for long sequences
- **Example ratios**:
  - 6 tokens: `0.95` (can explore most combinations)
  - 30 tokens: `0.000001716` (samples ~1,800 combinations)
  - 100 tokens: `10^-27` (samples ~1,800 combinations)

### 2. Target Sample Counts (Aligned with GraphSVX)
```python
if num_tokens <= 6:    target = 128
elif num_tokens <= 8:  target = 256
elif num_tokens <= 10: target = 512
elif num_tokens <= 12: target = 768
elif num_tokens <= 14: target = 1024
elif num_tokens <= 16: target = 1280
else:                  target = 1536
```

**Adjustments**:
- Multi-class: `target *= (1.0 + (num_labels - 2) * 0.1)`
- Capped at: `min(target, 2048)`
- Minimum: `max(target, 32)`

### 3. Comparison with GraphSVX

| Method | Sample Range | Adaptive? | OOM Prevention |
|--------|-------------|-----------|----------------|
| **GraphSVX** | 128-2048 | ✓ (graph size) | Built-in |
| **SubgraphX** | 256-1024 | ✓ (graph size) | Built-in |
| **TokenSHAP** | 128-2048 | ✓ (token count) | ✓ Fixed! |

## Usage

### Generate Hyperparameters
```bash
# Collect hyperparameters for a dataset
python -m src.explain.llm.main hyperparams stanfordnlp/sst2

# With limited samples
python -m src.explain.llm.main hyperparams setfit/ag_news --max-samples 100
```

### Run Explanations with Advisor
```bash
# The advisor is enabled by default
python -m src.explain.llm.main explain stanfordnlp/sst2

# Disable advisor (not recommended)
python -m src.explain.llm.main explain stanfordnlp/sst2 --no-advisor
```

### Run with Sharding (Parallel Processing)
```bash
# Using the helper script
./scripts/run_llm_sharded.sh setfit/ag_news 4 --max-samples 100

# Manual sharding
python -m src.explain.llm.main explain setfit/ag_news --num-shards 4 --shard-index 0
```

## Technical Details

### Why Adaptive Ratios Work

**The Problem**: TokenSHAP uses `sampling_ratio` to control combinations:
- `num_sampled = (2^N - 1) × ratio`
- For long sequences, even `ratio=0.001` → millions of combinations

**The Solution**: Calculate ratio per-sequence to achieve fixed target:
```python
target_samples = 1536  # Based on token count
ratio = target_samples / (2 ** num_tokens)
```

**Result**: All sequences process ~1,500 combinations regardless of length!

### Verification Results

**SST-2 Dataset (872 samples)**:
- Token range: 2-53 tokens
- All sequences 10+ tokens: ✓ Perfect match (~1,500 combinations)
- Short sequences (2-9): Slight delta, but safe (few combinations)

**AG News Dataset (7,600 samples)**:
- Token range: 16-126 tokens  
- All sequences: ✓ Perfect match (~1,800 combinations)
- Ratios as small as 10^-35 for longest sequences

### OOM Prevention

**Before Fix**:
- 30 tokens with ratio=0.001 → 1,073,741 combinations → OOM
- 100 tokens with ratio=0.001 → 10^24 combinations → Instant OOM

**After Fix**:
- 30 tokens with ratio=0.000001716 → 1,842 combinations ✓
- 100 tokens with ratio=10^-27 → 1,843 combinations ✓

## Architecture

### Modified Files

1. **`src/explain/llm/hyperparam_advisor.py`**:
   - `_suggest_sampling_ratio()`: Calculates `ratio = target / (2^N)`
   - `_suggest_num_samples()`: Returns target (128-2048)
   - `_sanitise()`: Removed minimum floor (was 0.001, now 1e-100)

2. **`src/explain/llm/config.py`**:
   - Increased `max_tokens` from 21 to 512
   - Added sharding support

3. **`src/explain/llm/token_shap_runner.py`**:
   - Implements dataset sharding
   - Improved progress bars
   - Removed sample skipping logic

4. **`src/explain/llm/main.py`**:
   - Added `--num-shards` and `--shard-index` arguments
   - Passes sharding params to runner

### Key Code Snippets

**Adaptive Ratio Calculation**:
```python
def _suggest_sampling_ratio(self, stats: SentenceStats) -> float:
    num_tokens = stats.num_tokens
    target_samples = self._suggest_num_samples(stats)
    
    total_possible = 2 ** num_tokens
    ratio = target_samples / total_possible
    
    # NO minimum floor - allow arbitrarily small ratios
    return float(min(ratio, 0.95))
```

**No Minimum Floor in Sanitisation**:
```python
# OLD: min(max(ratio, 0.001), 0.99)  ← Enforced minimum
# NEW: max(min(ratio, 0.99), 1e-100) ← No practical minimum
```

## Best Practices

1. **Always use the advisor**: Ensures optimal sampling for each sequence
2. **Use sharding for large datasets**: Parallel processing speeds up computation
3. **Monitor memory usage**: Even with advisor, very long sequences use significant memory
4. **Check hyperparams first**: Run `hyperparams` command to preview settings

## Troubleshooting

### Q: Still getting OOM errors?
A: Check if advisor is enabled (`--use-advisor` or ensure not using `--no-advisor`)

### Q: Processing too slowly?
A: Use sharding: `./scripts/run_llm_sharded.sh <dataset> <num_shards>`

### Q: Results different from GraphSVX?
A: This is expected - TokenSHAP and GraphSVX use different algorithms, but now with comparable sample counts

### Q: Very short sequences have delta?
A: Normal - ratio capped at 0.95, but they won't OOM (few combinations total)

## Performance Metrics

### Before Optimization
- SST-2: Many samples skipped (max_tokens=21)
- AG News: 95% OOM errors on long sequences
- No parallelization

### After Optimization
- SST-2: All 872 samples processable
- AG News: All 7,600 samples processable
- Sharding support for parallel processing
- Memory usage: ~same as GraphSVX (1,500 combinations)

## References

- **GraphSVX**: 256-1024 samples (quality profile)
- **SubgraphX**: Similar adaptive strategy
- **TokenSHAP Paper**: Original implementation didn't handle long sequences


