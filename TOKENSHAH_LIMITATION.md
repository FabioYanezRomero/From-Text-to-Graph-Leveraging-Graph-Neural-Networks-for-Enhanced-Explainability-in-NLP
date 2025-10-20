# TokenSHAP Limitation: Combination Explosion

## The Problem

**TokenSHAP generates ALL possible token combinations BEFORE sampling**, which causes memory issues for longer sequences.

### How TokenSHAP Works Internally:

1. **Generate all 2^N combinations** (where N = number of tokens)
2. **Then sample** from these combinations based on `sampling_ratio`
3. This is fundamentally different from GraphSVX which can directly sample

### Memory Impact:

| Tokens | Total Combinations | Memory Required | Status |
|--------|-------------------|-----------------|--------|
| 10 | 1,024 | ~8 KB | ✓ Safe |
| 16 | 65,536 | ~0.5 MB | ✓ Safe |
| 17 | 131,072 | ~1 MB | ⚠️ Borderline |
| 20 | 1,048,576 | ~8 MB | ⚠️ Risky |
| 24 | 16,777,216 | ~128 MB | ✗ Often OOM |
| 30 | 1,073,741,824 | ~8 GB | ✗ Guaranteed OOM |

## Why `sampling_ratio` Doesn't Help

Even with `sampling_ratio=0.001` (0.1%):
- **17 tokens**: Still generates 131K combinations first, then samples 131
- **24 tokens**: Still generates 16.7M combinations first, then samples 16K
- **The memory is consumed BEFORE sampling occurs**

## Current Implementation

### What the Hyperparameter Advisor Does:
```python
# For 24 tokens:
suggested_params = {
    "sampling_ratio": 0.001,           # Only 0.1%!
    "num_samples_override": 1536,      # Should limit to 1,536 samples
    "top_k_nodes": 8
}
```

### The Problem:
**TokenSHAP.analyze() doesn't accept `num_samples` parameter!**

```python
# GraphSVX (works correctly):
explainer.analyze(data, 
    sampling_ratio=0.01,
    num_samples_override=1536  # ✓ Limits total combinations
)

# TokenSHAP (limitation):
explainer.analyze(prompt,
    sampling_ratio=0.001,      # ✓ Works
    num_samples=1536           # ✗ Not supported!
)
```

### Current Behavior:

The code now:
1. **Warns** when sequences have > 2^16 combinations (may OOM)
2. **Processes anyway** (no skipping)
3. **Uses only `sampling_ratio`** (the only parameter TokenSHAP supports)

## What You'll See in Logs

```
WARNING - Sample 7: 17 tokens = 2^17 = 131,072 combinations (ratio=0.0010). 
          TokenSHAP generates all combinations before sampling - may cause OOM!
```

Then either:
- ✓ **Success**: Sample processed (took longer, used more memory)
- ✗ **OOM Error**: `numpy._core._exceptions._ArrayMemoryError: Unable to allocate X GiB`

## Recommendations

### Option 1: Accept the Limitation (Current)
- Process all samples
- Some will fail with OOM
- Those that succeed will have valid explanations

### Option 2: Pre-filter by Token Count
Modify the config to skip very long sequences:

```python
# In config.py
max_tokens: int = 16  # Limit to 2^16 = 65K combinations
```

This ensures no sample exceeds the safe threshold.

### Option 3: Use Batch Processing with Sharding
Distribute work across multiple processes/machines:

```bash
# Each shard processes different samples
# If one shard OOMs, others continue
./scripts/run_llm_sharded.sh setfit/ag_news 8
```

Failed samples in one shard don't affect others.

### Option 4: Modify TokenSHAP Library (Advanced)
Would need to modify the `token_shap` library itself to:
1. Generate combinations lazily/incrementally
2. Support `num_samples` parameter like GraphSVX
3. Sample during generation, not after

## Dataset-Specific Guidance

### SST-2 (avg 23 tokens, max 53)
- Most samples: 16-30 tokens = 65K to 1B combinations
- **Expected**: Some OOM errors on longest sequences
- **Recommendation**: Use sharding, monitor memory

### AG News (avg 50 tokens, max 126)
- Most samples: 40-80 tokens = 1T to 10^24 combinations
- **Expected**: High OOM rate
- **Recommendation**: 
  - Use `max_tokens=20` to limit to safe sequences
  - OR accept that many samples will fail
  - OR truncate sequences before processing

## Summary

**The fundamental issue**: TokenSHAP's design requires generating all combinations upfront, making it unsuitable for sequences > 16-20 tokens without memory constraints.

The hyperparameter advisor correctly suggests limiting samples, but **TokenSHAP doesn't support this parameter**. The `num_samples_override` value is calculated but cannot be used.

**Current approach**: Process everything, warn about potential OOM, let individual samples fail gracefully without stopping the entire run.


