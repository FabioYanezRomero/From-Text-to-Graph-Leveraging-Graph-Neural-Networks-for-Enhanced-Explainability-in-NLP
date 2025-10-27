# LLM TokenSHAP Sharding and Progress Bar Guide

## Summary of Changes

### 1. Fixed Missing `generate()` Method
**Issue**: TokenSHAP's baseline calculation expected a `generate()` method that was missing from `HFModelWrapper`.

**Fix**: Added `generate()` method to `HFModelWrapper` class that delegates to `__call__()`.

### 2. Removed max_tokens Restriction  
**Issue**: The default `max_tokens=21` was filtering out most samples:
- **SST-2**: 53% of samples skipped (466/872)
- **AG News**: ~100% of samples skipped (all sequences > 21 tokens)

**Fix**: Increased `max_tokens` to 512 to allow processing full sequences.

### 3. Added Sample Skipping Logging
**Issue**: No visibility into how many samples were being skipped.

**Fix**: Added logging and warnings:
```
INFO - Processed 406 samples, skipped 466 samples (exceeding max_tokens=512 or empty)
WARNING - Over 50% of samples were skipped (466/872). Consider increasing max_tokens...
```

### 4. Implemented Sharding Support
**Issue**: Sequential processing was too slow for large datasets.

**Fix**: Added parallel sharding support similar to GNN explainers:
- Process every N-th sample starting from shard index
- Separate output files per shard
- Progress bars show shard information

### 5. Fixed Progress Bar Issues
**Issue**: tqdm progress bars weren't updating properly.

**Fix**: 
- Properly configured tqdm with position and dynamic settings
- Shows shard info in progress bar description
- TokenSHAP's internal "Processing combinations" bar still appears but doesn't block main progress

## Usage

### Basic Usage (Single Process)

```bash
# Process all samples sequentially
python -m src.explain.llm.main explain stanfordnlp/sst2

# Process with max samples limit
python -m src.explain.llm.main explain setfit/ag_news --max-samples 100
```

### Sharding (Parallel Processing)

#### Manual Sharding

Run multiple terminals/processes simultaneously:

```bash
# Terminal 1: Process shard 0 of 4
python -m src.explain.llm.main explain setfit/ag_news --num-shards 4 --shard-index 0

# Terminal 2: Process shard 1 of 4  
python -m src.explain.llm.main explain setfit/ag_news --num-shards 4 --shard-index 1

# Terminal 3: Process shard 2 of 4
python -m src.explain.llm.main explain setfit/ag_news --num-shards 4 --shard-index 2

# Terminal 4: Process shard 3 of 4
python -m src.explain.llm.main explain setfit/ag_news --num-shards 4 --shard-index 3
```

#### Automated Sharding (Using Script)

```bash
# Process with 4 parallel shards
./scripts/run_llm_sharded.sh setfit/ag_news 4

# With additional arguments
./scripts/run_llm_sharded.sh stanfordnlp/sst2 8 --max-samples 800
```

### Command Line Arguments

```
--num-shards N         Number of shards for parallel processing (default: 1)
--shard-index I        Index of this shard, 0-based (default: 0)
--max-samples N        Maximum samples to process per shard
--no-advisor           Disable hyperparameter advisor
--sampling-ratio R     Override sampling ratio (disables advisor)
--top-k K              Number of top tokens to highlight (default: 5)
--no-raw               Don't store raw coalition data
```

## Output Files

### Without Sharding
```
outputs/insights/LLM/stanfordnlp/sst2/
├── token_shap.json          # Summary JSON
├── token_shap.csv           # Summary CSV
├── token_shap_records.json  # Raw records JSON
└── token_shap_records.pkl   # Raw records pickle
```

### With Sharding (--num-shards 4)
```
outputs/insights/LLM/stanfordnlp/sst2/
├── token_shap_shard00of04.json
├── token_shap_shard00of04.csv
├── token_shap_shard00of04_records.json
├── token_shap_shard00of04_records.pkl
├── token_shap_shard01of04.json
├── token_shap_shard01of04.csv
├── token_shap_shard01of04_records.json
├── token_shap_shard01of04_records.pkl
├── ... (shards 2 and 3)
```

## Progress Bar Display

### Single Process
```
TokenSHAP[stanfordnlp/sst2]:  10%|███▎         | 87/872 [02:15<18:23,  1.41s/it]
Processing combinations:   45%|██████▌      | 920/2048 [00:01<00:01, 845.12it/s]
```

### Sharded Process
```
TokenSHAP[stanfordnlp/sst2][1/4]:   5%|█▌       | 11/218 [00:45<14:12,  4.12s/it]
Processing combinations:   45%|██████▌      | 920/2048 [00:01<00:01, 845.12it/s]
```

The main progress bar shows:
- Dataset name
- Shard info (if using sharding)
- Progress percentage
- Samples processed/total
- Time elapsed and remaining
- Processing speed (samples/sec)

## Hyperparameter Collection

Collect suggested hyperparameters without running full explanations:

```bash
# Collect for all samples
python -m src.explain.llm.main hyperparams stanfordnlp/sst2

# Collect with sample limit
python -m src.explain.llm.main hyperparams setfit/ag_news --max-samples 1000
```

Output: `outputs/insights/LLM/{dataset}/hyperparams/token_shap_hyperparams.json`

## Performance Tips

### 1. Use Sharding for Large Datasets
- **AG News (7,600 samples)**: Use 4-8 shards
- **SST-2 (872 samples)**: Use 2-4 shards

### 2. Limit Long Sequences
Very long sequences cause combinatorial explosion:
- 20 tokens = 1M combinations
- 30 tokens = 1B combinations (OOM risk)

The hyperparameter advisor automatically reduces sampling ratios for long sequences, but you may still want to filter:

```bash
# Limit to reasonable token counts if needed
# Note: This would require custom filtering in the code
```

### 3. Monitor Memory Usage
Each shard runs independently, so:
- 4 shards = 4x memory usage
- Watch for OOM errors
- Reduce shards if memory is limited

### 4. Recommended Configurations

**For AG News (avg 50 tokens, 7,600 samples)**:
```bash
# Conservative (4 shards, ~2 hours on GPU)
./scripts/run_llm_sharded.sh setfit/ag_news 4

# Aggressive (8 shards, ~1 hour on GPU, needs more memory)  
./scripts/run_llm_sharded.sh setfit/ag_news 8
```

**For SST-2 (avg 23 tokens, 872 samples)**:
```bash
# Standard (2 shards, ~30 min on GPU)
./scripts/run_llm_sharded.sh stanfordnlp/sst2 2

# Fast (4 shards, ~15 min on GPU)
./scripts/run_llm_sharded.sh stanfordnlp/sst2 4
```

## Troubleshooting

### Memory Errors
```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate X GiB
```
**Solution**: Sequence too long for TokenSHAP. The advisor tries to prevent this, but some sequences may still fail. Consider:
1. Reducing num_shards (less parallel memory usage)
2. Adding max_tokens limit in config
3. Filtering long sequences beforehand

### Progress Bar Not Updating
The main progress bar should update after each sample. If frozen:
1. Check terminal supports tqdm (most modern terminals do)
2. TokenSHAP's internal "Processing combinations" bar may overlap briefly
3. Use `--no-raw` to reduce I/O if disk is slow

### Shard Files Missing
If some shard files are missing:
1. Check shard process logs for errors
2. Some shards may finish faster than others
3. Use the bash script which waits for all shards

## Dataset Statistics

### SST-2 (Validation Set)
- **Total samples**: 872
- **Token count**: 2-53 tokens (avg 23.2)
- **Processed with max_tokens=512**: 872 samples (100%)
- **Sampling ratios**: 0.001-0.5 (adaptive)

### AG News (Test Set)  
- **Total samples**: 7,600
- **Token count**: 16-126 tokens (avg 50.5)
- **Processed with max_tokens=512**: 7,600 samples (100%)
- **Sampling ratios**: ~0.0011 (very low due to longer sequences)
- **Multi-class**: 4 classes (vs SST-2's 2)

## Implementation Details

### Sharding Algorithm
```python
# Each shard processes every num_shards-th sample starting from shard_index
for idx, entry in enumerate(dataset):
    if idx % num_shards == shard_index:
        process(entry)
```

Example with 4 shards:
- Shard 0: samples 0, 4, 8, 12, ...
- Shard 1: samples 1, 5, 9, 13, ...
- Shard 2: samples 2, 6, 10, 14, ...
- Shard 3: samples 3, 7, 11, 15, ...

### Progress Bar Configuration
```python
tqdm(
    dataset,
    desc=f"TokenSHAP[{dataset}][{shard+1}/{num_shards}]",
    position=shard_index,
    leave=True,
    dynamic_ncols=True,
)
```

## Files Modified

1. `src/explain/llm/config.py`
   - Added `num_shards` and `shard_index` to `LLMExplainerRequest`
   - Increased `max_tokens` from 21 to 512

2. `src/explain/llm/token_shap_runner.py`
   - Implemented sharding logic
   - Fixed progress bar configuration
   - Added skip counting and logging
   - Updated output filenames for shards

3. `src/explain/llm/main.py`
   - Added `--num-shards` and `--shard-index` CLI arguments
   - Updated function signatures

4. `scripts/run_llm_sharded.sh`
   - New script for automated parallel sharding

## Next Steps

To further optimize:
1. **GPU batching**: Process multiple samples per batch (requires TokenSHAP changes)
2. **Distributed processing**: Use multiple GPUs/machines
3. **Checkpoint/resume**: Save progress and resume from last checkpoint
4. **Dynamic sharding**: Redistribute work if some shards finish early



