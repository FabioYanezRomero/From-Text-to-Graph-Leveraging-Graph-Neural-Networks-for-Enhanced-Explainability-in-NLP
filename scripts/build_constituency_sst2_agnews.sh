#!/usr/bin/env bash
set -euo pipefail

# Build constituency trees for SST-2 and AG News.

DATASETS=("stanfordnlp/sst2" "SetFit/ag_news")
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Get available subsets dynamically for each dataset
get_subsets() {
    local dataset=$1
    python3 - <<PY
from datasets import get_dataset_split_names
import sys
try:
    splits = get_dataset_split_names("$dataset")
    print(" ".join(splits))
except Exception as e:
    # Fallback to common splits if discovery fails
    if "$dataset" == "stanfordnlp/sst2":
        print("train validation test")
    else:
        print("train test")
PY
}

# Prefer a fine-tuned model provided via env/route (tokenizer compatibility)
if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
fi

# Set model name and checkpoint path based on dataset
if [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${FINETUNED_MODEL_NAME}"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${GRAPHTEXT_MODEL_NAME}"
else
  MODEL_NAME=${MODEL_NAME:-bert-base-uncased}
  echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; using default '${MODEL_NAME}'." >&2
fi

# Set checkpoint path based on dataset
if [[ -n "${SST2_CHECKPOINT:-}" || -n "${AGNEWS_CHECKPOINT:-}" ]]; then
  echo "[info] Using local fine-tuned checkpoints for constituency parsing"
fi

# Auto-fallback to CPU if CUDA is not available to PyTorch
if [[ "$DEVICE" == cuda:* || "$DEVICE" == cuda ]]; then
  if ! python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    echo "[warn] torch.cuda.is_available()==False; falling back to CPU." >&2
    DEVICE=cpu
  fi
fi

# Process each dataset with its available subsets
for ds in "${DATASETS[@]}"; do
    echo "Building constituency for ${ds} -> ${OUT_BASE}"

    # Set dataset-specific checkpoint path
    if [[ "$ds" == "stanfordnlp/sst2" && -n "${SST2_CHECKPOINT:-}" ]]; then
        CHECKPOINT_PATH="$SST2_CHECKPOINT"
        echo "[info] Using SST-2 checkpoint: $CHECKPOINT_PATH"
    elif [[ "$ds" == "SetFit/ag_news" && -n "${AGNEWS_CHECKPOINT:-}" ]]; then
        CHECKPOINT_PATH="$AGNEWS_CHECKPOINT"
        echo "[info] Using AG News checkpoint: $CHECKPOINT_PATH"
    else
        CHECKPOINT_PATH=""
        echo "[warn] No checkpoint found for $ds, using base model"
    fi

    # Get available subsets for this dataset
    SUBSETS_STR=$(get_subsets "$ds")
    # Convert string to array
    IFS=' ' read -r -a SUBSETS <<< "$SUBSETS_STR"

    echo "Available subsets for $ds: ${SUBSETS[*]}"

    # Build the command with optional checkpoint
    CMD="python -m src.graph_builders.tree_generator \
      --graph_type constituency \
      --dataset \"${ds}\" \
      --subsets ${SUBSETS[*]} \
      --batch_size \"${BATCH_SIZE}\" \
      --device \"${DEVICE}\" \
      --output_dir \"${OUT_BASE}\" \
      --model_name \"${MODEL_NAME}\""

    # Add checkpoint if available
    if [[ -n "$CHECKPOINT_PATH" ]]; then
        CMD="$CMD --weights_path \"$CHECKPOINT_PATH\""
    fi

    echo "[info] Running command: $CMD"
    eval "$CMD"
done

echo "Done. Trees saved under ${OUT_BASE}/<dataset>/<subset>/constituency/"
