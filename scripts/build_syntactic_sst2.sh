#!/usr/bin/env bash
set -euo pipefail

# Build syntactic dependency trees for SST-2 using the registry-based generator.

DATASET="stanfordnlp/sst2"
# Auto-detect all splits available in the dataset (e.g., train validation test)
mapfile -t SUBSETS < <(python - <<'PY'
from datasets import get_dataset_split_names
import sys
name = "stanfordnlp/sst2"
try:
    names = get_dataset_split_names(name)
    print("\n".join(names))
except Exception as e:
    # Fallback to common splits if discovery fails
    print("train\nvalidation\ntest")
PY
)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Model selection is not needed for syntactic tree building (uses Stanza).
# Keep env sourcing for parity but avoid printing confusing BERT warnings.
if [ -f "scripts/models.env" ]; then
  # shellcheck disable=SC1091
  source scripts/models.env
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

echo "Building syntactic trees for ${DATASET} -> ${OUT_BASE}"
python -m src.graph_builders.tree_generator \
  --graph_type syntactic \
  --dataset "${DATASET}" \
  --subsets "${SUBSETS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output_dir "${OUT_BASE}"

echo "Done. Trees saved under ${OUT_BASE}/${DATASET}/<subset>/syntactic/"
