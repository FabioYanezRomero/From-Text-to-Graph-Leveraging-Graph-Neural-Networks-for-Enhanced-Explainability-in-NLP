#!/usr/bin/env bash
set -euo pipefail

# Build syntactic dependency trees for SST-2 and AG News.

DATASETS=("stanfordnlp/sst2" "SetFit/ag_news")
SUBSETS=(train validation test)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Prefer a fine-tuned model provided via env/route
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

if [[ -n "${FINETUNED_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${FINETUNED_MODEL_NAME}"
elif [[ -n "${GRAPHTEXT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${GRAPHTEXT_MODEL_NAME}"
else
  MODEL_NAME=${MODEL_NAME:-bert-base-uncased}
  echo "[warn] FINETUNED_MODEL_NAME/GRAPHTEXT_MODEL_NAME not set; using default '${MODEL_NAME}'." >&2
fi

for ds in "${DATASETS[@]}"; do
  echo "Building syntactic for ${ds} -> ${OUT_BASE}"
  python -m src.graph_builders.tree_generator \
    --graph_type syntactic \
    --dataset "${ds}" \
    --subsets "${SUBSETS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output_dir "${OUT_BASE}" \
    --model_name "${MODEL_NAME}"
done

echo "Done. Trees saved under ${OUT_BASE}/<dataset>/<subset>/syntactic/"
