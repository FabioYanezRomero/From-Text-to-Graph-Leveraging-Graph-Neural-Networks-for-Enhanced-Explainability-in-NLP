#!/usr/bin/env bash
set -euo pipefail

# Build window graphs (word or token) for SST-2 and AG News across multiple window sizes.

SIZES=(${SIZES:-1 2 3 5 7})
DATASETS=("SetFit/ag_news" "stanfordnlp/sst2")
SUBSETS=(train validation test)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}

# Prefer a fine-tuned model provided via env/route (used by token-based builders)
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
  for k in "${SIZES[@]}"; do
    echo "Building window.word.k${k} for ${ds} -> ${OUT_BASE}"
    python -m src.graph_builders.tree_generator \
      --graph_type "window.word.k${k}" \
      --dataset "${ds}" \
      --subsets "${SUBSETS[@]}" \
      --batch_size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --output_dir "${OUT_BASE}" \
      --model_name "${MODEL_NAME}"
  done
done

echo "Done. Graphs saved under ${OUT_BASE}/<dataset>/<subset>/window.word.k<k>/"
