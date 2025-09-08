#!/usr/bin/env bash
set -euo pipefail

# Build syntactic dependency trees for AG News using the registry-based generator.

DATASET="SetFit/ag_news"
SUBSETS=(train test)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}
# Required by the current generator interface, not used by syntactic itself
MODEL_NAME=${MODEL_NAME:-bert-base-uncased}

echo "Building syntactic trees for ${DATASET} -> ${OUT_BASE}"
python -m src.graph_builders.tree_generator \
  --graph_type syntactic \
  --dataset "${DATASET}" \
  --subsets "${SUBSETS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output_dir "${OUT_BASE}" \
  --model_name "${MODEL_NAME}"

echo "Done. Trees saved under ${OUT_BASE}/${DATASET}/<subset>/syntactic/"

