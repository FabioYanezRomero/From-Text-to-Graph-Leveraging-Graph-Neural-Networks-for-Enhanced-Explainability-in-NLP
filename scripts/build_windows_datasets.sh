#!/usr/bin/env bash
set -euo pipefail

# Build word-window graphs for AG News and SST2 for multiple window sizes.

SIZES=(${SIZES:-2 3 5 7})
DATASETS=("SetFit/ag_news" "stanfordnlp/sst2")
SUBSETS=(train validation test)
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda:0}
OUT_BASE=${OUT_BASE:-outputs/graphs}
# Required by the current generator interface (used by some builders); okay to use base if not fine-tuned
MODEL_NAME=${MODEL_NAME:-bert-base-uncased}

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

