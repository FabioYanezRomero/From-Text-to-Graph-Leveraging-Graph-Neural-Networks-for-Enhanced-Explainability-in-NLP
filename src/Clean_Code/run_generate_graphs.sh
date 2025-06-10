#!/bin/bash
# Script to generate PyTorch geometric graphs for all splits in the structure expected by the GNN training script

EMBEDDINGS_DIR="/app/src/Clean_Code/output/embeddings/stanfordnlp/sst2/train"
DATASET="stanfordnlp/sst2"
EMBEDDING_MODEL="stanfordnlp_sst2"
SPECIAL_EMB_DIR="/app/src/Clean_Code/output/embeddings/stanfordnlp/sst2/train/special_embeddings/google-bert_bert-base-uncased/"
LLM_DIR="/app/src/Clean_Code/output/finetuned_llms/stanfordnlp"
MODALITY="constituency"
BATCH_SIZE=128
OUTPUT_DIR="/app/src/Clean_Code/output/pytorch_geometric"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/Graph_Generation/main.py"

for SPLIT in validation; do
    if [ "$SPLIT" = "validation" ]; then
        SPLIT_OUT="validation_llm_labels"
    else
        SPLIT_OUT="${SPLIT}_llm_labels"
    fi
    python "$PYTHON_SCRIPT" \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --dataset "$DATASET" \
        --embedding_model "$EMBEDDING_MODEL" \
        --split "$SPLIT" \
        --special_emb_dir "$SPECIAL_EMB_DIR" \
        --llm_dir "$LLM_DIR" \
        --modality "$MODALITY" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --split_out "$SPLIT_OUT"
done
