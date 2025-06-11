#!/bin/bash
# Usage: ./run_graph_embedding_pipeline.sh --dataset_name DATASET --split SPLIT --tree_dir TREE_DIR --output_dir OUTPUT_DIR [--model_name MODEL_NAME] [--device DEVICE]

# Default values
MODEL_NAME="bert-base-uncased"
DEVICE="cuda"
DATASET_NAME="stanfordnlp/sst2"
SPLIT="validation"
TREE_DIR="/app/src/Clean_Code/output/text_trees/stanfordnlp/sst2/validation/constituency"
OUTPUT_DIR="/app/src/Clean_Code/output/gnn_embeddings/stanfordnlp/sst2/validation"
BATCH_SIZE=128

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --tree_dir)
            TREE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --dataset_name DATASET --split SPLIT --tree_dir TREE_DIR --output_dir OUTPUT_DIR [--model_name MODEL_NAME] [--device DEVICE]"
            exit 0
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$DATASET_NAME" || -z "$SPLIT" || -z "$TREE_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --dataset_name DATASET --split SPLIT --tree_dir TREE_DIR --output_dir OUTPUT_DIR [--model_name MODEL_NAME] [--device DEVICE]"
    exit 1
fi

# Run the embedding pipeline
python3 /app/src/Clean_Code/Graph_Embedding_Pipeline/generate_graphs_with_embeddings.py \
    --dataset_name "$DATASET_NAME" \
    --split "$SPLIT" \
    --tree_dir "$TREE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
