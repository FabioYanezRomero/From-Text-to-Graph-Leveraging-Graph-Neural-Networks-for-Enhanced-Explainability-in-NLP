#!/bin/bash

# Script to generate GNN embeddings from fine-tuned models
# This script should be placed in the Clean_Code directory
# This script only handles embedding generation. Use generate_graphs.sh for graph generation.

# Set the Python path to include the parent directory
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(realpath $0)))

# Default parameters
dataset_name="stanfordnlp/sst2"
model_name=""  # Will be auto-detected if not provided
model_path=""  # Will be auto-detected if not provided
batch_size=16
chunk_size=50  # Size of chunks when saving embeddings to disk
max_length=128
output_dir="/app/src/Clean_Code/output"
results_dir="/app/src/Clean_Code/output/finetuned_llms"
metric="f1-score"
split="test"
auto_select_model=true

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dataset_name NAME      Dataset name (e.g., stanfordnlp/sst2, setfit/ag_news) [required]"
    echo "  --model_name NAME        Base model name (default: auto-detected from best checkpoint)"
    echo "  --model_path PATH        Path to the fine-tuned model (default: auto-selected best checkpoint)"
    echo "  --no_auto_select         Disable automatic selection of the best model checkpoint"
    echo "  --results_dir DIR        Directory containing fine-tuned model results"
    echo "                           (default: /app/src/Clean_Code/output/finetuned_llms)"
    echo "  --metric NAME            Metric to use for selecting the best model (default: f1-score)"
    echo "  --split NAME             Data split to use for evaluation (default: test)"
    echo "  --batch_size N           Batch size for embedding generation (default: 16)"
    echo "  --chunk_size N           Size of chunks when saving embeddings to disk (default: 50)"
    echo "  --max_length N           Maximum sequence length (default: 128)"
    echo "  --output_dir DIR         Output directory (default: /app/src/Clean_Code/output)"
    echo "  --help                   Show this help message"
    echo ""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset_name)
            dataset_name="$2"
            shift 2
            ;;
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            auto_select_model=false
            shift 2
            ;;
        --no_auto_select)
            auto_select_model=false
            shift
            ;;
        --results_dir)
            results_dir="$2"
            shift 2
            ;;
        --metric)
            metric="$2"
            shift 2
            ;;
        --split)
            split="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --chunk_size)
            chunk_size="$2"
            shift 2
            ;;
        --max_length)
            max_length="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$dataset_name" ]; then
    echo "Error: --dataset_name is required"
    show_help
fi

# Auto-select the best model checkpoint if enabled
if [ "$auto_select_model" = true ]; then
    echo "Finding the best checkpoint for $dataset_name based on $metric..."
    
    # Run the model selector to find the best checkpoint
    best_model_info=$(python -m Clean_Code.GNN_Embeddings.model_selector \
        --results_dir "$results_dir" \
        --dataset_name "$dataset_name" \
        --metric "$metric" \
        --split "$split")
    
    # Extract model path and name from the output
    model_path=$(echo "$best_model_info" | grep "Best model path:" | cut -d' ' -f4-)
    auto_model_name=$(echo "$best_model_info" | grep "Model name:" | cut -d' ' -f3-)
    best_score=$(echo "$best_model_info" | grep "Best $metric:" | cut -d' ' -f3-)
    
    if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
        echo "Error: Could not find a valid model checkpoint for $dataset_name"
        exit 1
    fi
    
    echo "Selected best checkpoint: $model_path"
    echo "Best $metric: $best_score"
    
    # Use auto-detected model name if not provided
    if [ -z "$model_name" ] && [ ! -z "$auto_model_name" ]; then
        model_name="$auto_model_name"
        echo "Using model name from config: $model_name"
    fi
fi

# If model_name is still empty, use a default
if [ -z "$model_name" ]; then
    model_name="google-bert/bert-base-uncased"
    echo "Using default model name: $model_name"
fi

# Check if model path exists
if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
    echo "Error: Model path is not valid or not provided"
    exit 1
fi

# Generate embeddings
echo "Generating embeddings for $dataset_name using $model_name"
echo "Using model checkpoint: $model_path"
embeddings_dir="$output_dir/embeddings/$dataset_name"
python -m Clean_Code.GNN_Embeddings.embedding_generator \
    --dataset_name "$dataset_name" \
    --model_name "$model_name" \
    --model_path "$model_path" \
    --batch_size "$batch_size" \
    --max_length "$max_length" \
    --chunk_size "$chunk_size" \
    --output_dir "$embeddings_dir"

# Check if embedding generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Embedding generation failed"
    exit 1
fi

echo "GNN embeddings generated successfully!"
echo "Embeddings saved to: $embeddings_dir"
echo "To generate graphs from these embeddings, run generate_graphs.sh"

exit 0
