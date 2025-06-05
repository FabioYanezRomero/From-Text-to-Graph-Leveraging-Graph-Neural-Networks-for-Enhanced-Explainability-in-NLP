#!/bin/bash
# This script only handles embedding generation. Use generate_graphs.sh for graph generation.

# Set the Python path to include the parent directory
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(realpath $0)))

# Default parameters
dataset_name="stanfordnlp/sst2"
model_name=""  # Will be auto-detected if not provided
model_path=""  # Will be auto-detected if not provided
batch_size=50     # Process multiple samples at once for faster processing
chunk_size=1000  # Number of samples per chunk file
output_dir="/app/src/Clean_Code/output"
results_dir="/app/src/Clean_Code/output/finetuned_llms"
metric="f1-score"
splits="all"  # Process all available splits by default
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
    echo "  --batch_size SIZE        Batch size for processing (default: 8)"
    echo "  --chunk_size SIZE        Number of samples per chunk file (default: 1000)"
    echo "  --output_dir DIR         Output directory (default: /app/src/Clean_Code/output)"
    echo "  --splits SPLITS          Dataset splits to process (comma-separated or 'all', default: all)"
    echo "  --help                   Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --splits)
            splits="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
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
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Auto-select best model checkpoint if enabled
if [ "$auto_select_model" = true ] && [ -z "$model_path" ]; then
    echo "Finding the best checkpoint for $dataset_name based on $metric..."
    
    # Find the best checkpoint if not specified
    best_checkpoint_output=$(python /app/src/Clean_Code/Utils/find_best_checkpoint.py \
        --results_dir "$results_dir" \
        --dataset_name "$dataset_name" \
        --metric "$metric")
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to find best checkpoint"
        exit 1
    fi
    
    # Extract model path and score from output
    model_path=$(echo "$best_checkpoint_output" | grep "Best checkpoint:" | cut -d' ' -f3-)
    score=$(echo "$best_checkpoint_output" | grep "Best $metric:" | cut -d' ' -f3-)
    
    # Check if model path was found
    if [ -z "$model_path" ]; then
        echo "Error: Could not extract model path from output"
        echo "Output was: $best_checkpoint_output"
        exit 1
    fi
    
    echo "Selected best checkpoint: $model_path"
    echo "Best $metric: $score"
    
    # Extract model name from config if available
    if [ -f "$(dirname "$model_path")/config.json" ]; then
        config_model_name=$(grep -o '"model_name": *"[^"]*"' "$(dirname "$model_path")/config.json" | cut -d'"' -f4)
        if [ ! -z "$config_model_name" ]; then
            model_name="$config_model_name"
            echo "Using model name from config: $model_name"
        fi
    fi
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

# Get available splits if 'all' is specified
if [ "$splits" = "all" ]; then
    # Use Python to get available splits
    available_splits=$(python -c "from datasets import load_dataset; print(','.join(load_dataset('$dataset_name').keys()))")
    if [ $? -ne 0 ] || [ -z "$available_splits" ]; then
        echo "Error: Failed to get available splits for dataset $dataset_name"
        exit 1
    fi
    splits="$available_splits"
    echo "Processing all available splits: $splits"
fi

# Process each split
IFS=',' read -ra SPLIT_ARRAY <<< "$splits"
for split in "${SPLIT_ARRAY[@]}"; do
    echo "\nProcessing split: $split"
    split_dir="$embeddings_dir/$split"
    mkdir -p "$split_dir"
    
    python -m Clean_Code.GNN_Embeddings.embedding_generator \
        --dataset_name "$dataset_name" \
        --model_name "$model_name" \
        --model_path "$model_path" \
        --batch_size "$batch_size" \
        --chunk_size "$chunk_size" \
        --output_dir "$split_dir" \
        --split "$split" \
        --cuda
    
    # Check if embedding generation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Embedding generation failed for split $split"
        exit 1
    fi
    
    echo "Successfully generated embeddings for split $split"
done

echo "\nEmbedding generation completed successfully for all splits"
echo "Embeddings saved to $embeddings_dir"