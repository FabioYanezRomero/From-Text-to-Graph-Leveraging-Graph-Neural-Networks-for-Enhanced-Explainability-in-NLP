#!/bin/bash
# Generate Constituency Trees
# This script calls the Tree_Generation module to generate constituency parse trees from text datasets

# Configuration Parameters
GRAPH_TYPE="constituency"  # "constituency", "syntactic", "semantic"
DATASET="stanfordnlp/sst2"   # "stanfordnlp/sst2", "SetFit/ag_news"
SUBSETS=("train")  # "train", "test", "validation"
BATCH_SIZE=2048
DEVICE="cuda:0"            # "cuda:0", "cpu"
OUTPUT_DIR="/app/src/Clean_Code/output/text_trees"

SCRIPT_DIR="$(dirname "$0")"

# Make the script executable
chmod +x "$SCRIPT_DIR/Tree_Generation/tree_generator.py"

# Run the tree generator with the configuration parameters
python -m "Clean_Code.Tree_Generation.tree_generator" \
  --graph_type "$GRAPH_TYPE" \
  --dataset "$DATASET" \
  --subsets "${SUBSETS[@]}" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT_DIR" \

echo "Constituency tree generation complete!"