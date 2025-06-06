#!/bin/bash

# Script to regenerate graphs after fixing the graph generation code
# This will use the existing embeddings to create new graphs with proper node features

# First, backup the existing graphs
BACKUP_DIR="/app/src/Clean_Code/output/embeddings/graphs_backup_$(date +%Y%m%d_%H%M%S)"
GRAPHS_DIR="/app/src/Clean_Code/output/embeddings/graphs"

if [ -d "$GRAPHS_DIR" ]; then
  echo "Backing up existing graphs to $BACKUP_DIR"
  mkdir -p "$BACKUP_DIR"
  cp -r "$GRAPHS_DIR"/* "$BACKUP_DIR"/
fi

# Regenerate the graphs for ag_news dataset
echo "Regenerating graphs for ag_news dataset..."

# Run the graph generation script with the fixed code
bash /app/src/Clean_Code/generate_graphs.sh \
  --dataset_name "setfit/ag_news" \
  --embeddings_dir "/app/src/Clean_Code/output/embeddings/setfit/ag_news" \
  --batch_size 200 \
  --edge_type "constituency" \
  --label_source "llm"

echo "Graph regeneration complete!"

# Verify the generated graphs
echo "Verifying generated graphs..."
python /app/verify_graphs.py --dataset "setfit/ag_news" --split "train"

# If you want to verify other splits as well
python /app/verify_graphs.py --dataset "setfit/ag_news" --split "val"
python /app/verify_graphs.py --dataset "setfit/ag_news" --split "test"

echo "Graph verification complete!"
