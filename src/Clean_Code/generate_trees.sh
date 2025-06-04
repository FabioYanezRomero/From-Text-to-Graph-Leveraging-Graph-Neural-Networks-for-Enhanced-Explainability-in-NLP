#!/bin/bash
# Generate Constituency Trees
# This script calls the Tree_Generation module to generate constituency parse trees from text datasets

SCRIPT_DIR="$(dirname "$0")"

# Make the script executable
chmod +x "$SCRIPT_DIR/Tree_Generation/tree_generator.py"

# Call the tree generator as a module with any arguments passed to this script
python -m "Clean_Code.Tree_Generation.tree_generator" "$@"

echo "Constituency tree generation complete!"
