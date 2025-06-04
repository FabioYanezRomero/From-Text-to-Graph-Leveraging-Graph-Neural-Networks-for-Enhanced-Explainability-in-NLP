"""
Graph Processor Main

This module provides the main entry point for the graph processor.
"""

import os
import logging
import argparse
from src.Clean_Code.Graph_Generation.graph_data_processor import process_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_dataset_graphs(dataset_name, batch_size=10, window_size=3, edge_type="window"):
    """
    Process embeddings for a dataset to create PyTorch Geometric graphs
    
    Args:
        dataset_name: Name of the dataset (e.g., 'sst2', 'ag_news')
        batch_size: Batch size for GNN data processing
        window_size: Window size for creating edges in graphs
        edge_type: Type of edges to create (window, fully_connected)
        
    Returns:
        output_dir: Directory containing the processed graphs
    """
    # Determine embeddings directory
    embeddings_dir = f"/app/src/Clean_Code/output/embeddings/{dataset_name}"
    
    if not os.path.exists(embeddings_dir):
        logger.error(f"Embeddings directory {embeddings_dir} does not exist")
        return None
    
    # Process embeddings
    output_dir = process_embeddings(
        dataset_name=dataset_name,
        embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        window_size=window_size,
        edge_type=edge_type
    )
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embeddings to create PyTorch Geometric graphs")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., sst2, ag_news)")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for GNN data processing")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for creating edges in graphs")
    parser.add_argument("--edge_type", type=str, default="window", choices=["window", "fully_connected"],
                        help="Type of edges to create")
    parser.add_argument("--embeddings_dir", type=str, default=None, 
                        help="Custom embeddings directory (default: /app/src/Clean_Code/output/embeddings/{dataset_name})")
    
    args = parser.parse_args()
    
    # Use custom embeddings directory if provided
    embeddings_dir = args.embeddings_dir
    if embeddings_dir is None:
        embeddings_dir = f"/app/src/Clean_Code/output/embeddings/{args.dataset_name}"
    
    # Process embeddings
    output_dir = process_embeddings(
        dataset_name=args.dataset_name,
        embeddings_dir=embeddings_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        edge_type=args.edge_type
    )
    
    if output_dir:
        logger.info(f"Graph processing completed successfully. Graphs saved to: {output_dir}")
    else:
        logger.error("Graph processing failed")
