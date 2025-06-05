"""
Test script for graph generation with pre-generated constituency trees.
"""

import os
import sys
import logging
import torch
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.Clean_Code.Graph_Generation.graph_data_processor import (
    load_embeddings,
    create_word_graphs,
    save_graphs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_graph_generation():
    """Test the graph generation pipeline with a small subset of data."""
    # Configuration
    input_dir = "/home/fabio/Documents/GitHub/From-Text-To-Graph/From-Text-to-Graph-Leveraging-Graph-Neural-Networks-for-Enhanced-Explainability-in-NLP/src/Clean_Code/output/embeddings"
    output_dir = "/home/fabio/Documents/GitHub/From-Text-To-Graph/From-Text-to-Graph-Leveraging-Graph-Neural-Networks-for-Enhanced-Explainability-in-NLP/src/Clean_Code/output/test_gnn_graphs"
    dataset = "sst2"
    split = "train"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a small subset of data (first chunk only)
    logger.info("Loading embeddings...")
    data = load_embeddings(input_dir, dataset, "stanfordnlp", split)
    
    if data is None:
        logger.error("Failed to load embeddings")
        return
        
    texts, labels, word_embeddings, sentence_embeddings = data
    
    # Use only a small subset for testing (first 10 samples)
    num_test_samples = 10
    texts = texts[:num_test_samples]
    labels = labels[:num_test_samples]
    word_embeddings = word_embeddings[:num_test_samples]
    sentence_embeddings = sentence_embeddings[:num_test_samples]
    
    logger.info(f"Testing with {len(texts)} samples")
    
    # Create graphs
    logger.info("Creating word graphs...")
    graphs = create_word_graphs(
        word_embeddings=word_embeddings,
        sentence_embeddings=sentence_embeddings,
        texts=texts,
        labels=labels,
        dataset_name=dataset,
        split=split
    )
    
    if not graphs:
        logger.error("No graphs were created")
        return
    
    # Filter out any None graphs
    graphs = [g for g in graphs if g is not None]
    logger.info(f"Successfully created {len(graphs)} graphs")
    
    # Print some statistics
    for i, graph in enumerate(graphs[:3]):  # Print info for first 3 graphs
        logger.info(f"\nGraph {i}:")
        logger.info(f"  Number of nodes: {graph.num_nodes}")
        logger.info(f"  Number of edges: {graph.edge_index.size(1) if graph.edge_index is not None else 0}")
        logger.info(f"  Node features shape: {graph.x.shape if hasattr(graph, 'x') else 'N/A'}")
        logger.info(f"  Label: {graph.y.item() if hasattr(graph, 'y') else 'N/A'}")
        logger.info(f"  Text: {graph.text[:100]}..." if hasattr(graph, 'text') and graph.text else "No text")
    
    # Save the graphs
    save_graphs(graphs, output_dir, batch_size=5, num_workers=2)
    logger.info(f"Test graphs saved to {output_dir}")

if __name__ == "__main__":
    test_graph_generation()
