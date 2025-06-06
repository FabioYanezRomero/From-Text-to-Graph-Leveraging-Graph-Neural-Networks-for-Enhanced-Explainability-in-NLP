#!/usr/bin/env python3
"""
Test script to verify graph generation with word embeddings.
This script loads a sample NPZ file and generates a graph to verify that
node features are properly assigned.
"""

import os
import sys
import logging
import numpy as np
import pickle as pkl
import torch
from torch_geometric.data import Data

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append('/app/src')
from Clean_Code.Graph_Generation.graph_data_processor import create_graph_from_tree, extract_tree_structure, create_word_graphs, load_constituency_tree

def find_sample_npz():
    """Find a sample NPZ file to test with"""
    # Direct path to a known NPZ file
    npz_path = '/app/src/Clean_Code/output/embeddings/stanfordnlp/sst2/validation/stanfordnlp_sst2/embedding_chunks/embeddings_chunk_0.npz'
    if os.path.exists(npz_path):
        return npz_path
    
    # Fallback to search
    embeddings_dir = '/app/src/Clean_Code/output/embeddings'
    for root, dirs, files in os.walk(embeddings_dir):
        for file in files:
            if file.endswith('.npz'):
                return os.path.join(root, file)
    
    return None

def find_sample_tree():
    """Find a sample tree file to test with"""
    # Direct path to a known tree file
    tree_path = '/app/src/Clean_Code/output/text_graphs/stanfordnlp/sst2/validation/constituency/0.pkl'
    if os.path.exists(tree_path):
        return tree_path
    
    # Fallback to search
    trees_dir = '/app/src/Clean_Code/output/text_graphs'
    for root, dirs, files in os.walk(trees_dir):
        for file in files:
            if file.endswith('.pkl'):
                return os.path.join(root, file)
    
    return None

def test_graph_generation():
    """Test graph generation with a sample NPZ file and tree"""
    # Find sample files
    npz_file = find_sample_npz()
    tree_file = find_sample_tree()
    
    if not npz_file:
        logger.error("No NPZ file found")
        return
    
    if not tree_file:
        logger.error("No tree file found")
        return
    
    logger.info(f"Testing with NPZ file: {npz_file}")
    logger.info(f"Testing with tree file: {tree_file}")
    
    # Load NPZ file
    chunk_data = np.load(npz_file, allow_pickle=True)
    logger.info(f"NPZ file keys: {list(chunk_data.keys())}")
    
    # Extract arrays
    word_embeddings = None
    if 'word_embeddings' in chunk_data:
        word_embeddings = chunk_data['word_embeddings']
        logger.info(f"Loaded word_embeddings with shape {word_embeddings.shape if hasattr(word_embeddings, 'shape') else 'unknown'}")
    
    # Handle different naming conventions for sentence embeddings
    sentence_embeddings = None
    if 'sentence_embeddings' in chunk_data:
        sentence_embeddings = chunk_data['sentence_embeddings']
    elif 'sent_embeddings' in chunk_data:
        sentence_embeddings = chunk_data['sent_embeddings']
    
    # Create empty texts list with the right length
    texts = []
    if word_embeddings is not None:
        texts = [""] * len(word_embeddings)
    
    # Initialize empty labels
    labels = [0] * len(word_embeddings) if word_embeddings is not None else []
    
    # Load tree file
    tree = load_constituency_tree(tree_file, sample_idx=0)
    
    if tree is None:
        logger.error("Could not extract a valid tree from the file")
        return
    
    logger.info(f"Tree type: {type(tree)}")
    
    # Extract tree structure
    node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(tree)
    logger.info(f"Tree has {len(node_features)} nodes, {len(edge_indices)} edges")
    logger.info(f"Found {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
    
    # Create graph
    if word_embeddings is not None and len(word_embeddings) > 0:
        word_embedding = word_embeddings[0]
        sentence_embedding = sentence_embeddings[0] if sentence_embeddings is not None else None
        text = texts[0] if texts else ""
        label = labels[0] if labels else 0
        
        graph_data = create_graph_from_tree(
            node_features, 
            edge_indices, 
            word_embedding, 
            sentence_embedding, 
            text, 
            label,
            leaf_nodes=leaf_nodes,
            word_nodes=word_nodes
        )
        
        logger.info(f"Created graph with node features shape: {graph_data.x.shape}")
        logger.info(f"Graph edge index shape: {graph_data.edge_index.shape}")
        
        # Verify the graph has node features
        if graph_data.x.shape[0] == 0:
            logger.error("Graph has no nodes")
        else:
            logger.info("Graph has valid node features!")
            logger.info(f"Number of nodes: {graph_data.x.shape[0]}")
            logger.info(f"Feature dimension: {graph_data.x.shape[1]}")
    else:
        logger.error("No word embeddings found")

if __name__ == "__main__":
    test_graph_generation()
