#!/usr/bin/env python3
"""
Script to verify that the graph generation fixes have resolved the empty node features issue.
"""

import os
import sys
import pickle as pkl
import numpy as np
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def verify_graphs(graph_dir):
    """
    Verify that all graphs in the directory have valid node features.
    
    Args:
        graph_dir: Directory containing graph pickle files
    
    Returns:
        dict: Statistics about the graphs
    """
    stats = {
        'total_graphs': 0,
        'valid_graphs': 0,
        'empty_features': 0,
        'node_counts': [],
        'feature_dims': []
    }
    
    # Find all graph files
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pkl')]
    logger.info(f"Found {len(graph_files)} graph files in {graph_dir}")
    
    # Process each file
    for graph_file in tqdm(graph_files, desc="Verifying graphs"):
        graph_path = os.path.join(graph_dir, graph_file)
        
        try:
            # Load the graph data
            with open(graph_path, 'rb') as f:
                graphs = pkl.load(f)
            
            # Process each graph in the file
            for graph in graphs:
                stats['total_graphs'] += 1
                
                # Check if node features are valid
                if hasattr(graph, 'x') and graph.x is not None:
                    if graph.x.shape[0] > 0:
                        stats['valid_graphs'] += 1
                        stats['node_counts'].append(graph.x.shape[0])
                        stats['feature_dims'].append(graph.x.shape[1])
                    else:
                        stats['empty_features'] += 1
                else:
                    stats['empty_features'] += 1
        
        except Exception as e:
            logger.error(f"Error processing {graph_file}: {str(e)}")
    
    # Calculate percentages
    if stats['total_graphs'] > 0:
        stats['valid_percent'] = (stats['valid_graphs'] / stats['total_graphs']) * 100
        stats['empty_percent'] = (stats['empty_features'] / stats['total_graphs']) * 100
    
    # Calculate average node count and feature dimension
    if stats['node_counts']:
        stats['avg_nodes'] = np.mean(stats['node_counts'])
        stats['min_nodes'] = np.min(stats['node_counts'])
        stats['max_nodes'] = np.max(stats['node_counts'])
    
    if stats['feature_dims']:
        stats['avg_dim'] = np.mean(stats['feature_dims'])
    
    return stats

def main():
    # Define the graph directories
    dataset_name = "ag_news"
    splits = ["train", "test", "val"]
    base_dir = "/app/data/graphs"
    
    # Verify graphs for each split
    for split in splits:
        graph_dir = os.path.join(base_dir, dataset_name, split)
        
        if os.path.exists(graph_dir):
            logger.info(f"Verifying graphs in {graph_dir}")
            stats = verify_graphs(graph_dir)
            
            # Print statistics
            logger.info(f"Statistics for {split} split:")
            logger.info(f"Total graphs: {stats['total_graphs']}")
            logger.info(f"Valid graphs: {stats['valid_graphs']} ({stats.get('valid_percent', 0):.2f}%)")
            logger.info(f"Empty features: {stats['empty_features']} ({stats.get('empty_percent', 0):.2f}%)")
            
            if 'avg_nodes' in stats:
                logger.info(f"Average node count: {stats['avg_nodes']:.2f} (min: {stats['min_nodes']}, max: {stats['max_nodes']})")
            
            if 'avg_dim' in stats:
                logger.info(f"Average feature dimension: {stats['avg_dim']:.2f}")
            
            logger.info("-" * 50)
        else:
            logger.warning(f"Directory {graph_dir} does not exist")

if __name__ == "__main__":
    main()
