#!/usr/bin/env python3
"""
Analyze a batch of tree files to understand their structure and content.
This will help us diagnose why some trees are not being processed correctly.
"""

import os
import sys
import logging
import pickle as pkl
import numpy as np
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append('/app/src')
from Clean_Code.Graph_Generation.graph_data_processor import extract_tree_structure

def analyze_tree_file(tree_file_path):
    """Analyze a single tree file"""
    logger.info(f"Analyzing tree file: {tree_file_path}")
    
    try:
        with open(tree_file_path, 'rb') as f:
            tree_data = pkl.load(f)
        
        logger.info(f"Tree data type: {type(tree_data)}")
        
        # Analyze the structure based on the type
        if isinstance(tree_data, list):
            logger.info(f"List length: {len(tree_data)}")
            
            # Analyze each item in the list
            for i, item in enumerate(tree_data[:5]):  # Limit to first 5 items
                logger.info(f"Item {i} type: {type(item)}")
                
                if isinstance(item, tuple) and len(item) > 0:
                    logger.info(f"  Tuple length: {len(item)}")
                    
                    # Analyze the first element (should be the tree)
                    tree = item[0]
                    logger.info(f"  Tree type: {type(tree)}")
                    
                    # If it's a NetworkX DiGraph, analyze it
                    if isinstance(tree, nx.DiGraph):
                        logger.info(f"  NetworkX DiGraph with {len(tree.nodes)} nodes and {len(tree.edges)} edges")
                        
                        # Extract tree structure
                        node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(tree)
                        logger.info(f"  Extracted {len(node_features)} nodes, {len(edge_indices)} edges")
                        logger.info(f"  Found {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
                        
                        # Check if word nodes are numeric
                        numeric_word_nodes = []
                        for node in word_nodes:
                            try:
                                if isinstance(node, int) or (isinstance(node, str) and node.isdigit()):
                                    numeric_word_nodes.append(node)
                            except Exception:
                                pass
                        
                        logger.info(f"  Found {len(numeric_word_nodes)} numeric word nodes")
                        
                        # Check node features
                        pos_tags = set()
                        for node_id, feature in node_features.items():
                            if isinstance(feature, str) and len(feature) <= 5:  # Likely a POS tag
                                pos_tags.add(feature)
                        
                        logger.info(f"  POS tags found: {pos_tags}")
                    
                    # If it's a list, it might be a custom tree format
                    elif isinstance(tree, list):
                        logger.info(f"  List with {len(tree)} elements")
                        
                        # Examine the first few elements to understand the structure
                        for j, elem in enumerate(tree[:5]):
                            logger.info(f"    Element {j} type: {type(elem)}")
                            
                            # If it's a NetworkX DiGraph, this is what we're looking for
                            if isinstance(elem, nx.DiGraph):
                                logger.info(f"    Found NetworkX DiGraph with {len(elem.nodes)} nodes and {len(elem.edges)} edges")
                                
                                # Extract tree structure from this individual tree
                                try:
                                    node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(elem)
                                    logger.info(f"    Extracted {len(node_features)} nodes, {len(edge_indices)} edges")
                                    logger.info(f"    Found {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
                                    
                                    # This is a valid tree, we can stop here
                                    break
                                except Exception as e:
                                    logger.error(f"    Error extracting tree structure: {str(e)}")
                        
                        # If we didn't find a valid tree in the first few elements, try the generic extraction
                        try:
                            node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(tree)
                            logger.info(f"  Extracted {len(node_features)} nodes, {len(edge_indices)} edges")
                            logger.info(f"  Found {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
                        except Exception as e:
                            logger.error(f"  Error extracting tree structure: {str(e)}")
                            
                        # If the tree is a list of trees, we need to modify our approach
                        if len(tree) > 0 and isinstance(tree[0], nx.DiGraph):
                            logger.info("  This appears to be a list of NetworkX DiGraphs (individual trees)")
                            logger.info("  This is the structure we need to handle in our graph generation code")
                            
                            # Count how many trees have valid structure
                            valid_trees = 0
                            for j, subtree in enumerate(tree[:10]):  # Check first 10 trees
                                if isinstance(subtree, nx.DiGraph):
                                    try:
                                        node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(subtree)
                                        if len(node_features) > 1 and len(edge_indices) > 0 and len(word_nodes) > 0:
                                            valid_trees += 1
                                    except Exception:
                                        pass
                            
                            logger.info(f"  Found {valid_trees} valid trees out of 10 checked")
                            
                            # Sample the first valid tree for detailed analysis
                            for j, subtree in enumerate(tree):
                                if isinstance(subtree, nx.DiGraph):
                                    try:
                                        node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(subtree)
                                        if len(node_features) > 1 and len(edge_indices) > 0 and len(word_nodes) > 0:
                                            logger.info(f"  Sample tree {j} has {len(node_features)} nodes, {len(edge_indices)} edges")
                                            logger.info(f"  Sample tree {j} has {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
                                            
                                            # Check node features
                                            pos_tags = set()
                                            for node_id, feature in node_features.items():
                                                if isinstance(feature, str) and len(feature) <= 5:  # Likely a POS tag
                                                    pos_tags.add(feature)
                                            
                                            logger.info(f"  POS tags found: {pos_tags}")
                                            break
                                    except Exception:
                                        pass
                    
                    # If it's a dictionary, it might be a custom tree format
                    elif isinstance(tree, dict):
                        logger.info(f"  Dictionary with {len(tree)} keys")
                        logger.info(f"  Keys: {list(tree.keys())}")
                        
                        # Try to extract tree structure
                        try:
                            node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(tree)
                            logger.info(f"  Extracted {len(node_features)} nodes, {len(edge_indices)} edges")
                            logger.info(f"  Found {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
                        except Exception as e:
                            logger.error(f"  Error extracting tree structure: {str(e)}")
        
        elif isinstance(tree_data, dict):
            logger.info(f"Dictionary with {len(tree_data)} keys")
            logger.info(f"Keys: {list(tree_data.keys())}")
        
        elif isinstance(tree_data, nx.DiGraph):
            logger.info(f"NetworkX DiGraph with {len(tree_data.nodes)} nodes and {len(tree_data.edges)} edges")
            
            # Extract tree structure
            node_features, edge_indices, leaf_nodes, word_nodes = extract_tree_structure(tree_data)
            logger.info(f"Extracted {len(node_features)} nodes, {len(edge_indices)} edges")
            logger.info(f"Found {len(leaf_nodes)} leaf nodes and {len(word_nodes)} word nodes")
        
        else:
            logger.warning(f"Unknown tree data type: {type(tree_data)}")
    
    except Exception as e:
        logger.error(f"Error analyzing tree file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def analyze_tree_batch():
    """Analyze a batch of tree files"""
    # Path to constituency trees
    trees_dir = '/app/src/Clean_Code/output/text_graphs/stanfordnlp/sst2/validation/constituency'
    
    if not os.path.exists(trees_dir):
        logger.error(f"Tree directory not found: {trees_dir}")
        return
    
    # Get all tree files
    tree_files = [os.path.join(trees_dir, f) for f in os.listdir(trees_dir) if f.endswith('.pkl')]
    logger.info(f"Found {len(tree_files)} tree files")
    
    # Analyze the first few files
    for tree_file in tree_files[:3]:  # Limit to first 3 files
        analyze_tree_file(tree_file)

if __name__ == "__main__":
    analyze_tree_batch()
