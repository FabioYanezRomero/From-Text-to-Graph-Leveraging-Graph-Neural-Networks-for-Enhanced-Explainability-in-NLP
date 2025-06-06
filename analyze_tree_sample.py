import os
import pickle
import networkx as nx
import argparse
from collections import Counter

def analyze_tree_sample(tree_file_path):
    """Analyze a sample tree file to understand its structure"""
    print(f"Analyzing tree file: {tree_file_path}")
    
    try:
        with open(tree_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it's a batch file or individual tree
        if isinstance(data, list):
            print("File contains a batch of trees")
            if len(data) > 0 and isinstance(data[0], tuple) and len(data[0]) == 2:
                trees, labels = data[0]
                print(f"Found {len(trees)} trees in the first batch tuple")
                tree = trees[0]  # Analyze the first tree
            else:
                print(f"Unexpected batch format: {type(data[0])}")
                return
        else:
            tree = data
            print("File contains a single tree")
        
        # Analyze the tree structure
        print(f"Tree type: {type(tree)}")
        
        if isinstance(tree, nx.DiGraph):
            print("\nAnalyzing NetworkX DiGraph tree:")
            print(f"Number of nodes: {tree.number_of_nodes()}")
            print(f"Number of edges: {tree.number_of_edges()}")
            
            # Check node attributes
            node_attrs = {}
            for node, attrs in tree.nodes(data=True):
                for key in attrs:
                    node_attrs[key] = node_attrs.get(key, 0) + 1
            
            print("\nNode attributes:")
            for attr, count in node_attrs.items():
                print(f"  - {attr}: present in {count} nodes")
            
            # Check edge attributes
            edge_attrs = {}
            for u, v, attrs in tree.edges(data=True):
                for key in attrs:
                    edge_attrs[key] = edge_attrs.get(key, 0) + 1
            
            print("\nEdge attributes:")
            for attr, count in edge_attrs.items():
                print(f"  - {attr}: present in {count} edges")
            
            # Check root and leaf nodes
            roots = [n for n, d in tree.in_degree() if d == 0]
            leaves = [n for n, d in tree.out_degree() if d == 0]
            
            print(f"\nRoot nodes ({len(roots)}): {roots[:5]}{'...' if len(roots) > 5 else ''}")
            print(f"Leaf nodes ({len(leaves)}): {leaves[:5]}{'...' if len(leaves) > 5 else ''}")
            
            # Check node types
            node_types = Counter([type(n) for n in tree.nodes()])
            print("\nNode types:")
            for node_type, count in node_types.items():
                print(f"  - {node_type}: {count} nodes")
            
            # Check if any nodes are numeric (potential word nodes)
            numeric_nodes = []
            for node in tree.nodes():
                try:
                    if isinstance(node, (int, float)) or (isinstance(node, str) and node.isdigit()):
                        numeric_nodes.append(node)
                except:
                    pass
            
            print(f"\nNumeric nodes ({len(numeric_nodes)}): {numeric_nodes[:5]}{'...' if len(numeric_nodes) > 5 else ''}")
            
            # Check for POS tags in node labels
            pos_tags = ['NNP', 'NN', 'VB', 'JJ', 'RB', 'DT', 'IN', 'CC', 'PRP', 'CD']
            nodes_with_pos = []
            
            for node, attrs in tree.nodes(data=True):
                if 'label' in attrs and attrs['label'] in pos_tags:
                    nodes_with_pos.append((node, attrs['label']))
            
            print(f"\nNodes with POS tags ({len(nodes_with_pos)}): {nodes_with_pos[:5]}{'...' if len(nodes_with_pos) > 5 else ''}")
            
            # Check for any nodes with embedding attributes
            nodes_with_embedding = []
            for node, attrs in tree.nodes(data=True):
                if 'embedding' in attrs:
                    nodes_with_embedding.append(node)
            
            print(f"\nNodes with embedding attribute ({len(nodes_with_embedding)}): {nodes_with_embedding[:5]}{'...' if len(nodes_with_embedding) > 5 else ''}")
            
            # Print a sample path from root to leaf
            if roots and leaves:
                try:
                    path = nx.shortest_path(tree, roots[0], leaves[0])
                    print("\nSample path from root to leaf:")
                    for node in path:
                        attrs = tree.nodes[node]
                        print(f"  - Node {node}: {attrs}")
                except nx.NetworkXNoPath:
                    print("\nNo path found from root to leaf")
        else:
            print("Tree is not a NetworkX DiGraph, cannot analyze further")
    
    except Exception as e:
        print(f"Error analyzing tree file: {str(e)}")
        import traceback
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Analyze a sample tree file")
    parser.add_argument("--tree_file", type=str, required=True, help="Path to the tree file to analyze")
    
    args = parser.parse_args()
    
    analyze_tree_sample(args.tree_file)

if __name__ == "__main__":
    main()
