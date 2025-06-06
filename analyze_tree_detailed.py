import os
import pickle
import sys
import networkx as nx

# Check if a file path was provided
if len(sys.argv) > 1:
    tree_file = sys.argv[1]
else:
    # Use a default tree file
    tree_file = "/app/src/Clean_Code/output/text_graphs/SetFit/ag_news/train/constituency/425.pkl"

print(f"Analyzing tree file: {tree_file}")

try:
    with open(tree_file, 'rb') as f:
        tree_data = pickle.load(f)
    
    print(f"Type of tree data: {type(tree_data)}")
    
    # If it's a list, check the first element
    if isinstance(tree_data, list):
        print(f"List length: {len(tree_data)}")
        if len(tree_data) > 0:
            if isinstance(tree_data[0], tuple):
                trees, labels = tree_data[0]
                print(f"First item is a tuple: (trees, labels)")
                print(f"  Trees type: {type(trees)}")
                print(f"  Trees length: {len(trees)}")
                print(f"  Labels type: {type(labels)}")
                print(f"  Labels length: {len(labels)}")
                
                # Examine the first tree
                if len(trees) > 0:
                    first_tree = trees[0]
                    print(f"\nFirst tree type: {type(first_tree)}")
                    
                    if isinstance(first_tree, nx.DiGraph):
                        print("\nAnalyzing NetworkX DiGraph structure:")
                        print(f"Number of nodes: {first_tree.number_of_nodes()}")
                        print(f"Number of edges: {first_tree.number_of_edges()}")
                        
                        # Print node attributes
                        print("\nNode attributes:")
                        for node, attrs in first_tree.nodes(data=True):
                            print(f"Node {node}: {attrs}")
                        
                        # Print edge attributes
                        print("\nEdge attributes:")
                        for u, v, attrs in first_tree.edges(data=True):
                            print(f"Edge ({u}, {v}): {attrs}")
                        
                        # Check if there are any root nodes (nodes with no incoming edges)
                        roots = [n for n, d in first_tree.in_degree() if d == 0]
                        print(f"\nRoot nodes: {roots}")
                        
                        # Check if there are any leaf nodes (nodes with no outgoing edges)
                        leaves = [n for n, d in first_tree.out_degree() if d == 0]
                        print(f"\nLeaf nodes: {leaves}")
                        
                        # Check if nodes have word embeddings or other useful attributes
                        print("\nChecking for word embeddings in nodes:")
                        embedding_nodes = [n for n, attrs in first_tree.nodes(data=True) if 'embedding' in attrs]
                        if embedding_nodes:
                            print(f"Found {len(embedding_nodes)} nodes with embeddings")
                            print(f"Sample node with embedding: {first_tree.nodes[embedding_nodes[0]]}")
                        else:
                            print("No nodes with 'embedding' attribute found")
                        
                        # Check for other common attributes
                        all_attrs = set()
                        for _, attrs in first_tree.nodes(data=True):
                            all_attrs.update(attrs.keys())
                        print(f"\nAll node attributes found: {all_attrs}")

except Exception as e:
    print(f"Error analyzing tree file: {str(e)}")
    import traceback
    traceback.print_exc()
