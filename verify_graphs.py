import os
import pickle
import torch
import argparse
from tqdm import tqdm

def analyze_graph_files(graph_dir):
    """Analyze graph files to check if they have proper node features"""
    print(f"Analyzing graph files in {graph_dir}")
    
    # Find all graph files
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pkl')]
    print(f"Found {len(graph_files)} graph files")
    
    # Stats
    total_graphs = 0
    empty_node_graphs = 0
    valid_graphs = 0
    node_counts = []
    edge_counts = []
    feature_dims = []
    
    # Analyze each file
    for file_name in tqdm(sorted(graph_files), desc="Analyzing graph files"):
        file_path = os.path.join(graph_dir, file_name)
        
        try:
            with open(file_path, 'rb') as f:
                graphs = pickle.load(f)
            
            # Check if it's a list of graphs
            if not isinstance(graphs, list):
                print(f"Warning: {file_path} does not contain a list of graphs")
                continue
            
            # Analyze each graph
            for i, graph in enumerate(graphs):
                total_graphs += 1
                
                # Check node features
                if hasattr(graph, 'x'):
                    if graph.x.shape[0] == 0:
                        empty_node_graphs += 1
                        print(f"Empty node features in {file_name}, graph {i}: x.shape = {graph.x.shape}")
                    else:
                        valid_graphs += 1
                        node_counts.append(graph.x.shape[0])
                        feature_dims.append(graph.x.shape[1])
                        
                        # Check if all node features are the same (e.g., all zeros or all random)
                        if torch.all(graph.x == 0):
                            print(f"All zero node features in {file_name}, graph {i}")
                        
                        # Check if node features are random (high variance)
                        if graph.x.var() > 0.9:
                            print(f"Likely random node features in {file_name}, graph {i} (variance: {graph.x.var().item():.2f})")
                
                # Check edge indices
                if hasattr(graph, 'edge_index'):
                    edge_counts.append(graph.edge_index.shape[1])
                    
                    # Check if edge indices are valid
                    if graph.edge_index.shape[1] > 0:
                        max_node_idx = graph.edge_index.max().item()
                        if hasattr(graph, 'x') and max_node_idx >= graph.x.shape[0]:
                            print(f"Invalid edge indices in {file_name}, graph {i}: max index {max_node_idx} >= node count {graph.x.shape[0]}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total graphs: {total_graphs}")
    print(f"Valid graphs with nodes: {valid_graphs} ({valid_graphs/total_graphs*100:.2f}%)")
    print(f"Empty node graphs: {empty_node_graphs} ({empty_node_graphs/total_graphs*100:.2f}%)")
    
    if node_counts:
        print(f"Node count: min={min(node_counts)}, max={max(node_counts)}, avg={sum(node_counts)/len(node_counts):.2f}")
    if edge_counts:
        print(f"Edge count: min={min(edge_counts)}, max={max(edge_counts)}, avg={sum(edge_counts)/len(edge_counts):.2f}")
    if feature_dims:
        print(f"Feature dimensions: {set(feature_dims)}")

def main():
    parser = argparse.ArgumentParser(description="Analyze graph files to check for proper node features")
    parser.add_argument("--graph_dir", type=str, default="/app/src/Clean_Code/output/embeddings/graphs", 
                        help="Directory containing graph files")
    parser.add_argument("--dataset", type=str, default="setfit/ag_news",
                        help="Dataset name (e.g., 'setfit/ag_news')")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (train, val, test)")
    
    args = parser.parse_args()
    
    # Construct the path to the graph files
    if '/' in args.dataset:
        provider, name = args.dataset.split('/', 1)
    else:
        provider, name = 'stanfordnlp', args.dataset
    
    # The actual directory structure uses split_llm_labels
    split_dir = f"{args.split}_llm_labels" if args.split in ['train', 'test'] else "validation_llm_labels"
    graph_dir = os.path.join(args.graph_dir, provider.lower(), name.lower(), split_dir)
    
    if not os.path.exists(graph_dir):
        print(f"Graph directory not found: {graph_dir}")
        return
    
    analyze_graph_files(graph_dir)

if __name__ == "__main__":
    main()
