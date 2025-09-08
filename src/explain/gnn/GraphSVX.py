import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool, global_max_pool, LayerNorm, MLP
# Import simplified GraphSVX implementation
import sys
import networkx as nx
from itertools import combinations
import os
import pickle
import json
import time
from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy
import pandas as pd

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# SHAP-based Graph Explainer Implementation (matching tokenSHAP approach)
class GraphSHAPExplainer:
    """
    A SHAP-based graph explainer that uses the same adaptive sampling strategy as tokenSHAP.
    This implementation focuses on node importance using coalition sampling similar to tokenSHAP's token masking.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def analyze(self, data, sampling_ratio=0.1, print_highlight_results=True, num_samples_override=None):
        """
        Analyze a single graph using SHAP-like node importance sampling.
        This mirrors the tokenSHAP.analyze() function but for graph nodes.
        
        Args:
            data: PyTorch Geometric Data object
            sampling_ratio: Ratio of total combinations to sample
            print_highlight_results: Whether to print detailed results
            
        Returns:
            Dictionary containing node importance scores and metadata
        """
        if data is None:
            raise ValueError("Data must be provided")
        
        # Get original prediction
        data = data.to(self.device)
        with torch.no_grad():
            original_output = self.model(data=data)
            original_pred = torch.softmax(original_output, dim=1)
            original_class = torch.argmax(original_pred, dim=1).item()
            original_confidence = original_pred[0][original_class].item()
        
        num_nodes = data.x.size(0)
        num_features = data.x.size(1)

        # Treat [CLS] and [SEP] (first and last tokens in BERT sequences) as always present.
        # Exclude them from sampling and importance to fairly match TokenSHAP behavior.
        special_indices = []
        if num_nodes >= 2:
            special_indices = [0, num_nodes - 1]
        content_indices = [i for i in range(num_nodes) if i not in special_indices]

        max_combinations = 2 ** max(len(content_indices), 0)
        # Allow overriding the exact number of samples to match TokenSHAP iteration counts
        num_samples = int(num_samples_override) if num_samples_override is not None else int(max_combinations * sampling_ratio)
        
        if print_highlight_results:
            print(f"Graph has {num_nodes} nodes, max combinations: {max_combinations}")
            print(f"Sampling {num_samples} combinations ({sampling_ratio:.1%})")
        
        # Initialize importance tracking
        node_importance = torch.zeros(num_nodes)
        combination_count = torch.zeros(num_nodes)  # Track how many times each node was sampled
        
        # Sample different node combinations (SHAP coalitions)
        sampled_combinations = []
        for _ in tqdm(range(num_samples)):
            # Generate random coalition of CONTENT nodes only (exclude special indices)
            coalition_size = random.randint(0, len(content_indices))
            if coalition_size == 0:
                selected_nodes = []
            else:
                selected_nodes = random.sample(content_indices, coalition_size)
            
            # Create masked version where only selected nodes are kept (others zeroed)
            masked_data = data.clone()
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            # Always keep special tokens active
            if len(special_indices) > 0:
                mask[special_indices] = True
            # Activate selected content nodes
            mask[selected_nodes] = True
            
            # Zero out non-selected nodes
            masked_data.x[~mask, :] = 0
            
            # Get prediction with this coalition
            with torch.no_grad():
                masked_output = self.model(data=masked_data)
                masked_pred = torch.softmax(masked_output, dim=1)
                coalition_confidence = masked_pred[0][original_class].item()
            
            # Calculate marginal contributions (SHAP values)
            for node_idx in selected_nodes:  # Only content nodes receive attribution
                # If node is in coalition, calculate its contribution
                if node_idx in selected_nodes:
                    # Remove this node and see the difference
                    coalition_without_node = [n for n in selected_nodes if n != node_idx]
                    
                    # Create coalition without this node
                    masked_data_without = data.clone()
                    mask_without = torch.zeros(num_nodes, dtype=torch.bool)
                    # Keep specials active even without this node
                    if len(special_indices) > 0:
                        mask_without[special_indices] = True
                    mask_without[coalition_without_node] = True
                    masked_data_without.x[~mask_without, :] = 0
                    
                    # Get prediction without this node
                    with torch.no_grad():
                        output_without = self.model(data=masked_data_without)
                        pred_without = torch.softmax(output_without, dim=1)
                        confidence_without = pred_without[0][original_class].item()
                    
                    # Marginal contribution of this node
                    marginal_contribution = coalition_confidence - confidence_without
                    node_importance[node_idx] += marginal_contribution
                    combination_count[node_idx] += 1
            
            # Store combination for analysis
            sampled_combinations.append({
                'selected_nodes': selected_nodes,
                'confidence': coalition_confidence,
                'coalition_size': len(selected_nodes)
            })
        
        # Normalize importance scores by the number of times each node was sampled
        for i in content_indices:
            if combination_count[i] > 0:
                node_importance[i] = node_importance[i] / combination_count[i]
        
        # Create DataFrame-like dictionary similar to tokenSHAP output
        # Convert sampled combinations to a structured format like tokenSHAP's DataFrame
        combinations_data = []
        for idx, combo in enumerate(sampled_combinations):
            # Create a binary representation of the coalition (like tokenSHAP does for tokens)
            coalition_binary = [1 if i in combo['selected_nodes'] else (1 if i in special_indices else 0) for i in range(num_nodes)]
            
            combinations_data.append({
                'combination_id': idx,
                'coalition_size': combo['coalition_size'],
                'coalition_nodes': combo['selected_nodes'],
                'coalition_binary': coalition_binary,
                'prediction_confidence': combo['confidence'],
                'prediction_class': original_class
            })
        
        # Create a DataFrame-like dictionary (mimicking tokenSHAP's df structure)
        df = pd.DataFrame(combinations_data)
        
        # Create results similar to tokenSHAP output structure
        results = {
            'df': df,  # Main result - DataFrame of combinations (like tokenSHAP)
            'node_importance': node_importance.cpu().numpy(),
            'combination_count': combination_count.cpu().numpy(),
            'original_prediction': {
                'class': original_class,
                'confidence': original_confidence,
                'probabilities': original_pred.cpu().numpy()[0]
            },
            'graph_info': {
                'num_nodes': num_nodes,
                'num_features': num_features,
                'num_edges': data.edge_index.size(1),
                'max_combinations': max_combinations,
                'num_samples': num_samples,
                'sampling_ratio': sampling_ratio
            }
        }
        
        if print_highlight_results:
            print(f"Original prediction: Class {original_class} (confidence: {original_confidence:.4f})")
            # Show top important nodes
            top_indices = np.argsort(node_importance.cpu().numpy())[-5:][::-1]
            top_scores = node_importance.cpu().numpy()[top_indices]
            print(f"Top 5 important nodes: {top_indices} with SHAP values: {top_scores}")
        
        return results

# Replace these with your actual file paths
import os as _os
_BASE = _os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs')
MODEL_PATH = _os.path.join(_BASE, "gnn_training/TransformerConvKNN4/final_model.pth")
DATASET_PATH = _os.path.join(_BASE, "embeddings/fully_connected/stanfordnlp/sst2/validation/validation/graphs_batch_000.pkl")
CONFIG_PATH = _os.path.join(_BASE, "gnn_training/TransformerConvKNN4/config.json")

# Load configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

print("Configuration loaded:")
print(json.dumps(config, indent=2))

# Using PyTorch Geometric's MLP class - no need to define our own

def apply_pooling(x, batch, pooling_type='mean'):
    """Apply pooling operation based on the specified type."""
    if batch is None:
        # For single graph, apply pooling across all nodes
        if pooling_type == 'mean':
            return x.mean(dim=0, keepdim=True)
        elif pooling_type == 'max':
            return x.max(dim=0, keepdim=True)[0]
        elif pooling_type == 'add' or pooling_type == 'sum':
            return x.sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
    else:
        # For batched graphs, use PyG pooling functions
        if pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif pooling_type == 'max':
            return global_max_pool(x, batch)
        elif pooling_type == 'add' or pooling_type == 'sum':
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

class GNNClassifier(nn.Module):
    """GNN Classifier matching the training architecture."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, 
                 module='TransformerConv', layer_norm=False, residual=False, pooling='mean', heads=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling
        self.module = module
        self.heads = heads
        
        # Build GNN layers
        self.gnn_layers = self._build_gnn_layers(input_dim, hidden_dim, num_layers, module, layer_norm, heads)
        
        # Adjust classifier input dimension based on whether we're using attention heads
        self.head_proj = None
        if module in ['TransformerConv', 'GATConv', 'GATv2Conv'] and heads > 1:
            # Add a projection layer to handle the concatenated heads
            self.head_proj = nn.Linear(hidden_dim * heads, hidden_dim)
            self.classifier = MLP([hidden_dim, hidden_dim // 2, output_dim], dropout=dropout)
        else:
            self.classifier = MLP([hidden_dim, hidden_dim // 2, output_dim], dropout=dropout)

    def _build_gnn_layers(self, input_dim, hidden_dim, num_layers, module, layer_norm, heads):
        """Build GNN layers based on the specified module type."""
        layers = nn.ModuleList()
        layer_norms = nn.ModuleList() if layer_norm else None
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            # Create TransformerConv layer
            if module == 'TransformerConv':
                conv = TransformerConv(in_dim, out_dim, heads=heads, dropout=self.dropout)
            else:
                raise ValueError(f"Unsupported module type: {module}")
            
            layers.append(conv)
            
            if layer_norm:
                layer_norms.append(LayerNorm(out_dim))
        
        self.layer_norms = layer_norms
        return layers

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        """Forward pass through the GNN."""
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else None
            
        # Process through GNN layers
        for i, conv in enumerate(self.gnn_layers):
            if self.residual and i > 0:
                prev_x = x
                
            # For TransformerConv, ensure input dimension matches expected dimension
            if self.module == 'TransformerConv' and hasattr(conv, 'in_channels'):
                if x.size(-1) != conv.in_channels:
                    # Project input to expected dimension
                    x = F.linear(x, torch.eye(conv.in_channels, x.size(-1), device=x.device))
            
            x = conv(x, edge_index)
            
            # Handle multi-head attention outputs for TransformerConv and GAT layers
            if (self.module in ['TransformerConv', 'GATConv', 'GATv2Conv'] and self.heads > 1):
                # For the last layer, keep all heads and project if needed
                if i == len(self.gnn_layers) - 1 and self.head_proj is not None:
                    # Project the concatenated heads to hidden_dim
                    x = self.head_proj(x)
    
            if self.residual and i > 0 and prev_x.size() == x.size():
                x = x + prev_x
                    
            x = F.relu(x)
            
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            if self.layer_norm and self.layer_norms is not None and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
                
        # Apply graph-level pooling
        x = apply_pooling(x, batch, self.pooling)
        
        # Apply classification head
        return self.classifier(x)

# Load dataset
print("Loading dataset...")
with open(DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)

print(f"Dataset loaded: {len(dataset)} graphs")
print(f"First graph: {dataset[0]}")

# Optionally align with tokenSHAP-selected samples if summary exists
TARGET_SUMMARY_PATHS = [
    "/app/explanations/LLM/sst-2/tokenSHAP_summary.json",
    "/app/tokenSHAP_summary.json",
    _os.path.join(_BASE, "explain/llm/tokenSHAP_summary.json"),
    "tokenSHAP_summary.json",
]
selected_indices = None
ts_index_to_num_samples = {}
ts_index_to_sampling_ratio = {}
for path in TARGET_SUMMARY_PATHS:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                ts_summary = json.load(f)
            # Keep the order sentence_1..sentence_10 and build per-index configs
            sorted_keys = sorted(ts_summary.keys(), key=lambda k: int(''.join(ch for ch in k if ch.isdigit()) or 0))
            selected_indices = []
            for k in sorted_keys:
                if "dataset_index" in ts_summary[k]:
                    idx = ts_summary[k]["dataset_index"]
                    selected_indices.append(idx)
                    # Store desired iteration count (TokenSHAP df rows)
                    if "result_shape" in ts_summary[k] and isinstance(ts_summary[k]["result_shape"], list) and len(ts_summary[k]["result_shape"]) > 0:
                        ts_index_to_num_samples[idx] = int(ts_summary[k]["result_shape"][0])
                    # Store TokenSHAP sampling ratio for logging
                    if "sampling_ratio" in ts_summary[k]:
                        ts_index_to_sampling_ratio[idx] = float(ts_summary[k]["sampling_ratio"])
            print(f"Aligning to tokenSHAP indices from {path}: {selected_indices}")
            break
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")

# Get input/output dimensions from the first example and config
in_channels = dataset[0].num_node_features
out_channels = 2  # Binary classification for SST-2

print(f"Input channels: {in_channels}, Output channels: {out_channels}")

# Instantiate model with configuration parameters
model = GNNClassifier(
    input_dim=in_channels,
    hidden_dim=config['hidden_dim'],
    output_dim=out_channels,
    num_layers=config['num_layers'],
    dropout=config['dropout'],
    module=config['module'],
    layer_norm=config['layer_norm'],
    residual=config['residual'],
    pooling=config['pooling'],
    heads=config['heads']
)

# Load the trained model weights
if os.path.exists(MODEL_PATH):
    print("Loading trained model weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint with 'model_state_dict' key")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model directly from state dict")
    
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Model contains {sum(p.numel() for p in model.parameters())} parameters")
else:
    print(f"Warning: Model checkpoint not found at {MODEL_PATH}")
    exit(1)

# Move model to device and set to eval mode
model = model.to(device)
model.eval()
print("Model loaded and ready!")

# Make a dataloader for batch loading
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set up the SHAP-based graph explainer (matching tokenSHAP approach)
explainer = GraphSHAPExplainer(model=model, device=device)

results = {}
graph_count = 0
# Prepare index list for iteration
# If tokenSHAP-aligned indices are provided, process EXACTLY those in order.
if selected_indices:
    indices_to_visit = selected_indices
else:
    # Fallback: visit in dataset order
    indices_to_visit = list(range(len(dataset)))
ptr = 0

print("Starting GraphSVX analysis...")

# Process graphs: if selected indices exist, process exactly that many; else cap at 10 as before
target_total = len(indices_to_visit) if selected_indices else 10
while graph_count < target_total and ptr < len(indices_to_visit):
    idx = indices_to_visit[ptr]
    data = dataset[idx]
    
    # Calculate number of nodes and derive content-node count that excludes [CLS]/[SEP]
    num_nodes = data.num_nodes if isinstance(data.num_nodes, int) else data.num_nodes.item()
    content_nodes = max(num_nodes - 2, 0) if num_nodes >= 2 else num_nodes
    
    max_combinations = 2 ** content_nodes
    print(f"\nGraph {graph_count+1}: {data}")
    num_edges = data.num_edges if isinstance(data.num_edges, int) else data.num_edges.item()
    print(f"Number of nodes: {num_nodes} (content: {content_nodes}), Number of edges: {num_edges}")
    print(f"Max combinations (content only): {max_combinations}")
    
    # Determine sampling ratio and exact iteration count to MATCH TokenSHAP when available
    sampling_ratio = ts_index_to_sampling_ratio[idx]
    
    print(f"Using sampling ratio: {sampling_ratio}")
    print("Starting Graph SHAP analysis...")
    
    try:
        start_time = time.time()
        
        # Move data to device and set batch to None for single graph
        data = data.to(device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        
        # Force flush stdout to ensure progress bar displays correctly
        sys.stdout.flush()
        
        # Match TokenSHAP iteration count if available
        num_samples_override = ts_index_to_num_samples.get(idx, None)
        explanation = explainer.analyze(
            data,
            sampling_ratio=sampling_ratio,
            print_highlight_results=True,
            num_samples_override=num_samples_override,
        )
        
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        print(f"Generated {len(explanation['df'])} combinations")
        print(explanation['df'].head(10))  # Show only first 10 rows to avoid overwhelming output
        
        # Store results with metadata - matching tokenSHAP structure exactly
        results[f"graph_{graph_count+1}"] = {
            "data": data.cpu(),  # Store graph data on CPU to save memory
            "result": explanation['df'],  # Main result - DataFrame (like tokenSHAP)
            "sampling_ratio": sampling_ratio,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "max_combinations": max_combinations,
            "elapsed_time": elapsed_time,
            "dataset_index": idx,
            "label": data.y.item() if hasattr(data, 'y') and data.y is not None else None,
            "node_importance": explanation['node_importance'],  # Store SHAP values separately
            "original_prediction": explanation['original_prediction']
        }
        
        graph_count += 1
        print(f"✓ Successfully analyzed graph {graph_count}/{target_total}")
        
    except ZeroDivisionError:
        print("Warning: All node combinations produced identical predictions. Skipping analysis.")
    except MemoryError:
        print(f"Memory error: Graph too large ({num_nodes} nodes). Skipping analysis.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        print(f"Skipping graph {idx}")
    
    ptr += 1

# Check if we processed the target count
if graph_count < target_total:
    print(f"\n⚠️  Warning: Only processed {graph_count} graphs out of 10 requested.")
    print(f"   Examined {ptr} graphs from candidate set (candidate set has {len(indices_to_visit)} graphs)")
else:
    print(f"\n✅ Successfully processed exactly {target_total} graphs!")

print(f"Analysis complete! Processed {graph_count} graphs out of {ptr} examined.")

# Save results to file (matching tokenSHAP naming)
output_file = "graphSHAP_results.pkl"
print(f"Saving results to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"Results saved successfully! File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

# Also save a summary in JSON format (matching tokenSHAP structure)
summary = {}
for graph_key, data in results.items():
    summary[graph_key] = {
        "num_nodes": data["num_nodes"],
        "num_edges": data["num_edges"],
        "max_combinations": data["max_combinations"],
        "sampling_ratio": data["sampling_ratio"],
        "elapsed_time": data["elapsed_time"],
        "dataset_index": data["dataset_index"],
        "label": data["label"],
        "prediction_class": data["original_prediction"]["class"] if "original_prediction" in data else None,
        "prediction_confidence": data["original_prediction"]["confidence"] if "original_prediction" in data else None,
        "result_shape": data["result"].shape if "result" in data and hasattr(data["result"], "shape") else None
    }

with open("graphSHAP_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved to graphSHAP_summary.json")
print(f"Total samples saved: {len(results)}")
