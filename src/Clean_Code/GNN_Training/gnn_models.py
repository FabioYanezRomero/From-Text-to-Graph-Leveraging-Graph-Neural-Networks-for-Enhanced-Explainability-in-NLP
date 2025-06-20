"""
GNN Models for Text Classification

This module contains Graph Neural Network models for text classification tasks.
It supports various GNN modules from PyTorch Geometric like GCNConv, GATConv, GraphConv, and SAGEConv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

class GNN_Classifier(torch.nn.Module):
    """
    Graph Neural Network Classifier for text classification tasks.
    
    This model supports various GNN modules and pooling strategies.
    """
    def __init__(self, 
                 input_dim=768, 
                 hidden_dim=256, 
                 output_dim=4,
                 num_layers=3, 
                 dropout=0.5, 
                 module='GCNConv', 
                 layer_norm=True, 
                 residual=True, 
                 pooling='max',
                 gat_heads=1, # Default, but ignored if using per-layer logic
                 gat_concat=True # Default, but ignored if using per-layer logic
                 ):
        """
        Initialize the GNN classifier.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            module: GNN module type ('GCNConv', 'GATConv', 'GraphConv', 'SAGEConv')
            layer_norm: Whether to use layer normalization
            residual: Whether to use residual connections
            pooling: Pooling strategy ('max', 'mean', 'add')
            gat_heads: (legacy) Number of attention heads for GATConv (default=1)
            gat_concat: (legacy) Whether to concatenate heads (True) or average (False)
        """
        super(GNN_Classifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.module = module
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        # Layer normalization
        self.layer_norms = nn.ModuleList() if layer_norm else None
        
        # Track the current feature dimension
        curr_dim = hidden_dim
        # --- Per-layer GAT heads logic ---
        if module == 'GATConv':
            if num_layers == 1:
                gat_heads_list = [8]
                gat_concat_list = [True]
            elif num_layers == 2:
                gat_heads_list = [8, 1]
                gat_concat_list = [True, False]
            else:
                # 8 heads in first, hidden_dim//2 in middle, 1 in last
                middle_heads = max(2, hidden_dim // 2)
                gat_heads_list = [8] + [middle_heads] * (num_layers - 2) + [1]
                gat_concat_list = [True] * (num_layers - 1) + [False]
        else:
            gat_heads_list = [None] * num_layers
            gat_concat_list = [None] * num_layers
        for i in range(num_layers):
            if module == 'GCNConv':
                self.convs.append(GCNConv(curr_dim, hidden_dim))
                curr_dim = hidden_dim
                if self.layer_norms is not None:
                    self.layer_norms.append(nn.LayerNorm(curr_dim))
            elif module == 'GATConv':
                heads = gat_heads_list[i]
                concat = gat_concat_list[i]
                if heads is not None and concat is not None:
                    self.convs.append(GATConv(curr_dim, hidden_dim, heads=heads, concat=concat))
                    if concat:
                        curr_dim = hidden_dim * heads
                    else:
                        curr_dim = hidden_dim
                    if self.layer_norms is not None:
                        self.layer_norms.append(nn.LayerNorm(curr_dim))
                else:
                    # Should not happen, but skip if heads/concat are None
                    pass
            elif module == 'GraphConv':
                self.convs.append(GraphConv(curr_dim, hidden_dim))
                curr_dim = hidden_dim
                if self.layer_norms is not None:
                    self.layer_norms.append(nn.LayerNorm(curr_dim))
            elif module == 'SAGEConv':
                self.convs.append(SAGEConv(curr_dim, hidden_dim))
                curr_dim = hidden_dim
                if self.layer_norms is not None:
                    self.layer_norms.append(nn.LayerNorm(curr_dim))
            else:
                raise ValueError(f"Unsupported GNN module: {module}")

        
        # Output classification layer (MLP after pooling)
        self.classifier = nn.Sequential(
            nn.Linear(curr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GNN classifier.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            Logits for each class [batch_size, output_dim]
        """
        # Initial feature projection
        h = self.input_proj(x)
        
        # Apply GNN layers with optional residual connections and layer normalization
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            # If GATConv with concat, output dim is hidden_dim * heads
            # If not concat, output dim is hidden_dim
            if self.residual and h.shape == h_new.shape:
                h = h_new + h
            else:
                h = h_new
            
            h = F.relu(h)
            
            if self.layer_norm:
                h = self.layer_norms[i](h)
            
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Apply pooling
        if self.pooling == 'max':
            h = global_max_pool(h, batch)
        elif self.pooling == 'mean':
            h = global_mean_pool(h, batch)
        elif self.pooling == 'add':
            h = global_add_pool(h, batch)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Classification
        return self.classifier(h)



class RGNN_Classifier(torch.nn.Module):
    """
    Relational Graph Neural Network Classifier for text classification tasks.
    
    This model supports relational GNN modules like RGCNConv and RGATConv.
    """
    def __init__(self, 
                 input_dim=768, 
                 hidden_dim=256, 
                 output_dim=4,
                 num_layers=3, 
                 dropout=0.5, 
                 module='RGCNConv', 
                 layer_norm=True, 
                 residual=True,
                 num_relations=3,
                 pooling='max'):
        """
        Initialize the RGNN classifier.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            module: GNN module type ('RGCNConv', 'RGATConv')
            layer_norm: Whether to use layer normalization
            residual: Whether to use residual connections
            num_relations: Number of relation types
            pooling: Pooling strategy ('max', 'mean', 'add')
        """
        super(RGNN_Classifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.module = module
        self.layer_norm = layer_norm
        self.residual = residual
        self.num_relations = num_relations
        self.pooling = pooling
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        # Layer normalization
        self.layer_norms = nn.ModuleList() if layer_norm else None
        
        # Initialize GNN layers based on the specified module type
        for i in range(num_layers):
            if module == 'RGCNConv':
                from torch_geometric.nn import RGCNConv
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            elif module == 'RGATConv':
                from torch_geometric.nn import RGATConv
                self.convs.append(RGATConv(hidden_dim, hidden_dim, num_relations=num_relations))
            else:
                raise ValueError(f"Unsupported GNN module: {module}")
            
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, edge_type, batch):
        """
        Forward pass through the RGNN classifier.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge relation types [num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            Logits for each class [batch_size, output_dim]
        """
        # Initial feature projection
        h = self.input_proj(x)
        
        # Apply GNN layers with optional residual connections and layer normalization
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_type)
            
            if self.residual and h.shape == h_new.shape:
                h = h_new + h
            else:
                h = h_new
            
            h = F.relu(h)
            
            if self.layer_norm:
                h = self.layer_norms[i](h)
            
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Apply pooling
        if self.pooling == 'max':
            h = global_max_pool(h, batch)
        elif self.pooling == 'mean':
            h = global_mean_pool(h, batch)
        elif self.pooling == 'add':
            h = global_add_pool(h, batch)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Classification
        return self.classifier(h)
