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
                 pooling='max'):
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
        
        # Initialize GNN layers based on the specified module type
        for i in range(num_layers):
            if module == 'GCNConv':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif module == 'GATConv':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif module == 'GraphConv':
                self.convs.append(GraphConv(hidden_dim, hidden_dim))
            elif module == 'SAGEConv':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
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
    
    def forward(self, x, edge_index, batch):
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
