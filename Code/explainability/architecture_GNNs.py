import torch
import torch_geometric
# homogeneous graphs modules
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GatedGraphConv, GraphConv, GATv2Conv, GatedGraphConv

# heterogeneous graphs modules
from torch_geometric.nn import RGCNConv, GINEConv, RGATConv, HEATConv, HANConv, HGTConv
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import LayerNorm
import copy
from arguments import *
from torch_geometric.nn import GCN, MLP
args = arguments()

""" 
In PyTorch Geometric (PyG), both Batch Normalization and Layer Normalization are used, but the choice between 
them depends on the specific characteristics of your graph data and the model architecture.

Batch Normalization (BatchNorm) normalizes the features across the batch dimension. This can be beneficial when you 
have sufficiently large batch sizes and when the batch statistics are a good estimate of the overall dataset statistics. 
However, in graph neural networks, the size and structure of each graph in a batch can vary significantly, which might 
make BatchNorm less effective.

Layer Normalization (LayerNorm), on the other hand, normalizes the features across the channel dimension for each sample 
independently. This makes it particularly suitable for graph data, where each graph can have a different number of nodes 
and structure. LayerNorm is often preferred in GNNs because it doesn’t rely on batch statistics and can be applied to 
graphs of varying sizes and shapes.

"""

""" General purpose MLP for classify after pooling layers from GNNs """


MODULE_DICT = {
    'GCNConv': GCNConv,
    'GATConv': GATConv,
    'SAGEConv': SAGEConv,
    'GINConv': GINConv,
    'GatedGraphConv': GatedGraphConv,
    'GraphConv': GraphConv,
    'GATv2Conv': GATv2Conv,
    'GatedGraphConv': GatedGraphConv
}


RMODULE_DICT = {
    'RGCNConv': RGCNConv,
    'RGATConv': RGATConv,
    'GINEConv': GINEConv,
    'HEATConv': HEATConv,
    'HANConv': HANConv,
    'HGTConv': HGTConv

}

""" class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels)) #hidden_channels
        for i in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            if i < len(self.lins) - 1:
                x1 = lin(x)
                # residual connection
                if x.size(-1) == x1.size(-1):
                    x = x + x1
                else:
                    x = x1
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = lin(x)
            
        return x
 """


""" With the following classes, we can create a GNN model with a specific number of layers, hidden size, output size, and dropout rate.
The main idea is to create a generic class for homogeneous graph modules and another for heterogeneous graph modules."""

class GNN_classifier(nn.Module):
    
    def __init__(self, size, num_layers, dropout, module, layer_norm=False, residual=False, pooling='mean', lin_transform=None):
        super(GNN_classifier, self).__init__()
        self.lin_transform = lin_transform
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gnn1 = nn.ModuleList()
        self.gnn2 = nn.ModuleList()
        if layer_norm:
            self.layer_norms1 = nn.ModuleList()
            self.layer_norms2 = nn.ModuleList()
        self.module = module
        self.residual = residual
        self.pooling = pooling
        self.size = size
        if self.lin_transform is not None:
            initial_size = self.size
            size = self.lin_transform
            """ self.weight1 = nn.Parameter(torch.Tensor(size, initial_size))
            self.weight2 = nn.Parameter(torch.Tensor(size, initial_size)) """
            self.weight1 = torch.empty(size, initial_size).to('cuda')
            self.weight2 = torch.empty(size, initial_size).to('cuda')
            #nn.init.xavier_uniform_(self.weight1)
            #nn.init.xavier_uniform_(self.weight2)
            nn.init.kaiming_uniform_(self.weight1)
            nn.init.kaiming_uniform_(self.weight2)
        for i in range(num_layers):
            if module in ['GCNConv', 'GraphConv', 'SAGEConv']:
                self.gnn1.append(MODULE_DICT[module](size, size))
            elif module in ['GATConv', 'GATv2Conv']:
                if i == 0:
                    heads = args['heads']
                    self.gnn1.append(MODULE_DICT[module](size, size, heads=heads))
                else:
                    self.gnn1.append(MODULE_DICT[module](size*heads, size, heads=heads))
            elif module in ['GINConv']:
                self.gnn1.append(MODULE_DICT[module](nn=nn.Sequential(nn.Linear(size, size), 
                                                                      nn.ReLU(), nn.Linear(size, size))))
            else:
                raise NotImplementedError
            if layer_norm:
                if module in ['GATConv', 'GATv2Conv']:
                    self.layer_norms1.append(LayerNorm(in_channels=size*heads))
                else:
                    self.layer_norms1.append(LayerNorm(in_channels=size))

        for i in range(num_layers):
            if module in ['GCNConv', 'GraphConv', 'SAGEConv']:
                self.gnn2.append(MODULE_DICT[module](size, size))
            elif module in ['GATConv', 'GATv2Conv']:
                if i == 0:
                    heads = args['heads']
                    self.gnn2.append(MODULE_DICT[module](size, size, heads=heads))
                else:
                    self.gnn2.append(MODULE_DICT[module](size*heads, size, heads=heads))
            elif module in ['GINConv']:
                self.gnn2.append(MODULE_DICT[module](nn=nn.Sequential(nn.Linear(size, size), 
                                                                      nn.ReLU(), nn.Linear(size, size))))
            else:
                raise NotImplementedError
            if layer_norm:
                if module in ['GATConv', 'GATv2Conv']:
                    self.layer_norms2.append(LayerNorm(in_channels=size*heads))
                else:
                    self.layer_norms2.append(LayerNorm(in_channels=size))
        
        if self.residual:
            if module in ['GATConv', 'GATv2Conv']:
                self.residuals1 = nn.ModuleList()
                self.residuals2 = nn.ModuleList()
                self.residuals1.append(nn.Linear(768, 768*heads))
                self.residuals2.append(nn.Linear(768, 768*heads))

        if module in ['GATConv', 'GATv2Conv']:
            size = size*heads
            hidden_size = 768
        else:
            hidden_size = size
        self.mlp = MLP(in_channels=size*2, hidden_channels=hidden_size, out_channels=4, dropout=0.2, num_layers=4)

    
    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        # Apply relu and dropout to all the layers except the final one
        if self.lin_transform is not None:
            x1 = F.linear(x1, self.weight1)
            x2 = F.linear(x2, self.weight2)
        
        for i, conv in enumerate(self.gnn1):
            prev_x1 = x1.clone()
            x1 = conv(x1, edge_index1)
            if self.residual:
                if self.module in ['GATConv', 'GATv2Conv'] and i==0:
                    x1 = x1 + self.residuals1[0](prev_x1)
                else:
                    x1 = x1 + prev_x1
            x1 = F.relu(x1)
            
            #Dropout is applied to the output of each layer except the last one
            if i < self.num_layers - 1:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
                if self.layer_norm:
                    x1 = self.layer_norms1[i](x1, batch1)
        if self.pooling == 'mean':
            x1 = global_mean_pool(x1, batch1)  # Pool to single embedding
        if self.pooling == 'max':
            x1 = global_max_pool(x1, batch1)
        if self.pooling == 'sum':
            x1 = global_add_pool(x1, batch1)
        
        for i, conv in enumerate(self.gnn2):
            prev_x2 = x2.clone()
            x2 = conv(x2, edge_index2)
            if self.residual:
                if self.module in ['GATConv', 'GATv2Conv'] and i ==0:
                    x2 = x2 + self.residuals2[0](prev_x2)
                else:
                    x2 = x2 + prev_x2
            x2 = F.relu(x2)
            if i < self.num_layers - 1:
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
                if self.layer_norm:
                    x2 = self.layer_norms2[i](x2, batch2)
        if self.pooling == 'mean':
            x2 = global_mean_pool(x2, batch2)  # Pool to single embedding
        if self.pooling == 'max':
            x2 = global_max_pool(x2, batch2)
        if self.pooling == 'sum':
            x2 = global_add_pool(x2, batch2)
        

        x = torch.cat([x1, x2], dim=1)
        x = self.mlp(x)

        return x
    
# Consider this class for RGCNCONV, RGATConv, GINEConv and HEATConv
class RGNN_classifier(nn.Module):
    def __init__(self, size, num_layers, dropout, module, num_relations, layer_norm=False, residual=False, pooling= 'mean'):
        super(RGNN_classifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gnn1 = nn.ModuleList()
        self.gnn2 = nn.ModuleList()
        self.layer_norms1 = nn.ModuleList()
        self.layer_norms2 = nn.ModuleList()
        self.module = module
        self.residual = residual
        self.num_relations = num_relations
        self.pooling = pooling
        # Layers for the first GNN
        for i in range(num_layers):
            if module in ['RGCNConv']:
                self.gnn1.append(RMODULE_DICT[module](in_channels=size, out_channels=size, num_relations=num_relations))
            elif module in ['RGATConv']:
                if i == 0:
                    heads = args['heads']
                    self.gnn1.append(RMODULE_DICT[module](in_channels=size, out_channels=size, heads=heads, num_relations=num_relations))
                else:
                    self.gnn1.append(RMODULE_DICT[module](in_channels=size*heads, out_channels=size, heads=heads, num_relations=num_relations))
            
            # GINEConv expects embeddings for edge instead of unique integers for each edge type
            elif module in ['GINEConv']:
                self.gnn1.append(RMODULE_DICT[module](nn=nn.Sequential(nn.Linear(in_features=size, out_features=size), 
                                                                      nn.ReLU(), nn.Linear(in_features=size, out_features=size))))
            else:
                raise NotImplementedError
            if layer_norm:
                if module in ['RGATConv']:
                    self.layer_norms1.append(LayerNorm(in_channels=size*heads))
                else:
                    self.layer_norms1.append(LayerNorm(in_channels=size))


        # Layers for the second GNN
        for i in range(num_layers):
            if module in ['RGCNConv']:
                self.gnn2.append(RMODULE_DICT[module](in_channels=size, out_channels=size, num_relations=num_relations))
            elif module in ['RGATConv']:
                if i == 0:
                    heads = args['heads']
                    self.gnn2.append(RMODULE_DICT[module](in_channels=size, out_channels=size, heads=heads, num_relations=num_relations))
                else:
                    self.gnn2.append(RMODULE_DICT[module](in_channels=size, out_channels=size, heads=heads, num_relations=num_relations))
            elif module in ['GINEConv']:
                self.gnn2.append(RMODULE_DICT[module](nn=nn.Sequential(nn.Linear(in_features=size, out_features=size), 
                                                                      nn.ReLU(), nn.Linear(in_features=size, out_features=size))))
            else:
                raise NotImplementedError
            if layer_norm:
                if module in ['RGATConv']:
                    self.layer_norms2.append(LayerNorm(in_channels=size*heads))
                else:
                    self.layer_norms2.append(LayerNorm(in_channels=size))
        
        if self.residual:
            if module in ['RGATConv']:
                self.residuals1 = nn.ModuleList()
                self.residuals2 = nn.ModuleList()
                self.residuals1.append(nn.Linear(768, 768*heads))
                self.residuals2.append(nn.Linear(768, 768*heads))

        if module in ['RGATConv']:
            size = size*heads
            hidden_size = 768
        else:
            hidden_size = 768
        self.mlp = MLP(size*2, hidden_size, 4, 3, dropout)

    
    def forward(self, x1, edge_index1, edge_type1, x2, edge_index2, edge_type2, batch1, batch2):
        # Apply relu and dropout to all the layers except the final one
        for i, conv in enumerate(self.gnn1):
            prev_x1 = x1.clone()
            x1 = conv(x=x1, edge_index=edge_index1, edge_type=edge_type1)
            if self.residual:
                if self.module in ['RGATConv'] and i==0:
                    x1 = x1 + self.residuals1[0](prev_x1)
                else:
                    x1 = x1 + prev_x1
            x1 = F.relu(x1)
            
            #Dropout is applied to the output of each layer except the last one
            if i < self.num_layers - 1:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
                if self.layer_norm:
                    x1 = self.layer_norms1[i](x1, batch1)
        if self.pooling == 'mean':
            x1 = global_mean_pool(x1, batch1)  # Pool to single embedding
        if self.pooling == 'max':
            x1 = global_max_pool(x1, batch1)
        if self.pooling == 'sum':
            x1 = global_add_pool(x1, batch1)
        
        for i, conv in enumerate(self.gnn2):
            prev_x2 = x2.clone()
            x2 = conv(x=x2, edge_index=edge_index2, edge_type=edge_type2)
            if self.residual:
                if self.module in ['RGATConv'] and i ==0:
                    x2 = x2 + self.residuals2[0](prev_x2)
                else:
                    x2 = x2 + prev_x2
            x2 = F.relu(x2)
            if i < self.num_layers - 1:
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
                if self.layer_norm:
                    x2 = self.layer_norms2[i](x2, batch2)
        
        x2 = global_mean_pool(x2, batch2)

        x = torch.cat([x1, x2], dim=1)
        x = self.mlp(x)

        return x
    

# FOR HGTConv and HANConv we should consider other class as metadata must be provided


class GCN_original(GCN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        self.GCN1 = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.GCN2 = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.mlp = MLP(in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout)
    
    
    def forward(self, x1, x2, edge_index1, edge_index2, batch1, batch2):
        x1 = self.GCN1(x1, edge_index1, batch=batch1)
        x2 = self.GCN2(x2, edge_index2, batch=batch2)
        x1 = global_max_pool(x1, batch1)
        x2 = global_max_pool(x2, batch2)
        x = torch.cat([x1, x2], dim=1)
        x = self.mlp(x)
        return x