import torch
from dig.xgraph.method import SubgraphX
from Clean_Code.GNN_Training.gnn_models import GNN_Classifier
from torch_geometric.data import Data

# ---- Minimal test: adjust these to your real data/model ----
# Dummy graph data (replace with a real Data object from your dataset)
x = torch.randn(10, 768)  # 10 nodes, 768 features
y = torch.tensor([1])     # single label
edge_index = torch.randint(0, 10, (2, 20))  # 20 edges

data = Data(x=x, edge_index=edge_index, y=y)

# Dummy model (replace with your trained model and correct dimensions)
model = GNN_Classifier(
    input_dim=768,
    hidden_dim=64,
    output_dim=2,
    num_layers=2,
    dropout=0.1,
    module='GCNConv',
    layer_norm=False,
    residual=False,
    pooling='max'
)

# Put model in eval mode and on CPU for test
model.eval()
device = torch.device('cpu')
model = model.to(device)

data = data.to(device)

# ---- SubgraphX explainer ----
explainer = SubgraphX(
    model=model,
    num_classes=2,
    device=device,
    num_hops=1,
    rollout=10,
    min_atoms=1,
    c_puct=1,
    expand_atoms=1,
    local_radius=1,
    sample_num=1,
    save_dir=None
)

# Run explanation (adjust arguments as needed for your DIG version)
try:
    explanation, related_pred = explainer.explain(
        x=data.x,
        edge_index=data.edge_index,
        label=int(data.y.item()),
        max_nodes=5
    )
    print('Explanation:', explanation)
    print('Related Prediction:', related_pred)
except Exception as e:
    print('Error running SubgraphX explain:', e)
