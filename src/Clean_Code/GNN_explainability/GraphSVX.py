import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv
from graphsvx import GraphSVXExplainer  # Assuming GraphSVX provides this interface
import os

# Replace these with your actual file paths
MODEL_PATH = "/path/to/your/model.pt"
DATASET_PATH = "/path/to/your/pyg_dataset.pt"

# Load your trained model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = TransformerConv(in_channels, out_channels)
        # Add any extra layers as per your architecture

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# Load dataset - assumes you saved it as a torch file list of Data objects
dataset = torch.load(DATASET_PATH)
# Or use the appropriate dataloader if it's a folder of files

# Get input/output dimensions from the first example, adjust as needed
in_channels = dataset[0].num_node_features
out_channels = int(dataset[0].y.max()) + 1 if hasattr(dataset[0], "y") else 2

# Instantiate and load your pre-trained model
model = GNNModel(in_channels, out_channels)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Make a dataloader for batch loading (if needed)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set up the GraphSVX explainer
explainer = GraphSVXExplainer(model=model, num_samples=100, device='cpu')

# Explain the first three graphs
for i, data in enumerate(loader):
    if i >= 3:
        break
    data = data.to('cpu')  # Move data to CPU if needed
    print(f"\nExplaining graph {i+1}:")
    explanation = explainer.explain_node(node_idx=None, data=data)
    # explanation is typically a dict containing feature attributions, node importances, etc.
    print(explanation)
