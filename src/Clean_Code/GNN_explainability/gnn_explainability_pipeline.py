import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
from metrics import ForwardPassCounter, measure_time, measure_memory, profile_hardware, estimate_flops

# Try to import torch_geometric and graphsvx
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError('torch_geometric is required for GNN explainability.')

try:
    from graphsvx import GraphSVX
except ImportError:
    raise ImportError('graphsvx is required for GNN explainability.')

def parse_args():
    parser = argparse.ArgumentParser(description="GNN Explainability with GraphSVX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt (full path)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save GraphSVX outputs")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of graph samples to explain")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to torch_geometric dataset file (e.g., .pt or directory)")
    return parser.parse_args()

def load_gnn_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def load_graph_dataset(dataset_path, num_samples):
    # Assume dataset is a torch_geometric Data object or list
    if dataset_path.endswith('.pt'):
        dataset = torch.load(dataset_path)
    else:
        # Could be a directory with multiple .pt files
        files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pt')]
        dataset = [torch.load(f) for f in files]
    if isinstance(dataset, list):
        return dataset[:num_samples]
    else:
        return [dataset]

def predict_proba(data_list, model, device, eval_counter=None):
    loader = DataLoader(data_list, batch_size=1)
    all_probs = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            probs = torch.nn.functional.softmax(out, dim=-1).cpu().numpy()
            all_probs.append(probs)
        if eval_counter is not None:
            eval_counter.increment(1)
    return all_probs

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = load_gnn_model(args.model_path, device)
    dataset = load_graph_dataset(args.dataset_path, args.num_samples)
    eval_counter = ForwardPassCounter()

    explanation_times = []
    model_evals = []
    input_sizes = []
    peak_memories = []
    hardware_stats = []
    flops_per_instance = []
    explanations = []

    for i, data in enumerate(dataset):
        eval_counter.reset()
        input_sizes.append(data.num_nodes)
        with measure_time() as t_fn, measure_memory() as m_fn:
            explainer = GraphSVX(model, data, device=args.device)
            explanation = explainer.explain_graph()
        explanation_times.append(t_fn())
        model_evals.append(eval_counter.get())
        peak_memories.append(m_fn())
        hardware_stats.append(profile_hardware())
        flops = estimate_flops(model, lambda x: x, data, device)
        flops_per_instance.append(flops)
        explanations.append(explanation)
        # Visualization: plot node importance if available
        if hasattr(explanation, 'node_imp'):
            plt.figure(figsize=(8,2))
            plt.bar(range(len(explanation.node_imp)), explanation.node_imp)
            plt.title(f"Graph {i} Node Importance")
            plt.savefig(os.path.join(args.output_dir, f"graphsvx_node_imp_{i}.png"), bbox_inches='tight')
            plt.close()

    metrics = {
        'explanation_time_per_instance': explanation_times,
        'model_evaluations_per_explanation': model_evals,
        'input_sizes': input_sizes,
        'peak_memory_bytes': peak_memories,
        'hardware_utilization': hardware_stats,
        'flops_per_instance': flops_per_instance,
        'mean_explanation_time': float(sum(explanation_times)/len(explanation_times)),
        'mean_model_evals': float(sum(model_evals)/len(model_evals)),
        'mean_peak_memory_bytes': float(sum(peak_memories)/len(peak_memories)),
        'mean_flops': float(sum([f for f in flops_per_instance if f is not None])/max(1, len([f for f in flops_per_instance if f is not None])))
    }
    with open(os.path.join(args.output_dir, 'metrics_graphsvx.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Done. Metrics and visualizations saved.")

if __name__ == "__main__":
    main()
