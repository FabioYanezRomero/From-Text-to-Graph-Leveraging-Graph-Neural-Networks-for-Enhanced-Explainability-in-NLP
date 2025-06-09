import os
import torch
import pickle as pkl
import networkx as nx
from torch_geometric.data import Data



def create_word_graphs(word_embedding_tuples_list, nx_graphs, special_embeddings=None, label_list=None, show_progress=True):
    """
    Create PyTorch Geometric graphs from constituency trees and word/special embeddings.

    Args:
        word_embedding_tuples_list: List of lists of (word, embedding) tuples, one per graph.
        nx_graphs: List of networkx DiGraph objects, one per graph.
        special_embeddings: Dict mapping special labels (e.g., «ROOT») to embeddings.
        label_list: Optional list of graph-level labels (e.g., for classification).
        show_progress: Show tqdm progress bar if True.
    Returns:
        List of torch_geometric.data.Data objects, each with x, edge_index, node_labels, and optionally y.
    """
    graphs = []
    min_len = min(len(word_embedding_tuples_list), len(nx_graphs), len(label_list) if label_list is not None else len(nx_graphs))
    safe_word_embedding_tuples_list = word_embedding_tuples_list[:min_len]
    safe_nx_graphs = nx_graphs[:min_len]
    safe_label_list = (label_list or [None]*len(nx_graphs))[:min_len]
    iterator = zip(safe_word_embedding_tuples_list, safe_nx_graphs, safe_label_list)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(list(iterator), desc="Generating graphs")

    for idx, (word_tuples, nx_tree, graph_label) in enumerate(iterator):
        # Build word lookup for this sample
        word_to_emb = {w: emb for w, emb in word_tuples}
        embedding_dim = None
        if word_to_emb:
            first_emb = next(iter(word_to_emb.values()))
            embedding_dim = first_emb.shape[0] if hasattr(first_emb, 'shape') else len(first_emb)
        elif special_embeddings:
            first_emb = next(iter(special_embeddings.values()))
            embedding_dim = first_emb.shape[0] if hasattr(first_emb, 'shape') else len(first_emb)
        else:
            print(f"[ERROR] No embeddings found for sample {idx}")
            graphs.append(None)
            continue

        # Node ordering: sorted by node id for reproducibility
        node_ids = list(nx_tree.nodes())
        node_labels = []
        node_features = []
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        for nid in node_ids:
            label = nx_tree.nodes[nid].get('label', None)
            if label is None:
                node_labels.append(None)
                node_features.append(torch.zeros(embedding_dim))
                continue
            # Attach word or special embedding
            if label == 'ROOT' or label == '«SENTENCE»':
                continue
            elif label.startswith('«') and label.endswith('»'):
                node_features.append(torch.tensor(special_embeddings[label], dtype=torch.float))
            elif label in word_to_emb:
                node_features.append(torch.tensor(word_to_emb[label], dtype=torch.float))
            else:
                print(f"[ERROR] No embedding found for node {nid}")
                node_features.append(torch.zeros(embedding_dim))
            node_labels.append(label)

        # Edge index
        edge_index = []
        # Eliminate the ROOT and SENTENCE nodes
        # TODO: Handle the case with ROOT and SENTENCE nodes with the special embeddings
        nx_tree.remove_node('ROOT')

        for node in nx_tree.nodes(data=True):
            if node[1]['label'] == '«SENTENCE»':
                nx_tree.remove_node(node[0])
                break
        for src, dst in nx_tree.edges():
            try:
                edge_index.append([node_id_to_idx[src], node_id_to_idx[dst]])
            except KeyError:
                print(f"KeyError: {src} or {dst} not found in node_id_to_idx")
                continue
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)

        data = Data(x=torch.stack(node_features), edge_index=edge_index, node_labels=node_labels)
        if graph_label is not None:
            data.y = torch.tensor([graph_label], dtype=torch.long)
        graphs.append(data)
    return graphs