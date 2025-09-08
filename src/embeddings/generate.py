import os
import argparse
import pickle as pkl
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from dicts import constituency_dict
import re

def load_trees(tree_dir, batch_size):
    """
    Load all constituency trees from numbered pickle files in the directory, sorted numerically.
    Each .pkl file is a list, whose [0] is a tuple, whose [0][0] is a list of DiGraph objects.
    Returns a list of batches, each batch being a list of graphs of size batch_size (except possibly the last).
    """
    files = [f for f in os.listdir(tree_dir) if re.fullmatch(r'\d+\.pkl', f)]
    if not files:
        raise FileNotFoundError(f"No numbered .pkl files found in {tree_dir}")
    files_sorted = sorted(files, key=lambda x: int(x.split('.')[0]))
    all_graphs = []
    for fname in files_sorted:
        with open(os.path.join(tree_dir, fname), "rb") as f:
            obj = pkl.load(f)
            # Expect obj to be a list, whose first element is a tuple, whose first element is a list of graphs
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], tuple) and len(obj[0]) > 0 and isinstance(obj[0][0], list):
                all_graphs.extend(obj[0][0])
            else:
                raise ValueError(f"File {fname} does not match expected structure: list->[tuple]->[list]")
    # Split into batches
    batches = [all_graphs[i:i+batch_size] for i in range(0, len(all_graphs), batch_size)]
    return batches

def load_sentences(dataset_name, split):
    ds = load_dataset(dataset_name, split=split)
    if 'sentence' in ds.column_names:
        return ds['sentence']
    elif 'text' in ds.column_names:
        return ds['text']
    else:
        raise ValueError("No suitable sentence/text column found in dataset.")



def is_special_label(label):
    """Detect if a label is a special (constituency or POS) label by dict or the «...» pattern."""
    return (
        isinstance(label, str)
        and (label in constituency_dict.values() or (label.startswith('«') and label.endswith('»')))
    )

def validate_graph_structure(graph, graph_idx=None):
    """Assert that all non-leaf nodes are special and all leaf nodes are words (not special)."""
    for nid, data in graph.nodes(data=True):
        out_degree = graph.out_degree(nid)
        label = data['label']
        if out_degree > 0:
            assert label in constituency_dict.values(), (
                f"Non-leaf node (id={nid}, label={label}) in graph {graph_idx} is not a valid special label!"
            )
        else:
            assert label not in constituency_dict.values(), (
                f"Leaf node (id={nid}, label={label}) in graph {graph_idx} is a special label, expected a word!"
            )

def normalize_special_labels(graph):
    """Replace any non-leaf node label that is a key in constituency_dict with its pretty label."""
    for nid, data in graph.nodes(data=True):
        if graph.out_degree(nid) > 0:
            label = data['label']
            if label in constituency_dict:
                data['label'] = constituency_dict[label]

def compute_special_embeddings(labels, model, tokenizer, device):
    """Compute CLS embeddings for each special label."""
    special_embeddings = {}
    model.eval()
    with torch.no_grad():
        for label in tqdm(labels, desc="Computing special embeddings"):
            inputs = tokenizer(label, return_tensors='pt').to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            special_embeddings[label] = cls_emb
    return special_embeddings

def get_word_embeddings(sentence, model, tokenizer, device):
    """Get wordpiece embeddings for each word in the sentence using robust alignment via offset overlaps."""
    inputs = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, truncation=True)
    offsets = inputs.pop('offset_mapping')  # Remove offset_mapping before passing to model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
    offsets = offsets.squeeze(0).tolist()
    words = sentence.split()
    word_embs = []

    # Compute character spans for each word
    char_idx = 0
    word_spans = []
    for word in words:
        # Skip leading spaces
        while char_idx < len(sentence) and sentence[char_idx].isspace():
            char_idx += 1
        start = char_idx
        end = start + len(word)
        word_spans.append((start, end))
        char_idx = end

    for word, (w_start, w_end) in zip(words, word_spans):
        token_indices = []
        for i, (tok, (t_start, t_end)) in enumerate(zip(tokens, offsets)):
            if tok in [tokenizer.cls_token, tokenizer.sep_token]:
                continue
            # Check for overlap between word and token spans
            if max(w_start, t_start) < min(w_end, t_end):
                token_indices.append(i)
        if token_indices:
            emb = hidden[token_indices].mean(axis=0)
        else:
            emb = np.zeros_like(hidden[0])
        word_embs.append(emb)
    return word_embs

def assign_embeddings_to_graph(graph, word_embs, special_embeddings):
    word_idx = 0
    for nid, data in graph.nodes(data=True):
        if data['type'] == 'word':
            if word_idx < len(word_embs):
                data['embedding'] = word_embs[word_idx]
            else:
                data['embedding'] = np.zeros_like(next(iter(special_embeddings.values())))
            word_idx += 1
        else:
            label = data['label']
            if label in special_embeddings:
                data['embedding'] = special_embeddings[label]
            else:
                data['embedding'] = np.zeros_like(next(iter(special_embeddings.values())))
    return graph

def clean_graph_whitespace_nodes(graph):
    """
    Remove non-leaf nodes with empty or whitespace-only labels,
    reconnecting their children to their parent(s) to preserve hierarchy.
    """
    problematic_labels = ['', '``', "''", '""', '`', "´", '“', '”', '‘', '’']
    nodes_to_remove = []
    for node, data in list(graph.nodes(data=True)):
        label = data.get('label', '')
        if label in problematic_labels or label.strip() == '':
            if graph.out_degree(node) > 0:
                nodes_to_remove.append(node)
    for node in nodes_to_remove:
        parents = list(graph.predecessors(node))
        children = list(graph.successors(node))
        for parent in parents:
            for child in children:
                graph.add_edge(parent, child)
        graph.remove_node(node)

def main():

    parser = argparse.ArgumentParser(description="Generate constituency or syntactic graphs with node embeddings for GNN training.")
    parser.add_argument('--graph_type', type=str, choices=['constituency', 'syntactic'], help='Type of graph to process', default='syntactic')
    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/sst2')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--tree_dir', type=str, default='/app/src/Clean_Code/output/text_trees/stanfordnlp/sst2/validation/syntactic')
    parser.add_argument('--output_dir', type=str, default='/app/src/Clean_Code/output/gnn_embeddings/stanfordnlp/sst2/validation/syntactic')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing graphs and sentences')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sentences = load_sentences(args.dataset_name, args.split)
    graph_batches = load_trees(args.tree_dir, args.batch_size)
    total_graphs = sum(len(batch) for batch in graph_batches)
    assert len(sentences) == total_graphs, f"Mismatch: {len(sentences)} sentences, {total_graphs} graphs"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)

    special_embeddings = {}
    if args.graph_type == 'constituency':
        # Define special_labels as all values in constituency_dict
        special_labels = set(constituency_dict.values())
        special_embeddings = compute_special_embeddings(list(special_labels), model, tokenizer, args.device)
    embedding_dim = next(iter(special_embeddings.values())).shape[0] if special_embeddings else model.config.hidden_size

    batch_counter = 0
    for batch_idx, graph_batch in enumerate(tqdm(graph_batches, desc='Processing batches', unit='batch')):
        batch_processed_graphs = []
        # Compute the sentence indices for this batch
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + len(graph_batch)
        sentence_batch = sentences[start_idx:end_idx]
        for idx_in_batch, (sentence, graph) in enumerate(tqdm(zip(sentence_batch, graph_batch), total=len(graph_batch), desc=f'Processing graphs in batch {batch_idx}', leave=False, unit='graph')):
            if args.graph_type == 'constituency':
                clean_graph_whitespace_nodes(graph)
                normalize_special_labels(graph)
                validate_graph_structure(graph, graph_idx=start_idx + idx_in_batch)
                word_embs = get_word_embeddings(sentence, model, tokenizer, args.device)
                word_idx = 0
                for nid, data in graph.nodes(data=True):
                    if is_special_label(data['label']):
                        data['embedding'] = special_embeddings.get(data['label'], np.zeros(embedding_dim))
                    else:
                        data['embedding'] = word_embs[word_idx] if word_idx < len(word_embs) else np.zeros_like(word_embs[0])
                        word_idx += 1
            else:
                # Use Stanza to tokenize the sentence as was done during tree generation
                import stanza
                if not hasattr(main, 'stanza_pipeline'):
                    main.stanza_pipeline = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, use_gpu=False)
                stanza_doc = main.stanza_pipeline(sentence)
                stanza_words = [word.text for sent in stanza_doc.sentences for word in sent.words]
                word_embs = get_word_embeddings(' '.join(stanza_words), model, tokenizer, args.device)
                if len(stanza_words) != len(word_embs):
                    raise ValueError(f"Stanza token count ({len(stanza_words)}) and embedding count ({len(word_embs)}) mismatch.")
                # Robust alignment: for each stanza token, find the first unmatched graph node with matching text
                stanza_words_norm = [w.strip().lower() for w in stanza_words]
                graph_word_nodes = [(nid, data) for nid, data in graph.nodes(data=True) if data.get('text', None) is not None]
                node_matched = [False] * len(graph_word_nodes)
                for token_idx, stanza_token in enumerate(stanza_words_norm):
                    found = False
                    for node_idx, (nid, data) in enumerate(graph_word_nodes):
                        node_word_norm = data['text'].strip().lower()
                        if not node_matched[node_idx] and node_word_norm == stanza_token:
                            data['embedding'] = word_embs[token_idx]
                            node_matched[node_idx] = True
                            found = True
                            break
                    if not found:
                        data['embedding'] = np.zeros_like(word_embs[0])
                        

                for (matched, (nid, data)) in zip(node_matched, graph_word_nodes):
                    if not matched:
                        data['embedding'] = np.zeros_like(word_embs[0])

            batch_processed_graphs.append(graph)
        batch_path = os.path.join(
            args.output_dir,
            f'{args.graph_type}_{args.split}_batch_{batch_idx:04d}_graphs_with_embeddings.pkl'
        )
        with open(batch_path, 'wb') as f:
            pkl.dump(batch_processed_graphs, f)
        print(f"Saved batch {batch_idx} with {len(batch_processed_graphs)} graphs to {batch_path}")

if __name__ == "__main__":
    main()
