import os
import argparse
import pickle as pkl
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from .dicts import constituency_dict
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

def _expr_from_special_label(label: str) -> str:
    """Return the natural-language expression for a special node label.

    Labels may come already pretty-formatted with guillemets (e.g., «NOUN PHRASE»).
    We strip decorative quotes and extra whitespace to form the text we feed to
    the language model for [CLS] pooling.
    """
    if not isinstance(label, str):
        return str(label)
    # strip guillemets and whitespace
    s = label.strip()
    if s.startswith('«') and s.endswith('»') and len(s) >= 2:
        s = s[1:-1]
    return s.strip()

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
            expr = _expr_from_special_label(label)
            inputs = tokenizer(expr, return_tensors='pt').to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            special_embeddings[label] = cls_emb
    return special_embeddings

def _encode_with_offsets(text, tokenizer, device):
    enc = tokenizer(text, return_tensors='pt', return_offsets_mapping=True, truncation=True)
    offsets = enc.pop('offset_mapping')
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, offsets.squeeze(0).tolist()


def _aggregate_subwords(hidden, offsets, spans, special_token_ids, agg='mean'):
    """Aggregate subword vectors into word vectors by span overlap.

    hidden: [seq_len, hidden_dim]
    offsets: list[(start,end)] for each subword token
    spans: list[(start,end)] for each target word/token
    special_token_ids: set of indices to ignore (CLS/SEP etc.)
    """
    out = []
    for (w_start, w_end) in spans:
        idxs = []
        for i, (t_start, t_end) in enumerate(offsets):
            if i in special_token_ids or t_start == t_end:
                continue
            if max(w_start, t_start) < min(w_end, t_end):
                idxs.append(i)
        if not idxs:
            out.append(np.zeros_like(hidden[0]))
        else:
            vecs = hidden[idxs]
            if agg == 'first':
                out.append(vecs[0])
            elif agg == 'sum':
                out.append(vecs.sum(axis=0))
            else:
                out.append(vecs.mean(axis=0))
    return out


def _spans_from_whitespace_words(sentence):
    spans = []
    i = 0
    for w in sentence.split():
        while i < len(sentence) and sentence[i].isspace():
            i += 1
        start = i
        end = start + len(w)
        spans.append((start, end))
        i = end
    return spans


def get_word_embeddings(sentence, model, tokenizer, device, spans=None, agg='mean'):
    """Compute word embeddings by aggregating subwords overlapping given spans.

    - If spans is None, uses whitespace tokenization spans.
    - Aggregation is mean by default (configurable).
    """
    enc, offsets = _encode_with_offsets(sentence, tokenizer, device)
    outputs = model(**enc)
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    input_ids = enc['input_ids'].squeeze(0).tolist()
    # Identify special tokens (CLS/SEP/PAD) by zero-length offsets or known ids
    special_idxs = set(i for i, (s, e) in enumerate(offsets) if e == s)
    # Build spans
    spans = spans or _spans_from_whitespace_words(sentence)
    return _aggregate_subwords(hidden, offsets, spans, special_idxs, agg=agg)

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

def _load_finetuned_weights_if_any(model, weights_path):
    if not weights_path:
        return model
    import torch
    if not os.path.isfile(weights_path):
        print(f"[warn] Weights file not found: {weights_path}; using base model weights.")
        return model
    try:
        sd = torch.load(weights_path, map_location='cpu')
        # unpack common wrappers
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        # strip potential 'module.' prefixes
        cleaned = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith('module.') else k
            cleaned[nk] = v
        # keep only keys present in the base model (ignore classifier heads etc.)
        base_sd = model.state_dict()
        filtered = {k: v for k, v in cleaned.items() if k in base_sd}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[info] Loaded finetuned weights: {weights_path}")
        if missing:
            print(f"[info] Missing keys (ok): {len(missing)}")
        if unexpected:
            print(f"[info] Ignored unexpected keys (heads, etc.): {len(unexpected)}")
    except Exception as e:
        print(f"[warn] Failed to load finetuned weights '{weights_path}': {e}; using base model weights.")
    return model


def main():

    parser = argparse.ArgumentParser(description="Generate constituency or syntactic graphs with node embeddings for GNN training.")
    parser.add_argument('--graph_type', type=str, choices=['constituency', 'syntactic'], help='Type of graph to process', default='syntactic')
    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/sst2')
    parser.add_argument('--split', type=str, default='validation')
    base = os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs')
    parser.add_argument('--tree_dir', type=str, default=f'{base}/graphs/stanfordnlp/sst2/validation/syntactic')
    parser.add_argument('--output_dir', type=str, default=f'{base}/embeddings/stanfordnlp/sst2/validation/syntactic')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--weights_path', type=str, default=os.environ.get('GRAPHTEXT_WEIGHTS_PATH', ''), help='Optional path to a finetuned .pt state_dict to load into the base model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing graphs and sentences')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sentences = load_sentences(args.dataset_name, args.split)
    graph_batches = load_trees(args.tree_dir, args.batch_size)
    total_graphs = sum(len(batch) for batch in graph_batches)
    assert len(sentences) == total_graphs, f"Mismatch: {len(sentences)} sentences, {total_graphs} graphs"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    # Optionally load a finetuned state_dict (e.g., model_epoch_X.pt) into the base model
    model = _load_finetuned_weights_if_any(model, args.weights_path)
    model = model.to(args.device)

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
                # Detect whether graph nodes are word-level or token-level
                nodes = list(graph.nodes(data=True))
                node_types = {d.get('type') for _, d in nodes}
                is_token_graph = node_types == {'token'} or ('token' in node_types and 'word' not in node_types)

                if is_token_graph:
                    # Token graphs: use HF tokenizer tokens without specials; one-to-one mapping by node id
                    enc = tokenizer(sentence, add_special_tokens=False, return_tensors='pt').to(args.device)
                    with torch.no_grad():
                        outputs = model(**enc)
                        hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()  # [seq_len, hid]
                    ordered_nodes = sorted([(nid, d) for nid, d in nodes], key=lambda x: x[0])
                    if len(ordered_nodes) != hidden.shape[0]:
                        print(f"[warn] token count mismatch nodes({len(ordered_nodes)}) vs model tokens({hidden.shape[0]}); truncating")
                    m = min(len(ordered_nodes), hidden.shape[0])
                    for i in range(m):
                        ordered_nodes[i][1]['embedding'] = hidden[i]
                    if m > 0:
                        zero_vec = np.zeros_like(hidden[0])
                        for _, d in ordered_nodes[m:]:
                            d['embedding'] = zero_vec
                else:
                    # Word graphs: Build character spans via Stanza tokenization, then aggregate subwords per span
                    import stanza
                    if not hasattr(main, 'stanza_pipeline'):
                        main.stanza_pipeline = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, use_gpu=False)
                    doc = main.stanza_pipeline(sentence)
                    spans = []
                    words = []
                    for sent in doc.sentences:
                        for tok in sent.tokens:
                            start = getattr(tok, 'start_char', None)
                            end = getattr(tok, 'end_char', None)
                            if start is None or end is None:
                                if not words:
                                    spans = _spans_from_whitespace_words(sentence)
                                    words = [w for s in doc.sentences for t in s.tokens for w in [t.text]]
                                else:
                                    pass
                            else:
                                spans.append((start, end))
                                words.append(tok.text)
                    word_embs = get_word_embeddings(sentence, model, tokenizer, args.device, spans=spans)
                    ordered_nodes = sorted([(nid, d) for nid, d in nodes], key=lambda x: x[0])
                    if len(ordered_nodes) != len(word_embs):
                        print(f"[warn] length mismatch words({len(ordered_nodes)}) vs embs({len(word_embs)}); truncating")
                    m = min(len(ordered_nodes), len(word_embs))
                    for i in range(m):
                        ordered_nodes[i][1]['embedding'] = word_embs[i]
                    if m > 0:
                        zero_vec = np.zeros_like(word_embs[0])
                        for _, d in ordered_nodes[m:]:
                            d['embedding'] = zero_vec

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
