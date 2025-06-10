import os
import argparse
import pickle as pkl
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

import re

def load_trees(tree_dir, split, batch_size):
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

# Provided constituency_dict (tag -> pretty label)
constituency_dict = {
    # POS TAGS
    'CC': '«COORDINATING CONJUNCTION»',
    'CD': '«CARDINAL NUMBER»',
    'DT': '«DETERMINER»',
    'EX': '«EXISTENTIAL THERE»',
    'FW': '«FOREIGN WORD»',
    'IN': '«PREPOSITION OR SUBORDINATING CONJUNCTION»',
    'JJ': '«ADJECTIVE»',
    'JJR': '«ADJECTIVE, COMPARATIVE»',
    'JJS': '«ADJECTIVE, SUPERLATIVE»',
    'LS': '«LIST MARKER»',
    'MD': '«MODAL VERB»',
    'NN': '«NOUN, SINGULAR OR MASS»',
    'NNS': '«NOUN, PLURAL»',
    'NNP': '«PROPER NOUN, SINGULAR»',
    'NNPS': '«PROPER NOUN, PLURAL»',
    'PDT': '«PREDETERMINER»',
    'POS': '«POSSESSIVE ENDING»',
    'PRP': '«PERSONAL PRONOUN»',
    'PRP$': '«POSSESSIVE PRONOUN»',
    'RB': '«ADVERB»',
    'RBR': '«ADVERB, COMPARATIVE»',
    'RBS': '«ADVERB, SUPERLATIVE»',
    'RP': '«PARTICLE»',
    'SYM': '«SYMBOL»',
    'TO': '«TO»',
    'UH': '«INTERJECTION»',
    'VB': '«VERB, BASE FORM»',
    'VBD': '«VERB, PAST TENSE»',
    'VBG': '«VERB, GERUND OR present participle»',
    'VBN': '«VERB, past participle»',
    'VBP': '«VERB, non-3rd person singular present»',
    'VBZ': '«VERB, 3rd person singular present»',
    'WDT': '«WH-DETERMINER»',
    'WP': '«WH-PRONOUN»',
    'WP$': '«WH-POSSESSIVE PRONOUN»',
    'WRB': '«WH-ADVERB»',
    # CONSTITUENCY TAGS
    'ROOT': '«ROOT»',  
    'SENTENCE': '«SENTENCE»', 
    'NP': '«NOUN PHRASE»',
    'VP': '«VERB PHRASE»',
    'PP': '«PREPOSITIONAL PHRASE»',
    'ADJP': '«ADJECTIVE PHRASE»',
    'ADVP': '«ADVERB PHRASE»',
    'SBAR': '«SUBORDINATE CLAUSE»',
    'PRT': '«PARTICLE»',
    'INTJ': '«INTERJECTION»',
    'CONJP': '«CONJUCTION PHRASE»',
    'LST': '«LIST MARKER»',
    'UCP': '«UNLIKE COORDINATED PHRASE»',
    'PRN': '«PARENTETICAL»',
    'FRAG': '«FRAGMENT»',
    'SINV': '«INVERTED SENTENCE»',
    'SBARQ': '«SUBORDINATE CLAUSE QUESTION»',
    'SQ': '«QUESTION»',
    'WHADJP': '«WH-ADJECTIVE PHRASE»',
    'WHAVP': '«WH-ADVERB PHRASE»',
    'WHNP': '«WH-NOUN PHRASE»',
    'WHPP': '«WH-PREPOSITIONAL PHRASE»',
    'RRC': '«REDUCED RELATIVE CLAUSE»',
    'NX': '«NOUN PHRASE (NO HEAD)»',
    'WHADVP': '«WH-ADVERB PHRASE»',
    'QP': '«QUANTIFIER PHRASE»',
    'NAC': '«NOT A CONSTITUENT»',
    'X': '«UNKNOWN»',
    'HYPH': '«HYPHEN»',
    'HVS': '«HYPHENATED VERB SUBSTITUTION»',
    'NML': '«NOMINALIZATION»',
    'LRB': '«LEFT PARENTHESIS»',
    'RRB': '«RIGHT PARENTHESIS»',
    '-LRB-': '«LEFT PARENTHESIS»',
    '-RRB-': '«RIGHT PARENTHESIS»',
}

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
    """Get wordpiece embeddings for each word in the sentence."""
    inputs = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, truncation=True).to(device)
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    # Map wordpieces back to words
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
    offsets = inputs['offset_mapping'].squeeze(0).tolist()
    words = sentence.split()
    word_embs = []
    for word in words:
        # Find the token indices that correspond to this word
        indices = [i for i, (tok, (start, end)) in enumerate(zip(tokens, offsets)) if tok not in [tokenizer.cls_token, tokenizer.sep_token] and word in sentence[start:end]]
        if indices:
            emb = hidden[indices].mean(axis=0)
        else:
            raise ValueError(f"Word '{word}' not found in sentence '{sentence}'")
        word_embs.append(emb)
    return word_embs

def assign_embeddings_to_graph(graph, sentence, word_embs, special_embeddings):
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

def main():
    parser = argparse.ArgumentParser(description='Generate NetworkX constituency graphs with embeddings')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--tree_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sentences = load_sentences(args.dataset_name, args.split)
    graphs = load_trees(args.tree_dir, args.split)

    assert len(sentences) == len(graphs), f"Mismatch: {len(sentences)} sentences, {len(graphs)} graphs"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)

    # Collect all unique special labels from all graphs
    special_labels = set()
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            if is_special_label(data['label']):
                special_labels.add(data['label'])
    # Optionally save found special labels for debugging
    debug_labels_path = os.path.join(args.output_dir, f'{args.split}_special_labels.txt')
    with open(debug_labels_path, 'w') as f:
        for label in sorted(special_labels):
            f.write(f'{label}\n')
    print(f"Found {len(special_labels)} special labels. Saved to {debug_labels_path}")

    special_embeddings = compute_special_embeddings(list(special_labels), model, tokenizer, args.device)
    embedding_dim = next(iter(special_embeddings.values())).shape[0] if special_embeddings else model.config.hidden_size

    processed_graphs = []
    for idx, (sentence, graph) in enumerate(tqdm(zip(sentences, graphs), total=len(sentences), desc='Processing graphs')):
        validate_graph_structure(graph, graph_idx=idx)
        word_embs = get_word_embeddings(sentence, model, tokenizer, args.device)
        word_idx = 0
        for nid, data in graph.nodes(data=True):
            if is_special_label(data['label']):
                data['embedding'] = special_embeddings.get(data['label'], np.zeros(embedding_dim))
            else:
                data['embedding'] = word_embs[word_idx] if word_idx < len(word_embs) else np.zeros(embedding_dim)
                word_idx += 1
        processed_graphs.append(graph)

    # Save all graphs as a pickle file
    with open(os.path.join(args.output_dir, f'{args.split}_graphs_with_embeddings.pkl'), 'wb') as f:
        pkl.dump(processed_graphs, f)
    print(f"Saved {len(processed_graphs)} graphs to {args.output_dir}")

if __name__ == "__main__":
    main()
