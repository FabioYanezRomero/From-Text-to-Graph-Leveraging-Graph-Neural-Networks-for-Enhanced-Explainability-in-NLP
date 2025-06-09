import os
import sys
import pickle as pkl
import torch
import networkx as nx
from graph_data_processor import create_word_graphs
from utils import load_embeddings, load_special_embeddings, load_labels

# ---- CONFIGURATION ----
EMBEDDINGS_DIR = '/app/src/Clean_Code/output/embeddings/stanfordnlp/sst2/train'
DATASET = 'stanfordnlp_sst2'
EMBEDDING_MODEL = 'stanfordnlp_sst2'
SPLIT = 'train'
SPECIAL_EMB_DIR = '/app/src/Clean_Code/output/embeddings/stanfordnlp/sst2/train/special_embeddings/google-bert_bert-base-uncased/'
LLM_DIR = '/app/src/Clean_Code/output/finetuned_llms/stanfordnlp'
MODALITY = 'constituency'
BATCH_SIZE = 128  # User-defined


def main():
    import numpy as np
    # Step 1: List and sort all tree files
    tree_base_dir = '/app/src/Clean_Code/output/text_trees'
    tree_dir = os.path.join(
        tree_base_dir,
        '/'.join(DATASET.split('_')),
        SPLIT,
        MODALITY,
    )
    if not os.path.exists(tree_dir):
        raise FileNotFoundError(f"Tree directory not found: {tree_dir}")
    tree_files = sorted([f for f in os.listdir(tree_dir) if f.endswith('.pkl')], key=lambda x: int(x.split('.')[0]))
    tree_paths = [os.path.join(tree_dir, fname) for fname in tree_files]

    # Step 2: Load LLM labels for best epoch
    labels_dict = load_labels(label_source="llm", split=SPLIT, dataset_name="stanfordnlp/sst2", llm_dir=LLM_DIR)

    # Step 3: Prepare for on-demand chunked embedding loading
    embedding_chunks_dir = os.path.join(EMBEDDINGS_DIR, EMBEDDING_MODEL, 'embedding_chunks')
    if not os.path.exists(embedding_chunks_dir):
        raise FileNotFoundError(f"Embedding chunks directory not found: {embedding_chunks_dir}")
    embedding_chunk_files = sorted(
        [f for f in os.listdir(embedding_chunks_dir) if f.endswith('.npz')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    # Step 4: Get chunk sizes and offsets
    chunk_sizes = []
    for fname in embedding_chunk_files:
        path = os.path.join(embedding_chunks_dir, fname)
        data = np.load(path, allow_pickle=True)
        chunk_sizes.append(len(data['word_embeddings']))
    chunk_offsets = [0]
    for sz in chunk_sizes:
        chunk_offsets.append(chunk_offsets[-1] + sz)

    def load_embedding_chunk(chunk_idx):
        path = os.path.join(embedding_chunks_dir, f'embeddings_chunk_{chunk_idx}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding chunk not found: {path}")
        data = np.load(path, allow_pickle=True)
        word_embeddings = data['word_embeddings']
        return word_embeddings

    # Step 5: Load special embeddings
    special_embs = load_special_embeddings(SPECIAL_EMB_DIR)
    if special_embs is None:
        print("Warning: Special embeddings not found.")

    # Step 6: No up-front flattening of all trees or file sources
    # Step 7: Build emb_word_embeddings as a list of lists (one per graph)
    emb_word_embeddings = []
    for chunk_idx in range(len(embedding_chunk_files)):
        word_embeddings_chunk = load_embedding_chunk(chunk_idx)  # list of (word, embedding) tuples for each graph in this chunk
        emb_word_embeddings.extend(word_embeddings_chunk)

    # Step 8: Global progress bar for all graphs
    from tqdm import tqdm
    total_graphs = sum(len(pkl.load(open(f, 'rb'))[0][0]) for f in tree_paths)
    progress = tqdm(total=total_graphs, desc="Processing all graphs")
    offset = 0
    import torch
    output_dir = f"/app/src/Clean_Code/output/pytorch_geometric/{DATASET}/{SPLIT}"
    os.makedirs(output_dir, exist_ok=True)

    for file_idx, file_path in enumerate(tree_paths):
        with open(file_path, 'rb') as f:
            trees = pkl.load(f)
        num_graphs_in_file = len(trees[0][0])
        file_embeddings = emb_word_embeddings[offset:offset + num_graphs_in_file]
        file_labels = [labels_dict.get(i, None) for i in range(offset, offset + num_graphs_in_file)]

        graph_idx = 0
        while graph_idx < num_graphs_in_file:
            batch_trees = trees[0][0][graph_idx:graph_idx + BATCH_SIZE]
            batch_embeddings = file_embeddings[graph_idx:graph_idx + BATCH_SIZE]
            batch_labels = file_labels[graph_idx:graph_idx + BATCH_SIZE]

            graphs = create_word_graphs(
                batch_embeddings,
                batch_trees,
                special_embeddings=special_embs,
                label_list=batch_labels,
                show_progress=True
            )
            # Save the batch of graphs
            batch_number = graph_idx // BATCH_SIZE
            output_file = os.path.join(
                output_dir,
                f"file_{file_idx:03d}_batch_{batch_number:03d}.pt"
            )
            torch.save(graphs, output_file)
            progress.update(len(graphs))
            graph_idx += BATCH_SIZE

        offset += num_graphs_in_file
    progress.close()

if __name__ == "__main__":
    main()
