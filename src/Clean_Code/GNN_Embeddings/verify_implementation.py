import os
import torch
import pickle as pkl
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import sys
import glob

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our functions
from GNN_Embeddings.embedding_generator import generate_special_embeddings, save_embeddings
from GNN_Embeddings.gnn_data_processor import load_embeddings, create_word_graphs

def count_graph_files(dataset_name, split):
    """Count the number of graph files for a dataset split"""
    if dataset_name == "ag_news":
        path = f"/app/src/Clean_Code/output/text_graphs/SetFit/{dataset_name}/{split}/constituency/"
    elif dataset_name == "sst2":
        path = f"/app/src/Clean_Code/output/text_graphs/stanfordnlp/{dataset_name}/{split}/constituency/"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    files = glob.glob(os.path.join(path, "*.pkl"))
    return len(files)

def load_graph_file(dataset_name, split, file_idx):
    """Load a graph file for a dataset split"""
    if dataset_name == "ag_news":
        path = f"/app/src/Clean_Code/output/text_graphs/SetFit/{dataset_name}/{split}/constituency/{file_idx}.pkl"
    elif dataset_name == "sst2":
        path = f"/app/src/Clean_Code/output/text_graphs/stanfordnlp/{dataset_name}/{split}/constituency/{file_idx}.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    try:
        with open(path, 'rb') as f:
            return pkl.load(f)
    except Exception as e:
        logger.error(f"Error loading graph file {path}: {str(e)}")
        return []

def generate_mock_embeddings(dataset_name, split, chunk_size=1):
    """Generate mock embeddings for testing"""
    logger.info(f"Generating mock embeddings for {dataset_name} {split}")
    
    # Count total graph files
    num_files = count_graph_files(dataset_name, split)
    logger.info(f"Found {num_files} graph files")
    
    # Create output directory
    output_dir = f"/app/src/Clean_Code/output/embeddings/{dataset_name}/{split}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate mock embeddings for each chunk
    for chunk_idx in range(0, num_files, chunk_size):
        chunk_word_embeddings = []
        chunk_sentence_embeddings = []
        chunk_texts = []
        chunk_labels = []
        
        # Process files in this chunk
        for file_idx in range(chunk_idx, min(chunk_idx + chunk_size, num_files)):
            # Load graph file
            graph_data = load_graph_file(dataset_name, split, file_idx)
            if not graph_data:
                continue
                
            # Extract text and label from graph
            for item in graph_data:
                # Each item is a tuple (text_list, label_tensor)
                text = item[0]
                
                # Handle label tensor - use first value if it's a tensor with multiple values
                if isinstance(item[1], torch.Tensor):
                    if item[1].numel() > 1:
                        # Take the first label if there are multiple
                        label = item[1][0].item()
                    else:
                        label = item[1].item()
                else:
                    label = item[1]
                
                # Generate mock word embeddings (would normally come from extract_embeddings)
                word_emb = np.random.rand(len(text), 768)
                sentence_emb = np.random.rand(768)
                
                chunk_word_embeddings.append(word_emb)
                chunk_sentence_embeddings.append(sentence_emb)
                chunk_texts.append(text)
                chunk_labels.append(label)
        
        # Skip empty chunks
        if not chunk_texts:
            logger.warning(f"Skipping empty chunk {chunk_idx}")
            continue
            
        # Save chunk embeddings
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        with open(os.path.join(chunk_dir, "word_embeddings.pkl"), 'wb') as f:
            pkl.dump(chunk_word_embeddings, f)
        
        with open(os.path.join(chunk_dir, "sentence_embeddings.pkl"), 'wb') as f:
            pkl.dump(chunk_sentence_embeddings, f)
        
        with open(os.path.join(chunk_dir, "metadata.pkl"), 'wb') as f:
            pkl.dump({"texts": chunk_texts, "labels": chunk_labels}, f)
        
        logger.info(f"Saved chunk {chunk_idx} with {len(chunk_texts)} samples")
    
    return output_dir

def generate_real_special_embeddings():
    """Generate real special embeddings for constituency tokens"""
    logger.info("Generating real special embeddings")
    
    # Load model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate special embeddings
    special_embeddings = generate_special_embeddings(model, tokenizer, device='cpu')
    
    # Save special embeddings
    output_dir = "/app/src/Clean_Code/output/embeddings/special"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "special_embeddings.pkl"), 'wb') as f:
        pkl.dump(special_embeddings, f)
    
    logger.info(f"Saved {len(special_embeddings)} special embeddings to {output_dir}")
    return special_embeddings

def verify_graph_creation(dataset_name, split, special_embeddings):
    """Verify graph creation using our embeddings"""
    logger.info(f"Verifying graph creation for {dataset_name} {split}")
    
    # Load embeddings
    embeddings_dir = f"/app/src/Clean_Code/output/embeddings/{dataset_name}/{split}"
    
    # Get all chunk directories
    chunk_dirs = sorted(glob.glob(os.path.join(embeddings_dir, "chunk_*")))
    logger.info(f"Found {len(chunk_dirs)} chunk directories")
    
    total_graphs = 0
    
    for chunk_dir in chunk_dirs:
        try:
            # Load chunk embeddings
            with open(os.path.join(chunk_dir, "word_embeddings.pkl"), 'rb') as f:
                word_embeddings = pkl.load(f)
            
            with open(os.path.join(chunk_dir, "sentence_embeddings.pkl"), 'rb') as f:
                sentence_embeddings = pkl.load(f)
            
            with open(os.path.join(chunk_dir, "metadata.pkl"), 'rb') as f:
                metadata = pkl.load(f)
            
            texts = metadata["texts"]
            labels = metadata["labels"]
            
            # Create graphs
            graphs = create_word_graphs(
                word_embeddings,
                sentence_embeddings,
                texts,
                labels,
                special_embeddings=special_embeddings
            )
            
            total_graphs += len(graphs)
            
            # Verify graphs - just check a few for brevity
            sample_size = min(3, len(graphs))
            for i, graph in enumerate(graphs[:sample_size]):
                logger.info(f"Graph {i}: {graph}")
                logger.info(f"Graph {i} shape: {graph.x.shape}")
                
            logger.info(f"Successfully created {len(graphs)} graphs from chunk {os.path.basename(chunk_dir)}")
            
            # Only process a few chunks for demonstration
            if total_graphs > 10:
                logger.info("Processed enough graphs for verification, stopping early")
                break
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_dir}: {str(e)}")
    
    logger.info(f"Created {total_graphs} graphs successfully")
    return total_graphs

def main():
    """Main verification function"""
    # Define datasets to verify
    datasets = ["ag_news", "sst2"]
    splits = ["train"]  # Can add "test", "validation" if needed
    
    # Generate special embeddings
    special_embeddings = generate_real_special_embeddings()
    
    # Process each dataset and split
    for dataset in datasets:
        for split in splits:
            try:
                # Generate mock embeddings
                generate_mock_embeddings(dataset, split, chunk_size=50)
                
                # Verify graph creation
                verify_graph_creation(dataset, split, special_embeddings)
            except Exception as e:
                logger.error(f"Error processing {dataset} {split}: {str(e)}")
                logger.error(f"Traceback: {sys.exc_info()}")

if __name__ == "__main__":
    main()
