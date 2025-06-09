from logging import getLogger
import os
import glob
import json
import pickle as pkl
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
import logging
logger = getLogger(__name__)


def load_embeddings(input_dir, batch_size=None):
    """
    Generator that yields (texts, word_embeddings) for each batch of samples.
    If batch_size is None, yields all samples in a chunk at once.
    Args:
        input_dir: Directory containing embeddings
        batch_size: Number of samples per batch (optional)
    Yields:
        texts: List of token lists (words) per sample in the batch
        word_embeddings: List of lists of word embedding vectors per sample in the batch
    """
    import glob
    import numpy as np
    import os
    import re

    dataset_name = input_dir.split('/')[-3:-1]
    dataset_name = '_'.join(dataset_name)
    chunks_dir = os.path.join(input_dir, dataset_name, 'embedding_chunks')
    def chunk_num_key(path):
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else -1
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, '*.npz')), key=chunk_num_key)

    buffer_texts = []
    buffer_word_embeddings = []

    for chunk_file in chunk_files:
        with np.load(chunk_file, allow_pickle=True) as data:
            chunk_word_embeddings = data['word_embeddings']
            texts = [[w for w, _ in sample] for sample in chunk_word_embeddings]
            word_embeddings = [[e for _, e in sample] for sample in chunk_word_embeddings]
            if batch_size is None:
                yield texts, word_embeddings
            else:
                buffer_texts.extend(texts)
                buffer_word_embeddings.extend(word_embeddings)
                while len(buffer_texts) >= batch_size:
                    yield (buffer_texts[:batch_size], buffer_word_embeddings[:batch_size])
                    buffer_texts = buffer_texts[batch_size:]
                    buffer_word_embeddings = buffer_word_embeddings[batch_size:]
    # Yield any remaining samples
    if batch_size is not None and buffer_texts:
        yield (buffer_texts, buffer_word_embeddings)


def load_labels(label_source: str, split: str = "train", dataset_name: str = None, llm_dir: str = None):
    """
    Loads labels for graph creation.
    Args:
        label_source: "hf" for HuggingFace, "llm" for LLM-predicted
        split: dataset split ("train", "test", etc.)
        dataset_name: HuggingFace dataset name/path (if using HuggingFace)
        llm_pred_dir: Directory containing LLM predictions (if using LLM)
        model_dir: Directory containing model checkpoints and classification reports (if using LLM with best epoch)
        label_column: Column name for labels in HuggingFace dataset
    Returns:
        labels: List of labels (aligned with sample order)
    """
    labels = []
    if label_source == "hf":
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("You must install the 'datasets' library to use HuggingFace dataset label loading.")
        if dataset_name is None:
            raise ValueError("dataset_name must be provided for HuggingFace label loading.")
        ds = load_dataset(dataset_name, split=split)
        labels = list(ds['labels'])
    elif label_source == "llm":
        if llm_dir is None:
            raise ValueError("llm_dir must be provided for LLM prediction label loading.")
        
        best_epoch = find_best_epoch(llm_dir)
        if best_epoch is None:
            raise ValueError("Could not determine best epoch, please check the llm_dir")
    else:
        raise ValueError("label_source must be either 'hf' or 'llm'.")

    # find the predictions.json file in the llm_dir through that dir and their subdirectories
    for root, dirs, files in os.walk(llm_dir):
        if 'predictions.json' in files:
            predictions_file = os.path.join(root, 'predictions.json')
            break
    
    if predictions_file is None:
        raise FileNotFoundError("predictions.json file not found in the llm_dir or its subdirectories")
    
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)
    labels = [item.get('predicted_label') for item in predictions_data if item.get('dataset') == split]
    labels_dict = {item['data_index']: item['predicted_label'] for item in predictions_data if item.get('dataset') == split and item.get('epoch') == best_epoch}
    return labels_dict

def load_special_embeddings(embeddings_dir):
    """
    Load special embeddings for constituency tokens from a specified directory.
    Args:
        embeddings_dir: Directory containing special embeddings
    Returns:
        special_embeddings: Dictionary of special embeddings, or None if not found
    """
    import os
    import pickle as pkl
    # Try several possible filenames
    possible_files = [
        os.path.join(embeddings_dir, 'special_embeddings.pkl'),
        os.path.join(embeddings_dir, 'special.pkl'),
        os.path.join(embeddings_dir, 'special_tokens.pkl')
    ]
    for path in possible_files:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pkl.load(f)
    print(f"[WARNING] No special embeddings found in {embeddings_dir}")
    return None

def find_best_epoch(model_dir, metric='f1-score', split='validation'):
    """Find the best epoch based on validation F1 score
    
    Args:
        model_dir: Directory containing model checkpoints and classification reports
        metric: Metric to use for selecting the best epoch (default: 'f1-score')
        split: Data split to use for evaluation (default: 'validation')
        
    Returns:
        Best epoch number or None if it cannot be determined
    """
    import glob
    import json
    import os
    
    # Find all classification reports for the specified split
    report_files = glob.glob(os.path.join(model_dir, f"classification_report_{split}_epoch*.json"))
    
    # If no validation reports are found, try test reports
    if not report_files and split == 'validation':
        logger.warning(f"No classification reports found for validation split, trying test split instead")
        report_files = glob.glob(os.path.join(model_dir, f"classification_report_test_epoch*.json"))
        split = 'test'
    
    if not report_files:
        logger.warning(f"No classification reports found in {model_dir} for split {split}")
        return None
    
    best_score = -1
    best_epoch = None
    
    # Iterate through all epochs
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Extract epoch number from filename
            epoch = int(os.path.basename(report_file).split('_')[-1].split('.')[0].replace('epoch', ''))
            
            # For multi-class classification, use macro avg F1-score
            # For binary classification, use weighted avg F1-score
            if 'macro avg' in report and metric in report['macro avg']:
                score = report['macro avg'][metric]
            elif 'weighted avg' in report and metric in report['weighted avg']:
                score = report['weighted avg'][metric]
            else:
                logger.warning(f"Metric {metric} not found in report {report_file}")
                continue
            
            logger.debug(f"Epoch {epoch}: {metric} = {score:.4f}")
            
            # Check if this is the best score so far
            if score > best_score:
                best_score = score
                best_epoch = epoch
        
        except Exception as e:
            logger.error(f"Error processing report file {report_file}: {e}")
            continue
    
    if best_epoch is not None:
        logger.info(f"Best {metric}: {best_score:.4f} (Epoch {best_epoch})")
    
    return best_epoch



def load_llm_predictions(predictions_file, split='test'):
    """Load LLM predictions from a JSON file using the best epoch
    
    Args:
        predictions_file: Path to the JSON file containing LLM predictions
        split: Data split to load predictions for ('train', 'validation', 'test')
        
    Returns:
        Dictionary mapping data index to predicted label
    """
    # Log the function call for debugging
    logger.debug(f"Loading LLM predictions from {predictions_file} for split {split}")
    import json
    import os
    import sys
    
    # Map split names to match those in the predictions file
    split_map = {
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    }
    
    split_name = split_map.get(split, split)
    
    try:
        # Get the model directory from the predictions file path
        model_dir = os.path.dirname(predictions_file)
        
        # Find the best epoch by analyzing the classification reports
        best_epoch = find_best_epoch(model_dir)
        
        if best_epoch is None:
            raise ValueError("Could not determine best epoch, please check the model directory.")
        
        # Check if there's an epoch numbering mismatch between classification reports and predictions
        # Load all predictions to check epoch numbering
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        # Get the unique epochs in the predictions file
        pred_epochs = set(item.get('epoch') for item in predictions_data if item.get('epoch') is not None)
        logger.info(f"Epochs in predictions file: {sorted(pred_epochs)}")
        
        # Check if the best_epoch exists in the predictions file
        if best_epoch not in pred_epochs:
            logger.warning(f"Best epoch {best_epoch} not found in predictions file epochs {sorted(pred_epochs)}")
            
            # Check if there's a 0/1-based indexing mismatch
            if best_epoch - 1 in pred_epochs:
                logger.info(f"Found epoch {best_epoch-1} in predictions file, likely a 0/1-based indexing mismatch")
                best_epoch = best_epoch - 1
            elif best_epoch + 1 in pred_epochs:
                logger.info(f"Found epoch {best_epoch+1} in predictions file, likely a 0/1-based indexing mismatch")
                best_epoch = best_epoch + 1
            else:
                # If best epoch is not found at all, use the latest available epoch
                if pred_epochs:
                    latest_epoch = max(pred_epochs)
                    logger.info(f"Using latest available epoch {latest_epoch} from predictions file")
                    best_epoch = latest_epoch
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        # Create a dictionary mapping data index to predicted label for the best epoch
        predictions = {}
        matched_epoch_count = 0
        
        for item in predictions_data:
            if item.get('dataset') == split_name and item.get('epoch') == best_epoch:
                predictions[item.get('data_index')] = item.get('predicted_label')
                matched_epoch_count += 1
        
        # If we didn't find any predictions for the best epoch, try adjacent epochs
        if matched_epoch_count == 0:
            logger.warning(f"No predictions found for epoch {best_epoch}, trying adjacent epochs")
            
            # Try epochs +/- 1 from the best epoch
            for offset in [-1, 1]:
                test_epoch = best_epoch + offset
                test_count = 0
                
                for item in predictions_data:
                    if item.get('dataset') == split_name and item.get('epoch') == test_epoch:
                        test_count += 1
                
                if test_count > 0:
                    logger.info(f"Found {test_count} predictions for epoch {test_epoch}, using this instead")
                    
                    # Use this epoch instead
                    for item in predictions_data:
                        if item.get('dataset') == split_name and item.get('epoch') == test_epoch:
                            predictions[item.get('data_index')] = item.get('predicted_label')
                    
                    best_epoch = test_epoch
                    break
        
        # Log some sample predictions for debugging
        sample_items = list(predictions.items())[:5]
        logger.debug(f"Sample predictions (data_index: predicted_label): {sample_items}")
        
        # Log statistics about the predictions
        logger.debug(f"Loaded {len(predictions)} LLM predictions for {split_name} split from best epoch {best_epoch}")
        
        # Log the range of data indices
        if predictions:
            min_idx = min(predictions.keys())
            max_idx = max(predictions.keys())
            logger.debug(f"Data index range: {min_idx} to {max_idx}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error loading LLM predictions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def determine_embedding_dimension(word_embeddings):
    """Determine the embedding dimension from available embeddings"""
    import torch
    
    embedding_dim = None
    if word_embeddings and len(word_embeddings) > 0:
        if isinstance(word_embeddings[0], torch.Tensor):
            embedding_dim = word_embeddings[0].shape[0]
        else:
            embedding_dim = len(word_embeddings[0])
    else:
        raise ValueError("No word embeddings provided")
    
    return embedding_dim



def save_graphs(graphs, output_dir, batch_size=32, num_workers=4):
    """Save a list of PyTorch Geometric Data objects to disk.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        output_dir: Directory to save the graphs
        batch_size: Number of graphs to save in each file
        num_workers: Number of worker processes to use
    """
    import os
    import torch
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataLoader to handle batching
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Save each batch
    for batch_idx, batch in enumerate(loader):
        batch_path = os.path.join(output_dir, f'batch_{batch_idx:04d}.pt')
        try:
            torch.save(batch, batch_path)
            logger.info(f"Saved batch {batch_idx} with {len(batch)} graphs to {batch_path}")
        except Exception as e:
            logger.error(f"Error saving batch {batch_idx}: {str(e)}")
    
    logger.info(f"Saved {len(graphs)} graphs in {len(loader)} batches to {output_dir}")


def find_best_epoch(model_dir, metric='f1-score', split='test'):
    """
    Find the best epoch by walking the directory, looking for classification_report_test_epoch*.json,
    and selecting the epoch with the best metric value (macro avg or weighted avg).
    Args:
        model_dir: Directory containing model checkpoints and classification reports
        metric: Metric to use for selecting the best epoch (default: 'f1-score')
        split: Unused, always uses test split as per user request
    Returns:
        Best epoch number or None if it cannot be determined
    """
    import os
    import json
    best_score = -1
    best_epoch = None
    for root, dirs, files in os.walk(model_dir):
        for fname in files:
            if fname.startswith('classification_report_test_epoch') and fname.endswith('.json'):
                try:
                    epoch_str = fname.split('epoch')[-1].split('.')[0]
                    epoch = int(epoch_str)
                    fpath = os.path.join(root, fname)
                    with open(fpath, 'r') as f:
                        report = json.load(f)
                    score = None
                    if 'macro avg' in report and metric in report['macro avg']:
                        score = report['macro avg'][metric]
                    elif 'weighted avg' in report and metric in report['weighted avg']:
                        score = report['weighted avg'][metric]
                    else:
                        logger.warning(f"Metric {metric} not found in report {fname}")
                        continue
                    logger.debug(f"Epoch {epoch}: {metric} = {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                except Exception as e:
                    logger.error(f"Error processing report file {fname}: {e}")
                    continue
    if best_epoch is not None:
        logger.info(f"Best {metric}: {best_score:.4f} (Epoch {best_epoch})")
    return best_epoch



def process_embeddings_batch(batch_word_embeddings, batch_texts, batch_labels, config, split_output_dir, batch_idx, special_embeddings=None):
    """Process a single batch of embeddings"""
    # Log label source
    label_source = config.get('label_source', 'original')
    if label_source == 'llm':
        logger.debug(f"Using LLM predictions as labels for batch {batch_idx}")
    
    # Extract dataset name from the path
    # We need to use the original dataset_name passed from process_embeddings
    # Instead of trying to extract it from the split_output_dir which might be incorrect
    
    # Get the dataset name from the config
    dataset_name = config.get('dataset_name', 'stanfordnlp/sst2')
    
    # Log the dataset name we're using
    logger.debug(f"Using dataset name: {dataset_name} for graph creation")

    
    # Extract split from the path
    split = os.path.basename(split_output_dir).replace('_llm_labels', '')
    
    # Create graphs
    batch_graphs = create_word_graphs(
        batch_word_embeddings,
        batch_texts,
        batch_labels,
        dataset_name,
        split,
        config['edge_type'],
        show_progress=False  # Disable progress bar for batch processing
    )
    
    # Save batch graphs with consistent numerical naming to preserve original order
    # Format with leading zeros to ensure correct sorting
    batch_graphs_path = os.path.join(split_output_dir, f"graphs_{config['edge_type']}_batch_{batch_idx:04d}.pkl")
    
    with open(batch_graphs_path, 'wb') as f:
        pkl.dump(batch_graphs, f)
    
    logger.debug(f"Saved {len(batch_graphs)} graphs to {batch_graphs_path}")
    
    return len(batch_graphs)


def process_embeddings(dataset_name, embeddings_dir, batch_size=200, edge_type='constituency', label_source='original', llm_predictions=None):
    """Process embeddings to create graph structures
    
    Args:
        dataset_name: Name of the dataset
        embeddings_dir: Directory containing embeddings
        batch_size: Batch size for processing
        edge_type: Type of edges to create (constituency)
        label_source: Source of labels ('original' or 'llm')
        llm_predictions: Path to LLM predictions JSON file (required if label_source is 'llm')
        
    Returns:
        output_dir: Directory containing the processed graphs
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(embeddings_dir)), 'graphs', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load special embeddings
    special_embeddings = load_special_embeddings(embeddings_dir)
    
    # Process each split with enhanced progress bar
    for split in tqdm(['train', 'validation', 'test'], desc="Processing splits", unit="split", position=0, leave=True, colour='blue'):
        split_dir = os.path.join(embeddings_dir, split)
        
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory {split_dir} does not exist, skipping")
            continue
        
        # Create output directory for this split
        # If using LLM predictions, create a separate directory
        if label_source == 'llm':
            split_output_dir = os.path.join(output_dir, f"{split}_llm_labels")
        else:
            split_output_dir = os.path.join(output_dir, split)
            
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Load LLM predictions if needed
        llm_pred_dict = None
        if label_source == 'llm' and llm_predictions:
            logger.info(f"Loading LLM predictions for {split} split from {llm_predictions}")
            if os.path.exists(llm_predictions):
                llm_pred_dict = load_llm_predictions(llm_predictions, split)
                if not llm_pred_dict:
                    logger.warning(f"No LLM predictions loaded for {split} split. Will use dummy labels.")
            else:
                logger.error(f"LLM predictions file not found: {llm_predictions}")
        
        # Get dataset information
        try:
            dataset_info, is_chunked = get_dataset_info(split_dir)
            logger.info(f"Processing {dataset_info['total_samples']} samples for {split} split")
        except Exception as e:
            logger.error(f"Error getting dataset information for {split_dir}: {str(e)}")
            continue
        
        # Save configuration
        config = {
            'edge_type': edge_type,
            'batch_size': batch_size,
            'total_samples': dataset_info['total_samples'],
            'label_source': label_source,
            'dataset_name': dataset_name
        }
        
        config_path = os.path.join(split_output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Process embeddings
        if is_chunked:
            # Process each chunk
            chunks_dir = os.path.join(split_dir, 'chunks')
            total_graphs = 0
            
            # Process one chunk at a time to manage memory with enhanced progress tracking
            total_chunks = len(dataset_info['chunk_dirs'])
            chunk_progress = tqdm(dataset_info['chunk_dirs'], desc=f"Processing chunks for {split}", unit="chunk", 
                                  position=1, leave=False, colour='cyan', total=total_chunks)
            for i, chunk_dir_name in enumerate(chunk_progress):
                chunk_dir = os.path.join(chunks_dir, chunk_dir_name)
                chunk_progress.set_postfix({"dir": chunk_dir_name, "progress": f"{i+1}/{total_chunks}"})
                
                try:
                    # Process each chunk file individually
                    chunk_files = [f for f in os.listdir(chunk_dir) if 
                                 (f.endswith('.pkl') and f.startswith('chunk_')) or 
                                 (f.endswith('.npz') and f.startswith('embeddings_chunk_'))]
                    
                    # Sort chunk files numerically
                    def extract_chunk_number(filename):
                        if filename.startswith('chunk_'):
                            return int(filename.replace('chunk_', '').replace('.pkl', ''))
                        elif filename.startswith('embeddings_chunk_'):
                            return int(filename.replace('embeddings_chunk_', '').replace('.npz', ''))
                    
                    chunk_files = sorted(chunk_files, key=extract_chunk_number)
                    
                    # Process each chunk file individually with enhanced progress tracking
                    total_files = len(chunk_files)
                    file_progress = tqdm(chunk_files, desc=f"Files in chunk {chunk_dir_name}", unit="file", 
                                         position=2, leave=False, colour='yellow', total=total_files)
                    for chunk_idx, chunk_file in enumerate(file_progress):
                        chunk_path = os.path.join(chunk_dir, chunk_file)
                        file_progress.set_postfix({"file": chunk_file, "progress": f"{chunk_idx+1}/{total_files}"})
                        
                        # Load single chunk file
                        try:
                            if chunk_file.endswith('.pkl'):
                                with open(chunk_path, 'rb') as f:
                                    chunk_data = pkl.load(f)
                                
                                # Extract data based on structure
                                if isinstance(chunk_data, dict):
                                    word_embeddings = chunk_data.get('word_embeddings', [])
                                    texts = chunk_data.get('texts', [])
                                    labels = chunk_data.get('labels', [])
                                elif isinstance(chunk_data, list) and len(chunk_data) == 3:
                                    word_embeddings, texts, labels = chunk_data
                                else:
                                    logger.warning(f"Unknown chunk data format in {chunk_path}")
                                    continue
                                    
                            elif chunk_file.endswith('.npz'):
                                chunk_data = np.load(chunk_path, allow_pickle=True)
                                logger.debug(f"NPZ file keys: {list(chunk_data.keys())}")
                                
                                # Extract arrays - handle different naming conventions and missing keys
                                word_embeddings = None
                                if 'word_embeddings' in chunk_data:
                                    word_embeddings = chunk_data['word_embeddings']
                                    logger.debug(f"Loaded word_embeddings with shape {word_embeddings.shape if hasattr(word_embeddings, 'shape') else 'unknown'}")
                                    
                                # We don't need texts for graph creation as we use the tree structure
                                # Create an empty list with the right length for compatibility
                                texts = []
                                if word_embeddings is not None:
                                    texts = [""] * len(word_embeddings)
                                
                                # Initialize empty labels - they will be filled from LLM predictions or original dataset
                                labels = []
                            
                            logger.info(f"Loaded {len(word_embeddings)} samples from {chunk_file}")
                            
                            # Calculate number of samples in this chunk
                            chunk_size = len(word_embeddings)
                            
                            # Process in smaller sub-batches with enhanced progress tracking
                            sub_batch_size = batch_size  # Use the full batch size as specified
                            total_batches = int(np.ceil(chunk_size / sub_batch_size))
                            
                            batch_progress = tqdm(range(0, chunk_size, sub_batch_size), 
                                                  desc=f"Batches in {os.path.basename(chunk_file)}", 
                                                  unit="batch", 
                                                  position=3, 
                                                  leave=False, 
                                                  colour='magenta',
                                                  total=total_batches)
                            for j in batch_progress:
                                end_idx = min(j + sub_batch_size, chunk_size)
                                batch_idx_display = j // sub_batch_size + 1
                                batch_progress.set_postfix({
                                    "samples": f"{j}-{end_idx}/{chunk_size}", 
                                    "batch": f"{batch_idx_display}/{total_batches}",
                                    "memory": f"{j/chunk_size:.1%}"
                                })
                                
                                # Extract sub-batch
                                sub_word_embeddings = word_embeddings[j:end_idx]
                                
                                # Handle texts safely
                                if texts is not None and len(texts) > 0:
                                    sub_texts = texts[j:end_idx]
                                else:
                                    sub_texts = []
                                
                                # Initialize batch_idx for this sub-batch
                                batch_idx = (i * 1000 + j) // sub_batch_size
                                
                                # Get the correct labels based on the source
                                if label_source == 'llm':
                                    if not llm_pred_dict:
                                        error_msg = "LLM predictions dictionary is empty but LLM label source was specified"
                                        logger.error(error_msg)
                                        raise ValueError(error_msg)
                                    
                                    sub_labels = []
                                    valid_indices = []
                                    missing_indices = []
                                    
                                    # Extract the chunk number from the filename
                                    chunk_num = extract_chunk_number(chunk_file)
                                    
                                    # Calculate the base index for this chunk
                                    # For LLM predictions, we need to track the global index across all chunks
                                    # Each chunk has a fixed size (sub_batch_size) in the LLM predictions indexing
                                    base_index = chunk_num * sub_batch_size
                                    
                                    # Log the base index for debugging
                                    logger.info(f"Base index for chunk {chunk_num}: {base_index}")
                                    
                                    # Special handling for the last chunk
                                    is_last_chunk = chunk_idx == total_files - 1
                                    
                                    for idx in range(j, min(end_idx, len(word_embeddings))):
                                        # For the last chunk, try multiple indexing strategies
                                        if is_last_chunk:
                                            # Try different indexing strategies for the last chunk
                                            strategies = [
                                                base_index + (idx - j),  # Base index + relative position (most accurate)
                                                chunk_num * sub_batch_size + (idx - j),  # Standard indexing
                                                idx,  # Try direct index
                                                len(llm_pred_dict) - (len(word_embeddings) - idx)  # Try reverse mapping from end
                                            ]
                                            
                                            # Try each strategy
                                            found = False
                                            for data_idx in strategies:
                                                if data_idx in llm_pred_dict:
                                                    sub_labels.append(llm_pred_dict[data_idx])
                                                    valid_indices.append(idx - j)
                                                    found = True
                                                    break
                                            
                                            if not found:
                                                # If all strategies fail, try to find the closest available index
                                                closest_idx = min(llm_pred_dict.keys(), key=lambda x: abs(x - strategies[0]))
                                                if abs(closest_idx - strategies[0]) < 100:  # Only use if reasonably close
                                                    sub_labels.append(llm_pred_dict[closest_idx])
                                                    valid_indices.append(idx - j)
                                                    logger.warning(f"Using closest available prediction {closest_idx} for index {strategies[0]} in last chunk")
                                                else:
                                                    missing_indices.append(strategies[0])
                                                    logger.warning(f"No LLM prediction found for last chunk index {strategies[0]}, skipping")
                                        else:
                                            # Standard indexing for non-last chunks
                                            # Use the base_index + relative position in the current chunk
                                            data_idx = base_index + (idx - j)
                                            
                                            if data_idx in llm_pred_dict:
                                                sub_labels.append(llm_pred_dict[data_idx])
                                                valid_indices.append(idx - j)
                                            else:
                                                missing_indices.append(data_idx)
                                                logger.warning(f"No LLM prediction found for data index {data_idx} in chunk {chunk_file}, skipping")
                                    
                                    # Log information about the LLM predictions
                                    logger.info(f"Applying LLM predictions to batch {j}-{end_idx} in chunk {chunk_num}")
                                    logger.info(f"Number of available LLM predictions: {len(llm_pred_dict)}")
                                    logger.info(f"Found {len(sub_labels)}/{end_idx-j} matches with LLM predictions")
                                else:
                                    # For non-LLM source, use original labels from the dataset
                                    # Try multiple approaches to find the original labels
                                    
                                    # First, check if labels were loaded with the chunk data
                                    if labels is not None and len(labels) > 0:
                                        sub_labels = labels[j:end_idx]
                                        logger.info(f"Using {len(sub_labels)} original labels from chunk data")
                                    else:
                                        # Try to load labels from a labels file if it exists
                                        labels_file = os.path.join(os.path.dirname(chunk_path), f"labels_{i}.json")
                                        labels_npz = os.path.join(os.path.dirname(chunk_path), f"labels_{i}.npz")
                                        
                                        if os.path.exists(labels_file):
                                            try:
                                                with open(labels_file, 'r') as f:
                                                    all_labels = json.load(f)
                                                sub_labels = all_labels[j:end_idx]
                                                logger.info(f"Loaded {len(sub_labels)} original labels from {labels_file}")
                                            except Exception as e:
                                                logger.error(f"Error loading original labels from {labels_file}: {str(e)}")
                                                # Continue trying other methods
                                                sub_labels = None
                                        elif os.path.exists(labels_npz):
                                            try:
                                                npz_data = np.load(labels_npz, allow_pickle=True)
                                                if 'labels' in npz_data:
                                                    all_labels = npz_data['labels']
                                                    sub_labels = all_labels[j:end_idx]
                                                    logger.info(f"Loaded {len(sub_labels)} original labels from {labels_npz}")
                                                else:
                                                    logger.error(f"No 'labels' key found in {labels_npz}")
                                                    sub_labels = None
                                            except Exception as e:
                                                logger.error(f"Error loading original labels from {labels_npz}: {str(e)}")
                                                sub_labels = None
                                        else:
                                            # If we can't find labels, check if we can extract them from metadata
                                            metadata_path = os.path.join(os.path.dirname(chunk_dir), 'metadata.json')
                                            if os.path.exists(metadata_path):
                                                try:
                                                    with open(metadata_path, 'r') as f:
                                                        metadata = json.load(f)
                                                    if 'labels' in metadata:
                                                        all_labels = metadata['labels']
                                                        # Calculate the correct indices for this chunk and batch
                                                        start_idx = sum(len(c) for c in word_embeddings[:i]) + j
                                                        end_idx = start_idx + (end_idx - j)
                                                        sub_labels = all_labels[start_idx:end_idx]
                                                        logger.info(f"Loaded {len(sub_labels)} original labels from metadata")
                                                    else:
                                                        logger.error(f"No 'labels' key found in metadata")
                                                        sub_labels = None
                                                except Exception as e:
                                                    logger.error(f"Error loading original labels from metadata: {str(e)}")
                                                    sub_labels = None
                                            else:
                                                sub_labels = None
                                        
                                        # If we still don't have labels, raise an error
                                        if sub_labels is None or len(sub_labels) == 0:
                                            error_msg = f"Could not find original labels for chunk {chunk_file}"
                                            logger.error(error_msg)
                                            raise ValueError(error_msg)
                                
                                # Process the batch with the obtained labels
                                try:
                                    num_graphs = process_embeddings_batch(
                                        sub_word_embeddings, sub_texts, sub_labels,
                                        config, split_output_dir, batch_idx, special_embeddings
                                    )
                                    
                                    # Log memory usage for debugging
                                    import psutil
                                    process = psutil.Process()
                                    logger.debug(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                                    
                                except Exception as e:
                                    logger.error(f"Error processing batch: {str(e)}")
                                    # Fall back to original labels if there was an error
                                    logger.info("Falling back to original labels due to error")
                                    num_graphs = process_embeddings_batch(
                                        sub_word_embeddings, sub_texts, sub_labels,
                                        config, split_output_dir, batch_idx, special_embeddings
                                    )
                                else:
                                    # This else block is redundant and causing duplicate processing
                                    # The batch has already been processed in the try block above
                                    pass
                                
                                total_graphs += num_graphs
                                
                                # Force garbage collection to free memory
                                import gc
                                gc.collect()
                                
                            # Clear variables to free memory
                            del word_embeddings, texts, labels, chunk_data
                            gc.collect()
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk file {chunk_path}: {str(e)}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_dir}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            logger.info(f"Processed {total_graphs} graphs for {split} split")
        
        else:
            # Load all embeddings
            try:
                # Load metadata
                metadata_path = os.path.join(split_dir, 'metadata.pkl')
                with open(metadata_path, 'rb') as f:
                    metadata = pkl.load(f)
                
                # Load word embeddings
                word_embeddings_path = os.path.join(split_dir, 'word_embeddings.pkl')
                with open(word_embeddings_path, 'rb') as f:
                    word_embeddings = pkl.load(f)
                
                # Process in batches with enhanced progress tracking
                total_samples = len(word_embeddings)
                total_graphs = 0
                
                # Use smaller sub-batches for large datasets
                sub_batch_size = batch_size  # Use the full batch size as specified
                total_batches = int(np.ceil(total_samples / sub_batch_size))
                
                batch_progress = tqdm(range(0, total_samples, sub_batch_size), 
                                      desc=f"Processing {split} batches", 
                                      unit="batch",
                                      position=1,
                                      leave=False,
                                      colour='cyan',
                                      total=total_batches)
                for i in batch_progress:
                    batch_end = min(i + sub_batch_size, total_samples)
                    batch_idx_display = i // sub_batch_size + 1
                    batch_progress.set_postfix({
                        "samples": f"{i}-{batch_end}/{total_samples}",
                        "batch": f"{batch_idx_display}/{total_batches}",
                        "progress": f"{i/total_samples:.1%}"
                    })
                    
                    # Get batch data
                    batch_word_embeddings = word_embeddings[i:batch_end]
                    batch_texts = metadata['texts'][i:batch_end]
                    batch_labels = metadata['labels'][i:batch_end]
                    
                    # Apply LLM predictions if needed
                    if label_source == 'llm' and llm_pred_dict:
                        # Try to match data indices to LLM predictions
                        modified_labels = []
                        for idx, label in enumerate(batch_labels):
                            # Use data index if available, otherwise use position in batch
                            data_idx = i + idx
                            if data_idx in llm_pred_dict:
                                modified_labels.append(llm_pred_dict[data_idx])
                            else:
                                modified_labels.append(label)
                                logger.warning(f"No LLM prediction found for data index {data_idx}, using original label")
                        
                        # Process batch with modified labels
                        num_graphs = process_embeddings_batch(
                            batch_word_embeddings, batch_texts, modified_labels,
                            config, split_output_dir, i // sub_batch_size, special_embeddings
                        )
                    else:
                        # Process batch with original labels
                        num_graphs = process_embeddings_batch(
                            batch_word_embeddings, batch_texts, batch_labels,
                            config, split_output_dir, i // sub_batch_size, special_embeddings
                        )
                    
                    total_graphs += num_graphs
                    
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                
                logger.info(f"Processed {total_graphs} graphs for {split} split")
                
            except Exception as e:
                logger.error(f"Error processing embeddings for {split_dir}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    
    return output_dir

