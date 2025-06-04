"""
Embedding Processor

This module handles the processing of embeddings for GNN tasks.
It includes functionality for selecting the best fine-tuned model
and generating embeddings for words, sentences, and special tokens.
"""

import os
import torch
import pickle as pkl
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import sys
import glob
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_best_model(results_dir, dataset_name, metric='f1-score', split='test'):
    """
    Find the best model checkpoint for a given dataset based on specified metric
    
    Args:
        results_dir: Directory containing fine-tuned model results
        dataset_name: Name of the dataset (e.g., 'sst2', 'ag_news')
        metric: Metric to use for selecting the best model (default: 'f1-score')
        split: Data split to use for evaluation (default: 'test')
        
    Returns:
        model_path: Path to the best model checkpoint
        model_name: Name of the base model
        best_score: Best metric score
    """
    logger.info(f"Finding best model for {dataset_name} based on {metric}...")
    
    # Determine provider based on dataset name
    if dataset_name == "sst2":
        provider = "stanfordnlp"
        full_dataset_name = f"{provider}/{dataset_name}"
    elif dataset_name == "ag_news":
        provider = "setfit"
        full_dataset_name = f"{provider}/{dataset_name}"
    else:
        provider, name = dataset_name.split('/')
        full_dataset_name = dataset_name
        dataset_name = name
    
    # Find all training runs for this dataset
    dataset_dirs = glob.glob(os.path.join(results_dir, provider, f"{dataset_name}_*"))
    
    if not dataset_dirs:
        logger.warning(f"No training runs found for dataset {full_dataset_name}")
        return None, None, None
    
    best_score = -1
    best_model_path = None
    best_run_dir = None
    best_epoch = None
    
    # Iterate through all training runs
    for run_dir in dataset_dirs:
        logger.info(f"Examining training run: {os.path.basename(run_dir)}")
        
        # Find all classification reports for the specified split
        report_files = glob.glob(os.path.join(run_dir, f"classification_report_{split}_epoch*.json"))
        
        # Check if model files exist
        model_files = glob.glob(os.path.join(run_dir, "model_epoch_*.pt"))
        
        if not report_files or not model_files:
            logger.warning(f"No classification reports or model files found in {run_dir}")
            continue
        
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
                
                logger.info(f"Epoch {epoch}: {metric} = {score:.4f}")
                
                # Check if this is the best score so far
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_run_dir = run_dir
                    
                    # Construct path to the corresponding model file
                    best_model_path = os.path.join(run_dir, f"model_epoch_{epoch}.pt")
                    
                    # Check if the model file exists
                    if not os.path.exists(best_model_path):
                        logger.warning(f"Model file {best_model_path} does not exist")
                        best_model_path = None
                        continue
            
            except Exception as e:
                logger.error(f"Error processing report file {report_file}: {e}")
    
    # Extract model name from run directory
    model_name = None
    if best_run_dir:
        # Try to find config.json to extract model name
        config_file = os.path.join(best_run_dir, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                model_name = config.get('model_name', None)
            except Exception as e:
                logger.error(f"Error reading config file {config_file}: {e}")
        
        # If model name not found in config, try to extract from run directory name
        if not model_name:
            run_name = os.path.basename(best_run_dir)
            parts = run_name.split('_')
            if len(parts) > 2:
                model_name = '_'.join(parts[2:])  # Assuming format: dataset_timestamp_model_name
    
    if best_model_path:
        logger.info(f"Best model found: {best_model_path}")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Best {metric}: {best_score:.4f}")
    else:
        logger.warning(f"No suitable model found for {full_dataset_name}")
    
    return best_model_path, model_name, best_score

def generate_embeddings_with_best_model(dataset_name, output_dir, results_dir="/app/src/Clean_Code/output/finetuned_llms", 
                                       metric="f1-score", split="test", batch_size=16, max_length=128, 
                                       chunk_size=50, device=None):
    """
    Generate embeddings using the best fine-tuned model for a dataset
    
    Args:
        dataset_name: Name of the dataset (e.g., 'sst2', 'ag_news')
        output_dir: Directory to save embeddings
        results_dir: Directory containing fine-tuned model results
        metric: Metric to use for selecting the best model
        split: Data split to use for evaluation
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        chunk_size: Size of chunks when saving embeddings to disk
        device: Device to use for inference (default: auto-detect)
        
    Returns:
        output_dir: Directory containing the saved embeddings
    """
    from Clean_Code.GNN_Embeddings.embedding_generator import extract_embeddings, generate_special_embeddings, save_embeddings
    
    # Determine device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find the best model
    model_path, model_name, best_score = find_best_model(results_dir, dataset_name, metric, split)
    
    if not model_path or not model_name:
        logger.error(f"Could not find a suitable model for {dataset_name}")
        return None
    
    logger.info(f"Using model {model_name} with {metric} = {best_score:.4f}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Load fine-tuned weights
    logger.info(f"Loading fine-tuned weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate special embeddings for constituency tokens
    special_embeddings = generate_special_embeddings(model, tokenizer, device=device)
    
    # Process each split
    for data_split in ['train', 'test', 'validation']:
        # Determine the correct path for text_graphs
        if dataset_name == "sst2":
            graph_dir = f"/app/src/Clean_Code/output/text_graphs/stanfordnlp/{dataset_name}/{data_split}/constituency"
        elif dataset_name == "ag_news":
            graph_dir = f"/app/src/Clean_Code/output/text_graphs/SetFit/{dataset_name}/{data_split}/constituency"
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue
        
        # Check if graph directory exists
        if not os.path.exists(graph_dir):
            logger.warning(f"Graph directory {graph_dir} does not exist, skipping {data_split} split")
            continue
        
        # Count graph files
        graph_files = sorted(glob.glob(os.path.join(graph_dir, "*.pkl")))
        if not graph_files:
            logger.warning(f"No graph files found in {graph_dir}, skipping {data_split} split")
            continue
        
        logger.info(f"Processing {len(graph_files)} graph files for {data_split} split")
        
        # Create output directory for this split
        split_output_dir = os.path.join(output_dir, dataset_name, data_split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process files in chunks
        for chunk_start in range(0, len(graph_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(graph_files))
            chunk_files = graph_files[chunk_start:chunk_end]
            
            # Initialize lists for this chunk
            chunk_texts = []
            chunk_labels = []
            
            # Load graph files for this chunk
            for graph_file in chunk_files:
                try:
                    with open(graph_file, 'rb') as f:
                        graph_data = pkl.load(f)
                    
                    # Extract texts and labels
                    for item in graph_data:
                        # Each item is a tuple (text_list, label_tensor)
                        text = item[0]
                        
                        # Handle label tensor - use first value if it's a tensor with multiple values
                        if isinstance(item[1], torch.Tensor):
                            if item[1].numel() > 1:
                                label = item[1][0].item()
                            else:
                                label = item[1].item()
                        else:
                            label = item[1]
                        
                        chunk_texts.append(text)
                        chunk_labels.append(label)
                
                except Exception as e:
                    logger.error(f"Error loading graph file {graph_file}: {e}")
            
            if not chunk_texts:
                logger.warning(f"No texts found in chunk {chunk_start}-{chunk_end}, skipping")
                continue
            
            logger.info(f"Generating embeddings for {len(chunk_texts)} texts in chunk {chunk_start//chunk_size}")
            
            # Generate word and sentence embeddings
            word_embeddings, sentence_embeddings = extract_embeddings(
                chunk_texts, model, tokenizer, 
                batch_size=batch_size, max_length=max_length, device=device
            )
            
            # Create chunk directory
            chunk_dir = os.path.join(split_output_dir, f"chunk_{chunk_start//chunk_size}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            # Save embeddings for this chunk
            save_embeddings(
                word_embeddings, sentence_embeddings, special_embeddings,
                chunk_texts, chunk_labels, chunk_dir
            )
            
            logger.info(f"Saved embeddings for chunk {chunk_start//chunk_size} to {chunk_dir}")
    
    logger.info(f"Embedding generation completed for {dataset_name}")
    return output_dir

# Graph processing functionality has been moved to Graph_Generation module

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for GNN models")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., sst2, ag_news)")
    parser.add_argument("--results_dir", type=str, default="/app/src/Clean_Code/output/finetuned_llms", 
                        help="Directory containing fine-tuned model results")
    parser.add_argument("--metric", type=str, default="f1-score", 
                        help="Metric to use for selecting the best model")
    parser.add_argument("--split", type=str, default="test", 
                        help="Data split to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for embedding generation")
    parser.add_argument("--chunk_size", type=int, default=50, 
                        help="Size of chunks when saving embeddings to disk")
    parser.add_argument("--max_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="/app/src/Clean_Code/output", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    embeddings_output_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_output_dir, exist_ok=True)
    
    # Generate embeddings
    generate_embeddings_with_best_model(
        dataset_name=args.dataset_name,
        output_dir=embeddings_output_dir,
        results_dir=args.results_dir,
        metric=args.metric,
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        chunk_size=args.chunk_size
    )
