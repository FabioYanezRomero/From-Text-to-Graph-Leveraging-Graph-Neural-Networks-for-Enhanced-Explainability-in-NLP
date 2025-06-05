"""
Graph Processor Main

This module provides the main entry point for processing word embeddings into graph structures
using pre-generated constituency trees and embeddings.
"""

import os
import sys
import logging
import argparse
import torch
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.Clean_Code.Graph_Generation.graph_data_processor import (
    load_embeddings,
    create_word_graphs,
    save_graphs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_embeddings(input_dir, output_dir, dataset, embedding_model='stanfordnlp', split='train', batch_size=32, num_workers=4):
    """Process word embeddings and create graph structures using pre-generated constituency trees.
    
    Args:
        input_dir: Base directory containing the input embeddings
        output_dir: Directory to save the output graphs
        dataset: Name of the dataset
        embedding_model: Name of the embedding model used
        split: Data split to process ('train', 'val', 'test')
        batch_size: Batch size for processing
        num_workers: Number of worker processes for data loading
    """
    logger.info(f"Processing {split} split of {dataset} dataset with {embedding_model} embeddings")
    logger.info(f"Using pre-generated constituency trees")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, f"{embedding_model}_{dataset}", split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    try:
        logger.info(f"Loading embeddings from {input_dir}")
        data = load_embeddings(input_dir, dataset, embedding_model, split)
        if data is None:
            logger.error("Failed to load embeddings")
            return
            
        texts, labels, word_embeddings, sentence_embeddings = data
        logger.info(f"Loaded {len(word_embeddings)} samples")
        
        # Create graphs using pre-generated constituency trees
        logger.info("Creating word graphs from pre-generated constituency trees...")
        graphs = create_word_graphs(
            word_embeddings=word_embeddings,
            sentence_embeddings=sentence_embeddings,
            texts=texts,
            labels=labels,
            dataset_name=dataset,
            split=split
        )
        
        if not graphs:
            logger.error("No graphs were created")
            return
        
        # Filter out any None graphs that might have been created due to errors
        graphs = [g for g in graphs if g is not None]
        if not graphs:
            logger.error("All graphs were None after filtering")
            return
        
        # Save the graphs
        logger.info(f"Saving {len(graphs)} graphs to {output_dir}")
        save_graphs(graphs, output_dir, batch_size, num_workers)
        
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Error in process_embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process word embeddings into graph structures using pre-generated constituency trees.')
    parser.add_argument('--input_dir', type=str, default='../output/embeddings',
                      help='Directory containing the input embeddings')
    parser.add_argument('--output_dir', type=str, default='../output/gnn_graphs',
                      help='Directory to save the output graphs')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['sst2', '20ng', 'wos', 'ag_news'],
                      help='Dataset name')
    parser.add_argument('--embedding_model', type=str, default='stanfordnlp',
                      help='Name of the embedding model used')
    parser.add_argument('--split', type=str, default='train',
                      choices=['train', 'test', 'val'],
                      help='Data split to process')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for data loading')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Process the embeddings
    process_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        embedding_model=args.embedding_model,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info("Graph generation completed successfully")
