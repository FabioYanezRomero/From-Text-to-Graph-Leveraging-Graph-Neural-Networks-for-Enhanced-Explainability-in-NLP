#!/usr/bin/env python3
"""
Tree Generator Module

This module provides functionality to generate constituency trees from text datasets.
It leverages the Graph_Generation package to create constituency parse trees.
"""

import os
import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

# Import local configuration
from .config import (
    GRAPH_TYPES, DEFAULT_MODELS, AVAILABLE_MODELS,
    DEFAULT_DATASETS, DEFAULT_SUBSETS, DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE, DEFAULT_OUTPUT_DIR, DEFAULT_DATA_DIR
)
from .constituency import ConstituencyTreeGenerator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate constituency trees from text datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=DEFAULT_DATASETS,
        help="Datasets to process (e.g., stanfordnlp/sst2 SetFit/ag_news)"
    )
    
    parser.add_argument(
        "--subsets", 
        nargs="+", 
        default=DEFAULT_SUBSETS,
        help="Subsets to process (e.g., train test validation)"
    )
    
    # Model arguments    
    parser.add_argument(
        "--constituency_model", 
        choices=AVAILABLE_MODELS["constituency"],
        default=DEFAULT_MODELS["constituency"],
        help="Model for constituency tree generation"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--device", 
        default=DEFAULT_DEVICE,
        help="Device to run on (e.g., cuda:0, cpu)"
    )
    
    # Config file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to JSON configuration file (overrides command-line arguments)"
    )
    
    return parser.parse_args()

def build_trees(dataset_name, subset, model=None, batch_size=DEFAULT_BATCH_SIZE, device=DEFAULT_DEVICE):
    """
    Build constituency trees from a dataset
    
    Args:
        dataset_name (str): Name of the dataset
        subset (str): Subset of the dataset (train, test, validation)
        model (str, optional): Model to use. If None, uses default model for constituency parsing
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    # Create appropriate tree generator
    generator = ConstituencyTreeGenerator(model=model, device=device)
    
    # Load dataset
    instance = load_dataset(dataset_name, split=subset)
    instance.set_format(type='torch')
    dataloader = DataLoader(dataset=instance, batch_size=batch_size, shuffle=False)
    
    # Create output directory
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, dataset_name, subset, "constituency")
    os.makedirs(output_path, exist_ok=True)
    
    # Process dataset
    iterator = 0
    for batch in tqdm(dataloader, desc=f"Processing {dataset_name}/{subset} constituency trees"):
        try:
            # Extract sentences and labels
            try:
                sentences = batch['sentence']
            except:
                sentences = batch['text']
            
            labels = batch['label']
            
            # Generate trees
            trees = generator.get_graph(sentences)
            processed_data = [(trees, labels)]
            
            # Save trees
            with open(f"{output_path}/{iterator}.pkl", 'wb') as f:
                pkl.dump(processed_data, f)
            
            iterator += 1
            
        except Exception as e:
            print(f"Error processing batch {iterator}: {str(e)}")
            continue


def process_datasets(datasets=None, subsets=None, models=None, batch_size=DEFAULT_BATCH_SIZE, device=DEFAULT_DEVICE):
    """
    Process multiple datasets to generate constituency trees
    
    Args:
        datasets (list, optional): List of datasets to process. If None, uses DEFAULT_DATASETS
        subsets (list, optional): List of subsets to process. If None, uses DEFAULT_SUBSETS
        models (dict, optional): Dict mapping to constituency model. If None, uses DEFAULT_MODELS
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    # Use defaults if not specified
    datasets = datasets or DEFAULT_DATASETS
    subsets = subsets or DEFAULT_SUBSETS
    models = models or DEFAULT_MODELS
    
    # Process each dataset
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # Create data directory
        data_dir = os.path.join(DEFAULT_DATA_DIR, dataset)
        os.makedirs(data_dir, exist_ok=True)
        
        # Process each subset
        for subset in subsets:

            if dataset == "SetFit/ag_news":
                if subset == "validation":
                    print("Skipping validation subset for SetFit/ag_news")
                    continue
            
            if dataset == "stanfordnlp/sst2":
                if subset == "test":
                    print("Skipping test subset for stanfordnlp/sst2")
                    continue

            print(f"Processing subset: {subset}")
            
            # Clear GPU memory
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Get model for constituency parsing
            model = models.get("constituency", DEFAULT_MODELS["constituency"])
            
            # Build trees
            build_trees(
                dataset_name=dataset,
                subset=subset,
                model=model,
                batch_size=batch_size,
                device=device
            )
            
            # Clear GPU memory again
            if device.startswith('cuda'):
                torch.cuda.empty_cache()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Override arguments with config file values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Create models dictionary
    models = {
        "constituency": args.constituency_model
    }
    
    # Process datasets - only for constituency trees
    process_datasets(
        datasets=args.datasets,
        subsets=args.subsets,
        models=models,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
