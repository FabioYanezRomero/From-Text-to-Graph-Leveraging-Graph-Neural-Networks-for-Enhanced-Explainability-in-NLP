#!/usr/bin/env python3
"""
Tree Generator Module

This module provides functionality to generate constituency trees from text datasets.
It leverages the Graph_Generation package to create constituency parse trees.
"""

import os
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
import pickle as pkl
from src.Clean_Code.Tree_Generation.constituency import ConstituencyTreeGenerator



def build_trees(graph_type, dataset_name, subset, batch_size, device, output_dir):
    """
    Build constituency trees from a dataset
    
    Args:
        graph_type (str): Type of graph to generate (constituency, syntactic, semantic)
        dataset_name (str): Name of the dataset
        subset (str): Subset of the dataset (train, test, validation)
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    # Create appropriate tree generator
    if graph_type == 'constituency':
        generator = ConstituencyTreeGenerator(device=device)
    else:
        raise NotImplementedError(f"Graph type {graph_type} is not supported.")
    # Load dataset
    instance = load_dataset(dataset_name, split=subset)
    instance.set_format(type='torch')
    dataloader = DataLoader(dataset=instance, batch_size=batch_size, shuffle=False)
    
    # Create output directory
    output_path = os.path.join(output_dir, dataset_name, subset, "constituency")
    os.makedirs(output_path, exist_ok=True)
    
    # Process dataset
    iterator = 0
    for batch in tqdm(dataloader, desc=f"Processing {dataset_name}/{subset} constituency trees"):
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


def process_dataset(graph_type, dataset, subsets, batch_size, device, output_dir):
    """
    Process multiple datasets to generate constituency trees
    
    Args:
        graph_type (str): Type of graph to generate (constituency, syntactic, semantic)
        dataset (str): Dataset to process
        subsets (list, optional): List of subsets to process. If None, uses DEFAULT_SUBSETS
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    
    # Process the dataset
    print(f"Processing dataset: {dataset}")
        
    # Create data directory
    os.makedirs(output_dir, exist_ok=True)
        
    # Handle each subset according to the given dataset
    subsets = subsets_handler(dataset, subsets)
    for subset in subsets:
        print(f"Processing subset: {subset}")
        
        # Clear GPU memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            
        # Build trees
        build_trees(
                graph_type=graph_type,
                dataset_name=dataset,
                subset=subset,
                batch_size=batch_size,
                device=device,
                output_dir=output_dir
            )
        
        # Clear GPU memory again
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def subsets_handler(dataset, subsets):
    if dataset == "SetFit/ag_news":
        if "validation" in subsets:
            subsets.remove("validation")
    elif dataset == "stanfordnlp/sst2":
        if "test" in subsets:
            subsets.remove("test")
    return subsets


def main(args):
    # Process datasets - only for constituency trees
    process_dataset(
        graph_type=args.graph_type,
        dataset=args.dataset,
        subsets=args.subsets,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_type", type=str, default="constituency")
    parser.add_argument("--dataset", type=str, default="SetFit/ag_news")
    parser.add_argument("--subsets", nargs='+', type=str, default=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="/app/src/Clean_Code/output")
    args = parser.parse_args()
    main(args)