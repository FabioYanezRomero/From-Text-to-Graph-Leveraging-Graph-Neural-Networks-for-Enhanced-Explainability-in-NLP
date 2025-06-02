"""
Graph Generator Module

This module provides functionality to generate different types of graphs from text datasets.
It supports syntactic, semantic, and constituency graph generation.
"""

import os
import json
import torch
import pickle as pkl
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

# Import configuration
from config import (
    GRAPH_TYPES, DEFAULT_MODELS, AVAILABLE_MODELS,
    DEFAULT_DATASETS, DEFAULT_SUBSETS, DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE, DEFAULT_OUTPUT_DIR, DEFAULT_DATA_DIR
)

class GraphGeneratorFactory:
    """Factory class to create appropriate graph generators based on type"""
    
    @staticmethod
    def create_generator(graph_type, model=None, device=DEFAULT_DEVICE):
        """
        Create a graph generator of the specified type
        
        Args:
            graph_type (str): Type of graph generator to create
            model (str, optional): Model name to use. If None, uses default model for the type
            device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
            
        Returns:
            BaseGraphGenerator: An instance of the appropriate graph generator
        
        Raises:
            ValueError: If graph_type is not supported
        """
        if graph_type not in GRAPH_TYPES:
            raise ValueError(f"Unsupported graph type: {graph_type}. Available types: {GRAPH_TYPES}")
        
        # Use default model if none specified
        if model is None:
            model = DEFAULT_MODELS[graph_type]
        
        # Validate model
        if model not in AVAILABLE_MODELS[graph_type]:
            raise ValueError(f"Unsupported model for {graph_type}: {model}. Available models: {AVAILABLE_MODELS[graph_type]}")
        
        # Import appropriate generator class based on type
        if graph_type == "syntactic":
            from .syntactic import SyntacticGraphGenerator
            return SyntacticGraphGenerator(model=model, device=device)
        elif graph_type == "semantic":
            from .semantic import SemanticGraphGenerator
            return SemanticGraphGenerator(model=model, device=device)
        elif graph_type == "constituency":
            from .constituency import ConstituencyGraphGenerator
            return ConstituencyGraphGenerator(model=model, device=device)


def build_graphs(dataset_name, subset, graph_type, model=None, batch_size=DEFAULT_BATCH_SIZE, device=DEFAULT_DEVICE):
    """
    Build graphs from a dataset
    
    Args:
        dataset_name (str): Name of the dataset
        subset (str): Subset of the dataset (train, test, validation)
        graph_type (str): Type of graph to generate
        model (str, optional): Model to use. If None, uses default model for the type
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    # Create appropriate graph generator
    generator = GraphGeneratorFactory.create_generator(graph_type, model, device)
    
    # Load dataset
    instance = load_dataset(dataset_name, split=subset)
    instance.set_format(type='torch')
    dataloader = DataLoader(dataset=instance, batch_size=batch_size, shuffle=False)
    
    # Create output directory
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, dataset_name, subset, graph_type)
    os.makedirs(output_path, exist_ok=True)
    
    # Process dataset
    iterator = 0
    for batch in tqdm(dataloader, desc=f"Processing {dataset_name}/{subset} with {graph_type} graphs"):
        try:
            # Extract sentences and labels
            try:
                sentences = batch['sentence']
            except:
                sentences = batch['text']
            
            labels = batch['label']
            
            # Generate graphs
            graphs = generator.get_graph(sentences)
            processed_data = [(graphs, labels)]
            
            # Save graphs
            with open(f"{output_path}/{iterator}.pkl", 'wb') as f:
                pkl.dump(processed_data, f)
            
            iterator += 1
            
        except Exception as e:
            print(f"Error processing batch {iterator}: {str(e)}")
            continue


def process_datasets(datasets=None, subsets=None, graph_types=None, 
                     models=None, batch_size=DEFAULT_BATCH_SIZE, device=DEFAULT_DEVICE):
    """
    Process multiple datasets with multiple graph types
    
    Args:
        datasets (list, optional): List of datasets to process. If None, uses DEFAULT_DATASETS
        subsets (list, optional): List of subsets to process. If None, uses DEFAULT_SUBSETS
        graph_types (list, optional): List of graph types to generate. If None, uses GRAPH_TYPES
        models (dict, optional): Dict mapping graph types to models. If None, uses DEFAULT_MODELS
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    # Use defaults if not specified
    datasets = datasets or DEFAULT_DATASETS
    subsets = subsets or DEFAULT_SUBSETS
    graph_types = graph_types or GRAPH_TYPES
    models = models or DEFAULT_MODELS
    
    # Process each dataset
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # Create data directory
        data_dir = os.path.join(DEFAULT_DATA_DIR, dataset)
        os.makedirs(data_dir, exist_ok=True)
        
        # Process each subset
        for subset in subsets:
            print(f"Processing subset: {subset}")
            
            # Process each graph type
            for graph_type in graph_types:
                print(f"Generating {graph_type} graphs")
                
                # Clear GPU memory
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                
                # Get model for this graph type
                model = models.get(graph_type, DEFAULT_MODELS[graph_type])
                
                # Build graphs
                build_graphs(
                    dataset_name=dataset,
                    subset=subset,
                    graph_type=graph_type,
                    model=model,
                    batch_size=batch_size,
                    device=device
                )
                
                # Clear GPU memory again
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    process_datasets()
