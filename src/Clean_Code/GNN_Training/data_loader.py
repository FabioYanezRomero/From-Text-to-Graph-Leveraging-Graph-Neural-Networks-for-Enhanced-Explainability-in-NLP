"""
Data Loaders for Graph Neural Networks

This module provides data loading utilities for graph data in PyTorch Geometric format.
"""

import os
import glob
import pickle
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    """
    Dataset for loading graph data from pickle files.
    
    This dataset loads graph data that has been preprocessed and stored in pickle files.
    Each pickle file contains a batch of graphs in PyTorch Geometric format.
    """
    def __init__(self, root_dir, transform=None, pre_transform=None, pre_filter=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory containing the graph data files
            transform: Transform to apply to each graph
            pre_transform: Transform to apply to each graph before saving to disk
            pre_filter: Filter to apply to each graph before saving to disk
        """
        super(GraphDataset, self).__init__(root_dir, transform, pre_transform, pre_filter)
        self.root_dir = root_dir
        self.file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        
        # Load the first batch to get information about the dataset
        if len(self.file_paths) > 0:
            sample_batch = torch.load(self.file_paths[0])
            if isinstance(sample_batch, Batch):
                self.num_classes = self._infer_num_classes(sample_batch)
                self.num_node_features = sample_batch.num_node_features
            else:
                raise ValueError(f"Expected PyTorch Geometric Batch, got {type(sample_batch)}")
        else:
            raise ValueError(f"No pickle files found in {root_dir}")
        
    def _infer_num_classes(self, batch):
        """
        Infer the number of classes from the batch.
        
        Args:
            batch: PyTorch Geometric Batch
            
        Returns:
            Number of classes
        """
        if hasattr(batch, 'y'):
            return len(torch.unique(batch.y))
        else:
            # Default to 4 classes if we can't infer
            return 4
    
    def len(self):
        """
        Get the number of files in the dataset.
        
        Returns:
            Number of files
        """
        return len(self.file_paths)
    
    def get(self, idx):
        """
        Get a batch of graphs from a file.
        
        Args:
            idx: Index of the file to load
            
        Returns:
            Batch of graphs
        """
        file_path = self.file_paths[idx]
        batch = torch.load(file_path)
        return batch


def load_graph_data(data_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Load graph data from a directory.
    
    Args:
        data_dir: Directory containing the graph data files
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading data
        
    Returns:
        dataset: GraphDataset object
        dataloader: DataLoader for the dataset
    """
    dataset = GraphDataset(data_dir)
    
    # Create a custom collate function that handles batches of batches
    def custom_collate(batch_list):
        # Each item in batch_list is already a batch, so we need to merge them
        all_data = []
        for batch in batch_list:
            # If batch is a Batch object, extract individual graphs
            if isinstance(batch, Batch):
                # Convert batch to list of Data objects
                data_list = batch.to_data_list()
                all_data.extend(data_list)
            else:
                all_data.append(batch)
        
        # Create a new batch from all graphs
        return Batch.from_data_list(all_data)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # We load one file at a time, which already contains multiple graphs
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    
    return dataset, dataloader
