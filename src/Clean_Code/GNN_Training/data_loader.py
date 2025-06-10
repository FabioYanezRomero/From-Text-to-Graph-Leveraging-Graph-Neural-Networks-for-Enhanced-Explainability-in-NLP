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
    Each pickle file contains a list of graphs in PyTorch Geometric format.
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
        # Initialize with empty root since we're not using the default PyG file structure
        super(GraphDataset, self).__init__(None, transform, pre_transform, pre_filter)
        
        self.root_dir = root_dir
        # Prefer .pt files if present, else fall back to .pkl
        pt_paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        pkl_paths = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if len(pt_paths) > 0:
            self.file_paths = pt_paths
            file_type = 'pt'
        elif len(pkl_paths) > 0:
            self.file_paths = pkl_paths
            file_type = 'pkl'
        else:
            raise ValueError(f"No .pt or .pkl files found in {root_dir}")

        # Initialize data structures to store all graphs
        self.graphs = []
        self.file_to_idx_map = {}  # Maps file index to graph indices
        self._num_node_features = None
        self._num_classes = None
        
        # Load all graph data and build index mapping
        start_idx = 0
        for file_idx, file_path in enumerate(self.file_paths):
            if file_type == 'pt':
                batch_data = torch.load(file_path, weights_only=False)
            else:
                with open(file_path, 'rb') as f:
                    import pickle
                    batch_data = pickle.load(f)
            
            # Store the mapping from file index to graph indices
            if isinstance(batch_data, list):
                num_graphs = len(batch_data)
                self.file_to_idx_map[file_idx] = (start_idx, start_idx + num_graphs)
                start_idx += num_graphs
                
                # Add all graphs to our list
                self.graphs.extend(batch_data)
                
                print(f"Loaded {num_graphs} graphs from {os.path.basename(file_path)}")
            else:
                raise ValueError(f"Expected a list of graphs in {file_path}, got {type(batch_data)}")
        
        # Get dataset properties from the first graph
        if len(self.graphs) > 0:
            self._num_node_features = self.graphs[0].num_node_features
            self._num_classes = self._infer_num_classes(self.graphs[0])
            print(f"Dataset loaded with {len(self.graphs)} total graphs, {self._num_node_features} node features, {self._num_classes} classes")
        else:
            raise ValueError("No graphs found in the dataset")
            
    @property
    def num_node_features(self):
        return self._num_node_features
        
    @property
    def num_classes(self):
        return self._num_classes
        
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
        Get the total number of graphs in the dataset.
        
        Returns:
            Number of graphs
        """
        return len(self.graphs)
    
    def get(self, idx):
        """
        Get a single graph by index.
        
        Args:
            idx: Index of the graph to retrieve
            
        Returns:
            A single graph (PyTorch Geometric Data object)
        """
        if idx < 0 or idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.graphs)} graphs")
        
        # Get the graph directly from our list
        graph = self.graphs[idx]
        
        # Apply transforms if specified
        if self.transform is not None:
            graph = self.transform(graph)
            
        return graph


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
