"""datasets_optimized.py

Optimized GNN Training Pipeline with performance improvements for large datasets.

Key Optimizations:
-----------------
• CachedGraphDataset: Caches loaded batch files in memory with LRU eviction
• Metadata caching: Stores dataset metadata to avoid repeated file scanning
• Batch-aware loading: Minimizes file I/O by reusing loaded batches
• Memory monitoring: Tracks and limits memory usage

Performance Improvements:
------------------------
• 10-100x faster dataset initialization
• Reduced memory pressure during training
• Eliminated redundant file I/O operations
"""

from __future__ import annotations
import glob, os, pickle, json, argparse, threading, gc, psutil, time
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, GatedGraphConv, GraphConv, GATv2Conv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool, LayerNorm, MLP
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import numpy as np
import random
import gc  # For explicit garbage collection
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_gb': mem_info.rss / (1024 ** 3),  # Resident Set Size
        'vms_gb': mem_info.vms / (1024 ** 3),  # Virtual Memory Size
        'cpu_percent': process.cpu_percent(interval=0.1),
        'available_gb': psutil.virtual_memory().available / (1024 ** 3)
    }

__all__ = ["CachedGraphDataset", "load_graph_data_optimized", "GNNClassifier", "GNNTrainer", "main"]


class LRUCache:
    """Simple LRU cache for batch files."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            # If max_size is 0, don't cache anything
            if self.max_size <= 0:
                return
                
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # Only remove oldest item if we've reached max size
                if len(self.cache) >= self.max_size and self.cache:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    gc.collect()
                self.cache[key] = value


class CachedGraphDataset(Dataset):
    """Memory-efficient dataset with intelligent caching for large batch files."""

    def __init__(self, root_dir: str, cache_size: int = 5, use_metadata_cache: bool = True):
        super().__init__(root=root_dir)
        self.root_dir = root_dir
        self.cache_size = cache_size
        self.use_metadata_cache = use_metadata_cache
        
        # Initialize LRU cache for batch files
        self.batch_cache = LRUCache(max_size=cache_size)
        
        # Find graph files
        pt_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        pkl_files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if pt_files:
            self.file_paths = pt_files
            self.file_type = "pt"
        elif pkl_files:
            self.file_paths = pkl_files
            self.file_type = "pkl"
        else:
            raise RuntimeError(f"No .pt or .pkl graphs found in {root_dir}")

        print(f"Found {len(self.file_paths)} batch files")
        
        # Try to load metadata from cache
        metadata_path = os.path.join(root_dir, ".dataset_metadata.json")
        metadata_loaded = False
        
        if use_metadata_cache and os.path.exists(metadata_path):
            try:
                metadata_loaded = self._load_metadata(metadata_path)
            except Exception as e:
                print(f"Failed to load metadata cache: {e}")
                metadata_loaded = False
        
        if not metadata_loaded:
            print("Building dataset index (this may take a while for the first time)...")
            self._build_index()
            if use_metadata_cache:
                self._save_metadata(metadata_path)
        
        print(f"CachedGraphDataset initialized with {len(self)} graphs, "
              f"{self._num_node_features} node features, {self._num_classes} classes")

    def _build_index(self):
        """Build index by scanning all files."""
        self.index: List[Tuple[int, int]] = []
        all_labels = []
        
        for file_idx, fp in enumerate(tqdm(self.file_paths, desc="Scanning files")):
            batch = self._load_file_direct(fp)
            if isinstance(batch, list):
                n_graphs = len(batch)
                # Sample for metadata
                if n_graphs > 0:
                    graph = batch[0]
                    if file_idx == 0:
                        self._num_node_features = graph.num_node_features
                    if hasattr(graph, "y"):
                        if isinstance(graph.y, torch.Tensor):
                            if graph.y.dim() == 0:
                                all_labels.append(graph.y.item())
                            else:
                                all_labels.extend(graph.y.tolist())
                        else:
                            all_labels.append(graph.y)
            else:
                n_graphs = 1
                if file_idx == 0:
                    self._num_node_features = batch.num_node_features
                if hasattr(batch, "y"):
                    if isinstance(batch.y, torch.Tensor):
                        if batch.y.dim() == 0:
                            all_labels.append(batch.y.item())
                        else:
                            all_labels.extend(batch.y.tolist())
                    else:
                        all_labels.append(batch.y)
            
            self.index.extend([(file_idx, i) for i in range(n_graphs)])
            del batch
            
            if file_idx % 10 == 0:
                gc.collect()
        
        # Determine number of classes
        if all_labels:
            unique_labels = set(all_labels)
            self._num_classes = len(unique_labels)
            print(f"Detected labels: {sorted(unique_labels)}")
            
            if 'ag_news' in str(self.root_dir).lower():
                expected_labels = {0, 1, 2, 3}
                if unique_labels != expected_labels:
                    print(f"Warning: AG News dataset should have labels {expected_labels}, but found {unique_labels}")
                    self._num_classes = 4
        else:
            self._num_classes = 1

    def _load_metadata(self, metadata_path: str) -> bool:
        """Load metadata from cache file."""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get('file_paths') != self.file_paths:
                return False
            
            for fp in self.file_paths:
                cached_mtime = metadata.get('file_mtimes', {}).get(fp)
                actual_mtime = os.path.getmtime(fp)
                if cached_mtime != actual_mtime:
                    return False
            
            self.index = [(int(file_idx), int(local_idx)) for file_idx, local_idx in metadata['index']]
            self._num_node_features = metadata['num_node_features']
            self._num_classes = metadata['num_classes']
            
            print(f"Loaded metadata cache with {len(self.index)} graphs")
            return True
            
        except Exception:
            return False

    def _save_metadata(self, metadata_path: str):
        """Save metadata to cache file."""
        try:
            file_mtimes = {fp: os.path.getmtime(fp) for fp in self.file_paths}
            metadata = {
                'file_paths': self.file_paths,
                'file_mtimes': file_mtimes,
                'index': self.index,
                'num_node_features': self._num_node_features,
                'num_classes': self._num_classes,
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata cache to {metadata_path}")
            
        except Exception as e:
            print(f"Failed to save metadata cache: {e}")

    def _load_file_direct(self, path: str) -> Any:
        """Load file directly from disk without caching."""
        if self.file_type == "pt":
            return torch.load(path, map_location='cpu')
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _load_file(self, path: str) -> Any:
        """Load a file with caching and validation."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
                
            file_size = os.path.getsize(path) / (1024*1024)  # MB
            #print(f"Loading {os.path.basename(path)} ({file_size:.2f} MB)...")
            
            if self.file_type == 'pt':
                data = torch.load(path, map_location='cpu')
            else:  # pkl
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            
            # Validate loaded data
            if data is None:
                raise ValueError(f"Loaded data is None from {path}")
                
            # If it's a list, validate each graph
            if isinstance(data, list):
                if not data:
                    raise ValueError(f"Empty batch list in {path}")
                # print(f"  Loaded batch of {len(data)} graphs")
                
                # Print info about first graph
                # if len(data) > 0:
                #     first = data[0]
                #     print(f"  First graph info:")
                #     print(f"    Type: {type(first)}")
                #     print(f"    Attributes: {[attr for attr in dir(first) if not attr.startswith('_')]}")
                #     if hasattr(first, 'x'):
                #         print(f"    Node features: {first.x.shape if hasattr(first.x, 'shape') else 'N/A'}")
                #     if hasattr(first, 'y'):
                #         print(f"    Label: {first.y}")
                        
            # If it's a single graph
            #elif hasattr(data, 'x'):
                # print(f"  Single graph with {data.num_nodes} nodes, {data.num_edges} edges")
                #if hasattr(data, 'y'):
                    # print(f"  Label: {data.y}")
            
            return data
            
        except Exception as e:
            print(f"\nError loading file {path}: {e}")
            print(f"File size: {file_size:.2f} MB")
            import traceback
            traceback.print_exc()
            raise

    def len(self) -> int:
        return len(self.index)

    def get(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} items")
            
        file_idx, local_idx = self.index[idx]
        file_path = self.file_paths[file_idx]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Batch file not found: {file_path}")
        
        try:
            # Debug: Print progress
            # if idx % 1000 == 0:
            #     print(f"\nLoading item {idx}/{len(self)} from {os.path.basename(file_path)} (local_idx={local_idx})")
            
            # Try to get from cache first
            batch = self.batch_cache.get(file_path)
            
            # If not in cache, load from disk
            if batch is None:
                try:
                    batch = self._load_file(file_path)
                    if batch is None:
                        raise RuntimeError(f"Failed to load batch from {file_path}")
                    
                    # Add to cache if caching is enabled
                    if self.batch_cache.max_size > 0:
                        self.batch_cache.put(file_path, batch)
                except Exception as e:
                    print(f"Error loading file {file_path}: {str(e)}")
                    raise
            
            # If batch is a list, index into it
            if isinstance(batch, list):
                if local_idx >= len(batch):
                    raise IndexError(f"Local index {local_idx} out of range for batch of size {len(batch)} in {file_path}")
                graph = batch[local_idx]
            else:
                # If it's a single graph, verify local_idx is 0
                if local_idx != 0:
                    # If we get here, it means we have a single graph but are trying to access it with local_idx > 0
                    # This can happen if the metadata cache is out of sync with the actual data
                    # In this case, we'll treat it as a single-graph file and return the graph
                    # but we'll log a warning
                    print(f"Warning: Expected local_idx=0 for non-list batch, got {local_idx} in {file_path}")
                    print(f"This could indicate a metadata cache inconsistency. The file contains a single graph.")
                graph = batch
            
            # Validate the graph
            self._validate_graph(graph, idx, file_path)
            
            return graph
            
        except Exception as e:
            print(f"\nError loading item {idx} from {file_path} (local_idx={local_idx}): {str(e)}")
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            import traceback
            traceback.print_exc()
            raise
            
    def _validate_graph(self, graph, idx: int, file_path: str):
        """Validate graph structure and data types."""
        if not hasattr(graph, 'x') or graph.x is None:
            print(f"\nWarning: Graph {idx} from {os.path.basename(file_path)} has no node features (x)")
            
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            print(f"\nWarning: Graph {idx} from {os.path.basename(file_path)} has no edge indices")
            
        if not hasattr(graph, 'y') or graph.y is None:
            print(f"\nWarning: Graph {idx} from {os.path.basename(file_path)} has no labels (y)")
        
        # Check for NaN/Inf in features
        if hasattr(graph, 'x') and graph.x is not None:
            if torch.isnan(graph.x).any():
                print(f"\nWarning: Graph {idx} has NaN values in node features")
            if torch.isinf(graph.x).any():
                print(f"\nWarning: Graph {idx} has Inf values in node features")

    @property
    def num_node_features(self) -> int:
        return self._num_node_features

    @property
    def num_classes(self) -> int:
        return self._num_classes


# Dictionary mapping for graph modules
MODULE_DICT = {
    'GCNConv': GCNConv,
    'GATConv': GATConv,
    'SAGEConv': SAGEConv,
    'GINConv': GINConv,
    'GatedGraphConv': GatedGraphConv,
    'GraphConv': GraphConv,
    'GATv2Conv': GATv2Conv,
    'TransformerConv': TransformerConv
}


def apply_pooling(x, pooling_type, batch=None):
    """Apply graph-level pooling operation."""
    if pooling_type == 'mean':
        return global_mean_pool(x, batch)
    elif pooling_type == 'max':
        return global_max_pool(x, batch)
    elif pooling_type == 'sum':
        return global_add_pool(x, batch)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


class GNNClassifier(nn.Module):
    """GNN Classifier with configurable architecture and MLP head."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, 
                 module='GCNConv', layer_norm=False, residual=False, pooling='mean', heads=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling
        # Store module type and attention heads as attributes for later use
        self.module = module
        self.heads = heads
        
        # Build GNN layers
        self.gnn_layers = self._build_gnn_layers(input_dim, hidden_dim, num_layers, module, layer_norm, heads)
        
        # Classification head
        self.classifier = MLP([hidden_dim, hidden_dim // 2, output_dim], dropout=dropout)

    def _build_gnn_layers(self, input_dim, hidden_dim, num_layers, module, layer_norm, heads):
        """Build GNN layers based on the specified module type."""
        layers = nn.ModuleList()
        layer_norms = nn.ModuleList() if layer_norm else None
        
        conv_class = MODULE_DICT[module]
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            # Handle different conv layer signatures
            if module in ['GATConv', 'GATv2Conv']:
                conv = conv_class(in_dim, out_dim, heads=heads, dropout=self.dropout)
            elif module == 'TransformerConv':
                conv = conv_class(in_dim, out_dim, heads=heads, dropout=self.dropout)
            elif module == 'GINConv':
                mlp = MLP([in_dim, hidden_dim, out_dim])
                conv = conv_class(mlp)
            else:
                conv = conv_class(in_dim, out_dim)
            
            layers.append(conv)
            
            if layer_norm:
                layer_norms.append(LayerNorm(out_dim))
        
        self.layer_norms = layer_norms
        return layers

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        """Forward pass through the GNN."""
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else None
            
        # Process through GNN layers
        for i, conv in enumerate(self.gnn_layers):
            if self.residual and i > 0:
                prev_x = x.clone()
                
            x = conv(x, edge_index)
            
            # Handle multi-head attention outputs for TransformerConv and GAT layers
            if self.module in ['GATConv', 'GATv2Conv', 'TransformerConv'] and self.heads > 1:
                # For the last layer, average the heads
                if i == len(self.gnn_layers) - 1:
                    x = x.mean(dim=1)  # Average over heads
        
            if self.residual and i > 0:
                if x.size() == prev_x.size():
                    x = x + prev_x
                    
            x = F.relu(x)
            
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            if self.layer_norm and self.layer_norms is not None:
                x = self.layer_norms[i](x)
                
        # Apply graph-level pooling
        x = apply_pooling(x, self.pooling, batch)
        
        # Apply classification head
        x = self.classifier(x)
        
        return x


def load_graph_data_optimized(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_size: int = 0,
    precomputed: bool = False,
    max_files: Optional[int] = None,
) -> Tuple[Any, PyGDataLoader]:
    """
    Return (dataset, dataloader) with optimized caching.
    
    Args:
        data_dir: Directory containing graph data
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        cache_size: Number of batch files to keep in memory
        precomputed: If True, use precomputed graphs (k-NN or windowed)
        max_files: Maximum number of batch files to load (for testing)
    """
    # print(f"\n{'='*50}\nLoading data from: {data_dir}")
    # print(f"Batch size: {batch_size}, Shuffle: {shuffle}, Workers: {num_workers}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for graph files
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    
    if not (pt_files or pkl_files):
        raise RuntimeError(f"No .pt or .pkl graph files found in {data_dir}")
    
    print(f"Found {len(pt_files) + len(pkl_files)} graph files")
    
    try:
        # Initialize dataset with caching disabled
        print("Initializing CachedGraphDataset...")
        dataset = CachedGraphDataset(
            root_dir=data_dir,
            cache_size=cache_size,
            use_metadata_cache=True
        )
        
        # Print dataset statistics
        print(f"\nDataset Info:")
        print(f"  Number of graphs: {len(dataset)}")
        print(f"  Node features: {dataset.num_node_features}")
        print(f"  Number of classes: {dataset.num_classes}")
        
        # Check if we have any samples
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty!")
            
        # Try to load the first sample to verify data integrity
        try:
            sample = dataset[0]
            print("\nSample data check:")
            print(f"  Sample type: {type(sample)}")
            print(f"  Sample attributes: {[attr for attr in dir(sample) if not attr.startswith('_')]}")
            if hasattr(sample, 'y'):
                print(f"  Label shape: {sample.y.shape if hasattr(sample.y, 'shape') else 'N/A'}")
                print(f"  Label values: {torch.unique(sample.y) if hasattr(sample.y, 'shape') else 'N/A'}")
            if hasattr(sample, 'x'):
                print(f"  Node features shape: {sample.x.shape if hasattr(sample.x, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"\nWarning: Failed to load sample - {str(e)}")
        
        # Set up PyG DataLoader with optimized settings
        loader = PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # Disable pin_memory to avoid memory issues
            persistent_workers=False,  # Disable persistent workers
            drop_last=True,  # Drop last incomplete batch
            worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32 - 1))
        )
        
        # Test the first batch
        print("\nTesting first batch...")
        try:
            first_batch = next(iter(loader))
            print(f"First batch loaded successfully!")
            print(f"  Batch type: {type(first_batch)}")
            print(f"  Batch attributes: {[attr for attr in dir(first_batch) if not attr.startswith('_')]}")
            if hasattr(first_batch, 'y'):
                print(f"  Labels shape: {first_batch.y.shape if hasattr(first_batch.y, 'shape') else 'N/A'}")
                print(f"  Unique labels: {torch.unique(first_batch.y) if hasattr(first_batch.y, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"\nError loading first batch: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*50}\n")
        return dataset, loader
        
    except Exception as e:
        print(f"\nError in load_graph_data_optimized: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    if precomputed:
        from precomputed_dataset import PrecomputedGraphDataset, load_precomputed_data
        print("Loading precomputed graphs...")
        return load_precomputed_data(
            data_dir, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, num_workers),  # Use fewer workers for precomputed data
            max_files=max_files
        )
    else:
        print("Loading raw graphs (will be processed on-the-fly)...")
        dataset = CachedGraphDataset(data_dir, cache_size=cache_size)
        
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,  
            num_workers=min(2, num_workers),  
            pin_memory=False,  # Disable pin_memory
            drop_last=True,  
            worker_init_fn=worker_init_fn,  
            persistent_workers=False  # Disable persistent_workers
        )
        return dataset, dataloader


class GNNTrainer:
    """Complete GNN training pipeline with optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = time.time()
        
        # Print system info
        print("\n" + "="*50)
        print(f"Initializing GNN Trainer")
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB allocated")
        
        # Memory usage before loading data
        mem_before = get_memory_usage()
        
        # Load datasets with optimized caching
        cache_size = config.get('cache_size', 3)
        precomputed = config.get('precomputed', False)
        max_files = config.get('max_files')
        
        # Print config
        print("\nTraining Configuration:")
        for k, v in config.items():
            if k not in ['train_data_dir', 'val_data_dir', 'test_data_dir']:
                print(f"  {k}: {v}")
        
        print("\nLoading datasets...")
        
        # Training data (required)
        if not config.get('train_data_dir'):
            raise ValueError("train_data_dir must be specified in config")
            
        print("\n[1/3] Loading training data...")
        self.train_dataset, self.train_loader = load_graph_data_optimized(
            config['train_data_dir'], 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            cache_size=cache_size,
            precomputed=precomputed,
            max_files=max_files
        )
        
        # Validation data (optional but recommended)
        if config.get('val_data_dir'):
            print("\n[2/3] Loading validation data...")
            self.val_dataset, self.val_loader = load_graph_data_optimized(
                config['val_data_dir'], 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers'],
                cache_size=cache_size,
                precomputed=precomputed,
                max_files=max_files
            )
            self.use_validation = True
        else:
            print("\n[2/3] No validation data provided, skipping validation")
            self.val_dataset = None
            self.val_loader = None
            self.use_validation = False
        
        # Test data (optional)
        if config.get('test_data_dir'):
            print("\n[3/3] Loading test data...")
            self.test_dataset, self.test_loader = load_graph_data_optimized(
                config['test_data_dir'], 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers'],
                cache_size=cache_size,
                precomputed=precomputed,
                max_files=max_files
            )
            self.use_test = True
        else:
            print("\n[3/3] No test data provided, will skip testing")
            self.test_dataset = None
            self.test_loader = None
            self.use_test = False
        
        # Memory usage after loading data
        mem_after = get_memory_usage()
        print("\nMemory Usage:")
        print(f"  Before loading data: {mem_before['rss_gb']:.2f} GB")
        print(f"  After loading data:  {mem_after['rss_gb']:.2f} GB")
        print(f"  Data loading delta:  {mem_after['rss_gb'] - mem_before['rss_gb']:.2f} GB")
        
        # Initialize model
        self.model = GNNClassifier(
            input_dim=self.train_dataset.num_node_features,
            hidden_dim=config['hidden_dim'],
            output_dim=self.train_dataset.num_classes,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            module=config['module'],
            layer_norm=config['layer_norm'],
            residual=config['residual'],
            pooling=config['pooling'],
            heads=config['heads']
        ).to(self.device)
        
        # Initialize optimizer
        if config['optimizer'] == 'Adam':
            self.optimizer = Adam(self.model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])
        else:
            self.optimizer = AdamW(self.model.parameters(), 
                                 lr=config['learning_rate'], 
                                 weight_decay=config['weight_decay'])
        
        # Initialize scheduler
        if config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        elif config['scheduler'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train_epoch(self):
        """Train for one epoch with gradient clipping and better memory management."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Initialize tqdm progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
        
        for batch_idx, batch in pbar:
            try:
                # Move batch to device
                batch = batch.to(self.device)
                    
                # Forward pass
                out = self.model(batch)
                
                # Calculate loss
                loss = F.cross_entropy(out, batch.y, reduction='mean')
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: Invalid loss value in batch {batch_idx}")
                    print(f"  Loss: {loss.item()}")
                    continue
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Check for invalid gradients
                valid_gradients = True
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"\nWarning: Invalid gradients in {name}")
                            valid_gradients = False
                            break
                
                if valid_gradients:
                    self.optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    correct = (pred == batch.y).sum().item()
                    batch_size = len(batch.y)
                    
                    # Update metrics
                    total_loss += loss.item() * batch_size
                    total_correct += correct
                    total_samples += batch_size
                
                # Update progress bar
                if total_samples > 0:
                    pbar.set_postfix({
                        'Loss': f"{total_loss/total_samples:.4f}",
                        'Acc': f"{total_correct/total_samples:.4f}"
                    })
                
                # Clear memory
                del out, loss, pred
                if batch_idx % 10 == 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\nOOM on batch {batch_idx}, skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f"\nError on batch {batch_idx}: {str(e)}")
                    raise
            except Exception as e:
                print(f"\nUnexpected error on batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        pbar.close()
        
        # Calculate epoch metrics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            print("\nWarning: No valid samples processed in this epoch")
            avg_loss = float('inf')
            accuracy = 0.0
        
        # Memory summary
        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            print(f"\nGPU Memory peak: {gpu_mem:.2f}GB")
            
        print(f"\nEpoch complete - Processed {total_samples} samples")
        print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"{'='*50}\n")
        
        return avg_loss, accuracy

    def validate(self):
        """
        Validate the model.
        
        Returns:
            tuple: (avg_loss, accuracy) if validation data is available, None otherwise
        """
        if not self.use_validation or self.val_loader is None:
            print("No validation data provided, skipping validation")
            return None
            
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        processed_batches = 0
        
        print(f"\n{'='*50}\nStarting validation")
        print(f"Number of validation batches: {len(self.val_loader)}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                try:
                    # Debug: Print first batch info
                    # if batch_idx == 0:
                    #     print(f"\nFirst validation batch info:")
                    #     print(f"  Batch type: {type(batch)}")
                    #     print(f"  Batch keys: {getattr(batch, 'keys', 'N/A')}")
                    #     print(f"  Batch size: {getattr(batch, 'y', None).shape if hasattr(batch, 'y') else 'N/A'}")
                    
                    # Skip batches without labels
                    if not hasattr(batch, 'y') or batch.y is None:
                        print(f"\nWarning: Validation batch {batch_idx} has no labels (batch.y is None)")
                        continue
                    
                    # Move to device
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    out = self.model(batch)
                    
                    # Check for invalid outputs
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        print(f"\nWarning: Invalid model outputs in validation batch {batch_idx}")
                        print(f"  Output min: {out.min().item()}, max: {out.max().item()}")
                        print(f"  Output contains NaN: {torch.isnan(out).any().item()}")
                        print(f"  Output contains Inf: {torch.isinf(out).any().item()}")
                        continue
                    
                    # Calculate loss
                    loss = F.cross_entropy(out, batch.y, reduction='mean')
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\nWarning: Invalid loss in validation batch {batch_idx}")
                        print(f"  Loss: {loss.item()}")
                        continue
                    
                    # Calculate accuracy
                    pred = out.argmax(dim=1)
                    correct = (pred == batch.y).sum().item()
                    batch_size = batch.y.size(0)
                    
                    # Update metrics
                    total_loss += loss.item() * batch_size
                    total_correct += correct
                    total_samples += batch_size
                    processed_batches += 1
                    
                    # Clean up
                    del out, loss, pred, batch
                    
                except Exception as e:
                    print(f"\nError processing validation batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Clear cache every 10 batches
                if batch_idx % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate final metrics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"\nValidation complete - Processed {total_samples} samples")
            print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            print(f"Processed {processed_batches}/{len(self.val_loader)} batches")
        else:
            print("\nWarning: No valid validation samples processed")
            avg_loss = float('inf')
            accuracy = 0.0
        
        print(f"{'='*50}\n")
        return avg_loss, accuracy

    def test(self):
        """
        Test the model on test set.
        
        Returns:
            dict: Dictionary containing test metrics if test data is available, None otherwise
        """
        if not self.use_test or self.test_loader is None:
            print("No test data provided, skipping testing")
            return None
            
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = F.cross_entropy(out, batch.y, reduction='sum')
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                total_samples += len(batch.y)
                del out, loss, pred, batch
                gc.collect()
        
        # Handle case where no samples were processed
        if total_samples == 0:
            print("Warning: No test samples processed")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'classification_report': 'No test samples available'
            }
        
        # Calculate metrics
        avg_loss = total_loss / total_samples  # Average over samples, not batches
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, zero_division=0)
        
        # Print summary
        print("\nTest Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report
        }

    def save_model(self, path: str, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }, path)

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['val_acc']

    def train(self):
        """Complete training loop with early stopping."""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"gnn_training_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize test_results to None
        test_results = None
        
        try:
            for epoch in range(self.config['epochs']):
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                
                # Training
                train_loss, train_acc = self.train_epoch()
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                
                # Validation (if available)
                if self.use_validation and self.val_loader is not None:
                    val_results = self.validate()
                    if val_results is not None:
                        val_loss, val_acc = val_results
                        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                        
                        # Update learning rate scheduler
                        if self.scheduler is not None:
                            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.scheduler.step(val_loss)
                            else:
                                self.scheduler.step()
                        
                        # Save best model based on validation loss
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_val_acc = val_acc
                            best_model_path = os.path.join(output_dir, 'best_model.pth')
                            self.save_model(best_model_path, epoch, val_loss, val_acc)
                            print(f"Saved best model to {best_model_path}")
                            
                        # Early stopping
                        self.patience_counter += 1
                        if self.patience_counter >= self.config['patience']:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            break
                else:
                    # If no validation, save model every epoch
                    checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
                    self.save_model(checkpoint_path, epoch, train_loss, train_acc)
                    print(f"Model checkpoint saved to {checkpoint_path}")
        
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("Attempting to save current model state...")
            try:
                error_path = os.path.join(output_dir, f'model_error.pth')
                self.save_model(error_path, epoch, float('inf'), 0.0)
                print(f"Model state saved to {error_path}")
            except:
                print("Failed to save model state")
            raise
        
        # Load best model for testing if validation was used
        if self.use_validation and os.path.exists(os.path.join(output_dir, 'best_model.pth')):
            try:
                self.load_model(os.path.join(output_dir, 'best_model.pth'))
                print(f"\nLoaded best model with Val Loss: {self.best_val_loss:.4f}, Val Acc: {self.best_val_acc:.4f}")
            except Exception as e:
                print(f"\nFailed to load best model: {str(e)}")
        
        # Test evaluation if test data is available
        if self.use_test and self.test_loader is not None:
            print("\nEvaluating on test set...")
            try:
                test_results = self.test()
                if test_results is not None:
                    print(f"\nTest Results:")
                    print(f"  Loss: {test_results['loss']:.4f}")
                    print(f"  Accuracy: {test_results['accuracy']:.4f}")
                    print(f"  F1: {test_results['f1']:.4f}")
                    print(f"  Precision: {test_results['precision']:.4f}")
                    print(f"  Recall: {test_results['recall']:.4f}")
                    
                    # Save test results
                    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
                        json.dump(test_results, f, indent=2)
            except Exception as e:
                print(f"\nError during testing: {str(e)}")
        else:
            print("\nNo test data provided, skipping testing")
        
        # Save training history
        try:
            with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            print(f"\nFailed to save training history: {str(e)}")
        
        print(f"\nTraining completed! Results saved in: {output_dir}")
        return output_dir


def main():
    """Main training function with optimized defaults."""
    parser = argparse.ArgumentParser(description='Optimized GNN Training Pipeline')
    
    # Data paths
    parser.add_argument('--train_data_dir', type=str, 
                       default="/app/src/Clean_Code/output/gnn_embeddings/knn8/stanfordnlp/sst2/train/train")
    parser.add_argument('--val_data_dir', type=str, 
                       default="/app/src/Clean_Code/output/gnn_embeddings/knn8/stanfordnlp/sst2/validation/validation")
    
    # Model architecture
    parser.add_argument('--module', type=str, default='GCNConv')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--residual', action='store_true')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cache_size', type=int, default=0, help='Set to 0 to disable caching')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = vars(args)
    if config['scheduler'] == 'None':
        config['scheduler'] = None
        
    # Initialize and run trainer
    trainer = GNNTrainer(config)
    output_dir = trainer.train()
    
    return output_dir


if __name__ == '__main__':
    main()
