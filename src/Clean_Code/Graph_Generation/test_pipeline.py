"""
Test Pipeline for Graph Generation

This script tests the complete pipeline for graph generation from embeddings.
"""

import os
import torch
import numpy as np
import pickle as pkl
import logging
import tempfile
import shutil
from torch_geometric.data import Data

from src.Clean_Code.Graph_Generation.graph_data_processor import (
    create_word_graphs,
    process_embeddings
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_embeddings(temp_dir, num_samples=5, embedding_dim=768, chunk_size=2):
    """Create mock embeddings for testing"""
    
    # Create dataset structure
    dataset_name = "mock_dataset"
    embeddings_dir = os.path.join(temp_dir, "embeddings", dataset_name)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create splits
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(embeddings_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create chunks directory
        chunks_dir = os.path.join(split_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Create chunks
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            chunk_dir = os.path.join(chunks_dir, f"chunk_{chunk_idx}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            # Calculate number of samples in this chunk
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_samples = end_idx - start_idx
            
            # Create word embeddings
            word_embeddings = []
            for _ in range(chunk_samples):
                # Random number of words (3-10)
                num_words = np.random.randint(3, 10)
                sample_embeddings = torch.randn(num_words, embedding_dim)
                word_embeddings.append(sample_embeddings)
            
            # Create sentence embeddings
            sentence_embeddings = [torch.randn(embedding_dim) for _ in range(chunk_samples)]
            
            # Create texts and labels
            texts = []
            for i in range(chunk_samples):
                num_words = len(word_embeddings[i])
                texts.append([f"word_{j}" for j in range(num_words)])
            
            labels = [np.random.randint(0, 2) for _ in range(chunk_samples)]
            
            # Create metadata
            metadata = {
                "texts": texts,
                "labels": labels
            }
            
            # Save to disk
            with open(os.path.join(chunk_dir, "word_embeddings.pkl"), "wb") as f:
                pkl.dump(word_embeddings, f)
            
            with open(os.path.join(chunk_dir, "sentence_embeddings.pkl"), "wb") as f:
                pkl.dump(sentence_embeddings, f)
            
            with open(os.path.join(chunk_dir, "metadata.pkl"), "wb") as f:
                pkl.dump(metadata, f)
    
    # Create special embeddings
    special_dir = os.path.join(temp_dir, "embeddings", "special")
    os.makedirs(special_dir, exist_ok=True)
    
    special_embeddings = {
        "NP": torch.randn(embedding_dim),
        "VP": torch.randn(embedding_dim),
        "S": torch.randn(embedding_dim)
    }
    
    with open(os.path.join(special_dir, "special_embeddings.pkl"), "wb") as f:
        pkl.dump(special_embeddings, f)
    
    return dataset_name, embeddings_dir

def test_pipeline():
    """Test the complete pipeline for graph generation"""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create mock embeddings
        dataset_name, embeddings_dir = create_mock_embeddings(temp_dir)
        
        # Process embeddings
        output_dir = process_embeddings(
            dataset_name=dataset_name,
            embeddings_dir=os.path.join(temp_dir, "embeddings", dataset_name),
            batch_size=2,
            window_size=2,
            edge_type="window"
        )
        
        # Check if output directory exists
        assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"
        
        # Check if graphs were created for each split
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(output_dir, split)
            assert os.path.exists(split_dir), f"Split directory {split_dir} does not exist"
            
            # Check if config file exists
            config_path = os.path.join(split_dir, "config.json")
            assert os.path.exists(config_path), f"Config file {config_path} does not exist"
            
            # Check if at least one batch of graphs exists
            graph_files = [f for f in os.listdir(split_dir) if f.startswith("graphs_")]
            assert len(graph_files) > 0, f"No graph files found in {split_dir}"
            
            # Load one batch of graphs
            graph_path = os.path.join(split_dir, graph_files[0])
            with open(graph_path, "rb") as f:
                graphs = pkl.load(f)
            
            # Check if graphs are valid
            assert len(graphs) > 0, f"No graphs found in {graph_path}"
            
            for graph in graphs:
                assert isinstance(graph, Data), f"Graph is not a torch_geometric.data.Data object"
                assert hasattr(graph, "x"), f"Graph does not have node features"
                assert hasattr(graph, "edge_index"), f"Graph does not have edge indices"
                assert hasattr(graph, "y"), f"Graph does not have labels"
        
        logger.info("Pipeline test passed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_pipeline()
