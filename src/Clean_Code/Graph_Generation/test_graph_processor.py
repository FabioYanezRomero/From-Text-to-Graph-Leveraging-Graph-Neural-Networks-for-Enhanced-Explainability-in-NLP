"""
Test Graph Processor

This module tests the graph processor functionality.
"""

import os
import torch
import pickle as pkl
import numpy as np
import logging
import unittest
from torch_geometric.data import Data

from src.Clean_Code.Graph_Generation.graph_data_processor import (
    create_word_graphs,
    load_special_embeddings,
    process_embeddings_batch
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestGraphProcessor(unittest.TestCase):
    """Test graph processor functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create test data
        self.embedding_dim = 768
        self.num_samples = 5
        self.num_tokens = 10
        
        # Create word embeddings
        self.word_embeddings = []
        for i in range(self.num_samples):
            sample_embeddings = []
            for j in range(self.num_tokens):
                sample_embeddings.append(np.random.randn(self.embedding_dim))
            self.word_embeddings.append(sample_embeddings)
        
        # Create sentence embeddings
        self.sentence_embeddings = []
        for i in range(self.num_samples):
            self.sentence_embeddings.append(np.random.randn(self.embedding_dim))
        
        # Create texts
        self.texts = []
        for i in range(self.num_samples):
            self.texts.append([f"token_{j}" for j in range(self.num_tokens)])
        
        # Create labels
        self.labels = []
        for i in range(self.num_samples):
            self.labels.append(i % 2)  # Binary labels
        
        # Create special embeddings
        self.special_embeddings = {
            "«SENTENCE»": np.random.randn(self.embedding_dim),
            "«NOUN PHRASE»": np.random.randn(self.embedding_dim),
            "«VERB PHRASE»": np.random.randn(self.embedding_dim)
        }
        
        # Create test directory
        self.test_dir = "/tmp/test_graph_processor"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def test_create_word_graphs(self):
        """Test create_word_graphs function"""
        logger.info("Testing create_word_graphs function...")
        
        # Create graphs with window edges
        window_graphs = create_word_graphs(
            self.word_embeddings,
            self.sentence_embeddings,
            self.texts,
            self.labels,
            special_embeddings=None,
            window_size=3,
            edge_type='window'
        )
        
        # Check graph properties
        self.assertEqual(len(window_graphs), self.num_samples)
        
        for i, graph in enumerate(window_graphs):
            # Check node features
            self.assertEqual(graph.x.shape, (self.num_tokens, self.embedding_dim))
            
            # Check edge index
            self.assertEqual(graph.edge_index.shape[0], 2)
            
            # Check label
            self.assertEqual(graph.y.item(), self.labels[i])
            
            # Check text
            self.assertEqual(len(graph.text), self.num_tokens)
            
            # Check sentence embedding
            self.assertEqual(graph.sentence_embedding.shape, (self.embedding_dim,))
        
        logger.info("Window graph tests passed!")
        
        # Create graphs with fully connected edges
        fully_graphs = create_word_graphs(
            self.word_embeddings,
            self.sentence_embeddings,
            self.texts,
            self.labels,
            special_embeddings=None,
            window_size=3,
            edge_type='fully_connected'
        )
        
        # Check graph properties
        self.assertEqual(len(fully_graphs), self.num_samples)
        
        for i, graph in enumerate(fully_graphs):
            # Check node features
            self.assertEqual(graph.x.shape, (self.num_tokens, self.embedding_dim))
            
            # Check edge index - fully connected should have n*(n-1) edges
            expected_edges = self.num_tokens * (self.num_tokens - 1)
            self.assertEqual(graph.edge_index.shape[1], expected_edges)
            
            # Check label
            self.assertEqual(graph.y.item(), self.labels[i])
            
            # Check text
            self.assertEqual(len(graph.text), self.num_tokens)
            
            # Check sentence embedding
            self.assertEqual(graph.sentence_embedding.shape, (self.embedding_dim,))
        
        logger.info("Fully connected graph tests passed!")
    
    def test_special_embeddings_integration(self):
        """Test integration of special embeddings"""
        logger.info("Testing special embeddings integration...")
        
        # Create texts with special tokens
        special_texts = [
            ["token_0", "«SENTENCE»", "token_2", "token_3", "token_4"],
            ["token_0", "token_1", "«NOUN PHRASE»", "token_3", "token_4"],
            ["token_0", "token_1", "token_2", "«VERB PHRASE»", "token_4"]
        ]
        
        # Create word embeddings for special texts
        special_word_embeddings = []
        for i in range(len(special_texts)):
            sample_embeddings = []
            for j in range(len(special_texts[i])):
                sample_embeddings.append(np.random.randn(self.embedding_dim))
            special_word_embeddings.append(sample_embeddings)
        
        # Create sentence embeddings for special texts
        special_sentence_embeddings = []
        for i in range(len(special_texts)):
            special_sentence_embeddings.append(np.random.randn(self.embedding_dim))
        
        # Create labels for special texts
        special_labels = [0, 1, 0]
        
        # Create graphs with special embeddings
        special_graphs = create_word_graphs(
            special_word_embeddings,
            special_sentence_embeddings,
            special_texts,
            special_labels,
            special_embeddings=self.special_embeddings,
            window_size=3,
            edge_type='window'
        )
        
        # Check graph properties
        self.assertEqual(len(special_graphs), len(special_texts))
        
        # Check that special embeddings were integrated
        for i, graph in enumerate(special_graphs):
            # Find special token in text
            special_tokens = [token for token in special_texts[i] if token in self.special_embeddings]
            
            if special_tokens:
                special_token = special_tokens[0]
                special_token_idx = special_texts[i].index(special_token)
                
                # Get node embedding for special token
                node_embedding = graph.x[special_token_idx].numpy()
                
                # Get special embedding
                special_embedding = self.special_embeddings[special_token]
                
                # Check that node embedding matches special embedding
                self.assertTrue(np.allclose(node_embedding, special_embedding))
        
        logger.info("Special embeddings integration tests passed!")
    
    def test_edge_creation(self):
        """Test edge creation"""
        logger.info("Testing edge creation...")
        
        # Create a simple text
        text = ["token_0", "token_1", "token_2", "token_3", "token_4"]
        
        # Create word embeddings
        word_embeddings = [np.random.randn(self.embedding_dim) for _ in range(len(text))]
        
        # Create sentence embedding
        sentence_embedding = np.random.randn(self.embedding_dim)
        
        # Create label
        label = 0
        
        # Create graph with window size 2
        window_graphs = create_word_graphs(
            [word_embeddings],
            [sentence_embedding],
            [text],
            [label],
            special_embeddings=None,
            window_size=2,
            edge_type='window'
        )
        
        # Check edge index
        window_graph = window_graphs[0]
        edge_index = window_graph.edge_index
        
        # Print edge index for debugging
        logger.info(f"Edge index shape: {edge_index.shape}")
        logger.info(f"Edge index: {edge_index}")
        
        # Check connections for each node
        for i in range(len(text)):
            # Get indices of nodes that this node connects to
            connections = edge_index[1][edge_index[0] == i].tolist()
            logger.info(f"Node {i} connects to: {connections}")
            
            # Check that connections are within window
            for j in connections:
                self.assertTrue(abs(i - j) <= 2)
        
        # Create fully connected graph
        fully_graphs = create_word_graphs(
            [word_embeddings],
            [sentence_embedding],
            [text],
            [label],
            special_embeddings=None,
            window_size=2,
            edge_type='fully_connected'
        )
        
        # Check edge index
        fully_graph = fully_graphs[0]
        edge_index_fully = fully_graph.edge_index
        
        # Print edge index for debugging
        logger.info(f"Fully connected edge index shape: {edge_index_fully.shape}")
        
        # Check that each node connects to all other nodes
        num_nodes = len(text)
        expected_edges = num_nodes * (num_nodes - 1)  # Excluding self-loops
        self.assertEqual(edge_index_fully.shape[1], expected_edges)
        
        return window_graphs, fully_graphs
    
    def test_process_embeddings_batch(self):
        """Test process_embeddings_batch function"""
        logger.info("Testing process_embeddings_batch function...")
        
        # Create config
        config = {
            'window_size': 3,
            'edge_type': 'window',
            'batch_size': 10
        }
        
        # Create output directory
        split_output_dir = os.path.join(self.test_dir, 'test_batch')
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process batch
        num_graphs = process_embeddings_batch(
            self.word_embeddings,
            self.sentence_embeddings,
            self.texts,
            self.labels,
            config,
            split_output_dir,
            0,
            self.special_embeddings
        )
        
        # Check number of graphs
        self.assertEqual(num_graphs, self.num_samples)
        
        # Check that batch file was created
        batch_file = os.path.join(split_output_dir, 'graphs_window_batch_0000.pkl')
        self.assertTrue(os.path.exists(batch_file))
        
        # Load batch file
        with open(batch_file, 'rb') as f:
            batch_graphs = pkl.load(f)
        
        # Check batch graphs
        self.assertEqual(len(batch_graphs), self.num_samples)
        
        logger.info("Batch processing tests passed!")

def main():
    """Run tests"""
    # Create test instance
    test = TestGraphProcessor()
    
    # Set up test data
    test.setUp()
    
    # Test graph creation
    logger.info("Testing PyTorch Geometric graph creation...")
    test.test_create_word_graphs()
    
    # Test special embeddings integration
    logger.info("Testing special embeddings integration...")
    test.test_special_embeddings_integration()
    
    # Test edge creation
    logger.info("Testing edge creation...")
    window_graphs, fully_graphs = test.test_edge_creation()
    
    # Test batch processing
    logger.info("Testing batch processing...")
    test.test_process_embeddings_batch()
    
    logger.info("All PyTorch Geometric graph tests passed!")

if __name__ == "__main__":
    main()
