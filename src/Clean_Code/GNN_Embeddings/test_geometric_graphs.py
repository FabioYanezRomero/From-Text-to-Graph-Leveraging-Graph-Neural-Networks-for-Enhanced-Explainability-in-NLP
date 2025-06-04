"""
Test script to verify PyTorch Geometric graph creation from embeddings
"""

import os
import sys
import logging
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# Add the project root to the path
sys.path.append('/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_graph_creation_from_embeddings():
    """Test creating PyTorch Geometric graphs from embeddings"""
    from src.Clean_Code.GNN_Embeddings.gnn_data_processor import create_word_graphs
    
    # Create mock data
    batch_size = 5
    seq_length = 10
    embedding_dim = 768
    
    # Create word embeddings (batch_size x seq_length x embedding_dim)
    word_embeddings = [torch.rand(seq_length, embedding_dim) for _ in range(batch_size)]
    
    # Create sentence embeddings (batch_size x embedding_dim)
    sentence_embeddings = [torch.rand(embedding_dim) for _ in range(batch_size)]
    
    # Create texts (batch_size x seq_length)
    texts = [["word" + str(j) for j in range(seq_length)] for _ in range(batch_size)]
    
    # Create labels
    labels = [i % 2 for i in range(batch_size)]
    
    # Create special embeddings
    special_embeddings = {
        "«SENTENCE»": torch.rand(embedding_dim),
        "«NOUN PHRASE»": torch.rand(embedding_dim),
        "«VERB PHRASE»": torch.rand(embedding_dim)
    }
    
    # Add some special tokens to texts
    texts[0][2] = "«SENTENCE»"
    texts[1][3] = "«NOUN PHRASE»"
    texts[2][4] = "«VERB PHRASE»"
    
    # Create graphs
    logger.info("Creating PyTorch Geometric graphs...")
    graphs = create_word_graphs(
        word_embeddings, 
        sentence_embeddings, 
        texts, 
        labels, 
        special_embeddings=special_embeddings,
        window_size=3, 
        edge_type='window'
    )
    
    # Verify graphs
    logger.info(f"Created {len(graphs)} graphs")
    
    for i, graph in enumerate(graphs):
        logger.info(f"Graph {i}:")
        logger.info(f"  Node features shape: {graph.x.shape}")
        logger.info(f"  Edge index shape: {graph.edge_index.shape}")
        logger.info(f"  Label: {graph.y}")
        logger.info(f"  Text length: {len(graph.text)}")
        logger.info(f"  Sentence embedding shape: {graph.sentence_embedding.shape}")
        
        # Check that the graph is a valid PyTorch Geometric Data object
        assert isinstance(graph, Data), "Graph should be a PyTorch Geometric Data object"
        
        # Check that node features have the right shape
        assert graph.x.shape[1] == embedding_dim, f"Node features should have dimension {embedding_dim}"
        
        # Check that edge index has the right shape (2 x num_edges)
        assert graph.edge_index.shape[0] == 2, "Edge index should have 2 rows"
        
        # Check that the label is correct
        assert graph.y.item() == labels[i], f"Label should be {labels[i]}"
        
        # Check that the text is stored
        assert hasattr(graph, 'text'), "Graph should have a text attribute"
        
        # Check that the sentence embedding is stored
        assert hasattr(graph, 'sentence_embedding'), "Graph should have a sentence_embedding attribute"
        assert graph.sentence_embedding.shape[0] == embedding_dim, f"Sentence embedding should have dimension {embedding_dim}"
    
    logger.info("All graph checks passed!")
    return graphs

def test_special_embeddings_integration():
    """Test that special embeddings are correctly integrated into graphs"""
    from src.Clean_Code.GNN_Embeddings.gnn_data_processor import create_word_graphs
    
    # Create mock data with special tokens
    batch_size = 3
    seq_length = 10
    embedding_dim = 768
    
    # Create word embeddings
    word_embeddings = [torch.rand(seq_length, embedding_dim) for _ in range(batch_size)]
    
    # Create sentence embeddings
    sentence_embeddings = [torch.rand(embedding_dim) for _ in range(batch_size)]
    
    # Create texts with special tokens
    texts = [
        ["word1", "word2", "«SENTENCE»", "word4", "word5", "word6", "word7", "word8", "word9", "word10"],
        ["word1", "word2", "word3", "«NOUN PHRASE»", "word5", "word6", "word7", "word8", "word9", "word10"],
        ["word1", "word2", "word3", "word4", "«VERB PHRASE»", "word6", "word7", "word8", "word9", "word10"]
    ]
    
    # Create labels
    labels = [0, 1, 0]
    
    # Create special embeddings with known values for testing
    special_embeddings = {
        "«SENTENCE»": torch.ones(embedding_dim),
        "«NOUN PHRASE»": torch.ones(embedding_dim) * 2,
        "«VERB PHRASE»": torch.ones(embedding_dim) * 3
    }
    
    # Create graphs
    logger.info("Creating graphs with special embeddings...")
    graphs = create_word_graphs(
        word_embeddings, 
        sentence_embeddings, 
        texts, 
        labels, 
        special_embeddings=special_embeddings,
        window_size=3, 
        edge_type='window'
    )
    
    # Verify special embeddings integration
    logger.info("Verifying special embeddings integration...")
    
    # Check first graph - should have «SENTENCE» at position 2
    graph0 = graphs[0]
    special_node_embedding = graph0.x[2]
    assert torch.allclose(special_node_embedding, special_embeddings["«SENTENCE»"]), \
        "Special embedding for «SENTENCE» not correctly integrated"
    
    # Check second graph - should have «NOUN PHRASE» at position 3
    graph1 = graphs[1]
    special_node_embedding = graph1.x[3]
    assert torch.allclose(special_node_embedding, special_embeddings["«NOUN PHRASE»"]), \
        "Special embedding for «NOUN PHRASE» not correctly integrated"
    
    # Check third graph - should have «VERB PHRASE» at position 4
    graph2 = graphs[2]
    special_node_embedding = graph2.x[4]
    assert torch.allclose(special_node_embedding, special_embeddings["«VERB PHRASE»"]), \
        "Special embedding for «VERB PHRASE» not correctly integrated"
    
    logger.info("Special embeddings correctly integrated into graphs!")
    return graphs

def test_edge_creation():
    """Test that edges are correctly created in the graphs"""
    from src.Clean_Code.GNN_Embeddings.gnn_data_processor import create_word_graphs
    
    # Create a simple test case with known structure
    seq_length = 5
    embedding_dim = 4
    
    # Create word embeddings for a single sample
    word_embeddings = [torch.tensor([[1.0, 2.0, 3.0, 4.0],  # word 0
                                    [5.0, 6.0, 7.0, 8.0],  # word 1
                                    [9.0, 10.0, 11.0, 12.0],  # word 2
                                    [13.0, 14.0, 15.0, 16.0],  # word 3
                                    [17.0, 18.0, 19.0, 20.0]])]  # word 4
    
    # Create sentence embedding
    sentence_embeddings = [torch.tensor([1.0, 2.0, 3.0, 4.0])]
    
    # Create text
    texts = [["word0", "word1", "word2", "word3", "word4"]]
    
    # Create label
    labels = [1]
    
    # Create graphs with window size 2
    logger.info("Creating graph with window size 2...")
    graphs_window2 = create_word_graphs(
        word_embeddings, 
        sentence_embeddings, 
        texts, 
        labels, 
        special_embeddings=None,
        window_size=2, 
        edge_type='window'
    )
    
    # Verify window edges
    graph = graphs_window2[0]
    edge_index = graph.edge_index
    
    logger.info(f"Edge index shape: {edge_index.shape}")
    logger.info(f"Edge index: {edge_index}")
    
    # For window size 2, each node should connect to 2 nodes before and 2 nodes after
    # plus itself, except at the boundaries
    
    # Expected connections for node 0: [0->0, 0->1, 0->2]
    # Expected connections for node 1: [1->0, 1->1, 1->2, 1->3]
    # Expected connections for node 2: [2->0, 2->1, 2->2, 2->3, 2->4]
    # Expected connections for node 3: [3->1, 3->2, 3->3, 3->4]
    # Expected connections for node 4: [4->2, 4->3, 4->4]
    
    # Count edges for each node
    num_nodes = word_embeddings[0].shape[0]
    for node in range(num_nodes):
        # Find all edges where this node is the source
        edges_from_node = edge_index[0] == node
        target_nodes = edge_index[1][edges_from_node]
        
        logger.info(f"Node {node} connects to: {target_nodes.tolist()}")
        
        # Check that connections are within the window
        for target in target_nodes:
            assert abs(node - target) <= 2, f"Node {node} connects to node {target} which is outside window size 2"
    
    # Create fully connected graph
    logger.info("Creating fully connected graph...")
    graphs_fully = create_word_graphs(
        word_embeddings, 
        sentence_embeddings, 
        texts, 
        labels, 
        special_embeddings=None,
        window_size=2, 
        edge_type='fully_connected'
    )
    
    # Verify fully connected edges
    graph_fully = graphs_fully[0]
    edge_index_fully = graph_fully.edge_index
    
    logger.info(f"Fully connected edge index shape: {edge_index_fully.shape}")
    
    # For fully connected, each node should connect to all other nodes (excluding self-loops)
    expected_edges = num_nodes * (num_nodes - 1)  # Excluding self-loops
    assert edge_index_fully.shape[1] == expected_edges, \
        f"Fully connected graph should have {expected_edges} edges, but has {edge_index_fully.shape[1]}"
    
    logger.info("Edge creation test passed!")
    return graphs_window2, graphs_fully

def main():
    """Main test function"""
    logger.info("Testing PyTorch Geometric graph creation...")
    
    # Test basic graph creation
    graphs = test_graph_creation_from_embeddings()
    
    # Test special embeddings integration
    special_graphs = test_special_embeddings_integration()
    
    # Test edge creation
    window_graphs, fully_graphs = test_edge_creation()
    
    logger.info("All PyTorch Geometric graph tests passed!")

if __name__ == "__main__":
    main()
