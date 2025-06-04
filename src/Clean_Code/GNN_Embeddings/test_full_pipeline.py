import os
import torch
import pickle as pkl
import logging
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our functions
from embedding_generator import generate_special_embeddings, save_embeddings
from gnn_data_processor import load_embeddings, create_word_graphs

def test_full_pipeline():
    """Test the full pipeline from embedding generation to graph creation"""
    logger.info("Testing full pipeline")
    
    # Set up test directory
    test_dir = os.path.join(os.getcwd(), 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test data
    texts = [
        ['This', 'is', 'a', 'test', '«SENTENCE»'],
        ['Another', 'test', 'with', '«NOUN PHRASE»', 'included'],
        ['«VERB PHRASE»', 'at', 'the', 'beginning']
    ]
    labels = [0, 1, 0]
    
    # Generate fake word embeddings (would normally come from extract_embeddings)
    word_embeddings = [np.random.rand(len(text), 768) for text in texts]
    sentence_embeddings = [np.random.rand(768) for _ in texts]
    
    # Load model and tokenizer for special embeddings
    logger.info("Loading model for special embeddings")
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate special embeddings
    special_embeddings = generate_special_embeddings(model, tokenizer, device='cpu')
    
    # Save all embeddings
    logger.info("Saving embeddings")
    save_embeddings(word_embeddings, sentence_embeddings, special_embeddings, texts, labels, test_dir)
    
    # Load embeddings
    logger.info("Loading embeddings")
    loaded_word_embeddings, loaded_sentence_embeddings, loaded_special_embeddings, loaded_texts, loaded_labels = load_embeddings(test_dir)
    
    # Verify loaded data
    assert len(loaded_word_embeddings) == len(word_embeddings)
    assert len(loaded_sentence_embeddings) == len(sentence_embeddings)
    assert loaded_special_embeddings is not None
    assert len(loaded_texts) == len(texts)
    assert len(loaded_labels) == len(labels)
    
    logger.info("Embeddings loaded successfully")
    
    # Create graphs
    logger.info("Creating graphs")
    graphs = create_word_graphs(
        loaded_word_embeddings,
        loaded_sentence_embeddings,
        loaded_texts,
        loaded_labels,
        special_embeddings=loaded_special_embeddings
    )
    
    # Verify graphs
    assert len(graphs) == len(texts)
    logger.info(f"Created {len(graphs)} graphs")
    
    # Check that special embeddings were used
    for i, graph in enumerate(graphs):
        logger.info(f"Graph {i}: {graph}")
        logger.info(f"Graph {i} shape: {graph.x.shape}")
    
    logger.info("Full pipeline test completed successfully")
    return True

if __name__ == "__main__":
    test_full_pipeline()
