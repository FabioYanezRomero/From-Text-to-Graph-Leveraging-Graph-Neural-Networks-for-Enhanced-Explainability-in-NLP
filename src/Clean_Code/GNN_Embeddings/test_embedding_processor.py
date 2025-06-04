"""
Test script for the embedding processor
"""

import os
import sys
import logging
import torch
import pickle as pkl
from tqdm import tqdm

# Add the project root to the path
sys.path.append('/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_find_best_model():
    """Test the find_best_model function"""
    from src.Clean_Code.GNN_Embeddings.embedding_processor import find_best_model
    
    # Test with SST2 dataset
    model_path, model_name, best_score = find_best_model(
        '/app/src/Clean_Code/output/finetuned_llms', 
        'sst2'
    )
    
    logger.info(f"Best model path: {model_path}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Best score: {best_score}")
    
    assert model_path is not None, "Model path should not be None"
    assert model_name is not None, "Model name should not be None"
    assert best_score > 0, "Best score should be greater than 0"
    
    return model_path, model_name, best_score

def test_generate_special_embeddings(model_name):
    """Test generating special embeddings"""
    from transformers import AutoModel, AutoTokenizer
    from src.Clean_Code.GNN_Embeddings.embedding_generator import generate_special_embeddings
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate special embeddings
    device = 'cpu'  # Use CPU for testing
    special_embeddings = generate_special_embeddings(model, tokenizer, device=device)
    
    logger.info(f"Generated {len(special_embeddings)} special embeddings")
    
    # Check a few special embeddings
    for token, embedding in list(special_embeddings.items())[:3]:
        logger.info(f"Token: {token}, Embedding shape: {embedding.shape}")
    
    assert len(special_embeddings) > 0, "Should have generated special embeddings"
    
    return special_embeddings

def test_chunked_embedding_storage():
    """Test the chunked embedding storage"""
    from src.Clean_Code.GNN_Embeddings.embedding_generator import save_embeddings
    
    # Create mock data
    word_embeddings = [torch.rand(10, 768) for _ in range(5)]
    sentence_embeddings = [torch.rand(768) for _ in range(5)]
    special_embeddings = {"token1": torch.rand(768), "token2": torch.rand(768)}
    texts = [["word1", "word2", "word3"] for _ in range(5)]
    labels = [0, 1, 0, 1, 0]
    
    # Create output directory
    output_dir = "/app/src/Clean_Code/output/test_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    save_embeddings(word_embeddings, sentence_embeddings, special_embeddings, texts, labels, output_dir)
    
    # Check if files were created
    assert os.path.exists(os.path.join(output_dir, "metadata.pkl")), "Metadata file should exist"
    assert os.path.exists(os.path.join(output_dir, "word_embeddings.pkl")), "Word embeddings file should exist"
    assert os.path.exists(os.path.join(output_dir, "sentence_embeddings.pkl")), "Sentence embeddings file should exist"
    assert os.path.exists(os.path.join(output_dir, "special_embeddings.pkl")), "Special embeddings file should exist"
    
    # Load and check embeddings
    with open(os.path.join(output_dir, "word_embeddings.pkl"), "rb") as f:
        loaded_word_embeddings = pkl.load(f)
    
    with open(os.path.join(output_dir, "sentence_embeddings.pkl"), "rb") as f:
        loaded_sentence_embeddings = pkl.load(f)
    
    with open(os.path.join(output_dir, "special_embeddings.pkl"), "rb") as f:
        loaded_special_embeddings = pkl.load(f)
    
    with open(os.path.join(output_dir, "metadata.pkl"), "rb") as f:
        loaded_metadata = pkl.load(f)
    
    # Verify loaded data
    assert len(loaded_word_embeddings) == len(word_embeddings), "Should have same number of word embeddings"
    assert len(loaded_sentence_embeddings) == len(sentence_embeddings), "Should have same number of sentence embeddings"
    assert len(loaded_special_embeddings) == len(special_embeddings), "Should have same number of special embeddings"
    assert loaded_metadata["texts"] == texts, "Texts should match"
    assert loaded_metadata["labels"] == labels, "Labels should match"
    
    logger.info("Chunked embedding storage test passed")
    
    return output_dir

def main():
    """Main test function"""
    logger.info("Testing embedding processor...")
    
    # Test finding the best model
    model_path, model_name, best_score = test_find_best_model()
    
    # Test generating special embeddings
    special_embeddings = test_generate_special_embeddings(model_name)
    
    # Test chunked embedding storage
    output_dir = test_chunked_embedding_storage()
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    main()
