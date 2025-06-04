import os
import torch
import pickle as pkl
import logging
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_special_embeddings_test(model_name='bert-base-uncased'):
    """Test function to generate special embeddings for constituency tokens"""
    logger.info("Testing special embeddings generation")
    
    # Define constituency dictionary for special tokens
    constituency_dict = {
        '«SENTENCE»': 'Sentence',
        '«NOUN PHRASE»': 'Noun phrase',
        '«VERB PHRASE»': 'Verb phrase',
    }
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = 'cpu'
    special_embeddings = {}
    model.eval()
    
    with torch.no_grad():
        for token, description in constituency_dict.items():
            logger.info(f"Processing token: {token}")
            # Tokenize the token
            inputs = tokenizer(description, return_tensors="pt").to(device)
            
            # Get model outputs with hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get the last hidden state
            if hasattr(outputs, 'hidden_states'):
                last_hidden_state = outputs.hidden_states[-1]
            else:
                last_hidden_state = outputs.last_hidden_state
            
            # Extract the CLS token embedding
            cls_embedding = last_hidden_state[:, 0, :].squeeze(0)
            
            # Store the embedding
            special_embeddings[token] = cls_embedding
            logger.info(f"Generated embedding shape: {cls_embedding.shape}")
    
    logger.info(f"Generated {len(special_embeddings)} special embeddings")
    
    # Save special embeddings to test file
    test_dir = os.path.join(os.getcwd(), 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    special_embeddings_path = os.path.join(test_dir, 'special_embeddings.pkl')
    
    with open(special_embeddings_path, 'wb') as f:
        pkl.dump(special_embeddings, f)
    
    logger.info(f"Saved special embeddings to {special_embeddings_path}")
    
    # Load special embeddings to verify
    with open(special_embeddings_path, 'rb') as f:
        loaded_embeddings = pkl.load(f)
    
    logger.info(f"Loaded {len(loaded_embeddings)} special embeddings")
    for token, embedding in loaded_embeddings.items():
        logger.info(f"Token: {token}, Shape: {embedding.shape}")
    
    return special_embeddings

if __name__ == "__main__":
    generate_special_embeddings_test()
