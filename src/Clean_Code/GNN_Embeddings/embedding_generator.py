#!/usr/bin/env python3
"""
GNN Embeddings Generator

This module generates embeddings for words in text datasets using fine-tuned language models.
It extracts embeddings from the fine-tuned models and prepares them for use in GNN models.
"""

import os
import torch
import pickle as pkl
import logging
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset class for text data"""
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.texts[idx], self.labels[idx]
        return self.texts[idx], 0

def load_dataset(dataset_name):
    """Load dataset from Hugging Face datasets"""
    from datasets import load_dataset
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Extract texts and labels based on dataset
    texts = {}
    labels = {}
    
    if dataset_name == 'stanfordnlp/sst2':
        for split in dataset.keys():
            if split == 'test':  # Skip test set as it has no labels
                continue
            texts[split] = dataset[split]['sentence']
            labels[split] = dataset[split]['label']
    
    elif dataset_name == 'setfit/ag_news':
        for split in dataset.keys():
            texts[split] = dataset[split]['text']
            labels[split] = dataset[split]['label']
    
    return texts, labels

def token_mapper(tokenizer, tokenized_sentences):
    """Map tokens to words in the original text"""
    # List of tokens to ignore when getting the embeddings
    ignore_tokens = []
    for token in tokenizer.special_tokens_map.values():
        if isinstance(token, str):
            ignore_tokens.append(tokenizer.convert_tokens_to_ids(token))
        elif isinstance(token, list):
            ignore_tokens.extend([tokenizer.convert_tokens_to_ids(t) for t in token if t is not None])
    
    token_mapping_list = []
    for tokenized_sentence in tokenized_sentences:
        token_mapping = {}
        word_iterator = 0
        token_iterator = 0
        first_iteration = True
        
        for i, token_id in enumerate(tokenized_sentence):
            if token_id in ignore_tokens:
                token_iterator += 1
                continue
                
            token = tokenizer.convert_ids_to_tokens(token_id.item())
            
            # Check if this is the start of a new word
            if not token.startswith('##'):
                if not first_iteration:
                    word_iterator += 1
                token_mapping[word_iterator] = [token_iterator]
                token_iterator += 1
                first_iteration = False
            else:
                # This is a continuation of the previous word
                token_mapping[word_iterator].append(token_iterator)
                token_iterator += 1
                
        token_mapping_list.append(token_mapping)
    
    return token_mapping_list

def save_embeddings(word_embeddings, sentence_embeddings, special_embeddings, texts, labels, output_dir):
    """Save embeddings and metadata to disk in a single file
    
    Args:
        word_embeddings: List of word embeddings for each text
        sentence_embeddings: List of sentence embeddings for each text
        special_embeddings: Dictionary of special embeddings for constituency trees
        texts: List of texts
        labels: List of labels
        output_dir: Output directory
        
    Returns:
        output_dir: Directory containing the saved embeddings
    """
    import os
    import pickle as pkl
    import logging
    
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of samples
    total_samples = len(word_embeddings)
    
    logger.info(f"Saving {total_samples} samples to {output_dir}")
    
    # Save metadata
    metadata = {
        'texts': texts,
        'labels': labels,
        'total_samples': total_samples
    }
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pkl.dump(metadata, f)
    
    # Save word embeddings
    word_embeddings_path = os.path.join(output_dir, 'word_embeddings.pkl')
    with open(word_embeddings_path, 'wb') as f:
        pkl.dump(word_embeddings, f)
    
    # Save sentence embeddings
    sentence_embeddings_path = os.path.join(output_dir, 'sentence_embeddings.pkl')
    with open(sentence_embeddings_path, 'wb') as f:
        pkl.dump(sentence_embeddings, f)
    
    # Save special embeddings if provided
    if special_embeddings:
        special_embeddings_path = os.path.join(output_dir, 'special_embeddings.pkl')
        with open(special_embeddings_path, 'wb') as f:
            pkl.dump(special_embeddings, f)
    
    logger.info(f"Saved embeddings to {output_dir}")
    return output_dir

def extract_embeddings(texts, model, tokenizer, batch_size=16, max_length=128, device='cuda'):
    """Extract embeddings from the model for each word in the texts"""
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_word_embeddings = []
    all_sentence_embeddings = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch_texts, _ in tqdm(dataloader, desc="Extracting embeddings"):
            # Tokenize texts
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding='max_length', 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            # Get token mappings
            mappings = token_mapper(tokenizer, inputs['input_ids'])
            
            # Get model outputs
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
            # Extract word embeddings using token mappings
            batch_word_embeddings = []
            
            for i, mapping in enumerate(mappings):
                sentence_word_embeddings = []
                
                for word_idx in mapping:
                    token_indices = mapping[word_idx]
                    word_token_embeddings = [last_hidden_states[i, token_idx, :] for token_idx in token_indices]
                    
                    # Average the embeddings if a word is split into multiple tokens
                    if len(word_token_embeddings) > 1:
                        word_embedding = torch.mean(torch.stack(word_token_embeddings), dim=0)
                    else:
                        word_embedding = word_token_embeddings[0]
                    
                    sentence_word_embeddings.append(word_embedding.cpu())
                
                batch_word_embeddings.append(sentence_word_embeddings)
            
            # Get sentence embeddings (CLS token)
            sentence_embeddings = last_hidden_states[:, 0, :].cpu()
            
            all_word_embeddings.extend(batch_word_embeddings)
            all_sentence_embeddings.extend(sentence_embeddings)
    
    return all_word_embeddings, all_sentence_embeddings

def generate_special_embeddings(model, tokenizer, device='cuda'):
    """Generate special embeddings for constituency tokens
    
    This function generates embeddings for special constituency tokens by extracting
    the CLS token representation for each token.
    """
    logger.info("Generating special embeddings for constituency tokens")
    
    # Define constituency dictionary for special tokens
    constituency_dict = {
        '«SENTENCE»': 'Sentence',
        '«REDUCED RELATIVE CLAUSE»': 'Reduced relative clause',
        '«NOUN PHRASE»': 'Noun phrase',
        '«VERB PHRASE»': 'Verb phrase',
        '«PREPOSITIONAL PHRASE»': 'Prepositional phrase',
        '«ADVERB PHRASE»': 'Adverb phrase',
        '«UNLIKE COORDINATED PHRASE»': 'Unlike coordinated phrase',
        '«ADJECTIVE PHRASE»': 'Adjective phrase',
        '«SUBORDINATE CLAUSE»': 'Subordinate clause',
        '«WH-ADVERB PHRASE»': 'Wh-adverb phrase',
        '«WHADVP»': 'Wh-adverb phrase',
        '«NX»': 'Nominal phrase',
        '«QUANTIFIER PHRASE»': 'Quantifier phrase',
        '«NOUN PHRASE (NO HEAD)»': 'Noun phrase (no head)',
        '«PARTICLE»': 'Particle',
        '«PARENTETICAL»': 'Parentetical',
        '«WH-NOUN PHRASE»': 'Wh-noun phrase',
        '«WH-ADJECTIVE PHRASE»': 'Wh-adjective phrase',
        '«NOT A CONSTITUENT»': 'Not a constituent',
        '«FRAGMENT»': 'Fragment',
        '«INVERTED SENTENCE»': 'Inverted sentence',
        '«INTERJECTION»': 'Interjection',
        '«WH-PREPOSITIONAL PHRASE»': 'Wh-prepositional phrase',
        '«QUESTION»': 'Question',
        '«SUBORDINATE CLAUSE QUESTION»': 'Subordinate clause question',
        '«CONJUCTION PHRASE»': 'Conjunction phrase',
        '«UNKNOWN»': 'Unknown',
        '«LIST MARKER»': 'List marker',
        'constituency relation': 'Constituency relation'
    }
    
    special_embeddings = {}
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for token, description in tqdm(constituency_dict.items(), desc="Generating special embeddings"):
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
            cls_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu()
            
            # Store the embedding
            special_embeddings[token] = cls_embedding
    
    logger.info(f"Generated special embeddings for {len(special_embeddings)} constituency tokens")
    return special_embeddings

def generate_embeddings(config):
    """Main function to generate embeddings"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    texts, labels = load_dataset(config['dataset_name'])
    
    # Create output directory
    output_dir = os.path.join(config['output_dir'], config['dataset_name'].replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Check if the model path exists
    if os.path.exists(config['model_path']):
        # Load the fine-tuned model
        model = AutoModel.from_pretrained(config['model_name'])
        model_state_dict = torch.load(config['model_path'])
        
        # Remove classifier weights if present (we only need the encoder)
        keys_to_remove = [k for k in model_state_dict.keys() if k.startswith('classifier')]
        for key in keys_to_remove:
            del model_state_dict[key]
            
        # Load the state dict
        model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Loaded fine-tuned model from {config['model_path']}")
    else:
        # Load the pre-trained model if fine-tuned model not found
        model = AutoModel.from_pretrained(config['model_name'])
        logger.warning(f"Fine-tuned model not found at {config['model_path']}, using pre-trained model")
    
    # Generate special embeddings for constituency tokens
    logger.info("Generating special embeddings for constituency tokens")
    special_embeddings = generate_special_embeddings(model, tokenizer, device=device)
    logger.info(f"Generated {len(special_embeddings)} special embeddings")
    
    # Process each split
    for split, split_texts in texts.items():
        logger.info(f"Processing {split} split")
        split_labels = labels[split]
        
        # Extract embeddings
        word_embeddings, sentence_embeddings = extract_embeddings(
            split_texts, 
            model, 
            tokenizer, 
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            device=device
        )
        
        # Save embeddings
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save embeddings to single files
        save_embeddings(word_embeddings, sentence_embeddings, special_embeddings, split_texts, split_labels, split_dir)
        
        logger.info(f"Saved embeddings to {split_dir}")
    
    logger.info(f"Embedding generation completed for {config['dataset_name']}")
    return output_dir

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate embeddings for text datasets using fine-tuned models')
    
    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., stanfordnlp/sst2, setfit/ag_news)')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name (e.g., google-bert/bert-base-uncased)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--chunk_size', type=int, default=50, help='Size of chunks when saving embeddings to disk')
    parser.add_argument('--output_dir', type=str, default='/app/src/Clean_Code/output/gnn_embeddings', help='Output directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Create config from arguments
    config = {
        'dataset_name': args.dataset_name,
        'model_name': args.model_name,
        'model_path': args.model_path,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'chunk_size': args.chunk_size,
        'output_dir': args.output_dir,
        'cuda': not args.no_cuda and torch.cuda.is_available()
    }
    
    # Generate embeddings
    generate_embeddings(config)

if __name__ == "__main__":
    main()
