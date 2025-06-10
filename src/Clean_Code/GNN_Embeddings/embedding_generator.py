#!/usr/bin/env python3
"""
GNN Embeddings Generator

This module generates embeddings for words in text datasets using fine-tuned language models.
It extracts embeddings from the fine-tuned models and prepares them for use in GNN models.
"""

import os
import sys
import json
import torch
import pickle as pkl
import logging
import argparse
import traceback
import numpy as np
import gc
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

def extract_embeddings(texts, model, tokenizer, batch_size=1, chunk_size=1000, device='cuda', output_dir=None):
    """Extract embeddings from the model for each word in the texts with memory optimizations.
    Processes full sequence length without truncation and saves to disk in chunks.
    
    Args:
        texts: List or dictionary of texts
        model: Pre-trained model
        tokenizer: Tokenizer
        batch_size: Number of samples to process at once (default: 1)
        chunk_size: Number of samples to process before saving to disk
        device: Device to use for inference
        output_dir: Directory to save embedding chunks (default: None, will create 'embedding_chunks' in current directory)
        
    Returns:
        output_dir: Directory containing the saved embedding chunks
    """
    # Create output directory for chunks if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'embedding_chunks')
    
    # Create a specific directory for chunks within the output directory
    chunks_dir = os.path.join(output_dir, 'embedding_chunks')
    os.makedirs(chunks_dir, exist_ok=True)
    
    logger.info(f"Saving embedding chunks to: {chunks_dir}")
    
    chunk_num = 0
    current_chunk_word = []
    current_chunk_sent = []
    
    # Convert texts to a list if it's a dictionary
    if isinstance(texts, dict):
        logger.info("Converting dictionary of texts to list")
        text_list = list(texts)
    else:
        text_list = texts
    
    # Process texts in batches
    num_texts = len(text_list)
    logger.info(f"Processing {num_texts} texts in batches of {batch_size}")
    
    for i in tqdm(range(0, num_texts, batch_size), desc="Processing batches"):
        try:
            # Get batch of texts
            batch_end = min(i + batch_size, num_texts)
            
            # Extract batch texts
            batch_texts = text_list[i:batch_end]
            
            # Clear CUDA cache before processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True  # Will use model's max length if needed
            ).to(device)
            
            # Get token mappings for the batch
            mappings = token_mapper(tokenizer, inputs['input_ids'])
            
            # Get model outputs with autocast for mixed precision
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                
                # Get the last hidden state
                last_hidden_state = outputs.last_hidden_state
                
                # Process each sample in the batch
                for j in range(len(batch_texts)):

                    input_ids = inputs['input_ids'][j].tolist()
                    # Get the sentence embedding (CLS token)
                    
                    sentence_embedding = last_hidden_state[j, 0, :].cpu().numpy()
                    
                    # Get word embeddings by averaging token embeddings
                    word_embeddings = []
                    mapping = mappings[j]
                    for word, token_indices in mapping.items():
                        if token_indices:
                            token_embeddings = [last_hidden_state[j, idx, :].cpu().numpy() for idx in token_indices]
                            word_embedding = np.mean(token_embeddings, axis=0)
                            word_tokens = [input_ids[idx] for idx in token_indices]
                            word_str = tokenizer.decode(word_tokens, skip_special_tokens=True)
                            word_embeddings.append((word_str, word_embedding))
                    # Now word_embeddings is a list of (word, embedding) tuples in order
                    current_chunk_word.append(word_embeddings)
                    current_chunk_sent.append(sentence_embedding)
                
                # Save chunk if it reaches the chunk size
                if len(current_chunk_word) >= chunk_size:
                    save_chunk(current_chunk_word, current_chunk_sent, chunks_dir, chunk_num)
                    chunk_num += 1
                    current_chunk_word = []
                    current_chunk_sent = []
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {str(e)}")
            if 'batch_texts' in locals() and batch_texts:
                logger.error(f"Batch size: {len(batch_texts)}")
                logger.error(f"First text in batch: {batch_texts[0][:100] if batch_texts[0] else 'Empty'}...")
            logger.error(traceback.format_exc())
            raise
    
    # Save any remaining samples
    if current_chunk_word:
        save_chunk(current_chunk_word, current_chunk_sent, chunks_dir, chunk_num)
    
    # Return paths to the saved chunks
    return chunks_dir

def save_chunk(word_embeddings, sent_embeddings, output_dir, chunk_num):
    """Save a chunk of embeddings to disk.
    
    Args:
        word_embeddings: List of word embeddings for each text
        sent_embeddings: List of sentence embeddings for each text
        output_dir: Output directory
        chunk_num: Chunk number
    """
    chunk_path = os.path.join(output_dir, f'embeddings_chunk_{chunk_num}.npz')
    
    # Convert to numpy arrays
    word_embeddings_np = np.array(word_embeddings, dtype=object)
    sent_embeddings_np = np.array(sent_embeddings)
    
    # Save compressed
    np.savez_compressed(
        chunk_path, 
        word_embeddings=word_embeddings_np, 
        sent_embeddings=sent_embeddings_np
    )
    
    logger.info(f"Saved chunk {chunk_num} with {len(word_embeddings)} samples to {chunk_path}")
    
    # Log memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def generate_word_embeddings(config):
     # Set device
    if config.get('cuda', False) and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if config.get('cuda', False) and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU instead")
        else:
            logger.info("Using CPU device")


    # Load dataset
    texts, labels = load_dataset(config['dataset_name'])
    
    # Create output directory
    output_dir = os.path.join(config['output_dir'], config['dataset_name'].replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Check if the model path exists
    if os.path.exists(config['model_path']):
        # Load the fine-tuned model
        model = AutoModel.from_pretrained(config['model_name'])
        model_state_dict = torch.load(config['model_path'], map_location=device)
        
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
    
    # Move model to device
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    # Enable evaluation mode
    model.eval()
    
    # Log memory usage if using CUDA
    if device.type == 'cuda':
        logger.info(f"CUDA memory after model loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Generate word embeddings for the dataset and save in chunks
    split = config['split']
    logger.info(f"Processing {split} split")
    
    # Get texts for the current split
    if isinstance(texts, dict) and split in texts:
        split_texts = texts[split]
        logger.info(f"Found {len(split_texts)} texts for split {split}")
    else:
        logger.info(f"Using all texts as split {split} was not found in the dictionary")
        split_texts = texts
    
    chunk_dir = extract_embeddings(
        split_texts,
        model,
        tokenizer,
        batch_size=config.get('batch_size', 1),
        chunk_size=config.get('chunk_size', 1000),
        device=device,
        output_dir=output_dir
    )
    
    logger.info(f"Embeddings saved in chunks to {chunk_dir}")
    
    return output_dir



def generate_special_embeddings(config):
    """Generate special embeddings for constituency tokens
    
    This function generates embeddings for special constituency tokens by extracting
    the CLS token representation for each token.
    """

    # Set device
    if config.get('cuda', False) and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if config.get('cuda', False) and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU instead")
        else:
            logger.info("Using CPU device")


    logger.info("Generating special embeddings for constituency tokens")
    
    # Define constituency dictionary for special tokens
    constituency_dict = {
    # POS TAGS
    'CC': '«COORDINATING CONJUNCTION»',
    'CD': '«CARDINAL NUMBER»',
    'DT': '«DETERMINER»',
    'EX': '«EXISTENTIAL THERE»',
    'FW': '«FOREIGN WORD»',
    'IN': '«PREPOSITION OR SUBORDINATING CONJUNCTION»',
    'JJ': '«ADJECTIVE»',
    'JJR': '«ADJECTIVE, COMPARATIVE»',
    'JJS': '«ADJECTIVE, SUPERLATIVE»',
    'LS': '«LIST MARKER»',
    'MD': '«MODAL VERB»',
    'NN': '«NOUN, SINGULAR OR MASS»',
    'NNS': '«NOUN, PLURAL»',
    'NNP': '«PROPER NOUN, SINGULAR»',
    'NNPS': '«PROPER NOUN, PLURAL»',
    'PDT': '«PREDETERMINER»',
    'POS': '«POSSESSIVE ENDING»',
    'PRP': '«PERSONAL PRONOUN»',
    'PRP$': '«POSSESSIVE PRONOUN»',
    'RB': '«ADVERB»',
    'RBR': '«ADVERB, COMPARATIVE»',
    'RBS': '«ADVERB, SUPERLATIVE»',
    'RP': '«PARTICLE»',
    'SYM': '«SYMBOL»',
    'TO': '«TO»',
    'UH': '«INTERJECTION»',
    'VB': '«VERB, BASE FORM»',
    'VBD': '«VERB, PAST TENSE»',
    'VBG': '«VERB, GERUND OR present participle»',
    'VBN': '«VERB, past participle»',
    'VBP': '«VERB, non-3rd person singular present»',
    'VBZ': '«VERB, 3rd person singular present»',
    'WDT': '«WH-DETERMINER»',
    'WP': '«WH-PRONOUN»',
    'WP$': '«WH-POSSESSIVE PRONOUN»',
    'WRB': '«WH-ADVERB»',
    # CONSTITUENCY TAGS
    'S': '«SENTENCE»',
    'NP': '«NOUN PHRASE»',
    'VP': '«VERB PHRASE»',
    'PP': '«PREPOSITIONAL PHRASE»',
    'ADJP': '«ADJECTIVE PHRASE»',
    'ADVP': '«ADVERB PHRASE»',
    'SBAR': '«SUBORDINATE CLAUSE»',
    'PRT': '«PARTICLE»',
    'INTJ': '«INTERJECTION»',
    'CONJP': '«CONJUCTION PHRASE»',
    'LST': '«LIST MARKER»',
    'UCP': '«UNLIKE COORDINATED PHRASE»',
    'PRN': '«PARENTETICAL»',
    'FRAG': '«FRAGMENT»',
    'SINV': '«INVERTED SENTENCE»',
    'SBARQ': '«SUBORDINATE CLAUSE QUESTION»',
    'SQ': '«QUESTION»',
    'WHADJP': '«WH-ADJECTIVE PHRASE»',
    'WHAVP': '«WH-ADVERB PHRASE»',
    'WHNP': '«WH-NOUN PHRASE»',
    'WHPP': '«WH-PREPOSITIONAL PHRASE»',
    'RRC': '«REDUCED RELATIVE CLAUSE»',
    'NX': '«NOUN PHRASE (NO HEAD)»',
    'WHADVP': '«WH-ADVERB PHRASE»',
    'QP': '«QUANTIFIER PHRASE»',
    'NAC': '«NOT A CONSTITUENT»',
    'X': '«UNKNOWN»',
    'HYPH': '«HYPHEN»',
    'HVS': '«HYPHENATED VERB SUBSTITUTION»',
    'NML': '«NOMINALIZATION»',
    'LRB': '«LEFT PARENTHESIS»',
    'RRB': '«RIGHT PARENTHESIS»',
    }

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Check if the model path exists
    if os.path.exists(config['model_path']):
        # Load the fine-tuned model
        model = AutoModel.from_pretrained(config['model_name'])
        model_state_dict = torch.load(config['model_path'], map_location=device)
        
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
    
    special_embeddings = {}
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for token, description in tqdm(constituency_dict.items(), desc="Generating special embeddings"):
            # IMPORTANT: We generate the embedding for the VALUE (description), NOT the key (token).
            # The mapping is: key (token) -> embedding of value (description)
            # This ensures the embedding represents the semantic meaning of the description.
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
            
            # Store the embedding with the DESCRIPTION (value) as the key
            special_embeddings[description] = cls_embedding
    
    logger.info(f"Generated special embeddings for {len(special_embeddings)} constituency descriptions (keys are now descriptions, not tags)")
    
    # Save special embeddings to a global directory (not per split)
    output_dir = os.path.join(config['output_dir'], 'special_embeddings', config['model_name'].replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)
    special_embeddings_path = os.path.join(output_dir, 'special_embeddings.pkl')

    # If file already exists, load and return it
    if os.path.exists(special_embeddings_path):
        logger.info(f"Special embeddings already exist at {special_embeddings_path}, loading instead of regenerating.")
        with open(special_embeddings_path, 'rb') as f:
            special_embeddings = pkl.load(f)
        return special_embeddings

    # Save special embeddings
    with open(special_embeddings_path, 'wb') as f:
        pkl.dump(special_embeddings, f)
    logger.info(f"Saved special embeddings to {special_embeddings_path}")
    return special_embeddings

def generate_embeddings(config):
    """
    Generate embeddings based on config flags.
    If both special_embeddings and word_embeddings are set, generate both.
    If neither is set, default to word embeddings for backward compatibility.
    """
    output_dir = os.path.join(config['output_dir'], config['dataset_name'].replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)

    special_flag = config.get('special_embeddings', False)
    word_flag = config.get('word_embeddings', False)

    # If neither flag is set, raise error
    if not special_flag and not word_flag:
        raise ValueError("At least one of --special_embeddings or --word_embeddings must be set")

    if special_flag:
        generate_special_embeddings(config)
    if word_flag:
        generate_word_embeddings(config)
    return output_dir

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate GNN embeddings from fine-tuned models')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_name', type=str, required=False, help='Name of the pre-trained model')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing (default: 1)')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Number of samples per chunk file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process')
    parser.add_argument('--special_embeddings', type=str, default='false', help='Generate special embeddings (true/false)')
    parser.add_argument('--word_embeddings', type=str, default='false', help='Generate word embeddings (true/false)')
    return vars(parser.parse_args())

def main():
    """Main entry point"""
    import json
    
    # Parse command line arguments
    config = parse_args()
    # Convert string parameters to booleans
    config['special_embeddings'] = str(config.get('special_embeddings', 'false')).lower() == 'true'
    config['word_embeddings'] = str(config.get('word_embeddings', 'false')).lower() == 'true'
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config['output_dir'], 'embedding_generation.log'))
        ]
    )
    
    # Log configuration
    logger.info("Starting embedding generation with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Generate embeddings based on flags
        output_dir = generate_embeddings(config)
        logger.info(f"Successfully completed embedding generation. Results saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
