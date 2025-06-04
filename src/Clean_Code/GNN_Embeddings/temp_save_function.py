def save_embeddings(word_embeddings, sentence_embeddings, texts, labels, output_dir, chunk_size=50):
    """Save embeddings and metadata to disk in smaller chunks
    
    Args:
        word_embeddings: List of word embeddings for each text
        sentence_embeddings: List of sentence embeddings for each text
        texts: List of texts
        labels: List of labels
        output_dir: Output directory
        chunk_size: Size of chunks to save
        
    Returns:
        chunks_dir: Directory containing the chunks
    """
    import os
    import pickle as pkl
    import logging
    
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of samples
    total_samples = len(word_embeddings)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division
    
    logger.info(f"Saving {total_samples} samples in {num_chunks} chunks of size {chunk_size}")
    
    # Save metadata for the full dataset (needed for dataset size information)
    metadata = {
        'texts': texts,
        'labels': labels,
        'num_chunks': num_chunks,
        'chunk_size': chunk_size,
        'total_samples': total_samples
    }
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pkl.dump(metadata, f)
    
    # Save embeddings in chunks
    chunks_dir = os.path.join(output_dir, 'chunks')
    os.makedirs(chunks_dir, exist_ok=True)
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        
        # Extract chunk data
        chunk_word_embeddings = word_embeddings[start_idx:end_idx]
        chunk_sentence_embeddings = sentence_embeddings[start_idx:end_idx]
        chunk_texts = texts[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        # Create chunk directory
        chunk_dir = os.path.join(chunks_dir, f"chunk_{chunk_idx}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save word embeddings for this chunk
        word_embeddings_path = os.path.join(chunk_dir, 'word_embeddings.pkl')
        with open(word_embeddings_path, 'wb') as f:
            pkl.dump(chunk_word_embeddings, f)
        
        # Save sentence embeddings for this chunk
        sentence_embeddings_path = os.path.join(chunk_dir, 'sentence_embeddings.pkl')
        with open(sentence_embeddings_path, 'wb') as f:
            pkl.dump(chunk_sentence_embeddings, f)
        
        # Save metadata for this chunk
        chunk_metadata = {
            'texts': chunk_texts,
            'labels': chunk_labels,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        chunk_metadata_path = os.path.join(chunk_dir, 'metadata.pkl')
        with open(chunk_metadata_path, 'wb') as f:
            pkl.dump(chunk_metadata, f)
        
        logger.info(f"Saved chunk {chunk_idx+1}/{num_chunks} with {len(chunk_texts)} samples")
    
    logger.info(f"Saved embeddings in {num_chunks} chunks to {output_dir}")
    return chunks_dir
