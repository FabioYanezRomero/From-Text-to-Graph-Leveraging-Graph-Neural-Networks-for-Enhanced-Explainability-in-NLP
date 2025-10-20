"""Word-level aggregation for token-level importance scores.

This module provides utilities to aggregate subword token scores into word-level
scores for more human-readable explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WordSpan:
    """Represents a word composed of one or more tokens."""
    
    word_text: str
    token_indices: Tuple[int, ...]
    start_char: int
    end_char: int


def detect_word_boundaries(
    tokens: Sequence[str],
    original_text: Optional[str] = None,
) -> List[WordSpan]:
    """
    Detect word boundaries from subword tokens.
    
    This handles common tokenization patterns:
    - BERT WordPiece: "##" prefix for continuation
    - GPT/RoBERTa BPE: "Ġ" prefix for word start
    - SentencePiece: "▁" prefix for word start
    
    Args:
        tokens: List of tokens (may include subword markers)
        original_text: Optional original text for character position mapping
    
    Returns:
        List of WordSpan objects representing complete words
    """
    words: List[WordSpan] = []
    current_word_tokens: List[int] = []
    current_word_text_parts: List[str] = []
    
    for idx, token in enumerate(tokens):
        # Check if this is a continuation token (WordPiece style)
        is_continuation = token.startswith("##")
        
        # Clean token text
        clean_token = token
        if token.startswith("##"):
            clean_token = token[2:]  # Remove ##
        elif token.startswith("Ġ"):
            clean_token = token[1:]  # Remove Ġ (GPT-style)
        elif token.startswith("▁"):
            clean_token = token[1:]  # Remove ▁ (SentencePiece)
        
        # If this is a continuation, add to current word
        if is_continuation and current_word_tokens:
            current_word_tokens.append(idx)
            current_word_text_parts.append(clean_token)
        else:
            # Start a new word
            # First, save the previous word if it exists
            if current_word_tokens:
                word_text = "".join(current_word_text_parts)
                words.append(WordSpan(
                    word_text=word_text,
                    token_indices=tuple(current_word_tokens),
                    start_char=-1,  # Character positions not tracked here
                    end_char=-1,
                ))
            
            # Start new word
            current_word_tokens = [idx]
            current_word_text_parts = [clean_token]
    
    # Don't forget the last word
    if current_word_tokens:
        word_text = "".join(current_word_text_parts)
        words.append(WordSpan(
            word_text=word_text,
            token_indices=tuple(current_word_tokens),
            start_char=-1,
            end_char=-1,
        ))
    
    return words


def aggregate_token_scores_to_words(
    token_importance: Sequence[float],
    tokens: Sequence[str],
    aggregation: str = "mean",
) -> Tuple[List[WordSpan], List[float]]:
    """
    Aggregate token-level importance scores to word-level scores.
    
    Args:
        token_importance: Importance score for each token
        tokens: List of tokens
        aggregation: Aggregation method - "mean", "sum", "max", or "first"
    
    Returns:
        Tuple of (word_spans, word_importance_scores)
    """
    word_spans = detect_word_boundaries(tokens)
    word_scores: List[float] = []
    
    for word_span in word_spans:
        # Get scores for all tokens in this word
        token_scores = [
            token_importance[idx]
            for idx in word_span.token_indices
            if idx < len(token_importance)
        ]
        
        if not token_scores:
            word_scores.append(0.0)
            continue
        
        # Aggregate scores
        if aggregation == "mean":
            score = sum(token_scores) / len(token_scores)
        elif aggregation == "sum":
            score = sum(token_scores)
        elif aggregation == "max":
            score = max(token_scores)
        elif aggregation == "first":
            score = token_scores[0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        word_scores.append(score)
    
    return word_spans, word_scores


def get_top_words(
    token_importance: Sequence[float],
    tokens: Sequence[str],
    k: int = 5,
    aggregation: str = "mean",
) -> Tuple[List[str], List[float], List[Tuple[int, ...]]]:
    """
    Get top-k most important words with their scores.
    
    Args:
        token_importance: Importance score for each token
        tokens: List of tokens
        k: Number of top words to return
        aggregation: How to aggregate token scores ("mean", "sum", "max", "first")
    
    Returns:
        Tuple of (top_words, top_scores, top_token_indices)
    """
    word_spans, word_scores = aggregate_token_scores_to_words(
        token_importance, tokens, aggregation
    )
    
    # Sort by importance
    scored_words = list(zip(word_spans, word_scores))
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k
    top_k = scored_words[:k]
    
    top_words = [w.word_text for w, _ in top_k]
    top_scores = [s for _, s in top_k]
    top_token_indices = [w.token_indices for w, _ in top_k]
    
    return top_words, top_scores, top_token_indices


def create_word_level_summary(
    record,
    aggregation: str = "mean",
    top_k: int = 5,
) -> Dict[str, object]:
    """
    Create a word-level summary from a token-level ExplanationRecord.
    
    Args:
        record: ExplanationRecord with token-level data
        aggregation: How to aggregate token scores
        top_k: Number of top words to include
    
    Returns:
        Dictionary with word-level fields
    """
    from .token_shap_runner import _extract_importances
    
    # Get token-level data
    token_text = record.extras.get("token_text", []) if hasattr(record, "extras") else []
    if not token_text:
        return {
            "word_level_available": False,
            "top_words": None,
            "top_word_scores": None,
        }
    
    token_importance = record.node_importance
    if not token_importance:
        return {
            "word_level_available": False,
            "top_words": None,
            "top_word_scores": None,
        }
    
    # Aggregate to word level
    word_spans, word_scores = aggregate_token_scores_to_words(
        token_importance, token_text, aggregation
    )
    
    # Get top words
    top_words, top_scores, top_token_indices = get_top_words(
        token_importance, token_text, k=top_k, aggregation=aggregation
    )
    
    return {
        "word_level_available": True,
        "num_words": len(word_spans),
        "words": [w.word_text for w in word_spans],
        "word_scores": word_scores,
        "top_words": top_words,
        "top_word_scores": top_scores,
        "top_word_token_indices": top_token_indices,
        "aggregation_method": aggregation,
    }





