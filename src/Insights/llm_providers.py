from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class TokenInfo:
    """
    Container conveying token-level metadata for LLM explanations.
    
    This is analogous to GraphInfo but for text/tokens instead of graph nodes.
    """

    token_text: Sequence[str]
    prompt: str
    num_tokens: int

    def text_for_indices(self, indices: Iterable[int]) -> List[str]:
        """Return token text for given indices."""
        return [
            self.token_text[i]
            for i in indices
            if 0 <= i < len(self.token_text)
        ]


class LLMExplanationProvider:
    """
    Provider for LLM explanation records that extracts token text from
    explanation records for enriching Insight summaries with human-readable
    token information.
    
    Unlike GraphArtifactProvider which loads external graph files, this
    provider extracts token information already stored in the explanation
    records themselves.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    def __call__(self, record) -> Optional[TokenInfo]:
        """
        Extract token information from an LLM explanation record.
        
        Args:
            record: ExplanationRecord with token_text in extras
        
        Returns:
            TokenInfo if token data is available, None otherwise
        """
        try:
            return self._extract_token_info(record)
        except (KeyError, AttributeError, TypeError) as exc:
            if self.strict:
                raise
            return None

    def _extract_token_info(self, record) -> Optional[TokenInfo]:
        """Extract token information from record.extras."""
        extras = getattr(record, "extras", {}) or {}
        
        # Extract token text
        token_text = extras.get("token_text")
        if not token_text:
            return None
        
        # Extract prompt
        prompt = extras.get("prompt", "")
        
        # Get number of tokens
        num_tokens = record.num_nodes or len(token_text)
        
        return TokenInfo(
            token_text=tuple(token_text),
            prompt=str(prompt),
            num_tokens=int(num_tokens),
        )





