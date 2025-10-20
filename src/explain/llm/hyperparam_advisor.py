from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from .config import TOKEN_SHAP_DEFAULTS

INT_PARAM_KEYS = {"num_samples_override", "top_k_nodes"}
FLOAT_PARAM_KEYS = {"sampling_ratio"}
BOOL_PARAM_KEYS = set()


@dataclass(frozen=True)
class ModelSpec:
    """Minimal description of the transformer model relevant for TokenSHAP tuning."""

    base_model_name: str
    num_labels: int
    max_length: int


@dataclass(frozen=True)
class DatasetContext:
    """Dataset-level metadata that influences TokenSHAP defaults."""

    dataset: str
    task_type: str
    backbone: str


@dataclass(frozen=True)
class SentenceStats:
    """Lightweight summary of a sentence needed for heuristic hyperparameter tuning."""

    num_tokens: int
    num_chars: int
    avg_token_length: float
    max_token_length: int
    
    @classmethod
    def from_tokens(cls, tokens: List[str]) -> "SentenceStats":
        """Compute sentence statistics from tokenized text."""
        num_tokens = len(tokens)
        if num_tokens == 0:
            return cls(
                num_tokens=0,
                num_chars=0,
                avg_token_length=0.0,
                max_token_length=0,
            )
        
        token_lengths = [len(token) for token in tokens]
        num_chars = sum(token_lengths)
        avg_token_length = num_chars / num_tokens
        max_token_length = max(token_lengths)
        
        return cls(
            num_tokens=num_tokens,
            num_chars=num_chars,
            avg_token_length=avg_token_length,
            max_token_length=max_token_length,
        )
    
    @property
    def is_very_short(self) -> bool:
        """Sentences with very few tokens (<=4)."""
        return self.num_tokens <= 4
    
    @property
    def is_short(self) -> bool:
        """Short sentences (5-8 tokens)."""
        return 5 <= self.num_tokens <= 8
    
    @property
    def is_medium(self) -> bool:
        """Medium-length sentences (9-16 tokens)."""
        return 9 <= self.num_tokens <= 16
    
    @property
    def is_long(self) -> bool:
        """Long sentences (>16 tokens)."""
        return self.num_tokens > 16
    
    @property
    def has_subword_tokens(self) -> bool:
        """Heuristic to detect if many subword tokens are present (##, Ġ prefixes)."""
        return self.avg_token_length < 3.5


class TokenSHAPHyperparameterAdvisor:
    """Suggests TokenSHAP hyperparameters conditioned on sentence/model characteristics."""

    def __init__(
        self,
        model_spec: ModelSpec,
        context: DatasetContext,
        *,
        locked_params: Optional[Dict[str, float]] = None,
        base_defaults: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model_spec = model_spec
        self.context = context
        self.base_defaults = self._cast_params(base_defaults or TOKEN_SHAP_DEFAULTS)
        self.locked_params = self._cast_params(locked_params or {})
        # Ensure locked values override defaults immediately
        for key, value in self.locked_params.items():
            self.base_defaults[key] = value

    def suggest(self, tokens: List[str]) -> Dict[str, float]:
        """Suggest hyperparameters for a specific sentence based on its tokens."""
        stats = SentenceStats.from_tokens(tokens)
        params = dict(self.base_defaults)

        if "sampling_ratio" not in self.locked_params:
            params["sampling_ratio"] = self._suggest_sampling_ratio(stats)

        if "num_samples_override" not in self.locked_params:
            params["num_samples_override"] = self._suggest_num_samples(stats)

        if "top_k_nodes" not in self.locked_params:
            params["top_k_nodes"] = self._suggest_top_k(stats)

        return self._sanitise(params, stats)

    def sanitise_for_sentence(
        self, params: Dict[str, float], tokens: List[str]
    ) -> Dict[str, float]:
        """Project arbitrary hyperparameters onto valid ranges for the supplied sentence."""
        stats = SentenceStats.from_tokens(tokens)
        return self._sanitise(dict(params), stats)

    def _cast_params(self, params: Dict[str, float]) -> Dict[str, float]:
        casted: Dict[str, float] = {}
        for key, value in params.items():
            if value is None:
                continue
            if key in INT_PARAM_KEYS:
                casted[key] = int(value)
            elif key in FLOAT_PARAM_KEYS:
                casted[key] = float(value)
            elif key in BOOL_PARAM_KEYS:
                casted[key] = (
                    bool(int(value))
                    if isinstance(value, (int, float))
                    else bool(value)
                )
            else:
                casted[key] = value
        return casted

    def _suggest_sampling_ratio(self, stats: SentenceStats) -> float:
        """
        Calculate sampling ratio to achieve a fixed target number of combinations.
        
        The ratio adapts to sequence length: ratio = target_samples / (2^num_tokens)
        This ensures all sequences generate the same number of combinations,
        regardless of length. For long sequences, the ratio becomes very small.
        
        Example:
        - 10 tokens, target=512: ratio = 512/1024 = 0.5
        - 30 tokens, target=1536: ratio = 1536/(2^30) = 0.0000014
        - 100 tokens, target=1536: ratio = 1536/(2^100) ≈ 10^-27 (extremely small!)
        """
        num_tokens = stats.num_tokens
        
        # Get target number of samples (128-2048, aligned with GraphSVX)
        target_samples = self._suggest_num_samples(stats)
        
        if target_samples is None:
            # Fallback for safety
            target_samples = 512
        
        # Calculate total possible combinations
        total_possible = 2 ** num_tokens
        
        # Calculate ratio to get exactly target_samples total combinations
        # Note: TokenSHAP samples from all 2^N combinations
        ratio = target_samples / total_possible
        
        # NO adjustments for subword/multiclass - we want exact target
        # NO minimum floor - ratio can be arbitrarily small for long sequences
        # Cap maximum at 0.95 for very short sequences
        return float(min(ratio, 0.95))

    def _suggest_num_samples(self, stats: SentenceStats) -> Optional[int]:
        """
        Suggest the number of samples to use in TokenSHAP.
        
        This provides an override to the sampling ratio when we want
        precise control over computation cost. Aligned with GraphSVX (128-2048 range).
        """
        num_tokens = stats.num_tokens
        
        # Base number of samples grows with sentence length
        # but caps at a reasonable limit (matching GraphSVX range: 128-2048)
        if num_tokens <= 6:
            baseline = 128
        elif num_tokens <= 8:
            baseline = 256
        elif num_tokens <= 10:
            baseline = 512
        elif num_tokens <= 12:
            baseline = 768
        elif num_tokens <= 14:
            baseline = 1024
        elif num_tokens <= 16:
            baseline = 1280
        else:
            baseline = 1536
        
        # Adjust for multi-class problems
        if self.model_spec.num_labels > 2:
            # More classes need more samples to capture variation
            class_factor = 1.0 + (self.model_spec.num_labels - 2) * 0.1
            baseline = int(baseline * class_factor)
        
        # Adjust for subword tokenization
        if stats.has_subword_tokens:
            # Subword tokens may have less semantic variation
            baseline = int(baseline * 0.9)
        
        # Cap at reasonable upper bound (no theoretical_max check - that causes OOM)
        estimate = min(baseline, 2048)
        
        # Ensure minimum samples for statistical validity
        estimate = max(estimate, 32)
        
        return int(estimate)

    def _suggest_top_k(self, stats: SentenceStats) -> int:
        """
        Suggest the number of top tokens to highlight.
        
        This should be proportional to sentence length but not too large.
        """
        num_tokens = stats.num_tokens
        
        if num_tokens <= 4:
            # Very short: highlight most tokens
            return max(2, num_tokens - 1)
        elif num_tokens <= 8:
            # Short: highlight ~60%
            return max(3, int(math.ceil(num_tokens * 0.6)))
        elif num_tokens <= 12:
            # Medium: highlight ~50%
            return max(4, int(math.ceil(num_tokens * 0.5)))
        elif num_tokens <= 16:
            # Long: highlight ~40%
            return min(8, max(5, int(math.ceil(num_tokens * 0.4))))
        elif num_tokens <= 20:
            # Very long: highlight ~30%
            return min(10, max(6, int(math.ceil(num_tokens * 0.3))))
        else:
            # Extremely long: cap at reasonable maximum
            return min(12, max(8, int(math.ceil(num_tokens * 0.25))))

    def _sanitise(
        self, params: Dict[str, float], stats: SentenceStats
    ) -> Dict[str, float]:
        """Ensure all parameters are within valid ranges."""
        cleaned: Dict[str, float] = {}
        
        for key in TOKEN_SHAP_DEFAULTS.keys():
            value = params.get(key, TOKEN_SHAP_DEFAULTS[key])
            if key in INT_PARAM_KEYS:
                cleaned[key] = int(value) if value is not None else None
            elif key in FLOAT_PARAM_KEYS:
                cleaned[key] = float(value)
            elif key in BOOL_PARAM_KEYS:
                cleaned[key] = bool(value)
            else:
                cleaned[key] = value
        
        # Sanitise sampling_ratio to be in (0, 1)
        # NO minimum - allow arbitrarily small ratios for long sequences
        cleaned["sampling_ratio"] = float(
            max(min(cleaned["sampling_ratio"], 0.99), 1e-100)
        )
        
        # Sanitise num_samples_override
        override = cleaned.get("num_samples_override")
        if override is not None:
            cleaned["num_samples_override"] = max(8, min(int(override), 4096))
        
        # Sanitise top_k_nodes to be within sentence length
        cleaned["top_k_nodes"] = max(
            1,
            min(
                int(cleaned["top_k_nodes"]),
                stats.num_tokens or int(cleaned["top_k_nodes"]),
            ),
        )
        
        return cleaned

