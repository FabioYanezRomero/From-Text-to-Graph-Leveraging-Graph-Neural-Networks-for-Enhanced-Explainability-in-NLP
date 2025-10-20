"""LLM explainability using TokenSHAP with adaptive hyperparameter tuning."""

from .config import (
    TOKEN_SHAP_DEFAULTS,
    DatasetProfile,
    LLMExplainerRequest,
    SamplingProfile,
    build_default_profiles,
)
from .hyperparam_advisor import (
    DatasetContext,
    ModelSpec,
    SentenceStats,
    TokenSHAPHyperparameterAdvisor,
)
from .model_loader import ModelBundle, load_dataset_split, load_finetuned_model
from .token_shap_runner import collect_token_shap_hyperparams, token_shap_explain
from .word_aggregation import (
    WordSpan,
    aggregate_token_scores_to_words,
    create_word_level_summary,
    detect_word_boundaries,
    get_top_words,
)

__all__ = [
    # Config
    "TOKEN_SHAP_DEFAULTS",
    "DatasetProfile",
    "LLMExplainerRequest",
    "SamplingProfile",
    "build_default_profiles",
    # Hyperparameter Advisor
    "TokenSHAPHyperparameterAdvisor",
    "ModelSpec",
    "DatasetContext",
    "SentenceStats",
    # Model Loading
    "ModelBundle",
    "load_finetuned_model",
    "load_dataset_split",
    # TokenSHAP Runner
    "token_shap_explain",
    "collect_token_shap_hyperparams",
    # Word-Level Aggregation
    "WordSpan",
    "detect_word_boundaries",
    "aggregate_token_scores_to_words",
    "get_top_words",
    "create_word_level_summary",
]

