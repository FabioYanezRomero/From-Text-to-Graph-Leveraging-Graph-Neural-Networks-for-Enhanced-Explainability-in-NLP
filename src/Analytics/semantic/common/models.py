from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class TokenAttribution:
    node_index: int
    token: str
    score: float
    position: float
    is_leaf: bool


@dataclass(slots=True)
class GraphSemanticSummary:
    graph_index: int
    label: int | None
    prediction_class: int | None
    prediction_confidence: float | None
    explanation_size: int
    unique_token_count: int
    semantic_density: float
    selected_tokens: List[TokenAttribution]
    graph_metadata: Dict[str, float]
    extras: Dict[str, float] = field(default_factory=dict)
