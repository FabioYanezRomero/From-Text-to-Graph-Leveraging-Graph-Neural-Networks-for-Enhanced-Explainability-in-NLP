from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RelatedPrediction:
    """Stores faithfulness-related prediction scores."""

    origin: Optional[float] = None
    masked: Optional[float] = None
    maskout: Optional[float] = None
    sparsity: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "RelatedPrediction":
        if not data:
            return cls()
        return cls(
            origin=data.get("origin"),
            masked=data.get("masked"),
            maskout=data.get("maskout"),
            sparsity=data.get("sparsity"),
        )


@dataclass(frozen=True)
class Coalition:
    """Represents a single coalition (subset of nodes) produced by an explainer."""

    nodes: Tuple[int, ...]
    confidence: float
    size: int
    combination_id: Optional[int] = None
    binary_mask: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_iterable(
        cls,
        nodes: Iterable[int],
        confidence: float,
        combination_id: Optional[int] = None,
        binary_mask: Optional[Sequence[int]] = None,
        size: Optional[int] = None,
        **metadata: Any,
    ) -> "Coalition":
        node_tuple = tuple(int(n) for n in nodes)
        coalition_size = size if size is not None else len(node_tuple)
        return cls(
            nodes=node_tuple,
            confidence=float(confidence),
            size=coalition_size,
            combination_id=combination_id,
            binary_mask=tuple(int(v) for v in binary_mask) if binary_mask is not None else None,
            metadata=metadata or {},
        )


@dataclass
class ExplanationRecord:
    """
    Canonical container for a single explanation instance.

    Attributes:
        dataset: Name of the dataset (e.g., 'ag-news').
        graph_type: Graph construction variant (e.g., 'skipgrams', 'window').
        method: Explainer name (e.g., 'graphsvx').
        run_id: Identifier for the experiment run that produced the explanation.
        graph_index: Index of the original graph/sample in the dataset.
        label: Ground-truth label (if available).
        prediction_class: Predicted label.
        prediction_confidence: Confidence assigned to the predicted label.
        num_nodes / num_edges: Graph sizes captured by the explainer artefact.
        node_importance: Continuous importance scores per node (GraphSVX-style).
        top_nodes: Ranked list of high-importance nodes.
        related_prediction: Faithfulness metrics (origin/masked/maskout/sparsity).
        hyperparams: Hyper-parameters used by the explainer.
        coalitions: List of coalitions evaluated by the explainer.
        extras: Additional raw fields not explicitly modelled above.
    """

    dataset: str
    graph_type: Optional[str]
    method: str
    run_id: Optional[str]
    graph_index: int
    label: Optional[int]
    prediction_class: Optional[int]
    prediction_confidence: Optional[float]
    num_nodes: Optional[int]
    num_edges: Optional[int]
    node_importance: Optional[Sequence[float]] = None
    top_nodes: Tuple[int, ...] = field(default_factory=tuple)
    related_prediction: RelatedPrediction = field(default_factory=RelatedPrediction)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    coalitions: List[Coalition] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def minimal_coalition(
        self,
        threshold: float,
        *,
        origin_confidence: Optional[float] = None,
    ) -> Optional[Coalition]:
        """
        Returns the smallest coalition whose confidence reaches the specified
        fraction of the origin confidence. If origin confidence is not provided,
        the value stored in related_prediction.origin is used.
        """
        if not self.coalitions:
            return None
        baseline = (
            origin_confidence
            if origin_confidence is not None
            else self.related_prediction.origin
        )
        if baseline is None:
            return None

        required = baseline * threshold
        eligible = [c for c in self.coalitions if c.confidence >= required]
        if not eligible:
            return None

        eligible.sort(key=lambda c: (c.size, -c.confidence))
        return eligible[0]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record into a JSON-friendly dictionary."""
        return {
            "dataset": self.dataset,
            "graph_type": self.graph_type,
            "method": self.method,
            "run_id": self.run_id,
            "graph_index": self.graph_index,
            "label": self.label,
            "prediction_class": self.prediction_class,
            "prediction_confidence": self.prediction_confidence,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_importance": list(self.node_importance) if self.node_importance is not None else None,
            "top_nodes": list(self.top_nodes),
            "related_prediction": {
                "origin": self.related_prediction.origin,
                "masked": self.related_prediction.masked,
                "maskout": self.related_prediction.maskout,
                "sparsity": self.related_prediction.sparsity,
            },
            "hyperparams": self.hyperparams,
            "coalitions": [
                {
                    "combination_id": c.combination_id,
                    "nodes": list(c.nodes),
                    "confidence": c.confidence,
                    "size": c.size,
                    "binary_mask": list(c.binary_mask) if c.binary_mask is not None else None,
                    "metadata": c.metadata,
                }
                for c in self.coalitions
            ],
            "extras": self.extras,
        }
