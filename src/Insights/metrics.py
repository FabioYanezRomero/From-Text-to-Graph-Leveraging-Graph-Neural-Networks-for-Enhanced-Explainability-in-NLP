from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import itertools

import networkx as nx

from .records import Coalition, ExplanationRecord

try:
    from .providers import GraphInfo  # type: ignore
except ImportError:  # pragma: no cover
    GraphInfo = None  # type: ignore

try:
    from .llm_providers import TokenInfo  # type: ignore
except ImportError:  # pragma: no cover
    TokenInfo = None  # type: ignore


def _compute_auc(values: Mapping[int, float], *, max_size: Optional[int]) -> Optional[float]:
    if not values or max_size in (None, 0):
        return None
    sorted_items = sorted(values.items())
    total = 0.0
    last_size = 0
    for size, value in sorted_items:
        width = size - last_size
        total += width * value
        last_size = size
    return total / max_size if max_size else None


@dataclass(frozen=True)
class CurveResult:
    """Stores size-confidence curves (e.g., insertion/deletion)."""

    values: Dict[int, float]
    origin: Optional[float]
    normalized: bool
    auc: Optional[float]

    def as_series(self) -> List[Tuple[int, float]]:
        """Return sorted (size, value) pairs."""
        return sorted(self.values.items(), key=lambda item: item[0])


def minimal_sufficient_size(record: ExplanationRecord, *, threshold: float = 0.9) -> Optional[int]:
    coalition = record.minimal_coalition(threshold)
    return coalition.size if coalition else None


def minimal_sufficient_statistics(
    records: Iterable[ExplanationRecord],
    *,
    threshold: float = 0.9,
) -> List[Tuple[int, int]]:
    stats: Dict[int, int] = {}
    for record in records:
        size = minimal_sufficient_size(record, threshold=threshold)
        if size is None:
            continue
        stats[size] = stats.get(size, 0) + 1
    return sorted(stats.items())


def insertion_curve(record: ExplanationRecord, *, normalize: bool = True) -> CurveResult:
    if not record.coalitions:
        return CurveResult(values={}, origin=None, normalized=normalize, auc=None)

    origin = record.related_prediction.origin
    curve: Dict[int, float] = {}
    for coalition in record.coalitions:
        best = curve.get(coalition.size)
        value = coalition.confidence
        if normalize and origin:
            value = value / origin
        if best is None or value > best:
            curve[coalition.size] = value

    max_size = record.num_nodes or (max(curve) if curve else None)
    auc = _compute_auc(curve, max_size=max_size)
    return CurveResult(values=curve, origin=origin, normalized=normalize, auc=auc)


def deletion_curve(record: ExplanationRecord, *, normalize: bool = True) -> CurveResult:
    """
    Uses mask-out confidence (if available) as a proxy deletion baseline.
    """
    origin = record.related_prediction.origin
    maskout = record.related_prediction.maskout
    if maskout is None:
        return CurveResult(values={}, origin=origin, normalized=normalize, auc=None)

    curve: Dict[int, float] = {}
    if normalize and origin:
        curve[0] = 1.0
        value = maskout / origin if origin else 0.0
    else:
        curve[0] = origin if origin is not None else maskout
        value = maskout

    num_nodes = record.num_nodes
    sparsity = record.related_prediction.sparsity
    removal_size: Optional[int] = None
    if sparsity is not None and num_nodes:
        removal_size = max(0, min(num_nodes, int(round((1.0 - sparsity) * num_nodes))))
    elif num_nodes:
        removal_size = num_nodes

    if removal_size is not None and removal_size > 0:
        curve[removal_size] = value
        max_size = max(removal_size, 1)
    else:
        max_size = num_nodes or 1

    auc = _compute_auc(curve, max_size=max_size)
    return CurveResult(values=curve, origin=origin, normalized=normalize, auc=auc)


def _normalised_drop(baseline: Optional[float], value: Optional[float], *, normalise: bool = True) -> Optional[float]:
    if baseline is None or value is None:
        return None
    drop = baseline - value
    if not normalise or baseline == 0:
        return drop
    return drop / abs(baseline)


def fidelity_plus(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Normalised drop when retaining only the important elements (sufficiency)."""

    return _normalised_drop(record.related_prediction.origin, record.related_prediction.masked, normalise=normalise)


def fidelity_minus(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Normalised drop when masking out the important elements (necessity)."""

    return _normalised_drop(record.related_prediction.origin, record.related_prediction.maskout, normalise=normalise)


def faithfulness(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Contrast between keeping and dropping important elements."""

    masked = record.related_prediction.masked
    maskout = record.related_prediction.maskout
    origin = record.related_prediction.origin
    if masked is None or maskout is None:
        return None
    score = masked - maskout
    if not normalise or origin in (None, 0):
        return score
    return score / abs(origin)


def deletion_auc(record: ExplanationRecord, *, normalise: bool = True) -> Optional[float]:
    """Convenience wrapper that returns the deletion AUC."""

    return deletion_curve(record, normalize=normalise).auc


def insertion_auc(record: ExplanationRecord, *, normalize: bool = True) -> Optional[float]:
    return insertion_curve(record, normalize=normalize).auc


def jaccard_overlap(nodes_a: Sequence[int], nodes_b: Sequence[int]) -> Optional[float]:
    if not nodes_a or not nodes_b:
        return None
    set_a = set(nodes_a)
    set_b = set(nodes_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else None


def top_nodes(record: ExplanationRecord, *, k: int) -> Sequence[int]:
    if record.top_nodes:
        return record.top_nodes[:k]
    if not record.node_importance:
        return ()
    ranked = sorted(
        enumerate(record.node_importance),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(idx for idx, _ in ranked[:k])


def stability_jaccard(
    record_a: ExplanationRecord,
    record_b: ExplanationRecord,
    *,
    k: int = 10,
) -> Optional[float]:
    top_a = top_nodes(record_a, k=k)
    top_b = top_nodes(record_b, k=k)
    return jaccard_overlap(top_a, top_b)


def induced_subgraph_metrics(graph: nx.Graph, nodes: Sequence[int]) -> Dict[str, Optional[float]]:
    if not nodes:
        return {
            "induced_num_nodes": 0,
            "induced_num_edges": 0,
            "components": 0,
            "density": None,
            "boundary_edges": 0,
            "cut_ratio": None,
            "avg_shortest_path": None,
        }

    node_set = set(nodes)
    subgraph = graph.subgraph(node_set).copy()
    undirected = subgraph.to_undirected()
    full_nodes = graph.number_of_nodes()

    boundary_edges = sum(
        1
        for u, v in graph.edges()
        if (u in node_set) ^ (v in node_set)
    )

    denominator = len(node_set) * (full_nodes - len(node_set))

    if subgraph.is_directed():
        component_count = nx.number_weakly_connected_components(subgraph)
    else:
        component_count = nx.number_connected_components(subgraph)

    metrics = {
        "induced_num_nodes": subgraph.number_of_nodes(),
        "induced_num_edges": subgraph.number_of_edges(),
        "components": component_count,
        "density": nx.density(undirected),
        "boundary_edges": boundary_edges,
        "cut_ratio": boundary_edges / denominator if denominator else None,
        "avg_shortest_path": None,
    }

    try:
        if component_count == 1 and undirected.number_of_nodes() > 1:
            metrics["avg_shortest_path"] = nx.average_shortest_path_length(undirected)
    except (nx.NetworkXError, ZeroDivisionError):
        metrics["avg_shortest_path"] = None

    return metrics


def stability_average(records: Sequence[ExplanationRecord], *, k: int = 10) -> Optional[float]:
    """Average pairwise stability (Jaccard of top-k nodes) across records."""

    pairs = list(itertools.combinations(records, 2))
    if not pairs:
        return None
    scores = [stability_jaccard(a, b, k=k) for a, b in pairs]
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def centrality_alignment(
    node_importance: Mapping[int, float],
    centrality_scores: Mapping[int, float],
) -> Optional[float]:
    import math

    common_nodes = set(node_importance) & set(centrality_scores)
    if len(common_nodes) < 2:
        return None

    importance_values = [node_importance[node] for node in common_nodes]
    centrality_values = [centrality_scores[node] for node in common_nodes]

    ranks_importance = _rank_values(importance_values)
    ranks_centrality = _rank_values(centrality_values)

    mean_rank_importance = sum(ranks_importance) / len(ranks_importance)
    mean_rank_centrality = sum(ranks_centrality) / len(ranks_centrality)

    numerator = sum(
        (r_i - mean_rank_importance) * (r_j - mean_rank_centrality)
        for r_i, r_j in zip(ranks_importance, ranks_centrality)
    )
    denominator = math.sqrt(
        sum((r - mean_rank_importance) ** 2 for r in ranks_importance)
        * sum((r - mean_rank_centrality) ** 2 for r in ranks_centrality)
    )
    return numerator / denominator if denominator else None


def _rank_values(values: Sequence[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1])

    ranks = [0.0] * len(values)
    current_rank = 1
    while current_rank <= len(values):
        start = current_rank - 1
        end = start
        while end + 1 < len(values) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        avg_rank = (current_rank + end + 1) / 2
        for idx in range(start, end + 1):
            ranks[indexed[idx][0]] = avg_rank
        current_rank = end + 2
    return ranks


def _default_graph_provider(_: ExplanationRecord) -> Optional[nx.Graph]:
    return None


def summarize_record(
    record: ExplanationRecord,
    *,
    sufficiency_threshold: float = 0.9,
    top_k: int = 10,
    graph: Optional[nx.Graph] = None,
    graph_provider: Callable[[ExplanationRecord], Optional[nx.Graph]] = _default_graph_provider,
    centrality_funcs: Optional[Mapping[str, Callable[[nx.Graph], Mapping[int, float]]]] = None,
) -> Dict[str, Any]:
    """
    Produce a dictionary summarising key metrics for an explanation record.
    """

    node_text: Optional[Sequence[str]] = None
    graph_payload = graph or (graph_provider(record) if graph_provider else None)

    # Handle both GraphInfo (for GNN explanations) and TokenInfo (for LLM explanations)
    if GraphInfo is not None and isinstance(graph_payload, GraphInfo):
        graph_obj = graph_payload.graph
        node_text = tuple(graph_payload.node_text)
    elif TokenInfo is not None and isinstance(graph_payload, TokenInfo):
        # For LLM explanations, we have tokens but no graph structure
        graph_obj = None
        node_text = tuple(graph_payload.token_text)
    else:
        graph_obj = graph_payload  # type: ignore[assignment]
    minimal = record.minimal_coalition(sufficiency_threshold)
    insertion = insertion_curve(record)
    deletion = deletion_curve(record)

    summary: Dict[str, Any] = {
        "dataset": record.dataset,
        "graph_type": record.graph_type,
        "method": record.method,
        "run_id": record.run_id,
        "graph_index": record.graph_index,
        "label": record.label,
        "prediction_class": record.prediction_class,
        "prediction_confidence": record.prediction_confidence,
        "origin_confidence": record.related_prediction.origin,
        "masked_confidence": record.related_prediction.masked,
        "maskout_confidence": record.related_prediction.maskout,
        "origin_distribution": list(record.related_prediction.origin_distribution)
        if record.related_prediction.origin_distribution is not None
        else None,
        "masked_distribution": list(record.related_prediction.masked_distribution)
        if record.related_prediction.masked_distribution is not None
        else None,
        "maskout_distribution": list(record.related_prediction.maskout_distribution)
        if record.related_prediction.maskout_distribution is not None
        else None,
        "sparsity": record.related_prediction.sparsity,
        "minimal_coalition_size": minimal.size if minimal else None,
        "minimal_coalition_confidence": minimal.confidence if minimal else None,
        "insertion_auc": insertion.auc,
        "deletion_auc": deletion.auc,
        "insertion_curve": insertion.as_series(),
        "deletion_curve": deletion.as_series(),
        "top_nodes": list(top_nodes(record, k=top_k)),
        "num_nodes": record.num_nodes,
        "num_edges": record.num_edges,
    }

    if node_text:
        summary["top_tokens"] = [node_text[idx] for idx in summary["top_nodes"] if 0 <= idx < len(node_text)]
    else:
        summary["top_tokens"] = None

    # Handle minimal coalition tokens
    if minimal and node_text:
        summary["minimal_coalition_tokens"] = [node_text[idx] for idx in minimal.nodes if 0 <= idx < len(node_text)]
    else:
        summary["minimal_coalition_tokens"] = None

    # Structural metrics only available for graph-based explanations
    if minimal and graph_obj is not None:
        summary["structural_metrics"] = induced_subgraph_metrics(graph_obj, minimal.nodes)
    else:
        summary["structural_metrics"] = None

    if graph_obj is not None and centrality_funcs and record.node_importance:
        importance_map = {idx: float(score) for idx, score in enumerate(record.node_importance)}
        centrality_results: Dict[str, Optional[float]] = {}
        for name, func in centrality_funcs.items():
            try:
                scores = func(graph_obj)
                centrality_results[name] = centrality_alignment(importance_map, scores)
            except Exception:
                centrality_results[name] = None
        summary["centrality_alignment"] = centrality_results
    else:
        summary["centrality_alignment"] = None

    summary["fidelity_plus"] = fidelity_plus(record)
    summary["fidelity_minus"] = fidelity_minus(record)
    summary["faithfulness"] = faithfulness(record)

    return summary


def summarize_records(
    records: Iterable[ExplanationRecord],
    *,
    sufficiency_threshold: float = 0.9,
    top_k: int = 10,
    graph_provider: Callable[[ExplanationRecord], Optional[nx.Graph]] = _default_graph_provider,
    centrality_funcs: Optional[Mapping[str, Callable[[nx.Graph], Mapping[int, float]]]] = None,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for record in records:
        summary = summarize_record(
            record,
            sufficiency_threshold=sufficiency_threshold,
            top_k=top_k,
            graph_provider=graph_provider,
            centrality_funcs=centrality_funcs,
        )
        summaries.append(summary)
    return summaries
