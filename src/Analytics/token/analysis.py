from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

from ..semantic.common.config import GraphSVXConfig, SubgraphXConfig
from ..semantic.common.data_loader import GraphArtifactLoader, load_json_records, load_prediction_lookup, load_subgraphx_results
from ..semantic.common.models import GraphSemanticSummary, TokenAttribution
from .selection import select_tokens


class AggregatedMetrics:
    def __init__(self) -> None:
        self.token_frequencies: Counter[str] = Counter()
        self.token_score_sums: Counter[str] = Counter()
        self.position_sums: Counter[str] = Counter()
        self.position_square_sums: Counter[str] = Counter()

    def register(self, token: str, score: float, position: float) -> None:
        key = token.lower()
        if not key:
            return
        self.token_frequencies[key] += 1
        self.token_score_sums[key] += score
        self.position_sums[key] += position
        self.position_square_sums[key] += position * position

    def to_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, float | str | int]] = []
        for token, frequency in self.token_frequencies.most_common():
            score_sum = self.token_score_sums[token]
            position_sum = self.position_sums[token]
            position_sq = self.position_square_sums[token]
            mean_score = score_sum / max(frequency, 1)
            mean_pos = position_sum / max(frequency, 1)
            var_pos = max(position_sq / max(frequency, 1) - mean_pos**2, 0.0)
            rows.append(
                {
                    "token": token,
                    "frequency": frequency,
                    "mean_score": mean_score,
                    "mean_position": mean_pos,
                    "position_std": math.sqrt(var_pos),
                }
            )
        return pd.DataFrame(rows)


def _compute_positions(node_order: Sequence[int]) -> Dict[int, float]:
    total = max(len(node_order) - 1, 1)
    return {idx: pos / total for pos, idx in enumerate(node_order)}


def _semantic_density(unique_tokens: int, explanation_size: int) -> float:
    if explanation_size <= 0:
        return 0.0
    return unique_tokens / float(explanation_size)


def _graph_metadata(graph: nx.Graph, selected_nodes: Sequence[int]) -> Dict[str, float]:
    induced_nodes = len(selected_nodes)
    if not induced_nodes:
        return {
            "num_nodes": float(graph.number_of_nodes()),
            "num_edges": float(graph.number_of_edges()),
            "induced_num_nodes": 0.0,
            "induced_num_edges": 0.0,
            "induced_density": 0.0,
            "induced_components": 0.0,
        }
    subgraph = graph.subgraph(selected_nodes)
    return {
        "num_nodes": float(graph.number_of_nodes()),
        "num_edges": float(graph.number_of_edges()),
        "induced_num_nodes": float(subgraph.number_of_nodes()),
        "induced_num_edges": float(subgraph.number_of_edges()),
        "induced_density": float(nx.density(subgraph)),
        "induced_components": float(nx.number_connected_components(subgraph.to_undirected())),
    }


def _graphsvx_scores(record: Mapping[str, object]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for idx, value in enumerate(record["node_importance"]):  # type: ignore[index]
        scores[idx] = float(value)
    return scores


def _subgraphx_scores(result) -> Dict[int, float]:
    max_without: Dict[int, float] = defaultdict(lambda: float("-inf"))
    max_with: Dict[int, float] = defaultdict(lambda: float("-inf"))

    for entry in result.explanation:
        coalition = set(entry.get("coalition", []))
        probability = float(entry.get("P", 0.0))
        for node in coalition:
            max_with[node] = max(max_with[node], probability)
        for node in max_without.keys():
            if node not in coalition:
                max_without[node] = max(max_without[node], probability)
        for node in coalition:
            max_without.setdefault(node, float("-inf"))

    baseline = float(result.related_prediction.get("masked", 0.0))
    scores: Dict[int, float] = {}
    for node, with_score in max_with.items():
        without = max_without.get(node, baseline)
        if without == float("-inf"):
            without = baseline
        scores[node] = with_score - without
    return scores


def _median_threshold(scores: Dict[int, float]) -> float:
    positives = [score for score in scores.values() if score > 0]
    if not positives:
        return 0.0
    return statistics.median(positives)


def _make_token_record(node_idx: int, token: str, score: float, position: float, is_leaf: bool) -> TokenAttribution:
    return TokenAttribution(
        node_index=node_idx,
        token=token,
        score=score,
        position=position,
        is_leaf=is_leaf,
    )


def analyse_graphsvx(
    cfg: GraphSVXConfig,
    loader: GraphArtifactLoader,
    stopwords: Set[str],
) -> Tuple[List[GraphSemanticSummary], pd.DataFrame]:
    records = load_json_records(cfg.path)
    aggregated = AggregatedMetrics()
    summaries: List[GraphSemanticSummary] = []

    for record in tqdm(records, desc=f"Semantic[{cfg.dataset}:{cfg.graph_type}]"):
        graph_info = loader.resolve(cfg.dataset, cfg.graph_type, record["graph_index"], cfg.backbone, cfg.split)
        if graph_info is None:
            continue
        node_order = list(graph_info.node_names)
        node_text = list(graph_info.node_text)
        positions = _compute_positions(node_order)
        scores = _graphsvx_scores(record)
        threshold = cfg.threshold if cfg.threshold is not None else _median_threshold(scores)
        selected = select_tokens(
            scores,
            node_order,
            node_text,
            positions,
            graph_info.graph,
            top_k=cfg.top_k,
            importance_threshold=threshold,
            restrict_to_leaves=False,
            stopwords=stopwords,
        )
        tokens = [_make_token_record(*entry) for entry in selected]
        for attr in tokens:
            aggregated.register(attr.token, attr.score, attr.position)

        metadata = _graph_metadata(graph_info.graph, [attr.node_index for attr in tokens])
        density = _semantic_density(len({attr.token for attr in tokens}), len(tokens))

        summaries.append(
            GraphSemanticSummary(
                graph_index=record["graph_index"],
                label=record.get("label"),
                prediction_class=record.get("prediction", {}).get("class") if record.get("prediction") else None,
                prediction_confidence=record.get("prediction", {}).get("confidence") if record.get("prediction") else None,
                explanation_size=len(tokens),
                unique_token_count=len({attr.token for attr in tokens}),
                semantic_density=density,
                selected_tokens=tokens,
                graph_metadata=metadata,
                extras={
                    "median_threshold": threshold,
                },
            )
        )

    return summaries, aggregated.to_frame()


def analyse_subgraphx(
    cfg: SubgraphXConfig,
    loader: GraphArtifactLoader,
    stopwords: Set[str],
) -> Tuple[List[GraphSemanticSummary], pd.DataFrame]:
    prediction_map = load_prediction_lookup(cfg.prediction_lookup or [])
    aggregated = AggregatedMetrics()
    summaries: List[GraphSemanticSummary] = []

    combined: Dict[int, object] = {}
    offset = 0
    for path in cfg.paths:
        results = load_subgraphx_results(path)
        for result in results:
            global_index = result.graph_index + offset
            if global_index in combined:
                continue
            result.graph_index = global_index
            combined[global_index] = result
        offset += len(results)

    for index in tqdm(sorted(combined), desc=f"Semantic[{cfg.dataset}:{cfg.graph_type}]"):
        result = combined[index]
        graph_info = loader.resolve(cfg.dataset, cfg.graph_type, index, cfg.backbone, cfg.split)
        if graph_info is None:
            continue
        node_order = list(graph_info.node_names)
        node_text = list(graph_info.node_text)
        positions = _compute_positions(node_order)
        scores = _subgraphx_scores(result)
        threshold = cfg.threshold if cfg.threshold is not None else _median_threshold(scores)
        selected = select_tokens(
            scores,
            node_order,
            node_text,
            positions,
            graph_info.graph,
            top_k=cfg.top_k,
            importance_threshold=threshold,
            restrict_to_leaves=True,
            stopwords=stopwords,
        )
        tokens = [_make_token_record(*entry) for entry in selected]
        for attr in tokens:
            aggregated.register(attr.token, attr.score, attr.position)

        metadata = _graph_metadata(graph_info.graph, [attr.node_index for attr in tokens])
        density = _semantic_density(len({attr.token for attr in tokens}), len(tokens))
        extras = {
            "masked_confidence": result.related_prediction.get("masked"),
            "maskout_confidence": result.related_prediction.get("maskout"),
            "sparsity": result.related_prediction.get("sparsity"),
            "median_threshold": threshold,
        }

        summaries.append(
            GraphSemanticSummary(
                graph_index=index,
                label=result.label,
                prediction_class=prediction_map.get(index),
                prediction_confidence=result.related_prediction.get("origin"),
                explanation_size=len(tokens),
                unique_token_count=len({attr.token for attr in tokens}),
                semantic_density=density,
                selected_tokens=tokens,
                graph_metadata=metadata,
                extras=extras,
            )
        )

    return summaries, aggregated.to_frame()
