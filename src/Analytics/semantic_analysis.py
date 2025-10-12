"""Semantic and structural analytics for explanation outputs.

This module ingests GraphSVX (window/skip-gram graphs) and SubgraphX
(tree-structured graphs) explanation artefacts and produces fine-grained
token-level attribution summaries plus complementary structural metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Set

import networkx as nx
import pandas as pd
from tqdm import tqdm

from src.Insights.providers import GraphArtifactProvider


# --------------------------------------------------------------------------- #
# Dataclasses capturing configuration and per-graph summaries
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GraphSVXConfig:
    """Configuration entry describing one GraphSVX artefact."""

    path: Path
    dataset: str
    graph_type: str
    split: str
    backbone: str
    top_k: int = 12
    threshold: Optional[float] = None


@dataclass(slots=True)
class SubgraphXConfig:
    """Configuration entry describing one SubgraphX artefact bundle."""

    paths: List[Path]
    dataset: str
    graph_type: str
    split: str
    backbone: str
    top_k: int = 12
    threshold: Optional[float] = None
    # Optional: JSON files to source prediction_class from (prior insights or runs)
    prediction_lookup: List[Path] | None = None


@dataclass(slots=True)
class TokenAttribution:
    """Token-level attribution details for a single graph."""

    node_index: int
    token: str
    score: float
    position: float
    is_leaf: bool


@dataclass(slots=True)
class GraphSemanticSummary:
    """Semantic and structural summary for a single explanation."""

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


@dataclass(slots=True)
class AggregatedMetrics:
    """Aggregated dataset-level metrics."""

    token_frequencies: Counter[str] = field(default_factory=Counter)
    token_score_sums: Counter[str] = field(default_factory=Counter)
    position_sums: Counter[str] = field(default_factory=Counter)
    position_square_sums: Counter[str] = field(default_factory=Counter)
    occurrences: Counter[str] = field(default_factory=Counter)

    def register(self, token: str, score: float, position: float) -> None:
        norm = (token or "").strip().lower()
        if not norm:
            return
        self.token_frequencies[norm] += 1
        self.token_score_sums[norm] += score
        self.position_sums[norm] += position
        self.position_square_sums[norm] += position * position
        self.occurrences[norm] += 1

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for token, count in self.token_frequencies.most_common():
            score_sum = self.token_score_sums[token]
            mean_score = score_sum / max(count, 1)
            pos_sum = self.position_sums[token]
            pos_sq_sum = self.position_square_sums[token]
            mean_pos = pos_sum / max(count, 1)
            var_pos = max(pos_sq_sum / max(count, 1) - mean_pos**2, 0.0)
            rows.append(
                {
                    "token": token,
                    "frequency": count,
                    "mean_score": mean_score,
                    "mean_position": mean_pos,
                    "position_std": math.sqrt(var_pos),
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def _normalise_dataset_name(dataset: str, backbone: str) -> str:
    """Return canonical dataset/backbone label."""
    if "/" in dataset:
        return dataset
    return f"{backbone}/{dataset}"


def _ensure_safe_globals() -> None:
    """Register PyG storage classes so torch.load accepts full Data objects."""
    try:
        from torch.serialization import add_safe_globals
        import torch_geometric.data.data as pyg_data
        import torch_geometric.data.storage as pyg_storage
    except Exception:
        # Either torch/torch_geometric is unavailable or this torch build lacks add_safe_globals.
        return

    try:
        add_safe_globals(
            [
                pyg_data.Data,
                pyg_data.DataTensorAttr,
                pyg_data.DataEdgeAttr,
                pyg_storage.BaseStorage,
                pyg_storage.EdgeStorage,
                pyg_storage.NodeStorage,
                pyg_storage.GlobalStorage,
            ]
        )
    except Exception:
        # Safe globals registration is best-effort; proceed even if it fails.
        pass


def _load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"GraphSVX artefact at {path} is not a JSON list.")
        return payload


def _resolve_graph_dir(nx_root: Path, dataset: str, graph_type: str, split: str) -> Path:
    base = nx_root / dataset / split
    if not base.exists():
        raise FileNotFoundError(f"NetworkX root {base} missing for dataset {dataset!s}.")
    # Graph type directories may carry extra suffixes (e.g. window.word.k1)
    direct = base / graph_type
    if direct.exists():
        return direct
    matches = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(graph_type))
    if not matches:
        raise FileNotFoundError(f"No graph directory matching '{graph_type}' under {base}")
    return matches[0]


def _load_prediction_lookup(paths: Sequence[Path]) -> Dict[int, int]:
    """Load a mapping graph_index -> prediction_class from prior artefacts.

    Supports two common payload shapes:
    - List of dicts with keys {"graph_index", "prediction": {"class": int}}
    - List of dicts with key "prediction_class": int
    """
    mapping: Dict[int, int] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            gi = item.get("graph_index")
            pred_class = None
            pred = item.get("prediction")
            if isinstance(pred, dict):
                pred_class = pred.get("class")
            if pred_class is None:
                pred_class = item.get("prediction_class")
            if isinstance(gi, int) and isinstance(pred_class, int):
                mapping[gi] = pred_class
    return mapping


def _semantic_density(unique_token_count: int, explanation_size: int) -> float:
    if explanation_size <= 0:
        return 0.0
    return unique_token_count / float(explanation_size)


def _format_label_suffix(label: object) -> str:
    """Return a filesystem-friendly suffix representing a label value."""
    if label is None:
        return "class_unknown"
    try:
        value = float(label)
        if float(int(value)) == value:
            return f"class_{int(value)}"
    except (TypeError, ValueError):
        pass
    text = str(label).strip()
    if not text:
        return "class_unknown"
    sanitized = text.replace(" ", "_").replace("/", "_")
    return f"class_{sanitized}"


def _token_is_structural(token: str) -> bool:
    """Heuristic to detect structural (non-lexical) labels in tree graphs."""
    stripped = token.strip()
    if not stripped:
        return True
    if stripped.startswith("«") and stripped.endswith("»"):
        return True
    if stripped.isupper() and len(stripped) <= 4:
        return True
    return False


def _compute_positions(node_order: Sequence[int]) -> Dict[int, float]:
    """Return normalised positions for each node index."""
    total = max(len(node_order) - 1, 1)
    return {idx: pos / total for pos, idx in enumerate(node_order)}


def _induced_metrics(graph: nx.Graph, selected_nodes: Sequence[int]) -> Dict[str, float]:
    """Compute structural metrics for the node-induced subgraph."""
    if not selected_nodes:
        return {
            "induced_num_nodes": 0.0,
            "induced_num_edges": 0.0,
            "induced_density": 0.0,
            "induced_components": 0.0,
        }
    subgraph = graph.subgraph(selected_nodes)
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()
    density = nx.density(subgraph)
    components = nx.number_connected_components(subgraph.to_undirected())
    return {
        "induced_num_nodes": float(num_nodes),
        "induced_num_edges": float(num_edges),
        "induced_density": float(density),
        "induced_components": float(components),
    }


def _default_stopwords() -> Set[str]:
    # A compact English stopword set (extendable via config).
    words = {
        "a","an","the","and","or","but","if","while","for","to","of","in","on","at","by","with","from",
        "as","that","this","these","those","is","am","are","was","were","be","been","being","it","its","it's",
        "he","she","they","we","you","i","me","him","her","them","us","my","your","our","their",
        "not","no","nor","so","than","then","too","very","can","could","should","would","may","might","will","shall",
        "do","does","did","doing","done","have","has","had","having",
        "there","here","also","just","only","over","under","up","down","out","into","about","after","before","between",
        "more","most","less","least","any","some","such","each","other","both","all","many","much","few","several",
    }
    # Common contractions and clitics
    contractions = {
        "'s", "’s", "n't", "n’t", "'re", "’re", "'ve", "’ve", "'ll", "’ll", "'d", "’d", "'m", "’m",
        "s", "t", "m", "re", "ve", "ll", "d", "nt", "wo",
        ";s", ";t", ";d", ";re", ";ve", ";ll",
    }
    # Common HTML entity residues
    html_residues = {"quot", "amp", "apos"}
    # Common punctuation-like tokens
    punct = set(list(".,:;!?()[]{}'\"`“”’—–-…/\\|+*=<>#@&%$^~"))
    # Also include backticks variants used in some corpora
    words.update({"``","''","--","\\a"})
    return words.union(punct).union(contractions).union(html_residues)


def _is_noise_token(token: str, stopwords: Set[str]) -> bool:
    t = (token or "").strip()
    if not t:
        return True
    lower = t.lower()
    if lower in stopwords:
        return True
    # Pure punctuation or numeric
    if all(ch in _PUNCT_CHARS for ch in t):
        return True
    if lower.isnumeric():
        return True
    # Tokens with digits but no letters (e.g., '4.', '6-', '1/')
    if any(ch.isdigit() for ch in t) and not any(ch.isalpha() for ch in t):
        return True
    # Semicolon clitic artifacts like ';s', ';t'
    if ";" in t and len(t) <= 3:
        return True
    return False


_PUNCT_CHARS = set(".,:;!?()[]{}'\"`“”’—–-…/\\|+*=<>#@&%$^~ ")


_GLOBAL_STOPWORDS: Set[str] = set()


# --------------------------------------------------------------------------- #
# SubgraphX loading helpers
# --------------------------------------------------------------------------- #


class _SubgraphXResult:
    """Lightweight placeholder matching pickled SubgraphXResult objects."""

    def __init__(self) -> None:
        self.graph_index: int = -1
        self.label: Optional[int] = None
        self.explanation: List[dict] = []
        self.related_prediction: Dict[str, float] = {}
        self.num_nodes: int = 0
        self.num_edges: int = 0
        self.hyperparams: Dict[str, float] = {}

    def __setstate__(self, state: Mapping[str, object]) -> None:  # pragma: no cover - invoked during pickle load
        self.__dict__.update(state)


SubgraphXResult = _SubgraphXResult  # Backwards compatibility for pickled artefacts.


class _SubgraphXStub:
    """Context manager patching sys.modules so pickle loads stub classes."""

    def __enter__(self):
        import sys
        import types

        self._modules = {}
        for name in (
            "src",
            "src.explain",
            "src.explain.gnn",
            "src.explain.gnn.subgraphx",
            "src.explain.gnn.subgraphx.main",
        ):
            self._modules[name] = sys.modules.get(name)

        src_module = types.ModuleType("src")
        explain_module = types.ModuleType("src.explain")
        gnn_module = types.ModuleType("src.explain.gnn")
        subgraphx_module = types.ModuleType("src.explain.gnn.subgraphx")
        main_module = types.ModuleType("src.explain.gnn.subgraphx.main")
        main_module.SubgraphXResult = _SubgraphXResult

        sys.modules.update(
            {
                "src": src_module,
                "src.explain": explain_module,
                "src.explain.gnn": gnn_module,
                "src.explain.gnn.subgraphx": subgraphx_module,
                "src.explain.gnn.subgraphx.main": main_module,
            }
        )

    def __exit__(self, exc_type, exc, traceback):
        import sys

        for name, module in getattr(self, "_modules", {}).items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_subgraphx_payload(path: Path) -> List[_SubgraphXResult]:
    with _SubgraphXStub():
        import pickle

        with path.open("rb") as handle:
            payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload type in {path}: {type(payload)}")
    return payload


# --------------------------------------------------------------------------- #
# Core analysis routines
# --------------------------------------------------------------------------- #


def _select_tokens(
    scores: Dict[int, float],
    node_order: Sequence[int],
    node_text: Sequence[str],
    positions: Mapping[int, float],
    graph: nx.Graph,
    *,
    top_k: int,
    importance_threshold: float,
    restrict_to_leaves: bool,
    stopwords: Set[str],
) -> List[TokenAttribution]:
    """Return ordered token attributions applying granularity constraints."""
    node_to_offset = {node: idx for idx, node in enumerate(node_order)}
    if graph.is_directed():
        degree_fn = graph.out_degree  # type: ignore[assignment]
        leaf_check = lambda deg: deg == 0
    else:
        degree_fn = graph.degree  # type: ignore[assignment]
        leaf_check = lambda deg: deg <= 1

    entries: List[TokenAttribution] = []
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for rank, (node_idx, score) in enumerate(sorted_nodes):
        if math.isnan(score):
            continue
        if score < importance_threshold:
            continue
        try:
            degree_value = degree_fn(node_idx)
        except Exception:  # pragma: no cover - defensive fallback
            degree_value = 0
        if restrict_to_leaves and not leaf_check(degree_value):
            continue
        if node_idx not in positions:
            continue
        offset = node_to_offset.get(node_idx)
        if offset is None or offset >= len(node_text):
            continue
        token = node_text[offset]
        if not token.strip():
            continue
        if _is_noise_token(token, stopwords):
            continue
        is_leaf = leaf_check(degree_value)
        entries.append(
            TokenAttribution(
                node_index=node_idx,
                token=token,
                score=float(score),
                position=float(positions[node_idx]),
                is_leaf=is_leaf,
            )
        )
        if len(entries) >= top_k:
            break
    return entries


def _graphsvx_scores(record: dict) -> Dict[int, float]:
    """Extract node importance mapping from a GraphSVX record."""
    scores = {}
    for idx, score in enumerate(record["node_importance"]):
        scores[idx] = float(score)
    return scores


def _subgraphx_scores(result: _SubgraphXResult) -> Dict[int, float]:
    """Estimate node contributions via coalition probability deltas."""
    max_without: Dict[int, float] = defaultdict(lambda: float("-inf"))
    max_with: Dict[int, float] = defaultdict(lambda: float("-inf"))

    for entry in result.explanation:
        coalition = set(entry.get("coalition", []))
        prob = float(entry.get("P", 0.0))
        for node in coalition:
            if prob > max_with[node]:
                max_with[node] = prob
        for node in max_without.keys():
            if node not in coalition and prob > max_without[node]:
                max_without[node] = prob
        for node in coalition:
            if node not in max_without:
                max_without[node] = float("-inf")

    baseline = float(result.related_prediction.get("masked", 0.0))
    scores: Dict[int, float] = {}
    for node, with_score in max_with.items():
        without = max_without.get(node, baseline)
        if without == float("-inf"):
            without = baseline
        scores[node] = with_score - without
    return scores


def _dataset_median(score_map: Mapping[int, float]) -> float:
    if not score_map:
        return 0.0
    positives = [score for score in score_map.values() if score > 0]
    if not positives:
        return 0.0
    return statistics.median(positives)


def analyse_graphsvx(
    config: GraphSVXConfig,
    provider: GraphArtifactProvider,
) -> Tuple[List[GraphSemanticSummary], pd.DataFrame]:
    """Process a GraphSVX artefact into per-graph and aggregate metrics."""
    records = _load_json(config.path)
    dataset_label = _normalise_dataset_name(config.dataset, config.backbone)
    tqdm_desc = f"GraphSVX[{dataset_label}:{config.graph_type}]"

    per_graph: List[GraphSemanticSummary] = []
    aggregated = AggregatedMetrics()
    medians: List[float] = []

    for record in tqdm(records, desc=tqdm_desc):
        info = provider(
            argparse.Namespace(
                dataset=config.dataset,
                graph_type=config.graph_type,
                graph_index=record["graph_index"],
                extras={"split": config.split, "backbone": config.backbone},
            )
        )
        if info is None:
            continue
        node_order = list(info.node_names)
        node_text = list(info.node_text)
        graph = info.graph
        positions = _compute_positions(node_order)

        scores = _graphsvx_scores(record)
        median_score = _dataset_median(scores)
        medians.append(median_score)
        threshold = config.threshold if config.threshold is not None else median_score

        selected = _select_tokens(
            scores,
            node_order,
            node_text,
            positions,
            graph,
            top_k=config.top_k,
            importance_threshold=threshold,
            restrict_to_leaves=False,
            stopwords=_GLOBAL_STOPWORDS,
        )
        for attribution in selected:
            aggregated.register(attribution.token, attribution.score, attribution.position)

        explanation_size = len(selected)
        unique_tokens = len({tok.token for tok in selected})
        metadata = {
            "num_nodes": float(graph.number_of_nodes()),
            "num_edges": float(graph.number_of_edges()),
        }
        metadata.update(_induced_metrics(graph, [item.node_index for item in selected]))
        density = _semantic_density(unique_tokens, explanation_size)

        per_graph.append(
            GraphSemanticSummary(
                graph_index=record["graph_index"],
                label=record.get("label"),
                prediction_class=record.get("prediction", {}).get("class"),
                prediction_confidence=record.get("prediction", {}).get("confidence"),
                explanation_size=explanation_size,
                unique_token_count=unique_tokens,
                semantic_density=density,
                selected_tokens=selected,
                graph_metadata=metadata,
                extras={
                    "median_threshold": median_score,
                },
            )
        )

    aggregate_frame = aggregated.to_frame()
    aggregate_frame["dataset"] = dataset_label
    aggregate_frame["graph_type"] = config.graph_type
    if medians:
        aggregate_frame.attrs["median_of_medians"] = float(statistics.median(medians))
    return per_graph, aggregate_frame


def analyse_subgraphx(
    config: SubgraphXConfig,
    provider: GraphArtifactProvider,
) -> Tuple[List[GraphSemanticSummary], pd.DataFrame]:
    """Process a SubgraphX artefact bundle while keeping memory usage low."""
    dataset_label = _normalise_dataset_name(config.dataset, config.backbone)
    tqdm_desc = f"SubgraphX[{dataset_label}:{config.graph_type}]"

    per_graph: List[GraphSemanticSummary] = []
    aggregated = AggregatedMetrics()
    medians: List[float] = []

    prediction_map: Dict[int, int] = {}
    if config.prediction_lookup:
        prediction_map = _load_prediction_lookup(config.prediction_lookup)

    offset = 0
    progress = tqdm(desc=tqdm_desc, unit="graph")
    try:
        for shard_idx, path in enumerate(config.paths):
            results = _load_subgraphx_payload(path)
            for result in results:
                original_index = result.graph_index
                graph_index = original_index + offset
                result.graph_index = graph_index
                setattr(
                    result,
                    "_aggregation_extras",
                    {"shard_index": shard_idx, "local_graph_index": original_index},
                )

                info = provider(
                    argparse.Namespace(
                        dataset=config.dataset,
                        graph_type=config.graph_type,
                        graph_index=graph_index,
                        extras={"split": config.split, "backbone": config.backbone},
                    )
                )
                if info is None:
                    progress.update(1)
                    continue

                node_order = list(info.node_names)
                node_text = list(info.node_text)
                graph = info.graph
                positions = _compute_positions(node_order)

                scores = _subgraphx_scores(result)
                median_score = _dataset_median(scores)
                medians.append(median_score)
                threshold = config.threshold if config.threshold is not None else median_score

                selected = _select_tokens(
                    scores,
                    node_order,
                    node_text,
                    positions,
                    graph,
                    top_k=config.top_k,
                    importance_threshold=threshold,
                    restrict_to_leaves=True,
                    stopwords=_GLOBAL_STOPWORDS,
                )
                selected = [item for item in selected if not _token_is_structural(item.token)]
                for attribution in selected:
                    aggregated.register(attribution.token, attribution.score, attribution.position)

                explanation_size = len(selected)
                unique_tokens = len({tok.token for tok in selected})
                metadata = {
                    "num_nodes": float(graph.number_of_nodes()),
                    "num_edges": float(graph.number_of_edges()),
                }
                metadata.update(_induced_metrics(graph, [item.node_index for item in selected]))
                density = _semantic_density(unique_tokens, explanation_size)

                extras = {
                    "masked_confidence": result.related_prediction.get("masked"),
                    "maskout_confidence": result.related_prediction.get("maskout"),
                    "sparsity": result.related_prediction.get("sparsity"),
                    "median_threshold": median_score,
                }
                extras.update(getattr(result, "_aggregation_extras", {}))

                per_graph.append(
                    GraphSemanticSummary(
                        graph_index=graph_index,
                        label=result.label,
                        prediction_class=prediction_map.get(graph_index),
                        prediction_confidence=result.related_prediction.get("origin"),
                        explanation_size=explanation_size,
                        unique_token_count=unique_tokens,
                        semantic_density=density,
                        selected_tokens=selected,
                        graph_metadata=metadata,
                        extras=extras,
                    )
                )
                progress.update(1)
            offset += len(results)
            del results
    finally:
        progress.close()

    aggregate_frame = aggregated.to_frame()
    aggregate_frame["dataset"] = dataset_label
    aggregate_frame["graph_type"] = config.graph_type
    if medians:
        aggregate_frame.attrs["median_of_medians"] = float(statistics.median(medians))
    return per_graph, aggregate_frame


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def _summaries_to_frame(
    summaries: Sequence[GraphSemanticSummary],
    dataset_label: str,
    graph_type: str,
) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        label = summary.label
        pred_class = summary.prediction_class
        is_correct = None
        if label is not None and pred_class is not None:
            is_correct = bool(label == pred_class)
        base = {
            "graph_index": summary.graph_index,
            "label": label,
            "prediction_class": pred_class,
            "prediction_confidence": summary.prediction_confidence,
            "explanation_size": summary.explanation_size,
            "unique_token_count": summary.unique_token_count,
            "semantic_density": summary.semantic_density,
            "dataset": dataset_label,
            "graph_type": graph_type,
            "is_correct": is_correct,
        }
        base.update(summary.graph_metadata)
        base.update(summary.extras)
        rows.append(base)
    return pd.DataFrame(rows)


def _tokens_to_frame(
    summaries: Sequence[GraphSemanticSummary],
    dataset_label: str,
    graph_type: str,
) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        label = summary.label
        pred_class = summary.prediction_class
        is_correct = None
        if label is not None and pred_class is not None:
            is_correct = bool(label == pred_class)
        for rank, attribution in enumerate(summary.selected_tokens, start=1):
            rows.append(
                {
                    "graph_index": summary.graph_index,
                    "token": attribution.token,
                    "score": attribution.score,
                    "position": attribution.position,
                    "rank": rank,
                    "dataset": dataset_label,
                    "graph_type": graph_type,
                    "label": label,
                    "prediction_class": pred_class,
                    "is_correct": is_correct,
                }
            )
    return pd.DataFrame(rows)


def _filter_frame(frame: pd.DataFrame, criteria: List[Tuple[str, object]]) -> pd.DataFrame:
    """Return rows matching all provided column/value pairs."""
    subset = frame
    for column, value in criteria:
        if column not in subset.columns:
            return subset.iloc[0:0].copy()
        if value is None:
            mask = subset[column].isna()
        else:
            mask = subset[column] == value
        subset = subset.loc[mask]
    return subset


def _build_partition_specs(summary_frame: pd.DataFrame) -> List[Tuple[str, List[Tuple[str, object]]]]:
    """Define partitions for correctness and label splits."""
    specs: List[Tuple[str, List[Tuple[str, object]]]] = []
    if "is_correct" in summary_frame.columns and summary_frame["is_correct"].notna().any():
        for flag, suffix in ((True, "correct"), (False, "incorrect")):
            mask = summary_frame["is_correct"] == flag
            if mask.any():
                specs.append((suffix, [("is_correct", flag)]))
    if "label" in summary_frame.columns:
        labels = (
            summary_frame["label"]
            .dropna()
            .unique()
            .tolist()
        )
        labels.sort(key=lambda item: str(item))
        for label in labels:
            label_mask = summary_frame["label"] == label
            if not label_mask.any():
                continue
            base_suffix = _format_label_suffix(label)
            specs.append((base_suffix, [("label", label)]))
            if "is_correct" in summary_frame.columns and summary_frame["is_correct"].notna().any():
                for flag, suffix in ((True, "correct"), (False, "incorrect")):
                    combined_mask = label_mask & (summary_frame["is_correct"] == flag)
                    if combined_mask.any():
                        specs.append(
                            (f"{base_suffix}_{suffix}", [("label", label), ("is_correct", flag)])
                        )
    return specs


def _write_partitioned_outputs(
    base_name: str,
    output_dir: Path,
    summary_frame: pd.DataFrame,
    tokens_frame: pd.DataFrame,
) -> None:
    """Persist partitioned summary and token tables mirroring structural splits."""
    specs = _build_partition_specs(summary_frame)
    for suffix, criteria in specs:
        subset_summary = _filter_frame(summary_frame, criteria)
        if subset_summary.empty:
            continue
        summary_path = output_dir / f"{base_name}_{suffix}_summary.csv"
        subset_summary.to_csv(summary_path, index=False)

        subset_tokens = _filter_frame(tokens_frame, criteria)
        if not subset_tokens.empty:
            tokens_path = output_dir / f"{base_name}_{suffix}_tokens.csv"
            subset_tokens.to_csv(tokens_path, index=False)


def _write_outputs(
    output_dir: Path,
    dataset_label: str,
    graph_type: str,
    summaries: Sequence[GraphSemanticSummary],
    aggregate: pd.DataFrame,
) -> None:
    base_name = f"{dataset_label.replace('/', '_')}_{graph_type}"
    target_dir = output_dir / base_name
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = _summaries_to_frame(summaries, dataset_label, graph_type)
    tokens_frame = _tokens_to_frame(summaries, dataset_label, graph_type)

    summary_path = target_dir / f"{base_name}_summary.csv"
    tokens_path = target_dir / f"{base_name}_tokens.csv"
    aggregate_path = target_dir / f"{base_name}_aggregate.csv"

    summary_frame.to_csv(summary_path, index=False)
    tokens_frame.to_csv(tokens_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    _write_partitioned_outputs(base_name, target_dir, summary_frame, tokens_frame)

    meta = {
        "dataset": dataset_label,
        "graph_type": graph_type,
        "num_graphs": len(summaries),
        "summary_csv": str(summary_path),
        "tokens_csv": str(tokens_path),
        "aggregate_csv": str(aggregate_path),
        "median_threshold": aggregate.attrs.get("median_of_medians"),
    }
    meta_path = target_dir / f"{base_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Configuration loading and CLI
# --------------------------------------------------------------------------- #


def _load_config(path: Path) -> Tuple[List[GraphSVXConfig], List[SubgraphXConfig], Path, Path, Set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nx_root = Path(payload.get("graph_roots", {}).get("nx", "outputs/graphs")).resolve()
    pyg_root = Path(payload.get("graph_roots", {}).get("pyg", "outputs/pyg_graphs")).resolve()
    graphsvx_entries: List[GraphSVXConfig] = []
    for entry in payload.get("graphsvx", []):
        graphsvx_entries.append(
            GraphSVXConfig(
                path=Path(entry["path"]).resolve(),
                dataset=entry["dataset"],
                graph_type=entry["graph_type"],
                split=entry["split"],
                backbone=entry["backbone"],
                top_k=entry.get("top_k", 12),
                threshold=entry.get("threshold"),
            )
        )
    subgraphx_entries: List[SubgraphXConfig] = []
    for entry in payload.get("subgraphx", []):
        subgraphx_entries.append(
            SubgraphXConfig(
                paths=[Path(p).resolve() for p in entry["paths"]],
                dataset=entry["dataset"],
                graph_type=entry["graph_type"],
                split=entry["split"],
                backbone=entry["backbone"],
                top_k=entry.get("top_k", 12),
                threshold=entry.get("threshold"),
                prediction_lookup=[Path(p).resolve() for p in entry.get("prediction_lookup", [])] if entry.get("prediction_lookup") else None,
            )
        )
    # Stopwords: inline list and/or files
    stopwords: Set[str] = _default_stopwords()
    for w in payload.get("stopwords", []) or []:
        if isinstance(w, str):
            stopwords.add(w.lower())
    for p in payload.get("stopwords_files", []) or []:
        try:
            text = Path(p).expanduser().read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    stopwords.add(line.lower())
        except Exception:
            continue
    return graphsvx_entries, subgraphx_entries, nx_root, pyg_root, stopwords


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Perform semantic and structural analysis over GraphSVX and SubgraphX artefacts.",
    )
    parser.add_argument(
        "--dataset",
        help="Optional dataset filter; processes only entries whose dataset matches this value.",
    )
    parser.add_argument(
        "--graph-type",
        help="Optional graph-type filter; processes only entries whose graph_type matches this value.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/semantic_analysis_config.json"),
        help="Path to semantic analysis configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/semantic_analysis"),
        help="Directory where analysis tables will be written.",
    )
    args = parser.parse_args(argv)

    graphsvx_cfgs, subgraphx_cfgs, nx_root, pyg_root, stopwords = _load_config(args.config.resolve())

    if args.dataset:
        target = args.dataset.strip()
        graphsvx_cfgs = [cfg for cfg in graphsvx_cfgs if cfg.dataset == target or _normalise_dataset_name(cfg.dataset, cfg.backbone) == target]
        subgraphx_cfgs = [cfg for cfg in subgraphx_cfgs if cfg.dataset == target or _normalise_dataset_name(cfg.dataset, cfg.backbone) == target]
    if args.graph_type:
        graphsvx_cfgs = [cfg for cfg in graphsvx_cfgs if cfg.graph_type == args.graph_type]
        subgraphx_cfgs = [cfg for cfg in subgraphx_cfgs if cfg.graph_type == args.graph_type]

    _ensure_safe_globals()
    global _GLOBAL_STOPWORDS
    _GLOBAL_STOPWORDS = stopwords
    provider = GraphArtifactProvider(graph_root=nx_root, pyg_root=pyg_root, strict=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in graphsvx_cfgs:
        dataset_label = _normalise_dataset_name(cfg.dataset, cfg.backbone)
        summaries, aggregate = analyse_graphsvx(cfg, provider)
        _write_outputs(args.output_dir, dataset_label, cfg.graph_type, summaries, aggregate)

    for cfg in subgraphx_cfgs:
        dataset_label = _normalise_dataset_name(cfg.dataset, cfg.backbone)
        summaries, aggregate = analyse_subgraphx(cfg, provider)
        _write_outputs(args.output_dir, dataset_label, cfg.graph_type, summaries, aggregate)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
