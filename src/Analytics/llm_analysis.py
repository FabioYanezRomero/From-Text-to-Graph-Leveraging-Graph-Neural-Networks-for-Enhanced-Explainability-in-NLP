"""Analytics helpers tailored to LLM-derived token attributions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from .fidelity_analysis import run_fidelity_analysis
from .postprocess.semantic_exports import run_exports as export_semantic_exports
from .semantic.common.config import _default_stopwords as _semantic_default_stopwords
from .semantic.common.models import GraphSemanticSummary, TokenAttribution
from .semantic.common.outputs import summaries_to_frame, tokens_to_frame, write_csv
from .token.analysis import AggregatedMetrics
from .utils import InsightFrame, load_insights, load_json_record, resolve_paths

_NUMERIC_EXTRAS: Tuple[str, ...] = (
    "origin_confidence",
    "masked_confidence",
    "maskout_confidence",
    "minimal_coalition_size",
    "minimal_coalition_confidence",
    "insertion_auc",
    "sparsity",
)


@dataclass(frozen=True)
class _SelectedToken:
    """Lightweight container linking token metadata before conversion."""

    index: int
    token: str
    score: float


def _coerce_float(value: object) -> float | None:
    """Best-effort coercion of heterogeneous numeric inputs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    """Robust int conversion returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_position(index: int, total: int) -> float:
    """Map a 0-based index on a sequence of ``total`` items to ``[0, 1]``."""
    if total <= 1:
        return 0.0
    return index / float(total - 1)


def _semantic_density(unique_tokens: int, explanation_size: int) -> float:
    if explanation_size <= 0:
        return 0.0
    return unique_tokens / float(explanation_size)


def _load_stopword_inventory(path: Path | None) -> Set[str]:
    """Load stopwords from a file or fall back to the semantic defaults."""
    inventory: Set[str] = set()
    if path is not None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
        for line in text.splitlines():
            token = line.strip().lower()
            if token:
                inventory.add(token)
    if not inventory:
        try:
            inventory.update({str(item).strip().lower() for item in _semantic_default_stopwords() if item})
        except Exception:
            inventory.update(DEFAULT_FALLBACK_STOPWORDS)
    return inventory


DEFAULT_FALLBACK_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "while",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "me",
    "him",
    "her",
    "them",
    "us",
    "my",
    "your",
    "our",
    "their",
    "not",
    "no",
    "nor",
    "so",
    "than",
    "then",
    "too",
    "very",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "will",
    "shall",
    "do",
    "does",
    "did",
    "doing",
    "done",
    "have",
    "has",
    "had",
    "having",
    "there",
    "here",
    "also",
    "just",
    "only",
    "over",
    "under",
    "up",
    "down",
    "out",
    "into",
    "about",
    "after",
    "before",
    "between",
    "more",
    "most",
    "less",
    "least",
    "any",
    "some",
    "such",
    "each",
    "other",
    "both",
    "all",
    "many",
    "much",
    "few",
    "several",
}


def _select_tokens(
    words: Sequence[object],
    scores: Sequence[object],
    top_k: int | None,
    stopwords: Set[str],
) -> List[_SelectedToken]:
    """Select the top scoring tokens from a word/score sequence."""
    cleaned: List[_SelectedToken] = []
    for index, (word, score) in enumerate(zip(words, scores)):
        token = str(word).strip()
        if not token:
            continue
        if token.lower() in stopwords:
            continue
        numeric_score = _coerce_float(score)
        if numeric_score is None:
            continue
        cleaned.append(_SelectedToken(index=index, token=token, score=numeric_score))
    cleaned.sort(key=lambda item: item.score, reverse=True)
    if top_k is not None and top_k > 0:
        return cleaned[: min(top_k, len(cleaned))]
    return cleaned


def _gather_extras(record: Mapping[str, object], selected_count: int, total_words: int) -> Dict[str, float]:
    """Collect optional numeric metadata to preserve in the summary CSV."""
    extras: Dict[str, float] = {}
    for key in _NUMERIC_EXTRAS:
        value = _coerce_float(record.get(key))
        if value is not None:
            extras[key] = value

    minimal_tokens = record.get("minimal_coalition_tokens")
    if isinstance(minimal_tokens, Sequence):
        extras["minimal_token_count"] = float(len(minimal_tokens))
    top_tokens = record.get("top_tokens")
    if isinstance(top_tokens, Sequence):
        extras["reported_top_token_count"] = float(len(top_tokens))

    extras["total_word_count"] = float(total_words)
    if total_words > 0:
        extras["selected_token_ratio"] = float(selected_count / total_words)

    word_level = record.get("word_level_available")
    if isinstance(word_level, bool):
        extras["word_level_available"] = 1.0 if word_level else 0.0

    return extras


def _build_summary(
    record: Mapping[str, object],
    dataset: str,
    graph_type: str,
    selections: Sequence[_SelectedToken],
    total_words: int,
    aggregator: AggregatedMetrics,
) -> GraphSemanticSummary | None:
    """Convert a raw insight dictionary into a structured summary."""
    if not selections:
        return None

    attributions: List[TokenAttribution] = []
    for token in selections:
        position = _normalise_position(token.index, total_words)
        attributions.append(
            TokenAttribution(
                node_index=token.index,
                token=token.token,
                score=token.score,
                position=position,
                is_leaf=True,
            )
        )
        aggregator.register(token.token, token.score, position)

    unique_count = len({attr.token.lower() for attr in attributions if attr.token})
    density = _semantic_density(unique_count, len(attributions))

    extras = _gather_extras(record, len(attributions), total_words)
    metadata: Dict[str, float] = {
        "sequence_length": float(total_words),
    }

    return GraphSemanticSummary(
        graph_index=_coerce_int(record.get("graph_index")) or 0,
        label=_coerce_int(record.get("label")),
        prediction_class=_coerce_int(record.get("prediction_class")),
        prediction_confidence=_coerce_float(record.get("prediction_confidence")),
        explanation_size=len(attributions),
        unique_token_count=unique_count,
        semantic_density=density,
        selected_tokens=attributions,
        graph_metadata=metadata,
        extras=extras,
    )


def _process_group(
    records: Sequence[Mapping[str, object]],
    dataset: str,
    graph_type: str,
    output_dir: Path,
    top_k: int | None,
    stopwords: Set[str],
) -> Dict[str, object]:
    """Produce analytics artefacts for a dataset/graph pair."""
    aggregator = AggregatedMetrics()
    summaries: List[GraphSemanticSummary] = []

    for record in records:
        words = record.get("words")
        scores = record.get("word_scores")
        if not isinstance(words, Sequence) or not isinstance(scores, Sequence):
            continue
        if len(words) != len(scores):
            continue

        selections = _select_tokens(words, scores, top_k, stopwords)
        summary = _build_summary(record, dataset, graph_type, selections, len(words), aggregator)
        if summary is not None:
            summaries.append(summary)

    tokens_frame = tokens_to_frame(summaries, dataset, graph_type)
    summary_frame = summaries_to_frame(summaries, dataset, graph_type)
    aggregate_frame = aggregator.to_frame()
    if not aggregate_frame.empty:
        aggregate_frame["dataset"] = dataset
        aggregate_frame["graph_type"] = graph_type

    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.csv"
    summary_path = output_dir / "summary.csv"
    aggregate_path = output_dir / "aggregate.csv"

    write_csv(tokens_frame, tokens_path)
    write_csv(summary_frame, summary_path)
    write_csv(aggregate_frame, aggregate_path)

    artefacts: Dict[str, object] = {
        "records": len(records),
        "summaries": len(summaries),
        "dataset": dataset,
        "graph_type": graph_type,
        "tokens_csv": str(tokens_path) if not tokens_frame.empty else None,
        "summary_csv": str(summary_path) if not summary_frame.empty else None,
        "aggregate_csv": str(aggregate_path) if not aggregate_frame.empty else None,
    }
    if top_k is not None and top_k > 0:
        artefacts["top_k"] = int(top_k)
    return artefacts


def _subset_insight(insight: InsightFrame, dataset: str, graph_type: str) -> InsightFrame | None:
    """Create a dataset/graph-specific view of an ``InsightFrame``."""
    if insight.data.empty:
        return None
    if "dataset" not in insight.data.columns or "graph_type" not in insight.data.columns:
        return None
    mask = (insight.data["dataset"] == dataset) & (insight.data["graph_type"] == graph_type)
    subset_data = insight.data.loc[mask].copy()
    if subset_data.empty:
        return None

    if insight.token_frame.empty:
        subset_tokens = insight.token_frame.copy()
    elif {"dataset", "graph_type"}.issubset(insight.token_frame.columns):
        subset_tokens = insight.token_frame.loc[
            (insight.token_frame["dataset"] == dataset) & (insight.token_frame["graph_type"] == graph_type)
        ].copy()
    else:
        valid_indices = set(subset_data["graph_index"])
        subset_tokens = insight.token_frame.loc[insight.token_frame["graph_index"].isin(valid_indices)].copy()

    subset_data.reset_index(drop=True, inplace=True)
    subset_tokens.reset_index(drop=True, inplace=True)
    return InsightFrame(data=subset_data, token_frame=subset_tokens)


def _generate_visualisations(
    general_dir: Path,
    folder_name: str,
    base_output: Path,
    token_csv_root: Path | None,
) -> Dict[str, List[str] | Dict[str, str] | str]:
    """Produce visual artefacts mirroring the GNN analytics workflow."""
    try:
        from src.visualisations.semantic_confidence import generate_confidence_threshold_visuals
        from src.visualisations.semantic_sparsity import generate_sparsity_visuals
        from src.visualisations.semantic_score import (
            generate_token_position_differences,
            generate_token_score_densities,
            generate_token_score_differences,
            generate_token_score_ranking,
        )
        from src.visualisations.semantic_token_frequency import generate_token_frequency_charts
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        return {"available": False, "reason": f"visualisation modules unavailable: {exc}"}

    def _to_str(paths: List[Path]) -> List[str]:
        return [str(path) for path in paths if isinstance(path, Path)]

    results: Dict[str, List[str] | Dict[str, str] | str] = {}
    tokens_csv = general_dir / "tokens.csv"
    summary_csv = general_dir / "summary.csv"
    if tokens_csv.exists():
        density_dir = base_output / "score" / "density" / folder_name
        density = generate_token_score_densities(general_dir, density_dir)
        if density:
            results["score_density"] = _to_str(density)

        ranking_dir = base_output / "score" / "ranking" / folder_name
        ranking = generate_token_score_ranking(general_dir, ranking_dir)
        if ranking:
            results.setdefault("score_ranking", []).extend(_to_str(ranking))

        token_freq_dir = base_output / "token" / folder_name / "frequency"
        frequency = generate_token_frequency_charts(general_dir, token_freq_dir)
        if frequency:
            results["token_frequency"] = _to_str(frequency)

    if token_csv_root is not None and token_csv_root.exists():
        difference_dir = base_output / "score" / "difference" / folder_name
        differences = generate_token_score_differences(token_csv_root, difference_dir)
        if differences:
            results["score_difference"] = _to_str(differences)

        position_dir = base_output / "position" / "difference" / folder_name
        position = generate_token_position_differences(token_csv_root, position_dir)
        if position:
            results["position_difference"] = _to_str(position)

    if summary_csv.exists():
        sparsity_dir = base_output / "sparsity" / folder_name / "plots"
        sparsity = generate_sparsity_visuals(general_dir, sparsity_dir)
        if sparsity:
            results["sparsity"] = _to_str(sparsity)

        confidence_dir = base_output / "confidence" / folder_name / "plots"
        confidence = generate_confidence_threshold_visuals(general_dir, confidence_dir)
        if confidence:
            results["confidence"] = _to_str(confidence)

    if not results:
        return {"available": False, "reason": "No eligible CSV artefacts found for visualisation."}
    return results


def run_llm_token_analysis(
    insight_paths: Sequence[str],
    output_root: Path,
    top_k: int | None = 12,
    stopwords_path: Path | None = None,
) -> Dict[str, object]:
    """Entry point orchestrating analytics from one or more LLM insight files."""
    resolved = resolve_paths(insight_paths)
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    total_records = 0
    for path in resolved:
        for record in load_json_record(path):
            dataset = record.get("dataset")
            graph_type = record.get("graph_type") or "tokens"
            if not isinstance(dataset, str) or not dataset:
                continue
            grouped.setdefault((dataset, str(graph_type)), []).append(record)
            total_records += 1

    output_root = output_root.resolve()
    general_root = output_root / "general"
    general_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "output_dir": str(output_root),
        "general_root": str(general_root),
        "total_records": total_records,
        "total_groups": len(grouped),
        "groups": {},
        "derived_roots": {
            "token": str(output_root / "token"),
            "sparsity": str(output_root / "sparsity"),
            "confidence": str(output_root / "confidence"),
            "score": str(output_root / "score"),
            "score_density": str(output_root / "score" / "density"),
            "score_difference": str(output_root / "score" / "difference"),
            "score_ranking": str(output_root / "score" / "ranking"),
            "position_difference": str(output_root / "position" / "difference"),
            "token_frequency": str(output_root / "token"),
            "fidelity": str(output_root / "fidelity"),
            "embedding": str(output_root / "embedding"),
        },
    }
    effective_top_k = None if top_k is None or top_k <= 0 else top_k
    stopwords = _load_stopword_inventory(stopwords_path)

    group_keys: Dict[Tuple[str, str], str] = {}
    for (dataset, graph_type), records in grouped.items():
        folder_name = f"{dataset.replace('/', '_')}_{graph_type}"
        group_dir = general_root / folder_name
        artefacts = _process_group(records, dataset, graph_type, group_dir, effective_top_k, stopwords)
        artefacts["folder"] = folder_name
        artefacts["general_dir"] = str(group_dir)
        group_key = f"{dataset}:{graph_type}"
        summary["groups"][group_key] = artefacts
        group_keys[(dataset, graph_type)] = group_key

    if not grouped:
        return summary

    export_semantic_exports(general_root, output_root)

    for artefacts in summary["groups"].values():
        folder_name = artefacts.get("folder")
        if not folder_name:
            continue
        token_dir = output_root / "token" / folder_name / "csv"
        sparsity_dir = output_root / "sparsity" / folder_name / "csv"
        confidence_dir = output_root / "confidence" / folder_name / "csv"
        score_dir = output_root / "score" / folder_name / "csv"
        if token_dir.exists():
            artefacts["token_exports"] = str(token_dir)
        if sparsity_dir.exists():
            artefacts["sparsity_exports"] = str(sparsity_dir)
        if confidence_dir.exists():
            artefacts["confidence_exports"] = str(confidence_dir)
        if score_dir.exists():
            artefacts["score_exports"] = str(score_dir)

    for artefacts in summary["groups"].values():
        folder_name = artefacts.get("folder")
        if not folder_name:
            continue
        general_dir = general_root / folder_name
        token_csv_dir = Path(artefacts["token_exports"]) if "token_exports" in artefacts else None
        visuals = _generate_visualisations(general_dir, folder_name, output_root, token_csv_dir)
        artefacts["visualisations"] = visuals

    embedding_root = output_root / "embedding"
    embedding_root.mkdir(parents=True, exist_ok=True)
    placeholder_payload = {
        "available": False,
        "reason": "LLM insight exports do not expose per-token node embeddings.",
    }

    insight_frame: InsightFrame | None = None
    try:
        insight_frame = load_insights([str(path) for path in resolved])
    except Exception:
        insight_frame = None

    for (dataset, graph_type), group_key in group_keys.items():
        artefacts = summary["groups"].get(group_key, {})
        folder_name = artefacts.get("folder")
        if not folder_name:
            continue

        embedding_dir = embedding_root / folder_name
        embedding_dir.mkdir(parents=True, exist_ok=True)
        embedding_summary_path = embedding_dir / "embedding_summary.json"
        embedding_summary_path.write_text(json.dumps(placeholder_payload, indent=2), encoding="utf-8")
        artefacts["embedding_summary"] = str(embedding_summary_path)

        if insight_frame is None:
            continue
        subset = _subset_insight(insight_frame, dataset, graph_type)
        if subset is None:
            continue
        fidelity_root = output_root / "fidelity" / folder_name
        result = run_fidelity_analysis(subset, fidelity_root, "label")
        fidelity_summary_path = fidelity_root / "fidelity_analysis_summary.json"
        fidelity_summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        artefacts["fidelity_dir"] = str(fidelity_root)
        artefacts["fidelity_summary"] = str(fidelity_summary_path)

    return summary
