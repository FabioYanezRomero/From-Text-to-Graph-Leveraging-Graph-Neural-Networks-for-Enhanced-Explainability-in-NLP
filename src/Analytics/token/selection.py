from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Set

import networkx as nx

_PUNCT_CHARS = set(".,:;!?()[]{}'\"`“”’—–-…/\\|+*=<>#@&%$^~ ")


def is_structural_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return True
    if stripped.startswith("«") and stripped.endswith("»"):
        return True
    if stripped.isupper() and len(stripped) <= 4:
        return True
    return False


def is_noise_token(token: str, stopwords: Set[str]) -> bool:
    t = (token or "").strip()
    if not t:
        return True
    lower = t.lower()
    if lower in stopwords:
        return True
    if all(ch in _PUNCT_CHARS for ch in t):
        return True
    if lower.isnumeric():
        return True
    if any(ch.isdigit() for ch in t) and not any(ch.isalpha() for ch in t):
        return True
    if ";" in t and len(t) <= 3:
        return True
    return False


def select_tokens(
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
) -> List[tuple[int, str, float, float, bool]]:
    node_to_offset = {node: idx for idx, node in enumerate(node_order)}
    if graph.is_directed():
        degree_fn = graph.out_degree  # type: ignore[assignment]
        leaf_check = lambda deg: deg == 0
    else:
        degree_fn = graph.degree  # type: ignore[assignment]
        leaf_check = lambda deg: deg <= 1

    entries: List[tuple[int, str, float, float, bool]] = []
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for node_idx, score in sorted_nodes:
        if math.isnan(score):
            continue
        if score < importance_threshold:
            continue
        try:
            degree_value = degree_fn(node_idx)
        except Exception:
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
        if is_noise_token(token, stopwords) or is_structural_token(token):
            continue
        is_leaf = leaf_check(degree_value)
        entries.append((node_idx, token, float(score), float(positions[node_idx]), is_leaf))
        if len(entries) >= top_k:
            break
    return entries
