from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple


@dataclass(slots=True)
class GraphSVXConfig:
    path: Path
    dataset: str
    graph_type: str
    split: str
    backbone: str
    top_k: int = 12
    threshold: Optional[float] = None


@dataclass(slots=True)
class SubgraphXConfig:
    paths: List[Path]
    dataset: str
    graph_type: str
    split: str
    backbone: str
    top_k: int = 12
    threshold: Optional[float] = None
    prediction_lookup: List[Path] | None = None


@dataclass(slots=True)
class SemanticConfig:
    graphsvx: List[GraphSVXConfig]
    subgraphx: List[SubgraphXConfig]
    nx_root: Path
    pyg_root: Path
    stopwords: Set[str]


def _default_stopwords() -> Set[str]:
    words = {
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
        "it's",
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
    contractions = {
        "'s",
        "’s",
        "n't",
        "n’t",
        "'re",
        "’re",
        "'ve",
        "’ve",
        "'ll",
        "’ll",
        "'d",
        "’d",
        "'m",
        "’m",
        "s",
        "t",
        "m",
        "re",
        "ve",
        "ll",
        "d",
        "nt",
        "wo",
        ";s",
        ";t",
        ";d",
        ";re",
        ";ve",
        ";ll",
    }
    html_residues = {"quot", "amp", "apos"}
    punct = set(".,:;!?()[]{}'\"`“”’—–-…/\\|+*=<>#@&%$^~ ")
    words.update({"``", "''", "--", "\\a"})
    return words.union(contractions, html_residues, punct)


def load_config(path: Path) -> SemanticConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nx_root = Path(payload.get("graph_roots", {}).get("nx", "outputs/graphs")).resolve()
    pyg_root = Path(payload.get("graph_roots", {}).get("pyg", "outputs/pyg_graphs")).resolve()

    def _graphsvx_entries() -> List[GraphSVXConfig]:
        entries: List[GraphSVXConfig] = []
        for item in payload.get("graphsvx", []) or []:
            entries.append(
                GraphSVXConfig(
                    path=Path(item["path"]).resolve(),
                    dataset=item["dataset"],
                    graph_type=item["graph_type"],
                    split=item["split"],
                    backbone=item["backbone"],
                    top_k=item.get("top_k", 12),
                    threshold=item.get("threshold"),
                )
            )
        return entries

    def _subgraphx_entries() -> List[SubgraphXConfig]:
        entries: List[SubgraphXConfig] = []
        for item in payload.get("subgraphx", []) or []:
            entries.append(
                SubgraphXConfig(
                    paths=[Path(p).resolve() for p in item["paths"]],
                    dataset=item["dataset"],
                    graph_type=item["graph_type"],
                    split=item["split"],
                    backbone=item["backbone"],
                    top_k=item.get("top_k", 12),
                    threshold=item.get("threshold"),
                    prediction_lookup=[Path(p).resolve() for p in item.get("prediction_lookup", [])] or None,
                )
            )
        return entries

    stopwords: Set[str] = _default_stopwords()
    for token in payload.get("stopwords", []) or []:
        if isinstance(token, str):
            stopwords.add(token.lower())
    for file_path in payload.get("stopwords_files", []) or []:
        try:
            content = Path(file_path).expanduser().read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    stopwords.add(line.lower())
        except FileNotFoundError:
            continue

    return SemanticConfig(
        graphsvx=_graphsvx_entries(),
        subgraphx=_subgraphx_entries(),
        nx_root=nx_root,
        pyg_root=pyg_root,
        stopwords=stopwords,
    )
