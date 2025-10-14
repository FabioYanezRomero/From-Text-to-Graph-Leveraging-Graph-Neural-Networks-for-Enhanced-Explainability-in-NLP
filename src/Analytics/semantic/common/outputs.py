from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import GraphSemanticSummary


def summaries_to_frame(summaries: List[GraphSemanticSummary], dataset: str, graph_type: str) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        if summary.label is None or summary.prediction_class is None:
            is_correct = None
        else:
            is_correct = bool(summary.label == summary.prediction_class)
        base = {
            "graph_index": summary.graph_index,
            "label": summary.label,
            "prediction_class": summary.prediction_class,
            "prediction_confidence": summary.prediction_confidence,
            "explanation_size": summary.explanation_size,
            "unique_token_count": summary.unique_token_count,
            "semantic_density": summary.semantic_density,
            "is_correct": is_correct,
            "dataset": dataset,
            "graph_type": graph_type,
        }
        base.update(summary.graph_metadata)
        base.update(summary.extras)
        rows.append(base)
    return pd.DataFrame(rows)


def tokens_to_frame(
    summaries: List[GraphSemanticSummary],
    dataset: str,
    graph_type: str,
) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        is_correct: Optional[bool]
        if summary.label is None or summary.prediction_class is None:
            is_correct = None
        else:
            is_correct = bool(summary.label == summary.prediction_class)
        for rank, attr in enumerate(summary.selected_tokens, start=1):
            rows.append(
                {
                    "graph_index": summary.graph_index,
                    "token": attr.token,
                    "score": attr.score,
                    "position": attr.position,
                    "rank": rank,
                    "label": summary.label,
                    "prediction_class": summary.prediction_class,
                    "is_correct": is_correct,
                    "dataset": dataset,
                    "graph_type": graph_type,
                }
            )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
