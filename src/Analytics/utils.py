"""Shared loading, normalisation, and parsing helpers for analytics scripts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


INSIGHT_COLUMNS = [
    "dataset",
    "graph_type",
    "method",
    "run_id",
    "graph_index",
    "label",
    "prediction_class",
    "prediction_confidence",
    "origin_confidence",
    "masked_confidence",
    "maskout_confidence",
    "sparsity",
    "minimal_coalition_size",
    "minimal_coalition_confidence",
    "insertion_auc",
    "num_nodes",
    "num_edges",
]


@dataclass
class InsightFrame:
    """Container bundling flattened insight data and token expansions."""

    data: pd.DataFrame
    token_frame: pd.DataFrame


def resolve_paths(raw_paths: Sequence[str]) -> List[Path]:
    """Resolve CLI-supplied paths and assert they exist."""
    paths = [Path(raw).expanduser().resolve() for raw in raw_paths]
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Insight file(s) missing:\n{joined}")
    return paths


def load_json_record(path: Path) -> List[dict]:
    """Load a single insight JSON file."""
    with path.open("r", encoding="utf-8") as handler:
        payload = json.load(handler)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of records in {path}")
        return payload


def flatten_records(records: Iterable[dict]) -> pd.DataFrame:
    """Normalise raw insight dictionaries into a dataframe."""
    flat_rows = []
    for record in records:
        row = {key: record.get(key) for key in INSIGHT_COLUMNS}

        row["accuracy"] = None
        label = record.get("label")
        prediction = record.get("prediction_class")
        if label is not None and prediction is not None:
            row["accuracy"] = int(label == prediction)

        structural = record.get("structural_metrics") or {}
        for key, value in structural.items():
            row[f"struct_{key}"] = value

        centrality = record.get("centrality_alignment") or {}
        for key, value in centrality.items():
            row[f"centrality_{key}"] = value

        insertion_curve = record.get("insertion_curve") or []
        if insertion_curve:
            row["insertion_curve_len"] = len(insertion_curve)
            row["insertion_curve_values"] = insertion_curve
        else:
            row["insertion_curve_values"] = []

        deletion_curve = record.get("deletion_curve") or []
        if deletion_curve:
            row["deletion_curve_len"] = len(deletion_curve)
            row["deletion_curve_values"] = deletion_curve
        else:
            row["deletion_curve_values"] = []

        row["top_tokens"] = record.get("top_tokens") or []
        row["minimal_coalition_tokens"] = record.get("minimal_coalition_tokens") or []

        row["fidelity_drop"] = _safe_diff(record.get("origin_confidence"), record.get("masked_confidence"))
        row["maskout_effect"] = _safe_diff(record.get("origin_confidence"), record.get("maskout_confidence"))
        row["graph_density"] = structural.get("density")
        row["boundary_edges"] = structural.get("boundary_edges")
        row["cut_ratio"] = structural.get("cut_ratio")
        row["components"] = structural.get("components")
        row["avg_shortest_path"] = structural.get("avg_shortest_path")

        row["explanation_size"] = record.get("minimal_coalition_size")
        row["top_token_count"] = len(row["top_tokens"])
        row["minimal_token_count"] = len(row["minimal_coalition_tokens"])

        flat_rows.append(row)
    frame = pd.DataFrame(flat_rows)
    return frame


def _safe_diff(a: float | None, b: float | None) -> float | None:
    """Return ``a - b`` while guarding against ``None`` and NaN."""
    if a is None or b is None:
        return None
    if any(math.isnan(val) for val in (a, b)):
        return None
    return float(a) - float(b)


def load_insights(paths: Sequence[str]) -> InsightFrame:
    """Load one or more insight files into an :class:`InsightFrame`."""
    resolved = resolve_paths(paths)
    payload: List[dict] = []
    for path in resolved:
        payload.extend(load_json_record(path))

    frame = flatten_records(payload)

    token_rows = []
    for idx, row in frame.iterrows():
        for token in row["top_tokens"]:
            token_rows.append({"graph_index": row["graph_index"], "token": token, "source": "top", "label": row["label"]})
        for token in row["minimal_coalition_tokens"]:
            token_rows.append({"graph_index": row["graph_index"], "token": token, "source": "minimal", "label": row["label"]})
    token_frame = pd.DataFrame(token_rows)
    return InsightFrame(data=frame, token_frame=token_frame)


def default_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a standard CLI parser for analytics modules."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "insight_paths",
        nargs="+",
        help="Insight JSON files to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics"),
        help="Directory where analysis artefacts will be written.",
    )
    parser.add_argument(
        "--group-key",
        default="label",
        help="Column to use when stratifying metrics (default: label).",
    )
    return parser
