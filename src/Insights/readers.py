from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from .records import Coalition, ExplanationRecord, RelatedPrediction


def _resolve_path(raw: str, base_dir: Optional[Path]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def _read_coalitions(csv_path: Path) -> List[Coalition]:
    coalitions: List[Coalition] = []
    if not csv_path.exists():
        return coalitions

    with csv_path.open("r", encoding="utf-8") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            combination_raw = row.get("combination_id")
            size_raw = row.get("coalition_size")

            try:
                nodes = ast.literal_eval(row.get("coalition_nodes", "[]"))
            except (SyntaxError, ValueError):
                nodes = []
            try:
                mask = ast.literal_eval(row.get("coalition_binary", "[]"))
            except (SyntaxError, ValueError):
                mask = None

            coalition = Coalition.from_iterable(
                nodes=nodes,
                confidence=float(row.get("confidence", 0.0)),
                combination_id=int(combination_raw) if combination_raw not in (None, "", "None") else None,
                binary_mask=mask,
                size=int(size_raw) if size_raw not in (None, "", "None") else None,
            )

            coalitions.append(coalition)
    coalitions.sort(key=lambda c: (c.size, -c.confidence, c.combination_id or 0))
    return coalitions


def load_graphsvx_records(
    json_path: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
    coalition_base: Optional[Path] = None,
) -> List[ExplanationRecord]:
    """
    Load a collection of ExplanationRecord instances from a GraphSVX JSON dump.
    """
    json_path = Path(json_path)
    dataset = dataset or json_path.stem.replace("graphsvx_results_", "")

    with json_path.open("r", encoding="utf-8") as handler:
        raw_items = json.load(handler)

    records: List[ExplanationRecord] = []
    for item in raw_items:
        coalitions_path = item.get("coalitions_path")
        coalitions = []
        if coalitions_path:
            resolved = _resolve_path(coalitions_path, coalition_base or json_path.parent)
            coalitions = _read_coalitions(resolved)

        prediction = item.get("prediction") or {}
        record = ExplanationRecord(
            dataset=dataset,
            graph_type=graph_type,
            method="graphsvx",
            run_id=run_id,
            graph_index=item.get("graph_index"),
            label=item.get("label"),
            prediction_class=prediction.get("class"),
            prediction_confidence=prediction.get("confidence"),
            num_nodes=item.get("num_nodes"),
            num_edges=item.get("num_edges"),
            node_importance=item.get("node_importance"),
            top_nodes=tuple(item.get("top_nodes", [])),
            related_prediction=RelatedPrediction.from_mapping(item.get("related_prediction")),
            hyperparams=item.get("hyperparams") or {},
            coalitions=coalitions,
            extras={
                key: value
                for key, value in item.items()
                if key
                not in {
                    "graph_index",
                    "label",
                    "num_nodes",
                    "num_edges",
                    "prediction",
                    "node_importance",
                    "top_nodes",
                    "related_prediction",
                    "hyperparams",
                    "coalitions_path",
                }
            },
        )
        records.append(record)
    return records


def load_subgraphx_records(
    json_path: Path,
    *,
    dataset: Optional[str] = None,
    graph_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> List[ExplanationRecord]:
    """
    Load ExplanationRecord instances from a SubgraphX summary JSON.
    """
    json_path = Path(json_path)
    dataset = dataset or json_path.stem.replace("subgraphx_results_", "")

    with json_path.open("r", encoding="utf-8") as handler:
        raw_items = json.load(handler)

    records: List[ExplanationRecord] = []
    for item in raw_items:
        record = ExplanationRecord(
            dataset=dataset,
            graph_type=graph_type,
            method="subgraphx",
            run_id=run_id,
            graph_index=item.get("graph_index"),
            label=item.get("label"),
            prediction_class=None,
            prediction_confidence=None,
            num_nodes=item.get("num_nodes"),
            num_edges=item.get("num_edges"),
            related_prediction=RelatedPrediction.from_mapping(item.get("related_prediction")),
            hyperparams=item.get("hyperparams") or {},
            coalitions=[],
            extras={
                key: value
                for key, value in item.items()
                if key
                not in {
                    "graph_index",
                    "label",
                    "num_nodes",
                    "num_edges",
                    "related_prediction",
                    "hyperparams",
                }
            },
        )
        records.append(record)
    return records
