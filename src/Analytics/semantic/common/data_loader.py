from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

from src.Insights.providers import GraphArtifactProvider


def ensure_safe_globals() -> None:
    try:
        from torch.serialization import add_safe_globals
        import torch_geometric.data.data as pyg_data
        import torch_geometric.data.storage as pyg_storage
    except Exception:
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
        pass


def load_json_records(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def load_prediction_lookup(paths: Sequence[Path]) -> Dict[int, int]:
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
            graph_index = item.get("graph_index")
            pred = item.get("prediction") or {}
            pred_class = None
            if isinstance(pred, dict):
                pred_class = pred.get("class")
            if pred_class is None:
                pred_class = item.get("prediction_class")
            if isinstance(graph_index, int) and isinstance(pred_class, int):
                mapping[graph_index] = pred_class
    return mapping


class GraphArtifactLoader:
    def __init__(self, nx_root: Path, pyg_root: Path) -> None:
        ensure_safe_globals()
        self.provider = GraphArtifactProvider(graph_root=nx_root, pyg_root=pyg_root, strict=True)

    def resolve(self, dataset: str, graph_type: str, graph_index: int, backbone: str, split: str):
        record = SimpleNamespace(
            dataset=dataset,
            graph_type=graph_type,
            graph_index=graph_index,
            extras={"backbone": backbone, "split": split},
        )
        return self.provider(record)


class SubgraphXResult:
    """Lightweight placeholder for SubgraphX pickled entries."""

    def __init__(self) -> None:
        self.graph_index: int = -1
        self.label: int | None = None
        self.explanation: List[dict] = []
        self.related_prediction: Dict[str, float] = {}
        self.num_nodes: int = 0
        self.num_edges: int = 0
        self.hyperparams: Dict[str, float] = {}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class _SubgraphXLoader:
    def __enter__(self):
        import sys
        import types

        self._modules = {}
        names = [
            "src",
            "src.explain",
            "src.explain.gnn",
            "src.explain.gnn.subgraphx",
            "src.explain.gnn.subgraphx.main",
        ]
        for name in names:
            self._modules[name] = sys.modules.get(name)

        src_module = types.ModuleType("src")
        explain_module = types.ModuleType("src.explain")
        gnn_module = types.ModuleType("src.explain.gnn")
        sub_module = types.ModuleType("src.explain.gnn.subgraphx")
        main_module = types.ModuleType("src.explain.gnn.subgraphx.main")
        main_module.SubgraphXResult = SubgraphXResult
        analytics_module = types.ModuleType("src.Analytics")
        semantic_module = types.ModuleType("src.Analytics.semantic")
        cli_module = types.ModuleType("src.Analytics.semantic.cli_pipeline")
        semantic_module.cli_pipeline = cli_module  # type: ignore[attr-defined]
        cli_module.SubgraphXResult = SubgraphXResult

        sys.modules.update(
            {
                "src": src_module,
                "src.explain": explain_module,
                "src.explain.gnn": gnn_module,
                "src.explain.gnn.subgraphx": sub_module,
                "src.explain.gnn.subgraphx.main": main_module,
                "src.Analytics": analytics_module,
                "src.Analytics.semantic": semantic_module,
                "src.Analytics.semantic.cli_pipeline": cli_module,
            }
        )

    def __exit__(self, exc_type, exc, tb):
        import sys

        for name, module in getattr(self, "_modules", {}).items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_subgraphx_results(path: Path) -> List[SubgraphXResult]:
    with _SubgraphXLoader():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected SubgraphX payload in {path}")
    return payload
