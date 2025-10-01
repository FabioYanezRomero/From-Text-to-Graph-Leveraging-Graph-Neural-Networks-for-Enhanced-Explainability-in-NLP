from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from src.explain.gnn.config import (
    DEFAULT_GNN_ROOT,
    DEFAULT_GRAPH_DATA_ROOT,
    ExplainerRequest,
    SUBGRAPHX_DEFAULTS,
    SUBGRAPHX_PROFILES,
)
from src.explain.gnn.model_loader import load_gnn_model, load_graph_split
from .custom_subgraphx import CustomSubgraphX


class MarginalSubgraphDataset(Dataset):
    """Compatibility shim for the DIG SubgraphX sampler."""

    def __init__(self, data: Data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device
        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask, dtype=torch.float32, device=self.device)
        self.include_mask = torch.tensor(include_mask, dtype=torch.float32, device=self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self) -> int:  # pragma: no cover - simple proxy method
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):  # pragma: no cover - DIG internal usage
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(
            self.X, self.edge_index, self.exclude_mask[idx]
        )
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(
            self.X, self.edge_index, self.include_mask[idx]
        )
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


# Monkey patch DIG so our dataset is used when constructing coalitions.
import dig.xgraph.method.shapley  # type: ignore  # pylint: disable=import-error

dig.xgraph.method.shapley.MarginalSubgraphDataset = MarginalSubgraphDataset  # type: ignore[attr-defined]


class UniversalDataModelWrapper(torch.nn.Module):
    """Normalise the model forward signature for DIG's expectations."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        if len(args) == 1 and isinstance(args[0], Data):
            data = args[0]
            return self.model(data=data)
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            x = args[0]
            edge_index = args[1]
            batch = None
            if len(args) >= 3:
                batch = args[2]
            elif "batch" in kwargs:
                batch = kwargs["batch"]
            return self.model(x=x, edge_index=edge_index, batch=batch)
        if "data" in kwargs:
            return self.model(data=kwargs["data"])
        return self.model(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


@dataclass
class SubgraphXResult:
    """Holds per-graph explanation artefacts."""

    graph_index: int
    label: Optional[int]
    explanation: object
    related_prediction: Dict[str, float]
    num_nodes: int
    num_edges: int

    def to_json(self) -> Dict[str, object]:  # pragma: no cover - serialisation helper
        return {
            "graph_index": self.graph_index,
            "label": self.label,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "related_prediction": self.related_prediction,
        }


def _merge_hyperparams(overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    params = dict(SUBGRAPHX_DEFAULTS)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    # Cast to expected types for DIG
    int_keys = {"num_hops", "rollout", "min_atoms", "expand_atoms", "local_radius", "sample_num", "max_nodes"}
    for key in int_keys:
        params[key] = int(params[key])
    params["c_puct"] = float(params["c_puct"])
    return params


def _make_slug(request: ExplainerRequest) -> str:
    dataset = str(request.dataset_subpath) if request.dataset_subpath != Path('.') else request.dataset
    parts = [request.backbone, dataset, request.graph_type, request.split]
    safe = [p.replace("/", "-") for p in parts if p]
    return "_".join(safe)


def _prepare_explainer(
    wrapper: UniversalDataModelWrapper,
    args: Dict[str, object],
    save_dir: Path,
    hyperparams: Dict[str, float],
) -> CustomSubgraphX:
    num_classes = int(args.get("num_classes") or args.get("output_dim", 2))
    save_dir.mkdir(parents=True, exist_ok=True)

    return CustomSubgraphX(
        model=wrapper,
        num_classes=num_classes,
        device=wrapper.device,
        num_hops=hyperparams["num_hops"],
        rollout=hyperparams["rollout"],
        min_atoms=hyperparams["min_atoms"],
        c_puct=hyperparams["c_puct"],
        expand_atoms=hyperparams["expand_atoms"],
        local_radius=hyperparams["local_radius"],
        sample_num=hyperparams["sample_num"],
        save_dir=str(save_dir),
    )


def explain_request(
    request: ExplainerRequest,
    *,
    progress: bool = True,
    hyperparams: Optional[Dict[str, float]] = None,
) -> Tuple[List[SubgraphXResult], Path, Path, Optional[Path]]:
    """Run SubgraphX on the dataset implied by the request."""

    device = torch.device(request.device) if request.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    updated_request = ExplainerRequest(
        dataset=request.dataset,
        graph_type=request.graph_type,
        backbone=request.backbone,
        split=request.split,
        method="subgraphx",
        device=str(device),
        checkpoint_name=request.checkpoint_name,
        gnn_root=request.gnn_root,
        graph_data_root=request.graph_data_root,
        hyperparams=request.hyperparams,
        profile=request.profile,
        num_shards=request.num_shards,
        shard_index=request.shard_index,
    )

    dataset, loader = load_graph_split(updated_request, batch_size=1, shuffle=False)
    model, train_args, run_dir = load_gnn_model(updated_request, dataset=dataset)

    wrapper = UniversalDataModelWrapper(model)
    profile_overrides = SUBGRAPHX_PROFILES.get((updated_request.profile or "").lower(), {})
    combined_overrides: Dict[str, float] = {}
    combined_overrides.update(profile_overrides)
    combined_overrides.update(updated_request.hyperparams or {})
    if hyperparams:
        combined_overrides.update(hyperparams)

    params = _merge_hyperparams(combined_overrides)
    artifact_dir = run_dir / "explanations" / "subgraphx" / _make_slug(updated_request)
    explainer = _prepare_explainer(wrapper, train_args, artifact_dir, params)

    results: List[SubgraphXResult] = []
    if progress:
        desc = f"SubgraphX[{updated_request.shard_index + 1}/{updated_request.num_shards}]"
        iterable = tqdm(
            loader,
            desc=desc,
            leave=False,
            position=updated_request.shard_index,
            dynamic_ncols=True,
        )
    else:
        iterable = loader

    for index, batch in enumerate(iterable):
        data: Data = batch.to(wrapper.device)
        label = int(data.y.item()) if hasattr(data, "y") and data.y is not None else None
        explanation, related_pred = explainer.explain(
            x=data.x,
            edge_index=data.edge_index,
            label=label,
            max_nodes=params["max_nodes"],
        )
        results.append(
            SubgraphXResult(
                graph_index=index,
                label=label,
                explanation=explanation,
                related_prediction=related_pred,
                num_nodes=data.num_nodes,
                num_edges=data.num_edges,
            )
        )

    summary = {
        "method": "subgraphx",
        "dataset": updated_request.dataset,
        "graph_type": updated_request.graph_type,
        "split": updated_request.split,
        "backbone": updated_request.backbone,
        "num_graphs": len(results),
        "hyperparams": params,
        "graphs": [entry.to_json() for entry in results],
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    raw_path: Optional[Path] = artifact_dir / "results.pkl"
    try:
        with raw_path.open("wb") as handle:
            pickle.dump(results, handle)
    except Exception as exc:  # pragma: no cover - safeguard for unexpected objects
        warnings.warn(f"Could not pickle SubgraphX results: {exc}")
        raw_path = None

    return results, artifact_dir, summary_path, raw_path


def _env_request() -> ExplainerRequest:
    dataset = os.getenv("GRAPHTEXT_DATASET", "sst2")
    graph_type = os.getenv("GRAPHTEXT_GRAPH_TYPE", "syntactic")
    backbone = os.getenv("GRAPHTEXT_BACKBONE", "stanfordnlp")
    split = os.getenv("GRAPHTEXT_SPLIT", "validation")
    device = os.getenv("GRAPHTEXT_DEVICE")
    gnn_root = Path(os.getenv("GRAPHTEXT_GNN_ROOT", str(DEFAULT_GNN_ROOT)))
    data_root = Path(os.getenv("GRAPHTEXT_GRAPH_ROOT", str(DEFAULT_GRAPH_DATA_ROOT)))

    return ExplainerRequest(
        dataset=dataset,
        graph_type=graph_type,
        backbone=backbone,
        split=split,
        device=device,
        gnn_root=gnn_root,
        graph_data_root=data_root,
    )


def main() -> None:  # pragma: no cover - CLI helper
    request = _env_request()
    results = explain_request(request)
    output_path = Path("subgraphx_results.json")
    output_path.write_text(json.dumps([entry.to_json() for entry in results], indent=2))
    print(f"Saved SubgraphX explanations to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    main()
