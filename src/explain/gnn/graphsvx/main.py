from __future__ import annotations

import json
import math
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.explain.gnn.config import (
    DEFAULT_GNN_ROOT,
    DEFAULT_GRAPH_DATA_ROOT,
    ExplainerRequest,
    GRAPH_SVX_DEFAULTS,
    GRAPH_SVX_PROFILES,
)
from src.explain.gnn.model_loader import load_gnn_model, load_graph_split


class GraphSHAPExplainer:
    """Minimal SHAP-style explainer inspired by the original GraphSVX implementation."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def analyze(
        self,
        data: Data,
        *,
        sampling_ratio: float,
        num_samples_override: Optional[int] = None,
        keep_special_tokens: bool = True,
    ) -> Dict[str, object]:
        if data is None:
            raise ValueError("Data must be provided")

        data = data.to(self.device)
        logits = self.model(data=data)
        probs = torch.softmax(logits, dim=1)
        predicted_class = int(torch.argmax(probs, dim=1))
        predicted_confidence = float(probs[0, predicted_class])

        num_nodes = data.x.size(0)
        special_indices = []
        if keep_special_tokens and num_nodes >= 2:
            special_indices = [0, num_nodes - 1]
        content_indices = [idx for idx in range(num_nodes) if idx not in special_indices]

        max_combinations = 2 ** max(len(content_indices), 0)
        if num_samples_override is not None:
            num_samples = min(int(num_samples_override), max_combinations)
        else:
            num_samples = max(1, int(math.ceil(max_combinations * sampling_ratio)))

        node_importance = torch.zeros(num_nodes, device=self.device)
        node_counter = torch.zeros(num_nodes, device=self.device)
        sampled_combinations: List[Dict[str, object]] = []

        for _ in range(num_samples):
            if not content_indices:
                selected_nodes: List[int] = []
            else:
                coalition_size = random.randint(0, len(content_indices))
                selected_nodes = random.sample(content_indices, coalition_size)

            mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            if special_indices:
                mask[special_indices] = True
            mask[selected_nodes] = True

            masked_data = data.clone()
            masked_data.x = masked_data.x.clone()
            masked_data.x[~mask] = 0

            masked_logits = self.model(data=masked_data)
            masked_probs = torch.softmax(masked_logits, dim=1)
            coalition_confidence = float(masked_probs[0, predicted_class])

            for node_idx in selected_nodes:
                without_nodes = [x for x in selected_nodes if x != node_idx]
                mask_without = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
                if special_indices:
                    mask_without[special_indices] = True
                mask_without[without_nodes] = True

                masked_without = data.clone()
                masked_without.x = masked_without.x.clone()
                masked_without.x[~mask_without] = 0

                output_without = self.model(data=masked_without)
                probs_without = torch.softmax(output_without, dim=1)
                confidence_without = float(probs_without[0, predicted_class])

                node_importance[node_idx] += coalition_confidence - confidence_without
                node_counter[node_idx] += 1

            sampled_combinations.append(
                {
                    "selected_nodes": selected_nodes,
                    "confidence": coalition_confidence,
                    "coalition_size": len(selected_nodes),
                }
            )

        for idx in range(num_nodes):
            if node_counter[idx] > 0:
                node_importance[idx] = node_importance[idx] / node_counter[idx]

        combinations_data = []
        for combo_id, combo in enumerate(sampled_combinations):
            coalition_binary = [
                1 if i in combo["selected_nodes"] else (1 if i in special_indices else 0)
                for i in range(num_nodes)
            ]
            combinations_data.append(
                {
                    "combination_id": combo_id,
                    "coalition_size": combo["coalition_size"],
                    "coalition_nodes": combo["selected_nodes"],
                    "coalition_binary": coalition_binary,
                    "confidence": combo["confidence"],
                }
            )

        df = pd.DataFrame(combinations_data)
        return {
            "node_importance": node_importance.detach().cpu(),
            "combinations": df,
            "original_prediction": {
                "class": predicted_class,
                "confidence": predicted_confidence,
            },
            "num_nodes": num_nodes,
            "num_edges": data.num_edges,
            "max_combinations": max_combinations,
            "sampled": num_samples,
        }


@dataclass
class GraphSVXResult:
    graph_index: int
    label: Optional[int]
    explanation: Dict[str, object]

    def to_json(self) -> Dict[str, object]:  # pragma: no cover - serialisation helper
        importance = self.explanation["node_importance"].tolist()
        return {
            "graph_index": self.graph_index,
            "label": self.label,
            "num_nodes": self.explanation["num_nodes"],
            "num_edges": self.explanation["num_edges"],
            "prediction": self.explanation["original_prediction"],
            "node_importance": importance,
            "coalitions_path": self.explanation.get("combinations_path"),
            "top_nodes": self.explanation.get("top_nodes", []),
        }


def _merge_hyperparams(overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    params = dict(GRAPH_SVX_DEFAULTS)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    params["sampling_ratio"] = float(params["sampling_ratio"])
    override = params.get("num_samples_override")
    params["num_samples_override"] = int(override) if override else None
    params["keep_special_tokens"] = bool(int(params.get("keep_special_tokens", 1)))
    params["top_k_nodes"] = int(params.get("top_k_nodes", 10))
    return params


def _make_slug(request: ExplainerRequest) -> str:
    dataset = str(request.dataset_subpath) if request.dataset_subpath != Path('.') else request.dataset
    parts = [request.backbone, dataset, request.graph_type, request.split]
    safe = [p.replace("/", "-") for p in parts if p]
    return "_".join(safe)


def explain_request(
    request: ExplainerRequest,
    *,
    progress: bool = True,
    hyperparams: Optional[Dict[str, float]] = None,
) -> Tuple[List[GraphSVXResult], Path, Path, Optional[Path]]:
    device = torch.device(request.device) if request.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    updated_request = ExplainerRequest(
        dataset=request.dataset,
        graph_type=request.graph_type,
        backbone=request.backbone,
        split=request.split,
        method="graphsvx",
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

    explainer = GraphSHAPExplainer(model=model, device=device)
    profile_overrides = GRAPH_SVX_PROFILES.get((updated_request.profile or "").lower(), {})
    combined_overrides: Dict[str, float] = {}
    combined_overrides.update(profile_overrides)
    combined_overrides.update(updated_request.hyperparams or {})
    if hyperparams:
        combined_overrides.update(hyperparams)

    params = _merge_hyperparams(combined_overrides)

    artifact_dir = run_dir / "explanations" / "graphsvx" / _make_slug(updated_request)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        desc = f"GraphSVX[{updated_request.shard_index + 1}/{updated_request.num_shards}]"
        iterable = tqdm(
            loader,
            desc=desc,
            leave=False,
            position=updated_request.shard_index,
            dynamic_ncols=True,
        )
    else:
        iterable = loader
    results: List[GraphSVXResult] = []
    for index, batch in enumerate(iterable):
        data: Data = batch.to(device)
        label = int(data.y.item()) if hasattr(data, "y") and data.y is not None else None
        explanation = explainer.analyze(
            data,
            sampling_ratio=params["sampling_ratio"],
            num_samples_override=params["num_samples_override"],
            keep_special_tokens=params["keep_special_tokens"],
        )

        combinations = explanation.get("combinations")
        if isinstance(combinations, pd.DataFrame):
            combos_path = artifact_dir / f"graph_{index:04d}_coalitions.csv"
            combinations.to_csv(combos_path, index=False)
            explanation["combinations_path"] = str(combos_path)
            explanation.pop("combinations")

        importance_tensor: torch.Tensor = explanation["node_importance"]
        ordered = torch.argsort(importance_tensor, descending=True)
        explanation["top_nodes"] = ordered[: params["top_k_nodes"]].tolist()
        results.append(GraphSVXResult(graph_index=index, label=label, explanation=explanation))

    summary = {
        "method": "graphsvx",
        "dataset": updated_request.dataset,
        "graph_type": updated_request.graph_type,
        "split": updated_request.split,
        "backbone": updated_request.backbone,
        "num_graphs": len(results),
        "hyperparams": params,
        "graphs": [entry.to_json() for entry in results],
    }

    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    raw_path: Optional[Path] = artifact_dir / "results.pkl"
    try:
        with raw_path.open("wb") as handle:
            pickle.dump(results, handle)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not pickle GraphSVX results: {exc}")
        raw_path = None

    return results, artifact_dir, summary_path, raw_path


def _env_request() -> ExplainerRequest:
    dataset = os.getenv("GRAPHTEXT_DATASET", "ag_news")
    graph_type = os.getenv("GRAPHTEXT_GRAPH_TYPE", "skipgrams")
    backbone = os.getenv("GRAPHTEXT_BACKBONE", "SetFit")
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
    output_path = Path("graphsvx_results.json")
    output_path.write_text(json.dumps([entry.to_json() for entry in results], indent=2))
    print(f"Saved GraphSVX explanations to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    main()
