from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import warnings
from dataclasses import dataclass, replace
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
from .hyperparam_advisor import (
    ArchitectureSpec,
    GraphContext,
    GraphSVXHyperparameterAdvisor,
    FLOAT_PARAM_KEYS,
    INT_PARAM_KEYS,
    BOOL_PARAM_KEYS,
)


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


def _build_node_masks(
    num_nodes: int,
    top_nodes: List[int],
    *,
    keep_special_tokens: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boolean masks for keeping and dropping the important nodes."""

    mask_keep = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if top_nodes:
        mask_keep[top_nodes] = True
    if keep_special_tokens and num_nodes >= 2:
        mask_keep[0] = True
        mask_keep[-1] = True
    if mask_keep.sum() == 0:
        mask_keep[0] = True

    mask_drop = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if top_nodes:
        mask_drop[top_nodes] = False
    if keep_special_tokens and num_nodes >= 2:
        mask_drop[0] = True
        mask_drop[-1] = True
    if not mask_drop.any():
        mask_drop[:] = True

    return mask_keep, mask_drop


def _confidence_with_mask(
    model: torch.nn.Module,
    data: Data,
    mask: torch.Tensor,
    *,
    device: torch.device,
    predicted_class: int,
) -> float:
    masked = data.clone().to(device)
    masked.x = masked.x.clone()
    masked.x[~mask] = 0
    logits = model(data=masked)
    probs = torch.softmax(logits, dim=1)
    return float(probs[0, predicted_class])


@dataclass
class GraphSVXResult:
    graph_index: int
    label: Optional[int]
    explanation: Dict[str, object]
    hyperparams: Dict[str, object]
    source: str

    # Populated per-graph confidence metrics matching SubgraphX semantics.
    related_prediction: Dict[str, float]

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
            "hyperparams": dict(self.hyperparams),
            "hyperparam_source": self.source,
            "related_prediction": dict(self.related_prediction),
        }

def _make_slug(request: ExplainerRequest) -> str:
    dataset = str(request.dataset_subpath) if request.dataset_subpath != Path('.') else request.dataset
    parts = [request.backbone, dataset, request.graph_type, request.split]
    if getattr(request, "num_shards", 1) > 1:
        parts.append(f"shard{request.shard_index + 1}of{request.num_shards}")
    safe = [p.replace("/", "-") for p in parts if p]
    return "_".join(safe)


def _extract_architecture_spec(model: torch.nn.Module, args: Dict[str, object]) -> ArchitectureSpec:
    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    num_layers = _coerce_int(args.get("num_layers") or args.get("gnn_layers") or getattr(model, "num_layers", None), 2)
    module = str(args.get("module") or getattr(model, "module", "GCNConv") or "GCNConv")
    heads = _coerce_int(args.get("heads") or args.get("attention_heads") or getattr(model, "heads", None), 1)
    return ArchitectureSpec(num_layers=num_layers, module=module, heads=heads)


def _make_context(request: ExplainerRequest) -> GraphContext:
    return GraphContext(dataset=str(request.dataset), graph_type=str(request.graph_type), backbone=str(request.backbone))


def collect_hyperparams(
    request: ExplainerRequest,
    *,
    progress: bool = True,
    output_path: Optional[Path] = None,
    max_graphs: Optional[int] = None,
) -> Path:
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

    profile_overrides = GRAPH_SVX_PROFILES.get((updated_request.profile or "").lower(), {})
    locked_overrides: Dict[str, float] = {}
    locked_overrides.update(profile_overrides)
    locked_overrides.update(updated_request.hyperparams or {})

    architecture_spec = _extract_architecture_spec(model, train_args)
    advisor = GraphSVXHyperparameterAdvisor(
        architecture=architecture_spec,
        context=_make_context(updated_request),
        locked_params=locked_overrides,
    )

    if output_path is not None:
        artifact_dir = output_path.parent
    else:
        artifact_dir = run_dir / "explanations" / "graphsvx" / (_make_slug(updated_request) + "_hyperparams")
        output_path = artifact_dir / "hyperparams.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    total = len(loader)
    if max_graphs is not None and max_graphs > 0:
        total = min(total, max_graphs)
    if progress:
        desc = f"CollectGraphSVX[{updated_request.shard_index + 1}/{updated_request.num_shards}]"
        iterable = tqdm(
            loader,
            desc=desc,
            leave=False,
            position=updated_request.shard_index,
            dynamic_ncols=True,
            total=total,
        )
    else:
        iterable = loader

    per_graph: List[Dict[str, object]] = []
    for index, batch in enumerate(iterable):
        if max_graphs is not None and index >= max_graphs:
            break
        data: Data = batch
        params = advisor.suggest(data)
        per_graph.append(
            {
                "graph_index": index,
                "num_nodes": int(getattr(data, "num_nodes", 0)),
                "num_edges": int(getattr(data, "num_edges", 0)),
                "hyperparams": dict(params),
            }
        )

    payload = {
        "method": "graphsvx",
        "dataset": updated_request.dataset,
        "graph_type": updated_request.graph_type,
        "split": updated_request.split,
        "backbone": updated_request.backbone,
        "num_shards": updated_request.num_shards,
        "shard_index": updated_request.shard_index,
        "base_defaults": dict(advisor.base_defaults),
        "locked_overrides": dict(advisor.locked_params),
        "per_graph": per_graph,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _load_precomputed_hparams(path: Path) -> Tuple[Dict[int, Dict[str, float]], Dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    entries = payload.get("per_graph")
    if entries is None:
        raise ValueError("Precomputed hyperparameter file missing 'per_graph'")

    mapping: Dict[int, Dict[str, float]] = {}
    for entry in entries:
        if "graph_index" not in entry:
            raise ValueError("Each entry must include 'graph_index'")
        index = int(entry["graph_index"])
        params = entry.get("hyperparams")
        if params is None:
            params = {k: v for k, v in entry.items() if k != "graph_index"}
        casted: Dict[str, float] = {}
        for key, value in params.items():
            if key in INT_PARAM_KEYS and value is not None:
                casted[key] = int(value)
            elif key in FLOAT_PARAM_KEYS:
                casted[key] = float(value)
            elif key in BOOL_PARAM_KEYS:
                casted[key] = bool(value)
            else:
                casted[key] = value
        mapping[index] = casted
    return mapping, payload


def explain_request(
    request: ExplainerRequest,
    *,
    progress: bool = True,
    hyperparams: Optional[Dict[str, float]] = None,
    precomputed_hparams: Optional[Dict[int, Dict[str, float]]] = None,
    precomputed_source: Optional[Path] = None,
    max_graphs: Optional[int] = None,
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

    profile_overrides = GRAPH_SVX_PROFILES.get((updated_request.profile or "").lower(), {})
    combined_overrides: Dict[str, float] = {}
    combined_overrides.update(profile_overrides)
    combined_overrides.update(updated_request.hyperparams or {})
    if hyperparams:
        combined_overrides.update(hyperparams)

    architecture_spec = _extract_architecture_spec(model, train_args)
    advisor = GraphSVXHyperparameterAdvisor(
        architecture=architecture_spec,
        context=_make_context(updated_request),
        locked_params=combined_overrides,
    )

    artifact_dir = run_dir / "explanations" / "graphsvx" / _make_slug(updated_request)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    precomputed_lookup = precomputed_hparams or {}
    explainer = GraphSHAPExplainer(model=model, device=device)

    total = len(loader)
    if max_graphs is not None and max_graphs > 0:
        total = min(total, max_graphs)
    if progress:
        desc = f"GraphSVX[{updated_request.shard_index + 1}/{updated_request.num_shards}]"
        iterable = tqdm(
            loader,
            desc=desc,
            leave=False,
            position=updated_request.shard_index,
            dynamic_ncols=True,
            total=total,
        )
    else:
        iterable = loader

    results: List[GraphSVXResult] = []
    per_graph_hparams: List[Dict[str, object]] = []
    for index, batch in enumerate(iterable):
        if max_graphs is not None and index >= max_graphs:
            break
        data: Data = batch.to(device)
        label = int(data.y.item()) if hasattr(data, "y") and data.y is not None else None
        source = "advisor"
        candidate = precomputed_lookup.get(index)
        if candidate is not None:
            graph_params = advisor.sanitise_for_graph(candidate, batch)
            source = "precomputed"
        else:
            graph_params = advisor.suggest(batch)

        explanation = explainer.analyze(
            data,
            sampling_ratio=graph_params["sampling_ratio"],
            num_samples_override=graph_params.get("num_samples_override"),
            keep_special_tokens=graph_params["keep_special_tokens"],
        )

        combinations = explanation.get("combinations")
        if isinstance(combinations, pd.DataFrame):
            combos_path = artifact_dir / f"graph_{index:04d}_coalitions.csv"
            combinations.to_csv(combos_path, index=False)
            explanation["combinations_path"] = str(combos_path)
            explanation.pop("combinations")

        importance_tensor: torch.Tensor = explanation["node_importance"]
        ordered = torch.argsort(importance_tensor, descending=True)
        explanation["top_nodes"] = ordered[: graph_params["top_k_nodes"]].tolist()
        top_nodes_list = explanation["top_nodes"]

        mask_keep, mask_drop = _build_node_masks(
            data.num_nodes,
            top_nodes_list,
            keep_special_tokens=graph_params["keep_special_tokens"],
            device=device,
        )

        predicted_class = explanation["original_prediction"]["class"]
        masked_conf = _confidence_with_mask(
            explainer.model,
            data,
            mask_keep,
            device=device,
            predicted_class=predicted_class,
        )
        maskout_conf = _confidence_with_mask(
            explainer.model,
            data,
            mask_drop,
            device=device,
            predicted_class=predicted_class,
        )
        kept_ratio = float(mask_keep.sum().item() / max(data.num_nodes, 1))
        related_pred = {
            "masked": masked_conf,
            "maskout": maskout_conf,
            "origin": explanation["original_prediction"]["confidence"],
            "sparsity": kept_ratio,
        }
        explanation["related_prediction"] = related_pred

        results.append(
            GraphSVXResult(
                graph_index=index,
                label=label,
                explanation=explanation,
                hyperparams=dict(graph_params),
                source=source,
                related_prediction=related_pred,
            )
        )
        per_graph_hparams.append({"graph_index": index, "source": source, **graph_params})

    summary = {
        "method": "graphsvx",
        "dataset": updated_request.dataset,
        "graph_type": updated_request.graph_type,
        "split": updated_request.split,
        "backbone": updated_request.backbone,
        "num_shards": updated_request.num_shards,
        "shard_index": updated_request.shard_index,
        "num_graphs": len(results),
        "hyperparams": {
            "base_defaults": dict(advisor.base_defaults),
            "locked_overrides": dict(advisor.locked_params),
            "precomputed_source": str(precomputed_source) if precomputed_source else None,
            "per_graph": per_graph_hparams,
        },
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


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="GraphSVX Explainability Runner")
    parser.add_argument("--collect-only", action="store_true", help="Only collect hyperparameters and exit.")
    parser.add_argument("--hyperparams-out", type=Path, help="Destination for collected hyperparameters JSON.")
    parser.add_argument("--hyperparams-in", type=Path, help="Reuse precomputed hyperparameters from JSON.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--dataset", type=str, help="Dataset identifier passed to the explainer.")
    parser.add_argument("--graph-type", type=str, help="Graph type within the dataset.")
    parser.add_argument("--backbone", type=str, help="Backbone model name.")
    parser.add_argument("--split", type=str, help="Dataset split to explain.")
    parser.add_argument("--device", type=str, help="Torch device (cpu/cuda:0).")
    parser.add_argument("--gnn-root", type=Path, help="Override path to trained GNN checkpoints.")
    parser.add_argument("--graph-data-root", type=Path, help="Override path to PyG graphs.")
    parser.add_argument("--checkpoint-name", type=str, help="Checkpoint filename to load.")
    parser.add_argument("--profile", type=str, help="GraphSVX profile key (fast/quality).")
    parser.add_argument("--num-shards", type=int, help="Total number of shards for parallel execution.")
    parser.add_argument("--shard-index", type=int, help="Index of this shard (0-based).")
    parser.add_argument("--max-graphs", type=int, help="Limit the number of graphs processed (for smoke tests).")
    args = parser.parse_args(argv)

    request = _env_request()
    overrides: Dict[str, object] = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.graph_type:
        overrides["graph_type"] = args.graph_type
    if args.backbone:
        overrides["backbone"] = args.backbone
    if args.split:
        overrides["split"] = args.split
    if args.device:
        overrides["device"] = args.device
    if args.gnn_root:
        overrides["gnn_root"] = args.gnn_root
    if args.graph_data_root:
        overrides["graph_data_root"] = args.graph_data_root
    if args.checkpoint_name:
        overrides["checkpoint_name"] = args.checkpoint_name
    if args.profile:
        overrides["profile"] = args.profile
    if args.num_shards is not None:
        overrides["num_shards"] = args.num_shards
    if args.shard_index is not None:
        overrides["shard_index"] = args.shard_index
    max_graphs = args.max_graphs if args.max_graphs and args.max_graphs > 0 else None

    if overrides:
        request = replace(request, **overrides)

    if args.collect_only:
        output_path = collect_hyperparams(
            request,
            progress=not args.no_progress,
            output_path=args.hyperparams_out,
            max_graphs=max_graphs,
        )
        print(f"Saved GraphSVX hyperparameters to {output_path}")
        return

    precomputed_lookup: Optional[Dict[int, Dict[str, float]]] = None
    precomputed_source: Optional[Path] = None
    if args.hyperparams_in:
        precomputed_lookup, payload = _load_precomputed_hparams(args.hyperparams_in)
        precomputed_source = Path(args.hyperparams_in).resolve()
        mismatches: List[str] = []
        for key in ("dataset", "graph_type", "split", "backbone"):
            expected = getattr(request, key)
            observed = payload.get(key)
            if observed is not None and str(observed) != str(expected):
                mismatches.append(f"{key}={observed} (expected {expected})")
        if mismatches:
            warnings.warn(
                "Precomputed hyperparameters metadata differs from request: "
                + "; ".join(mismatches)
            )

    results, artifact_dir, summary_path, _ = explain_request(
        request,
        progress=not args.no_progress,
        precomputed_hparams=precomputed_lookup,
        precomputed_source=precomputed_source,
        max_graphs=max_graphs,
    )

    output_path = Path("graphsvx_results.json")
    output_path.write_text(json.dumps([entry.to_json() for entry in results], indent=2))
    print(f"Saved GraphSVX explanations to {output_path}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    main()
