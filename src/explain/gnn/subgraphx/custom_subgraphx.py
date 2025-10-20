import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data

from dig.xgraph.method import SubgraphX


class CustomSubgraphX(SubgraphX):
    """Augmented SubgraphX explainer that captures probability distributions."""

    def __init__(self, *args, value_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_value_func = value_func
        if value_func is not None:
            warnings.warn(
                "Custom value_func injected into SubgraphX. This will override internal model calls."
            )

    def explain(  # type: ignore[override]
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: Optional[int],
        max_nodes: int = 5,
        node_idx: Optional[int] = None,
        saved_MCTSInfo_list: Optional[List[List]] = None,
        **kwargs,
    ):
        call_kwargs: Dict[str, object] = {
            "x": x,
            "edge_index": edge_index,
            "label": label,
            "max_nodes": max_nodes,
            "node_idx": node_idx,
            "saved_MCTSInfo_list": saved_MCTSInfo_list,
        }
        call_kwargs.update(kwargs)
        if self._custom_value_func is not None:
            call_kwargs["value_func"] = self._custom_value_func

        def _filtered(include_value_func: bool) -> Dict[str, object]:
            filtered: Dict[str, object] = {}
            for key, value in call_kwargs.items():
                if key == "value_func" and not include_value_func:
                    continue
                if key in {"x", "edge_index", "label", "value_func"} or value is not None:
                    filtered[key] = value
            return filtered

        try:
            results, related_pred = super().explain(**_filtered(include_value_func=True))
        except TypeError as exc:
            if "value_func" in call_kwargs and "unexpected keyword argument 'value_func'" in str(exc):
                results, related_pred = super().explain(**_filtered(include_value_func=False))
            else:
                raise

        try:
            self._augment_related_prediction(results, related_pred, x, edge_index, label)
        except Exception as exc:  # pragma: no cover - defensive guard
            warnings.warn(f"Failed to augment SubgraphX probabilities: {exc}")
        return results, related_pred

    def _augment_related_prediction(
        self,
        results,
        related_pred: Dict[str, object],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: Optional[int],
    ) -> None:
        if not isinstance(related_pred, dict):
            return

        origin_probs = self._predict_probs_from_inputs(x, edge_index)
        origin_distribution = origin_probs.detach().cpu().tolist()
        if origin_distribution:
            related_pred["origin_distribution"] = [float(v) for v in origin_distribution]

        target_index = self._resolve_target_index(label, origin_probs)
        if target_index is not None and 0 <= target_index < len(origin_distribution):
            related_pred["origin"] = float(origin_distribution[target_index])

        masked_distribution, maskout_distribution = self._compute_masked_distributions(results)

        if masked_distribution is not None:
            related_pred["masked_distribution"] = [float(v) for v in masked_distribution]
            if target_index is not None and 0 <= target_index < len(masked_distribution):
                related_pred["masked"] = float(masked_distribution[target_index])
        else:
            related_pred.setdefault("masked_distribution", None)

        if maskout_distribution is not None:
            related_pred["maskout_distribution"] = [float(v) for v in maskout_distribution]
            if target_index is not None and 0 <= target_index < len(maskout_distribution):
                related_pred["maskout"] = float(maskout_distribution[target_index])
        else:
            related_pred.setdefault("maskout_distribution", None)

    def _resolve_target_index(
        self, label: Optional[int], origin_probs: torch.Tensor
    ) -> Optional[int]:
        if label is not None:
            try:
                idx = int(label)
                if 0 <= idx < origin_probs.numel():
                    return idx
            except (TypeError, ValueError):
                pass
        if origin_probs.numel() == 0:
            return None
        return int(torch.argmax(origin_probs).item())

    def _compute_masked_distributions(
        self, results
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        base_data, coalition = self._extract_primary_data(results)
        if base_data is None or not coalition:
            return None, None

        num_nodes = base_data.num_nodes or base_data.x.size(0)
        if num_nodes <= 0:
            return None, None

        valid_nodes = sorted({idx for idx in coalition if 0 <= idx < num_nodes})
        if not valid_nodes:
            return None, None

        mask_keep = torch.zeros(num_nodes, dtype=torch.float32)
        mask_keep[valid_nodes] = 1.0
        mask_drop = torch.ones(num_nodes, dtype=torch.float32)
        mask_drop[valid_nodes] = 0.0

        masked_batch = self._build_batch(base_data, mask_keep)
        maskout_batch = self._build_batch(base_data, mask_drop)

        masked_probs = self._predict_probs_from_inputs(masked_batch)
        maskout_probs = self._predict_probs_from_inputs(maskout_batch)

        return (
            masked_probs.detach().cpu().tolist(),
            maskout_probs.detach().cpu().tolist(),
        )

    def _extract_primary_data(
        self, results
    ) -> Tuple[Optional[Data], Optional[Sequence[int]]]:
        if not results:
            return None, None
        entry = results[0]
        if isinstance(entry, list) and entry:
            entry = entry[0]
        if not isinstance(entry, dict):
            return None, None

        data_obj = entry.get("data")
        coalition = entry.get("coalition") or []
        if data_obj is None:
            return None, None

        if isinstance(data_obj, Batch):
            data_list = data_obj.to_data_list()
            if not data_list:
                return None, None
            base_data = data_list[0]
        elif isinstance(data_obj, Data):
            base_data = data_obj
        else:
            return None, None

        indices: List[int] = []
        for value in coalition:
            try:
                indices.append(int(value))
            except (TypeError, ValueError):
                continue
        return base_data, indices

    def _build_batch(self, data: Data, mask: torch.Tensor) -> Batch:
        masked = data.clone().cpu()
        mask = mask.to(masked.x.dtype)
        masked.x = masked.x * mask.unsqueeze(1)

        if self.subgraph_building_method == "split":
            node_mask = mask > 0.5
            row, col = masked.edge_index
            edge_mask = node_mask[row] & node_mask[col]
            masked.edge_index = masked.edge_index[:, edge_mask]
            if getattr(masked, "edge_attr", None) is not None:
                masked.edge_attr = masked.edge_attr[edge_mask]

        batch = Batch.from_data_list([masked])
        return batch.to(self.device)

    def _predict_probs_from_inputs(self, *model_args, **model_kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(*model_args, **model_kwargs)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        return probs
