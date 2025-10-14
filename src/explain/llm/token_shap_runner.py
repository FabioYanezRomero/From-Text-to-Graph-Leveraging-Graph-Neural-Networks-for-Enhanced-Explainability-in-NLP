from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from src.Insights.metrics import summarize_records
from src.Insights.records import Coalition, ExplanationRecord, RelatedPrediction
from src.Insights.reporting import export_summaries_csv, export_summaries_json

from .config import LLMExplainerRequest
from .model_loader import ModelBundle

LOGGER = logging.getLogger(__name__)

def _parse_response_scores(response: str) -> Tuple[int, float]:
    """Extract predicted class and confidence from TokenSHAP response string."""

    predicted: Optional[int] = None
    confidence: Optional[float] = None
    for part in response.split(","):
        part = part.strip()
        if part.startswith("Class:"):
            try:
                predicted = int(part.split(":", 1)[1].strip())
            except ValueError:
                predicted = None
        if part.startswith("Conf:"):
            try:
                confidence = float(part.split(":", 1)[1].strip())
            except ValueError:
                confidence = None
    if predicted is None:
        raise ValueError(f"Unable to parse predicted class from response: {response}")
    if confidence is None:
        raise ValueError(f"Unable to parse confidence from response: {response}")
    return predicted, confidence


def _extract_importances(
    df: pd.DataFrame,
    *,
    num_tokens: int,
) -> pd.Series:
    """Aggregate cosine similarity scores into per-token attributions."""

    importance = pd.Series(0.0, index=range(num_tokens), dtype="float64")
    for _, row in df.iterrows():
        coalition = row.get("Token_Indexes")
        weight = row.get("Cosine_Similarity")
        if coalition is None or weight is None:
            continue
        if not isinstance(coalition, (list, tuple)):
            try:
                coalition = list(coalition)
            except TypeError:
                continue
        for idx in coalition:
            try:
                idx = int(idx)
            except Exception:
                continue
            if 0 <= idx < num_tokens:
                importance[idx] += float(weight)
    return importance


def _top_nodes(importance: Iterable[float], *, k: int) -> List[int]:
    series = pd.Series(list(importance))
    ordered = series.sort_values(ascending=False).index.tolist()
    return ordered[:k]


def _coalitions_from_dataframe(df: pd.DataFrame, *, num_tokens: int) -> List[Coalition]:
    coalitions: List[Coalition] = []
    for combo_id, row in df.iterrows():
        coalition_raw = row.get("Token_Indexes")
        weight = row.get("Cosine_Similarity", 0.0)
        if coalition_raw is None:
            coalition_raw = []
        if not isinstance(coalition_raw, (list, tuple)):
            try:
                coalition_raw = list(coalition_raw)
            except TypeError:
                coalition_raw = []

        indices: List[int] = []
        for value in coalition_raw:
            try:
                idx = int(value)
            except Exception:
                continue
            if 0 <= idx < num_tokens:
                indices.append(idx)

        coalition = Coalition.from_iterable(
            nodes=indices,
            confidence=float(weight),
            combination_id=int(combo_id) if pd.notna(combo_id) else None,
            size=len(indices) if indices else None,
        )
        coalitions.append(coalition)
    return coalitions


def _build_record(
    *,
    request: LLMExplainerRequest,
    graph_index: int,
    prompt: str,
    label: Optional[int],
    token_text: List[str],
    importance: pd.Series,
    predicted_class: int,
    confidence: float,
    sampling_ratio: float,
    elapsed_time: float,
    coalitions: List[Coalition],
) -> ExplanationRecord:
    related_prediction = RelatedPrediction(
        origin=float(confidence),
        masked=None,
        maskout=None,
        sparsity=None,
    )

    top_indices = tuple(_top_nodes(importance, k=request.top_k_nodes))

    extras: Dict[str, object] = {
        "prompt": prompt,
        "elapsed_time": elapsed_time,
        "sampling_ratio": sampling_ratio,
        "max_tokens": request.max_tokens(),
        "max_length": request.max_length(),
        "token_text": token_text,
    }

    record = ExplanationRecord(
        dataset=request.insight_dataset(),
        graph_type=request.graph_type(),
        method=request.profile.method_name,
        run_id=request.profile.resolve_run_id(),
        graph_index=graph_index,
        label=label,
        prediction_class=predicted_class,
        prediction_confidence=float(confidence),
        num_nodes=len(token_text),
        num_edges=0,
        node_importance=importance.astype(float).tolist(),
        top_nodes=top_indices,
        related_prediction=related_prediction,
        hyperparams={
            "sampling_ratio": sampling_ratio,
        },
        coalitions=coalitions,
        extras=extras,
    )
    return record


def token_shap_explain(
    request: LLMExplainerRequest,
    model_bundle: ModelBundle,
    dataset,
    *,
    progress: bool = True,
) -> Tuple[
    List[ExplanationRecord],
    List[Dict[str, object]],
    Path,
    Path,
    Path,
    Optional[Path],
    Optional[Path],
]:
    """Run TokenSHAP over ``dataset`` returning explainability artefact paths."""

    try:
        from token_shap import TokenSHAP  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("token_shap package is required for LLM explainability") from exc

    tokenizer = model_bundle.tokenizer
    device = model_bundle.device

    class HFModelWrapper:
        """Adapter exposing a generate() method to TokenSHAP."""

        def __call__(self, prompt: str) -> str:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=request.max_length(),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model_bundle.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
                prob_0 = probabilities[0][0].item()
                prob_1 = probabilities[0][1].item()
            return (
                f"Class: {predicted_class}, P(0): {prob_0:.6f}, "
                f"P(1): {prob_1:.6f}, Conf: {confidence:.6f}"
            )

    class HFWordpieceSplitter:
        def __init__(self, tokenizer, include_special: bool = False, max_length: int = 512):
            self.tokenizer = tokenizer
            self.include_special = include_special
            self.max_length = max_length

        def split(self, text: str) -> List[str]:
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            ids = [tid for tid, mask in zip(enc["input_ids"], enc["attention_mask"]) if mask == 1]
            if not self.include_special:
                ids = [tid for tid in ids if tid not in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)]
            self._ids = ids
            self._tokens = self.tokenizer.convert_ids_to_tokens(ids)
            return list(self._tokens)

        def join(self, tokens_subset: List[str]) -> str:
            ids = self.tokenizer.convert_tokens_to_ids(tokens_subset)
            if self.include_special:
                if not ids or ids[0] != self.tokenizer.cls_token_id:
                    ids = [self.tokenizer.cls_token_id] + ids
                if ids[-1] != self.tokenizer.sep_token_id:
                    ids = ids + [self.tokenizer.sep_token_id]
            return self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

    hf_model = HFModelWrapper()
    splitter = HFWordpieceSplitter(tokenizer, max_length=request.max_length())
    explainer = TokenSHAP(hf_model, splitter)

    records: List[ExplanationRecord] = []
    output_dir = request.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = enumerate(dataset)
    if progress:
        iterator = tqdm(iterator, total=len(dataset), desc=f"TokenSHAP[{request.profile.key}]")

    processed = 0
    max_samples = request.effective_max_samples()
    for index, entry in iterator:
        if max_samples is not None and processed >= max_samples:
            break
        prompt = entry[request.profile.text_field]
        label = entry.get(request.profile.label_field)
        tokens = splitter.split(prompt)
        num_tokens = len(tokens)
        if num_tokens == 0 or num_tokens > request.max_tokens():
            continue

        ratio = request.sampling_ratio(num_tokens)
        start = time.perf_counter()
        df = explainer.analyze(prompt, sampling_ratio=ratio, print_highlight_text=False)
        elapsed = time.perf_counter() - start
        if not isinstance(df, pd.DataFrame):
            LOGGER.warning("TokenSHAP returned non DataFrame output for index %s", index)
            continue

        predicted_class, confidence = _parse_response_scores(df.iloc[0]["Response"])
        importance = _extract_importances(df, num_tokens=num_tokens)
        coalitions = _coalitions_from_dataframe(df, num_tokens=num_tokens)

        record = _build_record(
            request=request,
            graph_index=index,
            prompt=prompt,
            label=label,
            token_text=tokens,
            importance=importance,
            predicted_class=predicted_class,
            confidence=confidence,
            sampling_ratio=ratio,
            elapsed_time=elapsed,
            coalitions=coalitions,
        )
        records.append(record)

        if request.store_raw:
            coalitions_dir = output_dir / "coalitions"
            coalitions_dir.mkdir(parents=True, exist_ok=True)
            csv_path = coalitions_dir / f"graph_{index:04d}_coalitions.csv"
            df.to_csv(csv_path, index=False)
            record.extras["coalitions_path"] = str(csv_path)

        processed += 1

    if not records:
        return [], [], output_dir, Path(), Path(), None, None

    summaries = summarize_records(
        records,
        sufficiency_threshold=request.sufficiency_threshold,
        top_k=request.top_k_nodes,
        graph_provider=lambda _: None,
    )

    # Inject token-level text for readability similar to graph-based insights
    for record, summary in zip(records, summaries):
        token_lookup = record.extras.get("token_text", []) if isinstance(record.extras, dict) else []
        top_tokens = [token_lookup[idx] for idx in summary.get("top_nodes", []) if 0 <= idx < len(token_lookup)]
        summary["top_tokens"] = top_tokens
        minimal = record.minimal_coalition(request.sufficiency_threshold, origin_confidence=record.related_prediction.origin)
        if minimal:
            summary["minimal_coalition_tokens"] = [
                token_lookup[idx] for idx in minimal.nodes if 0 <= idx < len(token_lookup)
            ]
        else:
            summary["minimal_coalition_tokens"] = []

    basename = request.output_basename_or_default()
    summary_json = output_dir / f"{basename}.json"
    export_summaries_json(
        records,
        summary_json,
        sufficiency_threshold=request.sufficiency_threshold,
        top_k=request.top_k_nodes,
        summaries=summaries,
    )

    summary_csv = output_dir / f"{basename}.csv"
    export_summaries_csv(summaries, summary_csv)

    raw_json_path = output_dir / f"{basename}_records.json"
    raw_json_path.write_text(
        json.dumps([record.to_dict() for record in records], indent=2),
        encoding="utf-8",
    )

    raw_pickle_path: Optional[Path] = None
    if request.store_raw:
        raw_pickle_path = output_dir / f"{basename}_records.pkl"
        with raw_pickle_path.open("wb") as handle:
            pickle.dump(records, handle)

    return records, summaries, summary_json, summary_csv, raw_json_path, raw_pickle_path


