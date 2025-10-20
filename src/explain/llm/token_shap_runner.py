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

from src.Insights.llm_providers import LLMExplanationProvider
from src.Insights.metrics import summarize_records
from src.Insights.records import Coalition, ExplanationRecord, RelatedPrediction
from src.Insights.reporting import export_summaries_csv, export_summaries_json
from src.explain.common.fairness import FairMultimodalHyperparameterAdvisor
from src.utils.energy import EnergyMonitor

from .config import LLMExplainerRequest, TOKEN_SHAP_DEFAULTS
from .fair_sampling import compute_fair_hyperparams
from .hyperparam_advisor import (
    DatasetContext,
    ModelSpec,
    TokenSHAPHyperparameterAdvisor,
)
from .model_loader import ModelBundle
from .word_aggregation import create_word_level_summary

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
    top_indices: Tuple[int, ...],
    related_prediction: RelatedPrediction,
    hyperparams: Dict[str, object],
    masked_prompt: Optional[str] = None,
    maskout_prompt: Optional[str] = None,
) -> ExplanationRecord:
    extras: Dict[str, object] = {
        "prompt": prompt,
        "elapsed_time": elapsed_time,
        "sampling_ratio": sampling_ratio,
        "max_tokens": request.max_tokens(),
        "max_length": request.max_length(),
        "token_text": token_text,
    }
    if masked_prompt is not None:
        extras["masked_prompt"] = masked_prompt
    if maskout_prompt is not None:
        extras["maskout_prompt"] = maskout_prompt

    hyperparams_payload = dict(hyperparams)
    hyperparams_payload.setdefault("sampling_ratio", sampling_ratio)

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
        hyperparams=hyperparams_payload,
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
    use_advisor: bool = True,
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
    fairness_advisor = FairMultimodalHyperparameterAdvisor() if request.fair_comparison else None
    if fairness_advisor is not None:
        max_tokens_estimate = min(request.profile.max_tokens, request.profile.max_length)
        fair_defaults = fairness_advisor.tokenshap(num_tokens=max_tokens_estimate)
        request.top_k_nodes = max(request.top_k_nodes, int(fair_defaults.get("top_k_tokens", request.top_k_nodes)))
    
    # Initialize hyperparameter advisor if requested
    advisor: Optional[TokenSHAPHyperparameterAdvisor] = None
    if use_advisor and fairness_advisor is None:
        model_spec = ModelSpec(
            base_model_name=request.profile.base_model_name,
            num_labels=request.profile.num_labels,
            max_length=request.max_length(),
        )
        context = DatasetContext(
            dataset=request.profile.dataset_name,
            task_type="classification",
            backbone=request.profile.derive_backbone(),
        )
        advisor = TokenSHAPHyperparameterAdvisor(
            model_spec=model_spec,
            context=context,
        )
        LOGGER.info("Using TokenSHAP hyperparameter advisor for adaptive parameter selection")

    class HFModelWrapper:
        """Adapter exposing predict/generate methods compatible with TokenSHAP."""

        def __init__(self, model, tokenizer, *, device: torch.device, max_length: int) -> None:
            self._model = model
            self._tokenizer = tokenizer
            self._device = device
            self._max_length = max_length

        def _run(self, prompt: str) -> Tuple[int, float, List[float]]:
            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
            predicted_class = int(torch.argmax(probs).item())
            distribution = probs.detach().cpu().tolist()
            confidence = float(distribution[predicted_class])
            return predicted_class, confidence, distribution

        def __call__(self, prompt: str) -> str:
            predicted_class, confidence, distribution = self._run(prompt)
            prob_parts = ", ".join(f"P({idx}): {prob:.6f}" for idx, prob in enumerate(distribution))
            return f"Class: {predicted_class}, {prob_parts}, Conf: {confidence:.6f}"

        def generate(self, prompt: str) -> str:
            """Generate method expected by TokenSHAP for baseline calculation."""
            return self.__call__(prompt)

        def predict(self, prompt: str) -> Tuple[int, float, List[float]]:
            """Return predicted class, confidence, and full probability distribution."""
            return self._run(prompt)

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

    hf_model = HFModelWrapper(
        model_bundle.model,
        tokenizer,
        device=device,
        max_length=request.max_length(),
    )
    splitter = HFWordpieceSplitter(tokenizer, max_length=request.max_length())
    explainer = TokenSHAP(hf_model, splitter)

    records: List[ExplanationRecord] = []
    output_dir = request.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply sharding to dataset
    shard_index = request.shard_index
    num_shards = request.num_shards
    
    # Filter dataset for this shard (process every num_shards-th sample starting from shard_index)
    if num_shards > 1:
        dataset_list = list(enumerate(dataset))
        sharded_dataset = [(idx, entry) for idx, entry in dataset_list if idx % num_shards == shard_index]
        total_samples = len(sharded_dataset)
        LOGGER.info(
            "Shard %d/%d: Processing %d samples (every %d-th sample starting from %d)",
            shard_index + 1,
            num_shards,
            total_samples,
            num_shards,
            shard_index,
        )
    else:
        sharded_dataset = list(enumerate(dataset))
        total_samples = len(sharded_dataset)

    iterator = iter(sharded_dataset)
    if progress:
        desc = f"TokenSHAP[{request.profile.key}]"
        if num_shards > 1:
            desc = f"TokenSHAP[{request.profile.key}][{shard_index + 1}/{num_shards}]"
        iterator = tqdm(
            sharded_dataset,
            total=total_samples,
            desc=desc,
            position=shard_index,
            leave=True,
            dynamic_ncols=True,
        )

    processed = 0
    skipped = 0
    max_samples = request.effective_max_samples()
    energy_monitor = EnergyMonitor("TokenSHAP", output_dir=output_dir)
    energy_monitor.__enter__()
    try:
        for index, entry in iterator:
            if max_samples is not None and processed >= max_samples:
                break
            prompt = entry[request.profile.text_field]
            label = entry.get(request.profile.label_field)
            tokens = splitter.split(prompt)
        num_tokens = len(tokens)
        if num_tokens == 0 or num_tokens > request.max_tokens():
            skipped += 1
            continue

        # Use advisor if available, otherwise fall back to request's sampling strategy
        fairness_params: Optional[Dict[str, float]] = None
        if fairness_advisor is not None:
            fairness_params = fairness_advisor.tokenshap(num_tokens=num_tokens)
            ratio = float(fairness_params["sampling_ratio"])
            hyperparams = dict(fairness_params)
        elif advisor is not None:
            suggested_params = advisor.suggest(tokens)
            ratio = suggested_params["sampling_ratio"]
            # Store all suggested hyperparameters for later analysis
            hyperparams = dict(suggested_params)
            hyperparams.setdefault("sampling_ratio", ratio)
        else:
            ratio = request.sampling_ratio(num_tokens)
            hyperparams = {"sampling_ratio": ratio}
        
        # Log warning for sequences that may cause OOM, but process anyway
        max_combinations = 2 ** num_tokens
        if max_combinations > 65536:  # 2^16
            LOGGER.warning(
                "Sample %d: %d tokens = 2^%d = %s combinations (ratio=%.4f). "
                "TokenSHAP generates all combinations before sampling - may cause OOM!",
                index, num_tokens, num_tokens, f"{max_combinations:,}", ratio
            )
        
        start = time.perf_counter()
        # TokenSHAP may show internal progress bars for combinations
        # NOTE: TokenSHAP.analyze() only accepts sampling_ratio parameter
        # It does NOT support explicit num_samples limit like GraphSVX
        df = explainer.analyze(
            prompt, 
            sampling_ratio=ratio,
            print_highlight_text=False,
        )
        elapsed = time.perf_counter() - start
        if not isinstance(df, pd.DataFrame):
            LOGGER.warning("TokenSHAP returned non DataFrame output for index %s", index)
            continue

        parsed_class, parsed_confidence = _parse_response_scores(df.iloc[0]["Response"])
        importance = _extract_importances(df, num_tokens=num_tokens)
        coalitions = _coalitions_from_dataframe(df, num_tokens=num_tokens)

        target_top_k = fairness_params.get("top_k_tokens") if fairness_params else request.top_k_nodes
        target_top_k = max(1, min(int(target_top_k), num_tokens))
        top_indices = tuple(_top_nodes(importance, k=target_top_k))
        valid_top_indices = tuple(idx for idx in top_indices if 0 <= idx < num_tokens)
        important_index_set = set(valid_top_indices)

        origin_class, origin_confidence, origin_distribution = hf_model.predict(prompt)
        if origin_class != parsed_class or abs(origin_confidence - parsed_confidence) > 1e-5:
            LOGGER.debug(
                "TokenSHAP response mismatch at index %d: parsed=(%s, %.6f) model=(%s, %.6f)",
                index,
                parsed_class,
                parsed_confidence,
                origin_class,
                origin_confidence,
            )

        masked_tokens = [tokens[idx] for idx in range(num_tokens) if idx in important_index_set]
        if not masked_tokens and tokens:
            masked_tokens = [tokens[idx] for idx in valid_top_indices if 0 <= idx < num_tokens]
        if not masked_tokens:
            masked_tokens = tokens

        maskout_tokens = [tokens[idx] for idx in range(num_tokens) if idx not in important_index_set]
        if not maskout_tokens:
            maskout_tokens = tokens

        masked_prompt = splitter.join(masked_tokens)
        maskout_prompt = splitter.join(maskout_tokens)

        _, _, masked_distribution = hf_model.predict(masked_prompt)
        _, _, maskout_distribution = hf_model.predict(maskout_prompt)

        target_class = origin_class
        masked_confidence = masked_distribution[target_class] if masked_distribution else None
        maskout_confidence = maskout_distribution[target_class] if maskout_distribution else None
        sparsity = (len(important_index_set) / num_tokens) if num_tokens else None

        related_prediction = RelatedPrediction(
            origin=float(origin_confidence),
            masked=float(masked_confidence) if masked_confidence is not None else None,
            maskout=float(maskout_confidence) if maskout_confidence is not None else None,
            sparsity=sparsity,
            origin_distribution=tuple(float(v) for v in origin_distribution),
            masked_distribution=tuple(float(v) for v in masked_distribution) if masked_distribution else None,
            maskout_distribution=tuple(float(v) for v in maskout_distribution) if maskout_distribution else None,
        )

        record = _build_record(
            request=request,
            graph_index=index,
            prompt=prompt,
            label=label,
            token_text=tokens,
            importance=importance,
            predicted_class=origin_class,
            confidence=origin_confidence,
            sampling_ratio=ratio,
            elapsed_time=elapsed,
            coalitions=coalitions,
            top_indices=valid_top_indices,
            related_prediction=related_prediction,
            hyperparams=hyperparams,
            masked_prompt=masked_prompt,
            maskout_prompt=maskout_prompt,
        )
        records.append(record)

        if request.store_raw:
            coalitions_dir = output_dir / "coalitions"
            coalitions_dir.mkdir(parents=True, exist_ok=True)
            csv_path = coalitions_dir / f"graph_{index:04d}_coalitions.csv"
            df.to_csv(csv_path, index=False)
            record.extras["coalitions_path"] = str(csv_path)

            processed += 1
    finally:
        energy_monitor.__exit__(None, None, None)

    energy_data = energy_monitor.result
    if energy_data:
        energy_path = output_dir / "energy_metrics.json"
        energy_path.write_text(json.dumps(energy_data, indent=2))
        LOGGER.info("TokenSHAP energy metrics: %s", energy_data)

    LOGGER.info(
        "Processed %d samples, skipped %d samples (exceeding max_tokens=%d or empty)",
        processed,
        skipped,
        request.max_tokens(),
    )
    
    # Warn if a large portion of samples were skipped
    total_examined = processed + skipped
    if total_examined > 0 and skipped / total_examined > 0.5:
        LOGGER.warning(
            "Over 50%% of samples were skipped (%d/%d). Consider increasing max_tokens (current: %d) "
            "or using a dataset with shorter sequences.",
            skipped,
            total_examined,
            request.max_tokens(),
        )
    
    if not records:
        return [], [], output_dir, Path(), Path(), None, None

    # Use LLMExplanationProvider to automatically extract token text
    llm_provider = LLMExplanationProvider()
    
    summaries = summarize_records(
        records,
        sufficiency_threshold=request.sufficiency_threshold,
        top_k=request.top_k_nodes,
        graph_provider=llm_provider,  # Use LLM provider instead of None
    )
    
    # The LLMExplanationProvider automatically populates top_tokens and
    # minimal_coalition_tokens in summarize_record, so no manual injection needed
    
    # Add word-level aggregation to summaries
    for record, summary in zip(records, summaries):
        word_summary = create_word_level_summary(
            record,
            aggregation="mean",  # Average scores across subword tokens
            top_k=request.top_k_nodes,
        )
        summary.update(word_summary)

    basename = request.output_basename_or_default()
    
    # Add shard suffix if using multiple shards
    if request.num_shards > 1:
        basename = f"{basename}_shard{request.shard_index:02d}of{request.num_shards:02d}"
    
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


def collect_token_shap_hyperparams(
    request: LLMExplainerRequest,
    model_bundle: ModelBundle,
    dataset,
    *,
    output_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> Tuple[Path, List[Dict[str, object]]]:
    """
    Collect suggested hyperparameters for each sample in the dataset.
    
    This function analyzes each sentence and generates suggested hyperparameters
    using the TokenSHAPHyperparameterAdvisor, similar to the hyperparameter
    collection done for SubgraphX and GraphSVX.
    
    Args:
        request: Configuration for the LLM explainer
        model_bundle: Loaded model and tokenizer
        dataset: Dataset to analyze
        output_path: Optional path to save the hyperparameters JSON
        max_samples: Maximum number of samples to process
        progress: Whether to show progress bar
    
    Returns:
        Tuple of (output_path, per_sample_data)
    """
    tokenizer = model_bundle.tokenizer
    fairness_advisor = FairMultimodalHyperparameterAdvisor() if request.fair_comparison else None
    
    advisor: Optional[TokenSHAPHyperparameterAdvisor] = None
    if fairness_advisor is None:
        # Initialize model spec and context
        model_spec = ModelSpec(
            base_model_name=request.profile.base_model_name,
            num_labels=request.profile.num_labels,
            max_length=request.max_length(),
        )
        context = DatasetContext(
            dataset=request.profile.dataset_name,
            task_type="classification",
            backbone=request.profile.derive_backbone(),
        )
        # Create advisor
        advisor = TokenSHAPHyperparameterAdvisor(
            model_spec=model_spec,
            context=context,
        )
    
    # Determine output path
    if output_path is None:
        output_dir = request.output_dir()
        artifact_dir = output_dir / "hyperparams"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        basename = request.output_basename_or_default()
        output_path = artifact_dir / f"{basename}_hyperparams.json"
    else:
        artifact_dir = output_path.parent
        artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splitter for tokenization
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
            return self.tokenizer.convert_ids_to_tokens(ids)
    
    splitter = HFWordpieceSplitter(tokenizer, max_length=request.max_length())
    
    # Collect hyperparameters for each sample
    per_sample: List[Dict[str, object]] = []
    
    effective_max = max_samples or request.effective_max_samples()
    total = len(dataset) if effective_max is None else min(len(dataset), effective_max)
    
    iterator = enumerate(dataset)
    if progress:
        iterator = tqdm(
            iterator,
            total=total,
            desc=f"CollectTokenSHAPParams[{request.profile.key}]",
            dynamic_ncols=True,
        )
    
    for index, entry in iterator:
        if effective_max is not None and index >= effective_max:
            break
        
        prompt = entry[request.profile.text_field]
        tokens = splitter.split(prompt)
        num_tokens = len(tokens)
        
        if num_tokens == 0 or num_tokens > request.max_tokens():
            continue
        
        if fairness_advisor is not None:
            suggested_params = fairness_advisor.tokenshap(num_tokens=num_tokens)
        else:
            suggested_params = advisor.suggest(tokens)  # type: ignore[union-attr]
        
        # Compute sentence statistics for metadata
        from .hyperparam_advisor import SentenceStats
        stats = SentenceStats.from_tokens(tokens)
        
        per_sample.append({
            "sample_index": index,
            "num_tokens": num_tokens,
            "num_chars": stats.num_chars,
            "avg_token_length": stats.avg_token_length,
            "max_token_length": stats.max_token_length,
            "hyperparams": dict(suggested_params),
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for readability
            "fair_comparison": request.fair_comparison,
        })
    
    # Save to JSON
    payload = {
        "method": "token_shap",
        "dataset": request.profile.dataset_name,
        "backbone": request.profile.derive_backbone(),
        "split": request.profile.split,
        "max_tokens": request.max_tokens(),
        "max_length": request.max_length(),
        "num_labels": request.profile.num_labels,
        "base_model_name": request.profile.base_model_name,
        "total_samples": len(per_sample),
        "per_sample": per_sample,
    }
    
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "Collected hyperparameters for %d samples, saved to %s",
        len(per_sample),
        output_path,
    )
    
    return output_path, per_sample
