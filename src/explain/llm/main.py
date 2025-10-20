#!/usr/bin/env python3
"""Main entry point for LLM explainability using TokenSHAP."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .config import (
    DEFAULT_FINETUNED_ROOT,
    DEFAULT_INSIGHTS_ROOT,
    LLMExplainerRequest,
    build_default_profiles,
)
from .model_loader import load_dataset_split, load_finetuned_model
from .token_shap_runner import collect_token_shap_hyperparams, token_shap_explain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def run_token_shap(
    dataset_key: str,
    *,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    sampling_override: Optional[float] = None,
    top_k_nodes: int = 5,
    sufficiency_threshold: float = 0.9,
    finetuned_root: Path = DEFAULT_FINETUNED_ROOT,
    insights_root: Path = DEFAULT_INSIGHTS_ROOT,
    output_basename: Optional[str] = None,
    store_raw: bool = True,
    use_advisor: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
    fair_comparison: bool = False,
) -> None:
    """
    Run TokenSHAP explanations on a finetuned LLM.
    
    Args:
        dataset_key: Key for the dataset profile (e.g., "stanfordnlp/sst2")
        device: Device to use (cuda/cpu), auto-detected if None
        max_samples: Maximum number of samples to explain
        sampling_override: Override sampling ratio (use advisor if None)
        top_k_nodes: Number of top tokens to highlight
        sufficiency_threshold: Threshold for minimal coalition
        finetuned_root: Root directory for finetuned models
        insights_root: Root directory for output insights
        output_basename: Custom basename for output files
        store_raw: Whether to store raw coalition data
        use_advisor: Whether to use hyperparameter advisor
    """
    profiles = build_default_profiles(finetuned_root=finetuned_root)
    if dataset_key not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. Available: {available}"
        )
    
    profile = profiles[dataset_key]
    LOGGER.info("Loading model and dataset for %s", dataset_key)
    
    model_bundle = load_finetuned_model(profile, device=device)
    dataset = load_dataset_split(profile)
    
    request = LLMExplainerRequest(
        profile=profile,
        device=device,
        sampling_override=sampling_override,
        max_samples=max_samples,
        sufficiency_threshold=sufficiency_threshold,
        top_k_nodes=top_k_nodes,
        insights_root=insights_root,
        output_basename=output_basename,
        store_raw=store_raw,
        num_shards=num_shards,
        shard_index=shard_index,
        fair_comparison=fair_comparison,
    )
    
    LOGGER.info("Running TokenSHAP explanations (advisor=%s)", use_advisor)
    records, summaries, json_path, csv_path, raw_json, raw_pickle = token_shap_explain(
        request=request,
        model_bundle=model_bundle,
        dataset=dataset,
        progress=True,
        use_advisor=use_advisor,
    )
    
    LOGGER.info("Processed %d samples", len(records))
    LOGGER.info("Summary JSON: %s", json_path)
    LOGGER.info("Summary CSV: %s", csv_path)
    LOGGER.info("Raw records JSON: %s", raw_json)
    if raw_pickle:
        LOGGER.info("Raw records pickle: %s", raw_pickle)


def run_hyperparam_collection(
    dataset_key: str,
    *,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    output_path: Optional[Path] = None,
    finetuned_root: Path = DEFAULT_FINETUNED_ROOT,
    insights_root: Path = DEFAULT_INSIGHTS_ROOT,
    fair_comparison: bool = False,
) -> None:
    """
    Collect suggested hyperparameters for all samples without running explanations.
    
    This is useful for analyzing what hyperparameters the advisor suggests
    for different sentences in the dataset.
    
    Args:
        dataset_key: Key for the dataset profile (e.g., "stanfordnlp/sst2")
        device: Device to use (cuda/cpu), auto-detected if None
        max_samples: Maximum number of samples to analyze
        output_path: Custom path for output JSON
        finetuned_root: Root directory for finetuned models
        insights_root: Root directory for output insights
    """
    profiles = build_default_profiles(finetuned_root=finetuned_root)
    if dataset_key not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. Available: {available}"
        )
    
    profile = profiles[dataset_key]
    LOGGER.info("Loading model and dataset for %s", dataset_key)
    
    model_bundle = load_finetuned_model(profile, device=device)
    dataset = load_dataset_split(profile)
    
    request = LLMExplainerRequest(
        profile=profile,
        device=device,
        max_samples=max_samples,
        insights_root=insights_root,
        fair_comparison=fair_comparison,
    )
    
    LOGGER.info("Collecting hyperparameters for all samples")
    saved_path, per_sample = collect_token_shap_hyperparams(
        request=request,
        model_bundle=model_bundle,
        dataset=dataset,
        output_path=output_path,
        max_samples=max_samples,
        progress=True,
    )
    
    LOGGER.info("Collected hyperparameters for %d samples", len(per_sample))
    LOGGER.info("Saved to: %s", saved_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run TokenSHAP explainability on finetuned LLMs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Run TokenSHAP explanations",
    )
    explain_parser.add_argument(
        "dataset",
        type=str,
        help="Dataset key (e.g., stanfordnlp/sst2, setfit/ag_news)",
    )
    explain_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    explain_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to explain",
    )
    explain_parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=None,
        help="Override sampling ratio (disables advisor)",
    )
    explain_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens to highlight",
    )
    explain_parser.add_argument(
        "--no-advisor",
        action="store_true",
        help="Disable hyperparameter advisor",
    )
    explain_parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Don't store raw coalition data",
    )
    explain_parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards for parallel processing",
    )
    explain_parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard (0-based, must be < num-shards)",
    )
    explain_parser.add_argument(
        "--fair",
        action="store_true",
        help="Enable fair multimodal hyperparameter alignment",
    )
    
    # Hyperparams command
    hyperparams_parser = subparsers.add_parser(
        "hyperparams",
        help="Collect suggested hyperparameters for all samples",
    )
    hyperparams_parser.add_argument(
        "dataset",
        type=str,
        help="Dataset key (e.g., stanfordnlp/sst2, setfit/ag_news)",
    )
    hyperparams_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    hyperparams_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze",
    )
    hyperparams_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output path for hyperparameters JSON",
    )
    hyperparams_parser.add_argument(
        "--fair",
        action="store_true",
        help="Enable fair multimodal hyperparameter alignment",
    )
    
    args = parser.parse_args()
    
    if args.command == "explain":
        run_token_shap(
            dataset_key=args.dataset,
            device=args.device,
            max_samples=args.max_samples,
            sampling_override=args.sampling_ratio,
            top_k_nodes=args.top_k,
            use_advisor=not args.no_advisor,
            store_raw=not args.no_raw,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            fair_comparison=args.fair,
        )
    elif args.command == "hyperparams":
        run_hyperparam_collection(
            dataset_key=args.dataset,
            device=args.device,
            max_samples=args.max_samples,
            output_path=args.output,
            fair_comparison=args.fair,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

