"""Enhanced data loaders for new insights structure with sharding support.

This module provides comprehensive loading capabilities for:
- Sharded GNN summary files (part0001.json, part0002.json, etc.)
- LLM token SHAP exports
- Agreement/ranking metrics
- Stratified analysis by class and correctness
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np


@dataclass
class InsightMetadata:
    """Metadata about loaded insights."""
    
    total_records: int
    datasets: List[str]
    graph_types: List[str]
    methods: List[str]
    has_gnn: bool
    has_llm: bool
    has_agreement: bool
    source_files: List[str]


@dataclass
class EnhancedInsightFrame:
    """Enhanced container for insight data with comprehensive metrics."""
    
    data: pd.DataFrame
    token_frame: pd.DataFrame
    agreement_frame: Optional[pd.DataFrame]
    metadata: InsightMetadata


def load_json_file(path: Path) -> Union[List[dict], dict]:
    """Load a single JSON file with error handling."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {path}: {e}")
    except Exception as e:
        raise IOError(f"Failed to read {path}: {e}")


def load_sharded_summaries(base_path: Path) -> List[dict]:
    """Load sharded summary files (e.g., skipgrams_summaries.json + parts).
    
    Args:
        base_path: Path to the base summary file (e.g., skipgrams_summaries.json)
        
    Returns:
        List of all records from all shards
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Summary file not found: {base_path}")
    
    # Load the index file
    index = load_json_file(base_path)
    
    # If it's a list, it's not sharded
    if isinstance(index, list):
        return index
    
    # If it's a dict, it contains shard information
    if isinstance(index, dict) and "shards" in index:
        records = []
        parent_dir = base_path.parent
        
        for shard_name in index["shards"]:
            shard_path = parent_dir / shard_name
            if not shard_path.exists():
                print(f"Warning: Shard file not found: {shard_path}")
                continue
                
            shard_data = load_json_file(shard_path)
            if isinstance(shard_data, list):
                records.extend(shard_data)
            else:
                print(f"Warning: Unexpected shard format in {shard_path}")
        
        print(f"Loaded {len(records)} records from {len(index['shards'])} shards")
        return records
    
    # Unknown format
    raise ValueError(f"Unexpected summary file format in {base_path}")


def load_llm_token_shap(base_dir: Path) -> List[dict]:
    """Load LLM token SHAP exports, handling sharded files.
    
    Args:
        base_dir: Directory containing token_shap*.json files
        
    Returns:
        List of all token SHAP records
    """
    records = []
    
    # Look for token_shap.json first
    main_file = base_dir / "token_shap.json"
    if main_file.exists():
        data = load_json_file(main_file)
        if isinstance(data, list):
            return data
    
    # Look for sharded files: token_shap_shard00of03.json, etc.
    shard_files = sorted(base_dir.glob("token_shap_shard*.json"))
    
    for shard_file in shard_files:
        shard_data = load_json_file(shard_file)
        if isinstance(shard_data, list):
            records.extend(shard_data)
    
    if not records:
        raise FileNotFoundError(f"No token_shap files found in {base_dir}")
    
    print(f"Loaded {len(records)} LLM records from {len(shard_files)} shards")
    return records


def load_agreement_metrics(agreement_file: Path) -> List[dict]:
    """Load agreement/ranking comparison metrics.
    
    Args:
        agreement_file: Path to *_agreement.json file
        
    Returns:
        List of agreement records
    """
    if not agreement_file.exists():
        raise FileNotFoundError(f"Agreement file not found: {agreement_file}")
    
    data = load_json_file(agreement_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in agreement file: {agreement_file}")
    
    return data


def flatten_gnn_record(record: dict) -> dict:
    """Flatten a GNN insight record into a flat dictionary."""
    flat = {
        # Basic identifiers
        "dataset": record.get("dataset"),
        "graph_type": record.get("graph_type"),
        "method": record.get("method"),
        "run_id": record.get("run_id"),
        "graph_index": record.get("graph_index"),
        
        # Labels and predictions
        "label": record.get("label"),
        "prediction_class": record.get("prediction_class"),
        "prediction_confidence": record.get("prediction_confidence"),
        "is_correct": record.get("is_correct"),
        
        # Model type
        "model_type": "gnn",
        
        # Confidences
        "origin_confidence": record.get("origin_confidence"),
        "masked_confidence": record.get("masked_confidence"),
        "maskout_confidence": record.get("maskout_confidence"),
        
        # Distributions (handle None values)
        "origin_distribution_0": (record.get("origin_distribution") or [None, None])[0],
        "origin_distribution_1": (record.get("origin_distribution") or [None, None])[1] if len(record.get("origin_distribution") or []) > 1 else None,
        "masked_distribution_0": (record.get("masked_distribution") or [None, None])[0],
        "masked_distribution_1": (record.get("masked_distribution") or [None, None])[1] if len(record.get("masked_distribution") or []) > 1 else None,
        "maskout_distribution_0": (record.get("maskout_distribution") or [None, None])[0],
        "maskout_distribution_1": (record.get("maskout_distribution") or [None, None])[1] if len(record.get("maskout_distribution") or []) > 1 else None,
        
        # Sparsity and coalition metrics
        "sparsity": record.get("sparsity"),
        "minimal_coalition_size": record.get("minimal_coalition_size"),
        "minimal_coalition_confidence": record.get("minimal_coalition_confidence"),
        
        # AUC metrics
        "insertion_auc": record.get("insertion_auc"),
        "deletion_auc": record.get("deletion_auc"),
        
        # Graph structure
        "num_nodes": record.get("num_nodes"),
        "num_edges": record.get("num_edges"),
        
        # Contrastivity
        "origin_contrastivity": record.get("origin_contrastivity"),
        "masked_contrastivity": record.get("masked_contrastivity"),
        "maskout_contrastivity": record.get("maskout_contrastivity"),
        "origin_second_class": record.get("origin_second_class"),
        "origin_second_confidence": record.get("origin_second_confidence"),
        
        # Deltas and robustness
        "masked_delta": record.get("masked_delta"),
        "maskout_delta": record.get("maskout_delta"),
        "robustness_score": record.get("robustness_score"),
        
        # Monotonicity
        "maskout_drop_monotonicity": record.get("maskout_drop_monotonicity"),
        "maskout_conf_monotonicity": record.get("maskout_conf_monotonicity"),
        "sufficiency_conf_monotonicity": record.get("sufficiency_conf_monotonicity"),
        "sufficiency_drop_monotonicity": record.get("sufficiency_drop_monotonicity"),
        
        # Fidelity and faithfulness
        "fidelity_plus": record.get("fidelity_plus"),
        "fidelity_minus": record.get("fidelity_minus"),
        "faithfulness": record.get("faithfulness"),
        "faithfulness_monotonicity": record.get("faithfulness_monotonicity"),
        
        # Tokens
        "top_tokens": record.get("top_tokens", []),
        "minimal_coalition_tokens": record.get("minimal_coalition_tokens", []),
        
        # Store curves as JSON strings for later analysis
        "insertion_curve": record.get("insertion_curve"),
        "deletion_curve": record.get("deletion_curve"),
    }
    
    # Structural metrics
    structural = record.get("structural_metrics") or {}
    flat["struct_induced_num_nodes"] = structural.get("induced_num_nodes")
    flat["struct_induced_num_edges"] = structural.get("induced_num_edges")
    flat["struct_components"] = structural.get("components")
    flat["struct_density"] = structural.get("density")
    flat["struct_boundary_edges"] = structural.get("boundary_edges")
    flat["struct_cut_ratio"] = structural.get("cut_ratio")
    flat["struct_avg_shortest_path"] = structural.get("avg_shortest_path")
    
    # Centrality alignment
    centrality = record.get("centrality_alignment") or {}
    flat["centrality_degree"] = centrality.get("degree")
    flat["centrality_betweenness"] = centrality.get("betweenness")
    flat["centrality_closeness"] = centrality.get("closeness")
    
    # Derived metrics
    flat["fidelity_drop"] = _safe_diff(flat["origin_confidence"], flat["masked_confidence"])
    flat["maskout_effect"] = _safe_diff(flat["origin_confidence"], flat["maskout_confidence"])
    flat["compactness"] = flat["sparsity"]  # Compactness is essentially sparsity
    
    return flat


def flatten_llm_record(record: dict) -> dict:
    """Flatten an LLM token SHAP record into a flat dictionary."""
    flat = {
        # Basic identifiers
        "dataset": record.get("dataset"),
        "graph_type": record.get("graph_type", "tokens"),
        "method": record.get("method", "token_shap_llm"),
        "run_id": record.get("run_id"),
        "graph_index": record.get("graph_index"),
        
        # Labels and predictions
        "label": record.get("label"),
        "prediction_class": record.get("prediction_class"),
        "prediction_confidence": record.get("prediction_confidence"),
        "is_correct": record.get("is_correct"),
        
        # Model type
        "model_type": "llm",
        
        # Confidences
        "origin_confidence": record.get("origin_confidence"),
        "masked_confidence": record.get("masked_confidence"),
        "maskout_confidence": record.get("maskout_confidence"),
        
        # Distributions (handle None values)
        "origin_distribution_0": (record.get("origin_distribution") or [None, None])[0],
        "origin_distribution_1": (record.get("origin_distribution") or [None, None])[1] if len(record.get("origin_distribution") or []) > 1 else None,
        "masked_distribution_0": (record.get("masked_distribution") or [None, None])[0],
        "masked_distribution_1": (record.get("masked_distribution") or [None, None])[1] if len(record.get("masked_distribution") or []) > 1 else None,
        "maskout_distribution_0": (record.get("maskout_distribution") or [None, None])[0],
        "maskout_distribution_1": (record.get("maskout_distribution") or [None, None])[1] if len(record.get("maskout_distribution") or []) > 1 else None,
        
        # Sparsity and coalition metrics
        "sparsity": record.get("sparsity"),
        "minimal_coalition_size": record.get("minimal_coalition_size"),
        "minimal_coalition_confidence": record.get("minimal_coalition_confidence"),
        
        # AUC metrics
        "insertion_auc": record.get("insertion_auc"),
        "deletion_auc": record.get("deletion_auc"),
        
        # Graph structure (tokens don't have edges)
        "num_nodes": record.get("num_nodes") or record.get("num_words"),
        "num_edges": record.get("num_edges", 0),
        
        # Contrastivity
        "origin_contrastivity": record.get("origin_contrastivity"),
        "masked_contrastivity": record.get("masked_contrastivity"),
        "maskout_contrastivity": record.get("maskout_contrastivity"),
        "origin_second_class": record.get("origin_second_class"),
        "origin_second_confidence": record.get("origin_second_confidence"),
        
        # Deltas and robustness
        "masked_delta": record.get("masked_delta"),
        "maskout_delta": record.get("maskout_delta"),
        "robustness_score": record.get("robustness_score"),
        
        # Monotonicity
        "maskout_drop_monotonicity": record.get("maskout_drop_monotonicity"),
        "maskout_conf_monotonicity": record.get("maskout_conf_monotonicity"),
        "sufficiency_conf_monotonicity": record.get("sufficiency_conf_monotonicity"),
        "sufficiency_drop_monotonicity": record.get("sufficiency_drop_monotonicity"),
        
        # Fidelity and faithfulness
        "fidelity_plus": record.get("fidelity_plus"),
        "fidelity_minus": record.get("fidelity_minus"),
        "faithfulness": record.get("faithfulness"),
        "faithfulness_monotonicity": record.get("faithfulness_monotonicity"),
        
        # Tokens
        "top_tokens": record.get("top_tokens") or record.get("top_words", []),
        "minimal_coalition_tokens": record.get("minimal_coalition_tokens", []),
        "top_word_scores": record.get("top_word_scores", []),
        
        # Store curves as JSON strings for later analysis
        "insertion_curve": record.get("insertion_curve"),
        "deletion_curve": record.get("deletion_curve"),
        
        # LLM-specific (no structural metrics for tokens)
        "struct_induced_num_nodes": None,
        "struct_induced_num_edges": None,
        "struct_components": None,
        "struct_density": None,
        "struct_boundary_edges": None,
        "struct_cut_ratio": None,
        "struct_avg_shortest_path": None,
        
        "centrality_degree": None,
        "centrality_betweenness": None,
        "centrality_closeness": None,
    }
    
    # Derived metrics
    flat["fidelity_drop"] = _safe_diff(flat["origin_confidence"], flat["masked_confidence"])
    flat["maskout_effect"] = _safe_diff(flat["origin_confidence"], flat["maskout_confidence"])
    flat["compactness"] = flat["sparsity"]
    
    return flat


def flatten_agreement_record(record: dict) -> dict:
    """Flatten an agreement record."""
    return {
        "dataset": record.get("dataset"),
        "graph_index": record.get("graph_index"),
        "label": record.get("label"),
        "is_correct": record.get("is_correct"),
        "method_a": record.get("method_a"),
        "method_b": record.get("method_b"),
        "graph_type_a": record.get("graph_type_a"),
        "graph_type_b": record.get("graph_type_b"),
        "run_id_a": record.get("run_id_a"),
        "run_id_b": record.get("run_id_b"),
        "top_k": record.get("top_k"),
        "overlap_count": record.get("overlap_count"),
        "rbo": record.get("rbo"),
        "spearman": record.get("spearman"),
        "kendall": record.get("kendall"),
        "feature_overlap_ratio": record.get("feature_overlap_ratio"),
        "stability_jaccard": record.get("stability_jaccard"),
        "kl_divergence": record.get("kl_divergence"),
    }


def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Compute a - b safely."""
    if a is None or b is None:
        return None
    if np.isnan(a) or np.isnan(b):
        return None
    return float(a) - float(b)


def create_token_frame(records: List[dict]) -> pd.DataFrame:
    """Create a token-level dataframe from insight records."""
    token_rows = []
    
    for record in records:
        graph_index = record.get("graph_index")
        label = record.get("label")
        dataset = record.get("dataset")
        graph_type = record.get("graph_type")
        is_correct = record.get("is_correct")
        
        for token in record.get("top_tokens", []):
            token_rows.append({
                "graph_index": graph_index,
                "token": token,
                "source": "top",
                "label": label,
                "dataset": dataset,
                "graph_type": graph_type,
                "is_correct": is_correct,
            })
        
        for token in record.get("minimal_coalition_tokens", []):
            token_rows.append({
                "graph_index": graph_index,
                "token": token,
                "source": "minimal",
                "label": label,
                "dataset": dataset,
                "graph_type": graph_type,
                "is_correct": is_correct,
            })
    
    return pd.DataFrame(token_rows)


def load_insights_from_directory(
    base_dir: Path,
    *,
    load_gnn: bool = True,
    load_llm: bool = True,
    load_agreement: bool = True,
) -> EnhancedInsightFrame:
    """Load all insights from a directory structure.
    
    Expected structure:
        base_dir/
            GNN/
                <model>/
                    <dataset>/
                        skipgrams_summaries.json
                        syntactic_summaries.json
                        ...
                        skipgrams_agreement.json
                        ...
            LLM/
                <model>/
                    <dataset>/
                        token_shap.json (or shards)
    
    Args:
        base_dir: Root directory containing GNN/ and LLM/ subdirectories
        load_gnn: Whether to load GNN insights
        load_llm: Whether to load LLM insights
        load_agreement: Whether to load agreement metrics
        
    Returns:
        EnhancedInsightFrame with all loaded data
    """
    all_records = []
    agreement_records = []
    source_files = []
    
    base_path = Path(base_dir)
    
    # Load GNN insights
    if load_gnn:
        gnn_dir = base_path / "GNN"
        if gnn_dir.exists():
            for model_dir in gnn_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for dataset_dir in model_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    
                    # Load summary files
                    for summary_file in dataset_dir.glob("*_summaries.json"):
                        print(f"Loading GNN summaries: {summary_file}")
                        records = load_sharded_summaries(summary_file)
                        all_records.extend(records)
                        source_files.append(str(summary_file))
                    
                    # Load agreement files
                    if load_agreement:
                        for agreement_file in dataset_dir.glob("*_agreement.json"):
                            print(f"Loading agreement metrics: {agreement_file}")
                            records = load_agreement_metrics(agreement_file)
                            agreement_records.extend(records)
                            source_files.append(str(agreement_file))
    
    # Load LLM insights
    if load_llm:
        llm_dir = base_path / "LLM"
        if llm_dir.exists():
            for model_dir in llm_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for dataset_dir in model_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    
                    try:
                        print(f"Loading LLM insights: {dataset_dir}")
                        records = load_llm_token_shap(dataset_dir)
                        all_records.extend(records)
                        source_files.append(str(dataset_dir / "token_shap*.json"))
                    except FileNotFoundError as e:
                        print(f"Warning: {e}")
    
    if not all_records:
        raise ValueError(f"No insight records found in {base_dir}")
    
    # Flatten records based on type
    flattened = []
    for record in all_records:
        model_type = record.get("graph_type")
        if model_type in ["tokens", None] or record.get("method", "").endswith("llm"):
            flattened.append(flatten_llm_record(record))
        else:
            flattened.append(flatten_gnn_record(record))
    
    # Create dataframes
    data_frame = pd.DataFrame(flattened)
    token_frame = create_token_frame(all_records)
    
    # Create agreement frame
    agreement_frame = None
    if agreement_records:
        agreement_frame = pd.DataFrame([flatten_agreement_record(r) for r in agreement_records])
    
    # Create metadata
    metadata = InsightMetadata(
        total_records=len(data_frame),
        datasets=data_frame["dataset"].dropna().unique().tolist() if "dataset" in data_frame.columns else [],
        graph_types=data_frame["graph_type"].dropna().unique().tolist() if "graph_type" in data_frame.columns else [],
        methods=data_frame["method"].dropna().unique().tolist() if "method" in data_frame.columns else [],
        has_gnn="gnn" in data_frame["model_type"].values if "model_type" in data_frame.columns else False,
        has_llm="llm" in data_frame["model_type"].values if "model_type" in data_frame.columns else False,
        has_agreement=agreement_frame is not None and not agreement_frame.empty,
        source_files=source_files,
    )
    
    print(f"\nLoaded {metadata.total_records} records:")
    print(f"  Datasets: {metadata.datasets}")
    print(f"  Graph types: {metadata.graph_types}")
    print(f"  Methods: {metadata.methods}")
    print(f"  Has GNN: {metadata.has_gnn}")
    print(f"  Has LLM: {metadata.has_llm}")
    print(f"  Has agreement: {metadata.has_agreement}")
    
    return EnhancedInsightFrame(
        data=data_frame,
        token_frame=token_frame,
        agreement_frame=agreement_frame,
        metadata=metadata,
    )


def load_insights_from_files(file_paths: Sequence[str]) -> EnhancedInsightFrame:
    """Load insights from specific files (backward compatible).
    
    Args:
        file_paths: List of paths to insight JSON files
        
    Returns:
        EnhancedInsightFrame with loaded data
    """
    all_records = []
    source_files = []
    
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Try to load as sharded summaries
        try:
            records = load_sharded_summaries(path)
            all_records.extend(records)
            source_files.append(str(path))
        except ValueError:
            # Try to load as regular JSON
            data = load_json_file(path)
            if isinstance(data, list):
                all_records.extend(data)
                source_files.append(str(path))
    
    if not all_records:
        raise ValueError("No records found in provided files")
    
    # Flatten records
    flattened = []
    for record in all_records:
        model_type = record.get("graph_type")
        if model_type in ["tokens", None] or record.get("method", "").endswith("llm"):
            flattened.append(flatten_llm_record(record))
        else:
            flattened.append(flatten_gnn_record(record))
    
    data_frame = pd.DataFrame(flattened)
    token_frame = create_token_frame(all_records)
    
    metadata = InsightMetadata(
        total_records=len(data_frame),
        datasets=data_frame["dataset"].dropna().unique().tolist() if "dataset" in data_frame.columns else [],
        graph_types=data_frame["graph_type"].dropna().unique().tolist() if "graph_type" in data_frame.columns else [],
        methods=data_frame["method"].dropna().unique().tolist() if "method" in data_frame.columns else [],
        has_gnn="gnn" in data_frame["model_type"].values if "model_type" in data_frame.columns else False,
        has_llm="llm" in data_frame["model_type"].values if "model_type" in data_frame.columns else False,
        has_agreement=False,
        source_files=source_files,
    )
    
    return EnhancedInsightFrame(
        data=data_frame,
        token_frame=token_frame,
        agreement_frame=None,
        metadata=metadata,
    )

