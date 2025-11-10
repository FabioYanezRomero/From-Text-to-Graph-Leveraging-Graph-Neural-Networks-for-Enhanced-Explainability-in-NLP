#!/usr/bin/env python3
"""Export full per-module analytics tables for downstream aggregation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset


ANALYTIC_ROOTS: Mapping[str, Path] = {
    "auc": Path("outputs/analytics/auc"),
    "consistency": Path("outputs/analytics/consistency"),
    "progression": Path("outputs/analytics/progression"),
    "fidelity": Path("outputs/analytics/fidelity"),
}

METHOD_GRAPHS: Mapping[str, Sequence[str]] = {
    "graphsvx": ("skipgrams", "window"),
    "subgraphx": ("constituency", "syntactic"),
    "token_shap_llm": ("tokens",),
}

BASE_DROP_COLUMNS = {
    "method",
    "backbone",
    "dataset",
    "dataset_raw",
    "dataset_backbone",
    "graph_type",
    "run_id",
    "split",
    "graph_index",
}


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    label: str
    hf_id: str
    split: str
    text_field: str
    label_field: str
    label_text_field: str | None = None
    label_names: Mapping[int, str] | None = None


DATASETS: Mapping[str, DatasetSpec] = {
    "setfit_ag_news": DatasetSpec(
        slug="setfit_ag_news",
        label="AG News (SetFit)",
        hf_id="SetFit/ag_news",
        split="test",
        text_field="text",
        label_field="label",
        label_text_field="label_text",
    ),
    "stanfordnlp_sst2": DatasetSpec(
        slug="stanfordnlp_sst2",
        label="SST-2 (StanfordNLP)",
        hf_id="stanfordnlp/sst2",
        split="validation",
        text_field="sentence",
        label_field="label",
        label_text_field=None,
        label_names={0: "negative", 1: "positive"},
    ),
}


def dataset_text_lookup(spec: DatasetSpec) -> Dict[int, Dict[str, str | int | None]]:
    dataset = load_dataset(spec.hf_id, split=spec.split)
    lookup: Dict[int, Dict[str, str | int | None]] = {}
    for idx in range(len(dataset)):
        record = dataset[idx]
        text = record.get(spec.text_field)
        label_value = record.get(spec.label_field)
        label_text = None
        if spec.label_text_field and spec.label_text_field in record:
            label_text = record[spec.label_text_field]
        elif spec.label_names is not None and label_value in spec.label_names:
            label_text = spec.label_names[label_value]
        lookup[idx] = {
            "text": text,
            "label_id": int(label_value) if label_value is not None else None,
            "label_text": label_text,
        }
    return lookup


def analytic_path(kind: str, method: str, dataset: str, graph: str) -> Path:
    path = ANALYTIC_ROOTS[kind] / method / dataset / f"{graph}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} data for {dataset} · {method} · {graph}: {path}")
    return path


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep_cols = {"global_graph_index"}
    drop_cols = [col for col in BASE_DROP_COLUMNS if col in df.columns]
    trimmed = df.drop(columns=drop_cols, errors="ignore").copy()
    rename_map = {col: f"{prefix}_{col}" for col in trimmed.columns if col not in keep_cols}
    return trimmed.rename(columns=rename_map)


def to_bool(series: pd.Series) -> pd.Series:
    def cast(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return bool(value)

    return series.apply(cast)


def merge_analytics(dataset: str, method: str, graph: str) -> pd.DataFrame:
    fidelity_df = pd.read_csv(analytic_path("fidelity", method, dataset, graph))
    fidelity_df = fidelity_df.copy()
    fidelity_df["global_graph_index"] = fidelity_df["global_graph_index"].astype(int)

    merged = fidelity_df
    for kind in ("consistency", "progression", "auc"):
        df = pd.read_csv(analytic_path(kind, method, dataset, graph))
        df = df.copy()
        df["global_graph_index"] = df["global_graph_index"].astype(int)
        prefixed = prefix_columns(df, kind)
        merged = merged.merge(prefixed, on="global_graph_index", how="inner", validate="one_to_one")

    merged["is_correct"] = to_bool(merged["is_correct"])
    return merged


def enrich_module_frame(
    df: pd.DataFrame,
    spec: DatasetSpec,
    lookup: Mapping[int, Mapping[str, str | int | None]],
    method: str,
    graph: str,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["dataset_slug"] = spec.slug
    enriched["dataset_label"] = spec.label
    enriched["method"] = method
    enriched["graph_type"] = graph
    enriched["module_id"] = f"{method}:{graph}"
    enriched["global_graph_index"] = enriched["global_graph_index"].astype(int)
    enriched["text"] = enriched["global_graph_index"].map(lambda idx: lookup.get(int(idx), {}).get("text"))
    enriched["label_text"] = enriched["global_graph_index"].map(lambda idx: lookup.get(int(idx), {}).get("label_text"))
    enriched["label_id"] = enriched["global_graph_index"].map(lambda idx: lookup.get(int(idx), {}).get("label_id"))
    columns_order = [
        "dataset_slug",
        "dataset_label",
        "method",
        "graph_type",
        "module_id",
        "global_graph_index",
        "label_id",
        "label_text",
        "label",
        "prediction_class",
        "prediction_confidence",
        "is_correct",
        "text",
    ]
    remaining_cols = [col for col in enriched.columns if col not in columns_order]
    return enriched.loc[:, columns_order + remaining_cols]


def write_module_csv(
    df: pd.DataFrame,
    dataset_slug: str,
    method: str,
    graph: str,
    output_dir: Path,
) -> Path:
    module_dir = output_dir / dataset_slug
    module_dir.mkdir(parents=True, exist_ok=True)
    filename = f"module_dataset_{dataset_slug}_{method}_{graph}.csv"
    output_path = module_dir / filename
    df.to_csv(output_path, index=False)
    print(f"  • {method}:{graph} -> {output_path}")
    return output_path


def process_dataset(dataset_slug: str, output_dir: Path) -> List[Path]:
    if dataset_slug not in DATASETS:
        raise KeyError(f"Unknown dataset slug: {dataset_slug}")

    spec = DATASETS[dataset_slug]
    lookup = dataset_text_lookup(spec)
    written: List[Path] = []

    for method, graphs in METHOD_GRAPHS.items():
        for graph in graphs:
            try:
                merged = merge_analytics(dataset_slug, method, graph)
            except FileNotFoundError:
                continue
            enriched = enrich_module_frame(merged, spec, lookup, method, graph)
            written.append(write_module_csv(enriched, dataset_slug, method, graph, output_dir))

    if not written:
        raise RuntimeError(f"No module datasets were produced for '{dataset_slug}'.")
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-module analytics datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS.keys()),
        help="Dataset slug(s) to process (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/use_case/module_datasets"),
        help="Directory where module CSVs will be stored.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = args.dataset or sorted(DATASETS.keys())
    for dataset_slug in datasets:
        print(f"Dataset: {dataset_slug}")
        process_dataset(dataset_slug, args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

