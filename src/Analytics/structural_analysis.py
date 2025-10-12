"""Explore how structural metrics relate to explanation behaviour.

Extended: compute per-graph analytics (degree stats, centralities, NSCC) and
persist them alongside labels and predicted classes for later aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse
from types import SimpleNamespace
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import skew as scipy_skew  # type: ignore

from src.Insights.providers import GraphArtifactProvider
from tqdm import tqdm

from .utils import InsightFrame, default_argument_parser, load_insights

STRUCT_COLUMNS = [
    "num_nodes",
    "num_edges",
    "struct_induced_num_nodes",
    "struct_induced_num_edges",
    "struct_density",
    "struct_components",
    "struct_boundary_edges",
    "struct_cut_ratio",
    "struct_avg_shortest_path",
]

EXPLANATION_COLUMNS = [
    "prediction_confidence",
    "fidelity_drop",
    "maskout_effect",
    "sparsity",
    "minimal_coalition_size",
    "insertion_auc",
    "minimal_coalition_confidence",
]

GRAPH_METRIC_COLUMNS = [
    "num_nodes",
    "num_edges",
    "avg_degree",
    "max_degree",
    "var_degree",
    "skew_degree",
    "nscc",
    "avg_betweenness",
    "max_betweenness",
    "avg_closeness",
    "max_closeness",
    "avg_eigenvector",
    "max_eigenvector",
]


def _safe_frame(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Select available columns, ignoring the ones that are missing."""
    available = [col for col in columns if col in frame.columns]
    return frame[available].copy()


def _hist(series: pd.Series, title: str, path: Path) -> None:
    """Persist a histogram for a structural series."""
    values = series.dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color="#C44E52", edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _scatter(x: pd.Series, y: pd.Series, title: str, path: Path) -> None:
    """Plot a scatter chart relating structural and explanation metrics."""
    x_values = x.dropna()
    y_values = y.dropna()
    aligned = pd.concat([x_values, y_values], axis=1, join="inner").dropna()
    if aligned.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(aligned.iloc[:, 0], aligned.iloc[:, 1], alpha=0.4, s=16, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_structural_analysis(insight: InsightFrame, output_dir: Path) -> dict:
    """Create structural distribution plots and correlation artefacts."""
    frame = insight.data
    plots_dir = output_dir / "structural_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    structural_frame = _safe_frame(frame, STRUCT_COLUMNS)
    explanation_frame = _safe_frame(frame, EXPLANATION_COLUMNS)

    for column in structural_frame.columns:
        _hist(structural_frame[column], f"{column.replace('_', ' ').title()} Distribution", plots_dir / f"{column}_hist.png")

    correlation_matrix = pd.concat([structural_frame, explanation_frame], axis=1).corr()
    corr_path = output_dir / "structure_explanation_correlation.csv"
    correlation_matrix.to_csv(corr_path)

    scatter_pairs = [
        ("struct_density", "fidelity_drop"),
        ("struct_density", "insertion_auc"),
        ("struct_induced_num_nodes", "minimal_coalition_size"),
        ("num_nodes", "sparsity"),
    ]
    for x_col, y_col in scatter_pairs:
        if x_col in frame.columns and y_col in frame.columns:
            _scatter(
                frame[x_col],
                frame[y_col],
                f"{x_col} vs {y_col}",
                plots_dir / f"{x_col}_vs_{y_col}.png",
            )

    centrality_cols = [col for col in frame.columns if col.startswith("centrality_")]
    centrality_correlations = {}
    if centrality_cols:
        centrality_correlations = (
            frame[centrality_cols + [col for col in EXPLANATION_COLUMNS if col in frame.columns]]
            .corr()
            .loc[centrality_cols, :]
        )
        centrality_correlations.to_csv(output_dir / "centrality_alignment_correlations.csv")

    return {
        "structural_columns": list(structural_frame.columns),
        "explanation_columns": list(explanation_frame.columns),
        "centrality_columns": centrality_cols,
        "correlation_matrix_path": str(corr_path),
    }


def _nscc(graph: nx.Graph) -> int:
    if graph.is_directed():
        return nx.number_strongly_connected_components(graph)  # type: ignore[arg-type]
    return nx.number_connected_components(graph)


def _ensure_safe_globals() -> None:
    """Register PyG storage classes so torch.load accepts full Data objects."""
    try:
        from torch.serialization import add_safe_globals
        import torch_geometric.data.data as pyg_data
        import torch_geometric.data.storage as pyg_storage

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
        # If torch or torch_geometric are unavailable, proceed without changes
        pass


def _degree_stats(graph: nx.Graph) -> dict:
    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
    if degrees.size == 0:
        return {
            "avg_degree": 0.0,
            "max_degree": 0.0,
            "var_degree": 0.0,
            "skew_degree": 0.0,
        }
    return {
        "avg_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "var_degree": float(degrees.var() if degrees.size > 1 else 0.0),
        "skew_degree": float(scipy_skew(degrees) if degrees.size > 2 else 0.0),
    }


def _centralities(graph: nx.Graph) -> dict:
    if graph.number_of_nodes() == 0:
        zero = {"avg": 0.0, "max": 0.0}
        return {
            "betweenness": zero,
            "closeness": zero,
            "eigenvector": zero,
        }
    # Betweenness/closeness (normalized where applicable)
    betw = nx.betweenness_centrality(graph, normalized=True)
    clos = nx.closeness_centrality(graph)
    # Eigenvector: guard for convergence issues on directed graphs
    try:
        if graph.is_directed():
            ev = nx.eigenvector_centrality_numpy(graph.to_undirected())
        else:
            ev = nx.eigenvector_centrality_numpy(graph)
    except Exception:
        # Fallback to zeros on failure
        ev = {n: 0.0 for n in graph.nodes()}

    def _avg_max(d: dict) -> tuple[float, float]:
        vals = np.array(list(d.values()), dtype=float)
        if vals.size == 0:
            return 0.0, 0.0
        return float(vals.mean()), float(vals.max())

    betw_avg, betw_max = _avg_max(betw)
    clos_avg, clos_max = _avg_max(clos)
    ev_avg, ev_max = _avg_max(ev)
    return {
        "betweenness": {"avg": betw_avg, "max": betw_max},
        "closeness": {"avg": clos_avg, "max": clos_max},
        "eigenvector": {"avg": ev_avg, "max": ev_max},
    }


def run_graph_analytics(
    insight: InsightFrame,
    output_dir: Path,
    *,
    backbone: str | None = None,
    split: str | None = None,
) -> Path:
    """Compute per-graph analytics and persist a CSV.

    Uses GraphArtifactProvider to load NetworkX graphs corresponding to
    each insight row, and computes degree/centrality metrics plus NSCC.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_safe_globals()
    provider = GraphArtifactProvider(strict=True)
    rows = []
    # Deduplicate by (dataset, graph_type, graph_index) to avoid repeated rows
    key_cols = [c for c in ("dataset", "graph_type", "graph_index") if c in insight.data.columns]
    frame = insight.data
    if not key_cols:
        raise ValueError("InsightFrame missing required columns for graph lookup.")

    seen = set()
    # Best-effort descriptor (stable in config-mode)
    try:
        first_ds = str(frame.iloc[0].get("dataset")) if not frame.empty else ""
        first_gt = str(frame.iloc[0].get("graph_type")) if not frame.empty else ""
        desc = f"Struct[{first_ds}:{first_gt}]"
    except Exception:
        desc = "Struct[graphs]"

    for _, r in tqdm(frame.iterrows(), total=len(frame), desc=desc):
        key = tuple(r[c] for c in key_cols)
        if key in seen:
            continue
        seen.add(key)
        dataset = str(r.get("dataset"))
        graph_type = str(r.get("graph_type")) if r.get("graph_type") is not None else None
        graph_index = int(r.get("graph_index"))

        # Build a record for the provider. If dataset already contains backbone (e.g. 'SetFit/ag_news'), use it.
        extras = {}
        if backbone:
            extras["backbone"] = backbone
        if split:
            extras["split"] = split
        record = SimpleNamespace(dataset=dataset, graph_type=graph_type, graph_index=graph_index, extras=extras)

        info = provider(record)
        if info is None:
            continue
        G = info.graph
        deg = _degree_stats(G)
        cen = _centralities(G)
        rows.append(
            {
                "dataset": dataset,
                "graph_type": graph_type,
                "graph_index": graph_index,
                "label": r.get("label"),
                "prediction_class": r.get("prediction_class"),
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "avg_degree": deg["avg_degree"],
                "max_degree": deg["max_degree"],
                "var_degree": deg["var_degree"],
                "skew_degree": deg["skew_degree"],
                "nscc": _nscc(G),
                "avg_betweenness": cen["betweenness"]["avg"],
                "max_betweenness": cen["betweenness"]["max"],
                "avg_closeness": cen["closeness"]["avg"],
                "max_closeness": cen["closeness"]["max"],
                "avg_eigenvector": cen["eigenvector"]["avg"],
                "max_eigenvector": cen["eigenvector"]["max"],
            }
        )

    df = pd.DataFrame(rows)
    out_path = output_dir / "structural_graph_analytics.csv"
    df.to_csv(out_path, index=False)
    _write_group_correlations(df, output_dir)
    return out_path


def _write_group_correlations(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = [col for col in GRAPH_METRIC_COLUMNS if col in frame.columns]
    if not metrics:
        return

    def _corr(subset: pd.DataFrame) -> pd.DataFrame | None:
        subset = subset.dropna(subset=metrics)
        if subset.shape[0] < 2:
            return None
        return subset[metrics].corr()

    def _save(name: str, corr: pd.DataFrame | None) -> None:
        if corr is None:
            return
        path = output_dir / f"structural_correlations_{name}.csv"
        corr.to_csv(path)

    # Overall correct / incorrect split
    if {"label", "prediction_class"}.issubset(frame.columns):
        tmp = frame.copy()
        tmp["is_correct"] = tmp["label"] == tmp["prediction_class"]
        for flag, group in tmp.groupby("is_correct"):
            corr = _corr(group)
            suffix = "correct" if flag else "incorrect"
            _save(suffix, corr)

        # Per-class correlations (by true label)
        if "label" in tmp.columns:
            label_offset = 0
            numeric_labels = pd.to_numeric(tmp["label"], errors="coerce")
            if numeric_labels.notna().any() and numeric_labels.min() == 0:
                label_offset = 1

            def _format_label(label_val) -> str:
                try:
                    iv = int(label_val)
                    if float(iv) == float(label_val):
                        display = iv + label_offset
                        return f"class{display}"
                except Exception:
                    pass
                return f"class_{label_val}".replace("/", "_")

            for label_value, group in tmp.groupby("label"):
                corr = _corr(group)
                label_suffix = _format_label(label_value)
                _save(label_suffix, corr)
                for flag, sub in group.groupby("is_correct"):
                    scorr = _corr(sub)
                    outcome = "correct" if flag else "incorrect"
                    _save(f"{label_suffix}_{outcome}", scorr)


def _load_prediction_lookup(paths: List[Path]) -> pd.DataFrame:
    """Load graph_index -> (label, prediction_class) from JSON lists."""
    rows = []
    for p in paths:
        try:
            payload = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            gi = item.get("graph_index")
            label = item.get("label")
            pred = item.get("prediction") or {}
            pred_class = pred.get("class") if isinstance(pred, dict) else None
            if pred_class is None:
                pred_class = item.get("prediction_class")
            if isinstance(gi, int):
                rows.append({"graph_index": gi, "label": label, "prediction_class": pred_class})
    return pd.DataFrame(rows)


def _load_struct_config(config_path: Path) -> List[dict]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tasks: List[dict] = []
    for section in ("graphsvx", "subgraphx"):
        for entry in cfg.get(section, []) or []:
            dataset = entry["dataset"]
            graph_type = entry["graph_type"]
            split = entry["split"]
            backbone = entry["backbone"]
            lookup_paths: List[Path] = []
            if section == "graphsvx" and entry.get("path"):
                lookup_paths.append(Path(entry["path"]).resolve())
            if entry.get("prediction_lookup"):
                lookup_paths.extend(Path(p).resolve() for p in entry["prediction_lookup"])
            lookup = _load_prediction_lookup(lookup_paths)
            if lookup.empty:
                continue
            lookup = lookup.sort_values(by=["graph_index"]).drop_duplicates(subset=["graph_index"], keep="first")
            df = lookup.copy()
            df["dataset"] = dataset
            df["graph_type"] = graph_type
            tasks.append({
                "dataset": dataset,
                "graph_type": graph_type,
                "backbone": backbone,
                "split": split,
                "frame": df,
            })
    return tasks


def _run_structural_from_config(config_path: Path, output_dir: Path) -> None:
    tasks = _load_struct_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tqdm(tasks, desc="Struct[datasets]"):
        df = task["frame"]
        insight_like = SimpleNamespace(data=df)
        subdir = output_dir / f"{task['dataset'].replace('/', '_')}_{task['graph_type']}"
        subdir.mkdir(parents=True, exist_ok=True)
        run_graph_analytics(
            insight_like,
            subdir,
            backbone=task["backbone"],
            split=task["split"],
        )


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for structural analytics."""
    # Pre-parse to detect config-mode and avoid requiring positional insight_paths
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)

    if known.config:
        parser = argparse.ArgumentParser(description="Analyse structural metrics (config mode).")
        parser.add_argument("--config", type=Path, required=True, help="Semantic-style config to batch structural analytics")
        parser.add_argument("--output-dir", type=Path, default=Path("outputs/analytics/structural"))
        args = parser.parse_args(argv)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _ensure_safe_globals()
        _run_structural_from_config(args.config.resolve(), args.output_dir)
        return 0

    # Legacy insight-driven mode
    parser = default_argument_parser("Analyse structural metrics and their relation to explanations.")
    parser.add_argument("--backbone", help="Backbone override when resolving graph artefacts (e.g., SetFit)")
    parser.add_argument("--split", help="Split override when resolving graph artefacts (e.g., test, validation)")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_safe_globals()
    insight = load_insights(args.insight_paths)
    summary = run_structural_analysis(insight, args.output_dir)
    # Also compute graph analytics with optional overrides
    try:
        run_graph_analytics(
            insight,
            args.output_dir,
            backbone=getattr(args, "backbone", None),
            split=getattr(args, "split", None),
        )
    except Exception:
        pass
    summary_path = args.output_dir / "structural_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
