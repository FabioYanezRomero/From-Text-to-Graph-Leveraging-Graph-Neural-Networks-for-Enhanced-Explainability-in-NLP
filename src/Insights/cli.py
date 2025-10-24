from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import networkx as nx

from .metrics import pairwise_agreement, summarize_records
from .providers import GraphArtifactProvider
from .llm_providers import LLMExplanationProvider
from .readers import (
    discover_llm_records,
    discover_graphsvx_runs,
    discover_subgraphx_runs,
    load_llm_records,
    load_graphsvx_records,
    load_graphsvx_run_records,
    load_subgraphx_records,
    load_subgraphx_run_records,
)
from .reporting import (
    export_agreement_csv,
    export_agreement_json,
    export_minimal_size_histogram,
    export_summaries_csv,
    export_summaries_json,
)
from .records import ExplanationRecord

DEFAULT_GRAPH_TYPES = ("skipgrams", "window", "constituency")


def _infer_graph_type(path: Path) -> str | None:
    for part in path.parts:
        lower = part.lower()
        for candidate in DEFAULT_GRAPH_TYPES:
            if candidate in lower:
                return candidate
    return None


def _infer_run_id(path: Path) -> str | None:
    for parent in path.parents:
        if parent.name and parent.name[0].isdigit():
            return parent.name
    return None


def collect_records(args: argparse.Namespace) -> List[ExplanationRecord]:
    records: List[ExplanationRecord] = []
    coalition_base = Path(args.coalition_base) if args.coalition_base else None

    enable_subgraphx_predictions = not getattr(args, "disable_subgraphx_predictions", False)

    for path_str in args.graphsvx_json:
        path = Path(path_str)
        inferred_graph_type = _infer_graph_type(path)
        inferred_run = _infer_run_id(path)
        dataset = args.graphsvx_dataset
        graph_type = args.graphsvx_graph_type or inferred_graph_type
        run_id = args.graphsvx_run_id or inferred_run

        loaded = load_graphsvx_records(
            path,
            dataset=dataset,
            graph_type=graph_type,
            run_id=run_id,
            coalition_base=coalition_base,
        )
        records.extend(loaded)

    run_dirs: List[Path] = [Path(path_str) for path_str in getattr(args, "graphsvx_run_dirs", [])]
    if getattr(args, "graphsvx_root", None):
        run_dirs.extend(discover_graphsvx_runs(Path(args.graphsvx_root)))

    seen: set[Path] = set()
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        loaded = load_graphsvx_run_records(
            resolved,
            dataset=args.graphsvx_dataset,
            graph_type=args.graphsvx_graph_type,
            run_id=args.graphsvx_run_id,
        )
        records.extend(loaded)

    for path_str in args.subgraphx_json:
        path = Path(path_str)
        inferred_graph_type = _infer_graph_type(path)
        inferred_run = _infer_run_id(path)
        dataset = args.subgraphx_dataset
        graph_type = args.subgraphx_graph_type or inferred_graph_type
        run_id = args.subgraphx_run_id or inferred_run

        loaded = load_subgraphx_records(
            path,
            dataset=dataset,
            graph_type=graph_type,
            run_id=run_id,
        )
        records.extend(loaded)

    subgraphx_run_dirs: List[Path] = [Path(p) for p in getattr(args, "subgraphx_run_dirs", [])]
    if getattr(args, "subgraphx_root", None):
        subgraphx_run_dirs.extend(discover_subgraphx_runs(Path(args.subgraphx_root)))

    seen_sx: set[Path] = set()
    for run_dir in subgraphx_run_dirs:
        resolved = run_dir.resolve()
        if not resolved.exists():
            continue

        candidate_dirs: List[Path]
        results_path = resolved / "results.pkl"
        if results_path.exists():
            candidate_dirs = [resolved]
        else:
            candidate_dirs = [
                candidate.resolve()
                for candidate in discover_subgraphx_runs(resolved)
            ]

        for candidate in candidate_dirs:
            candidate_path = candidate.resolve()
            if candidate_path in seen_sx or not candidate_path.exists():
                continue
            seen_sx.add(candidate_path)
            loaded = load_subgraphx_run_records(
                candidate_path,
                dataset=args.subgraphx_dataset,
                graph_type=args.subgraphx_graph_type,
                run_id=args.subgraphx_run_id,
                enable_predictions=enable_subgraphx_predictions,
            )
            records.extend(loaded)

    llm_paths: List[Path] = [Path(p) for p in getattr(args, "llm_records", [])]
    if getattr(args, "llm_root", None):
        llm_paths.extend(discover_llm_records(Path(args.llm_root)))

    seen_llm: set[Path] = set()
    for raw_path in llm_paths:
        resolved = raw_path.resolve()
        if resolved in seen_llm or not resolved.exists():
            continue
        seen_llm.add(resolved)
        loaded = load_llm_records(
            resolved,
            dataset=args.llm_dataset,
            graph_type=args.llm_graph_type,
            run_id=args.llm_run_id,
        )
        records.extend(loaded)

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise explainability artefacts from GraphSVX/SubgraphX/TokenSHAP.",
    )
    parser.add_argument(
        "--graphsvx-json",
        nargs="*",
        default=[],
        help="Paths to GraphSVX result JSON files.",
    )
    parser.add_argument(
        "--graphsvx-run-dirs",
        nargs="*",
        default=[],
        help="GraphSVX run directories under outputs/gnn_models (each containing results.pkl).",
    )
    parser.add_argument(
        "--graphsvx-root",
        help="Root directory to discover GraphSVX runs automatically (e.g., outputs/gnn_models).",
    )
    parser.add_argument(
        "--graph-root",
        default="outputs/graphs",
        help="Root directory containing NetworkX graph artefacts.",
    )
    parser.add_argument(
        "--pyg-root",
        default="outputs/pyg_graphs",
        help="Root directory containing PyG graph tensors.",
    )
    parser.add_argument(
        "--subgraphx-json",
        nargs="*",
        default=[],
        help="Paths to SubgraphX result JSON files.",
    )
    parser.add_argument(
        "--subgraphx-run-dirs",
        nargs="*",
        default=[],
        help="SubgraphX run directories under outputs/gnn_models (each containing results.pkl).",
    )
    parser.add_argument(
        "--subgraphx-root",
        help="Root directory to discover SubgraphX runs automatically.",
    )
    parser.add_argument("--graphsvx-dataset", help="Override dataset name for GraphSVX records.")
    parser.add_argument("--graphsvx-graph-type", help="Override graph type for GraphSVX records.")
    parser.add_argument("--graphsvx-run-id", help="Override run id for GraphSVX records.")
    parser.add_argument("--subgraphx-dataset", help="Override dataset name for SubgraphX records.")
    parser.add_argument("--subgraphx-graph-type", help="Override graph type for SubgraphX records.")
    parser.add_argument("--subgraphx-run-id", help="Override run id for SubgraphX records.")
    parser.add_argument(
        "--llm-records",
        nargs="*",
        default=[],
        help="Paths to TokenSHAP LLM record files (JSON or pickle).",
    )
    parser.add_argument(
        "--llm-root",
        help="Root directory to discover TokenSHAP record files automatically.",
    )
    parser.add_argument("--llm-dataset", help="Override dataset name for LLM records.")
    parser.add_argument("--llm-graph-type", help="Override graph type for LLM records.")
    parser.add_argument("--llm-run-id", help="Override run id for LLM records.")
    parser.add_argument(
        "--coalition-base",
        help="Base directory to resolve GraphSVX coalition CSV paths against.",
    )
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=0.9,
        help="Threshold (fraction of origin confidence) for minimal sufficiency.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top nodes to store per record.",
    )
    parser.add_argument("--output-json", help="Path to write detailed record summaries (JSON).")
    parser.add_argument("--output-csv", help="Path to write summaries as CSV (requires pandas).")
    parser.add_argument("--minimal-csv", help="Path to write minimal coalition size histogram CSV.")
    parser.add_argument(
        "--agreement-json",
        help="Path to write pairwise agreement metrics across explainers (JSON).",
    )
    parser.add_argument(
        "--agreement-csv",
        help="Path to write pairwise agreement metrics across explainers (CSV, requires pandas).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary statistics without writing files.",
    )
    parser.add_argument(
        "--disable-structure",
        action="store_true",
        help="Skip loading graph artefacts (structural metrics and token mapping remain null).",
    )
    parser.add_argument(
        "--disable-centrality",
        action="store_true",
        help="Disable centrality alignment even when structural graphs are available.",
    )
    parser.add_argument(
        "--disable-subgraphx-predictions",
        action="store_true",
        help="Skip recomputing predictions for SubgraphX shards (avoids large tensor loads).",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.9,
        help="Rank-biased overlap persistence parameter (0 < p < 1).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    records = collect_records(args)
    if not records:
        parser.error("No explanations found. Provide at least one JSON path.")

    graph_provider: GraphArtifactProvider | None = None
    centrality_funcs: Dict[str, Callable[[nx.Graph], Dict[int, float]]] | None = None
    if not args.disable_structure:
        try:
            graph_provider = GraphArtifactProvider(args.graph_root, args.pyg_root, strict=False)
            if not args.disable_centrality:
                centrality_funcs = {
                    "degree": lambda g: nx.degree_centrality(g.to_undirected()),
                    "betweenness": lambda g: nx.betweenness_centrality(g.to_undirected(), normalized=True),
                    "closeness": lambda g: nx.closeness_centrality(g.to_undirected()),
                }
        except Exception:  # pragma: no cover - graceful fallback
            graph_provider = None
            centrality_funcs = None

    llm_provider = LLMExplanationProvider()

    def _graph_lookup(record: ExplanationRecord):
        if graph_provider is None:
            return None
        try:
            return graph_provider(record)
        except Exception:
            return None

    def combined_provider(record: ExplanationRecord):
        graph_info = _graph_lookup(record)
        if graph_info is not None:
            return graph_info
        return llm_provider(record)

    provider_callable = combined_provider if (graph_provider or llm_provider) else (lambda _: None)
    centrality_callable = centrality_funcs if graph_provider and centrality_funcs else None

    summaries = summarize_records(
        records,
        sufficiency_threshold=args.sufficiency_threshold,
        top_k=args.top_k,
        graph_provider=provider_callable,
        centrality_funcs=centrality_callable,
    )

    if args.dry_run or (not args.output_json and not args.output_csv):
        print(f"Loaded {len(records)} explanations.", file=sys.stdout)
        minimal_sizes = [summary["minimal_coalition_size"] for summary in summaries if summary["minimal_coalition_size"] is not None]
        if minimal_sizes:
            avg_size = sum(minimal_sizes) / len(minimal_sizes)
            print(f"Average minimal coalition size (threshold={args.sufficiency_threshold}): {avg_size:.2f}", file=sys.stdout)
        else:
            print("No minimal coalitions satisfied the threshold.", file=sys.stdout)

    if args.output_json:
        export_summaries_json(
            records,
            Path(args.output_json),
            sufficiency_threshold=args.sufficiency_threshold,
            top_k=args.top_k,
            summaries=summaries,
        )

    if args.output_csv:
        try:
            export_summaries_csv(summaries, Path(args.output_csv))
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.minimal_csv:
        export_minimal_size_histogram(
            records,
            Path(args.minimal_csv),
            threshold=args.sufficiency_threshold,
        )

    agreement_entries = pairwise_agreement(
        records,
        top_k=args.top_k,
        rbo_p=args.rbo_p,
    )

    if args.agreement_json:
        export_agreement_json(agreement_entries, Path(args.agreement_json))

    if args.agreement_csv:
        try:
            export_agreement_csv(agreement_entries, Path(args.agreement_csv))
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.dry_run and agreement_entries:
        print(f"Computed {len(agreement_entries)} pairwise agreement entries.", file=sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
