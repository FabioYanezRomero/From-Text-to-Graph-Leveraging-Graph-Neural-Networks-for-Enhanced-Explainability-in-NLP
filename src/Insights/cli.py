from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

from .metrics import summarize_records
from .readers import load_graphsvx_records, load_subgraphx_records
from .reporting import (
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

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise explainability artefacts from GraphSVX/SubgraphX.",
    )
    parser.add_argument(
        "--graphsvx-json",
        nargs="*",
        default=[],
        help="Paths to GraphSVX result JSON files.",
    )
    parser.add_argument(
        "--subgraphx-json",
        nargs="*",
        default=[],
        help="Paths to SubgraphX result JSON files.",
    )
    parser.add_argument("--graphsvx-dataset", help="Override dataset name for GraphSVX records.")
    parser.add_argument("--graphsvx-graph-type", help="Override graph type for GraphSVX records.")
    parser.add_argument("--graphsvx-run-id", help="Override run id for GraphSVX records.")
    parser.add_argument("--subgraphx-dataset", help="Override dataset name for SubgraphX records.")
    parser.add_argument("--subgraphx-graph-type", help="Override graph type for SubgraphX records.")
    parser.add_argument("--subgraphx-run-id", help="Override run id for SubgraphX records.")
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
        "--dry-run",
        action="store_true",
        help="Print summary statistics without writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    records = collect_records(args)
    if not records:
        parser.error("No explanations found. Provide at least one JSON path.")

    summaries = summarize_records(
        records,
        sufficiency_threshold=args.sufficiency_threshold,
        top_k=args.top_k,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
