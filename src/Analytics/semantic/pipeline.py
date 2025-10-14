from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from .common.config import SemanticConfig, load_config
from .common.data_loader import GraphArtifactLoader
from .common.models import GraphSemanticSummary
from .common.outputs import summaries_to_frame, tokens_to_frame, write_csv
from ..token.analysis import analyse_graphsvx, analyse_subgraphx


class SemanticPipeline:
    """Coordinates semantic token, density, and context analytics."""

    def __init__(self, cfg: SemanticConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = GraphArtifactLoader(cfg.nx_root, cfg.pyg_root)

    def run_tokens(self) -> None:
        for entry in self.cfg.graphsvx:
            summaries, aggregate = analyse_graphsvx(entry, self.loader, self.cfg.stopwords)
            self._emit_outputs(summaries, aggregate, entry.dataset, entry.graph_type)
        for entry in self.cfg.subgraphx:
            summaries, aggregate = analyse_subgraphx(entry, self.loader, self.cfg.stopwords, self.output_dir)
            # Always merge results from shard files to avoid OOM
            summaries = self._merge_shard_results(entry.dataset, entry.graph_type)
            # Don't call _emit_outputs since merge already created the files
            # Just write the aggregate file
            core_name = f"{entry.dataset.replace('/', '_')}_{entry.graph_type}"
            folder = self.output_dir / core_name
            folder.mkdir(parents=True, exist_ok=True)
            from .common.outputs import write_csv
            if not aggregate.empty:
                aggregate_copy = aggregate.copy()
                aggregate_copy["dataset"] = entry.dataset
                aggregate_copy["graph_type"] = entry.graph_type
                write_csv(aggregate_copy, folder / "aggregate.csv")

    def _emit_outputs(
        self,
        summaries: list[GraphSemanticSummary],
        aggregate: pd.DataFrame,
        dataset: str,
        graph_type: str,
    ) -> None:
        core_name = f"{dataset.replace('/', '_')}_{graph_type}"
        folder = self.output_dir / core_name
        folder.mkdir(parents=True, exist_ok=True)
        tokens_df = tokens_to_frame(summaries, dataset, graph_type)
        summary_df = summaries_to_frame(summaries, dataset, graph_type)
        if not aggregate.empty:
            aggregate = aggregate.copy()
            aggregate["dataset"] = dataset
            aggregate["graph_type"] = graph_type
        write_csv(tokens_df, folder / "tokens.csv")
        write_csv(summary_df, folder / "summary.csv")
        write_csv(aggregate, folder / "aggregate.csv")

    def _merge_shard_results(self, dataset: str, graph_type: str) -> list[GraphSemanticSummary]:
        """Merge results from shard CSV files into a single list of summaries."""
        from .common.models import GraphSemanticSummary

        core_name = f"{dataset.replace('/', '_')}_{graph_type}"
        folder = self.output_dir / core_name

        summaries = []
        summary_files = list(folder.glob("summary_shard*.csv"))
        token_files = list(folder.glob("tokens_shard*.csv"))

        if not summary_files:
            return summaries

        # Merge summary files
        summary_dfs = []
        for summary_file in sorted(summary_files):
            df = pd.read_csv(summary_file)
            summary_dfs.append(df)
        if summary_dfs:
            merged_summary_df = pd.concat(summary_dfs, ignore_index=True)
            write_csv(merged_summary_df, folder / "summary.csv")

        # Merge token files
        token_dfs = []
        for token_file in sorted(token_files):
            df = pd.read_csv(token_file)
            token_dfs.append(df)
        if token_dfs:
            merged_token_df = pd.concat(token_dfs, ignore_index=True)
            write_csv(merged_token_df, folder / "tokens.csv")

        # Note: We return empty list since summaries are now written to merged files
        # The pipeline will handle the output correctly
        return summaries


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/semantic_analysis_config.json"),
        help="Semantic analytics configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/general"),
        help="Directory for semantic analytics artefacts.",
    )
    return parser
