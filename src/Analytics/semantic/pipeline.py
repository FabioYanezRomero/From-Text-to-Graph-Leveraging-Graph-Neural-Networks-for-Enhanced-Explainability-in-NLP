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
            summaries, aggregate = analyse_subgraphx(entry, self.loader, self.cfg.stopwords)
            self._emit_outputs(summaries, aggregate, entry.dataset, entry.graph_type)

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
        default=Path("outputs/analytics/semantic"),
        help="Directory for semantic analytics artefacts.",
    )
    return parser
