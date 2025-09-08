from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ..registry import GRAPH_BUILDERS


@dataclass
class BuildArgs:
    graph_type: str
    dataset: str
    subsets: List[str]
    batch_size: int = 256
    device: str = "cuda:0"
    output_dir: str = "./outputs/graphs"


class BaseGraphBuilder:
    name = "base"

    def process_dataset(self, args: BuildArgs) -> None:
        raise NotImplementedError


@GRAPH_BUILDERS.register("constituency")
class ConstituencyBuilder(BaseGraphBuilder):
    name = "constituency"

    def process_dataset(self, args: BuildArgs) -> None:
        # Delegate to the new intuitive path (shim to legacy)
        from src.graph_builders import tree_generator as tg
        tg.process_dataset(
            graph_type="constituency",
            dataset=args.dataset,
            subsets=args.subsets,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )


@GRAPH_BUILDERS.register("syntactic")
class SyntacticBuilder(BaseGraphBuilder):
    name = "syntactic"

    def process_dataset(self, args: BuildArgs) -> None:
        from src.graph_builders import tree_generator as tg
        tg.process_dataset(
            graph_type="syntactic",
            dataset=args.dataset,
            subsets=args.subsets,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )


# Note: Additional builders (e.g., n-grams, skip-grams, KNN, windowed) can be
# added by creating new classes here or in separate modules and registering with
# @GRAPH_BUILDERS.register("new_name").
