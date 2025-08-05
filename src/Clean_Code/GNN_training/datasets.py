"""datasets.py

LazyGraphDataset: a memory-efficient dataset that streams graphs from disk on
first access instead of loading everything into RAM at construction time.

Assumptions
-----------
• Graph files are stored in a directory (*root_dir*).  Each file is either:
    – a single `torch_geometric.data.Data` object, or
    – a *list* of such objects that belong to the same split (legacy export).
• Files can be in ``.pt`` (preferred) or ``.pkl`` format.

The dataset builds a lightweight index mapping **global graph index → (file
path, local index inside the file)** so that `__getitem__` can locate the graph
quickly.  Only the first graph of the first file is temporarily loaded to
retrieve `num_node_features` and an initial guess of `num_classes`.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import List, Tuple

import torch
from torch_geometric.data import Data, Dataset

__all__ = ["LazyGraphDataset", "load_graph_data"]


class LazyGraphDataset(Dataset):
    """Stream graphs on-demand.

    Parameters
    ----------
    root_dir: str
        Directory that contains ``*.pt`` or ``*.pkl`` (pickled) graph files.
    """

    def __init__(self, root_dir: str):
        super().__init__(root=root_dir)
        self.root_dir = root_dir

        # Prefer .pt, otherwise .pkl
        pt_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        pkl_files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if pt_files:
            self.file_paths = pt_files
            self.file_type = "pt"
        elif pkl_files:
            self.file_paths = pkl_files
            self.file_type = "pkl"
        else:
            raise RuntimeError(f"No .pt or .pkl graphs found in {root_dir}")

        # Build (file_idx, local_idx) index list without holding graphs in RAM
        self.index: List[Tuple[int, int]] = []
        for file_idx, fp in enumerate(self.file_paths):
            # For speed we *only* need len() of the file. We still need to load
            # once, but we discard it immediately.
            batch = self._load_file(fp)
            if isinstance(batch, list):
                n_graphs = len(batch)
            else:
                n_graphs = 1
            self.index.extend([(file_idx, i) for i in range(n_graphs)])
            # release memory quickly
            del batch

        # Read the very first graph to get feature / class info
        first_graph = self[0]
        self._num_node_features = first_graph.num_node_features
        if hasattr(first_graph, "y"):
            self._num_classes = int(torch.unique(first_graph.y).numel())
        else:
            self._num_classes = 1  # fallback
        # Clean up
        del first_graph

        print(
            f"LazyGraphDataset initialised with {len(self)} graphs, "
            f"{self._num_node_features} node features, ~{self._num_classes} classes",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_file(self, path: str):
        if self.file_type == "pt":
            return torch.load(path, weights_only=False)
        # pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Required by torch_geometric.data.Dataset
    # ------------------------------------------------------------------
    def len(self) -> int:  # noqa: D401, naming matches pytorch geometric
        return len(self.index)

    def get(self, idx: int) -> Data:  # type: ignore[override]
        file_idx, local_idx = self.index[idx]
        fp = self.file_paths[file_idx]
        batch = self._load_file(fp)
        if isinstance(batch, list):
            data = batch[local_idx]
        else:
            # When file stores a single graph, local_idx must be 0
            assert local_idx == 0, "Index mismatch for single-graph file"
            data = batch
        return data

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def num_node_features(self) -> int:  # noqa: D401
        return self._num_node_features

    @property
    def num_classes(self) -> int:  # noqa: D401
        return self._num_classes


# ---------------------------------------------------------------------------
# Data-loader helper mirroring original API
# ---------------------------------------------------------------------------
from torch_geometric.loader import DataLoader  # noqa: E402, isort:skip


def load_graph_data(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """Return (dataset, dataloader) like Clean_Code.GNN_Training.data_loader."""

    dataset = LazyGraphDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader
