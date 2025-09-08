from __future__ import annotations
from dataclasses import dataclass
import sys
import subprocess


@dataclass
class ExplainConfig:
    method: str = "subgraphx"
    # For the legacy SubgraphX/AutoGOAL runner most params are internal.


def run_subgraphx_autogoal(cfg: ExplainConfig) -> None:
    """
    Execute the legacy AutoGOAL-based SubgraphX optimization entry.
    Note: The legacy script expects its own paths (best model, data dirs) in-file.
    """
    _ = cfg  # Reserved for future parameterization
    args = [sys.executable, "-m", "src.explain.subgraphx.main"]
    subprocess.run(args, check=True)
