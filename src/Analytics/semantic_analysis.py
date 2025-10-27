"""Backward-compatible entry point for semantic analytics.

The semantic analytics tooling now lives in ``src.Analytics.semantic``.
This wrapper exposes prior helpers (e.g. ``_default_stopwords``) and delegates
to the modular pipeline so existing scripts keep working:

    python3 -m src.Analytics.semantic.cli_pipeline --config configs/semantic_analysis_config.json
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

from src.Analytics.semantic.common.config import (
    _default_stopwords as _config_default_stopwords,
    load_config,
)
from src.Analytics.semantic.pipeline import SemanticPipeline, build_argument_parser


_GLOBAL_STOPWORDS: Set[str] = set()


def _default_stopwords() -> Set[str]:
    """Backward-compatible accessor returning the default stopword inventory."""
    return set(_config_default_stopwords())


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser("Run semantic analytics (deprecated wrapper).")
    args = parser.parse_args(argv)
    cfg = load_config(args.config.resolve())
    global _GLOBAL_STOPWORDS
    _GLOBAL_STOPWORDS = set(cfg.stopwords)
    pipeline = SemanticPipeline(cfg, args.output_dir.resolve())
    pipeline.run_tokens()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
