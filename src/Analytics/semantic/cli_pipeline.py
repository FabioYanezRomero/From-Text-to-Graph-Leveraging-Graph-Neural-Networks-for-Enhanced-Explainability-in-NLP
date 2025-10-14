from __future__ import annotations

from .pipeline import SemanticPipeline, build_argument_parser, load_config

try:  # pragma: no cover - optional dependency for compatibility
    from Analytics import semantic_analysis as legacy_semantic_module  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    legacy_semantic_module = None


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser("Run full semantic analytics pipeline (tokens, summary, aggregate).")
    args = parser.parse_args(argv)
    cfg = load_config(args.config.resolve())
    if legacy_semantic_module is not None:
        try:
            legacy_semantic_module._GLOBAL_STOPWORDS = set(cfg.stopwords)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best-effort
            pass
    pipeline = SemanticPipeline(cfg, args.output_dir.resolve())
    pipeline.run_tokens()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
