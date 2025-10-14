from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _label_slug(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unlabeled"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().replace("/", "_").replace(" ", "_")
    if number.is_integer():
        return f"class{int(number)}"
    return str(value).strip().replace("/", "_").replace(" ", "_")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _split_by_correct(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if "is_correct" not in df.columns:
        return results
    for flag, subset in df.groupby("is_correct", dropna=False):
        if subset.empty:
            continue
        suffix = "correct" if bool(flag) else "incorrect"
        results[suffix] = subset
    return results


def _split_by_label(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if "label" not in df.columns:
        return results
    for label, subset in df.groupby("label", dropna=False):
        if subset.empty:
            continue
        results[_label_slug(label)] = subset
    return results


def export_tokens(folder: Path, token_df: pd.DataFrame, token_root: Path) -> None:
    if token_df.empty:
        return
    dest = token_root / folder.name / "csv"
    _write_csv(token_df, dest / "tokens.csv")
    for suffix, subset in _split_by_correct(token_df).items():
        _write_csv(subset, dest / f"tokens_{suffix}.csv")
    for label_slug, subset in _split_by_label(token_df).items():
        _write_csv(subset, dest / f"tokens_{label_slug}.csv")
        for suffix, split_subset in _split_by_correct(subset).items():
            _write_csv(split_subset, dest / f"tokens_{label_slug}_{suffix}.csv")


def export_summary(folder: Path, summary_df: pd.DataFrame, sparsity_root: Path, confidence_root: Path) -> None:
    if summary_df.empty:
        return
    for root in (sparsity_root, confidence_root):
        dest = root / folder.name / "csv"
        _write_csv(summary_df, dest / "summary.csv")
        for suffix, subset in _split_by_correct(summary_df).items():
            _write_csv(subset, dest / f"summary_{suffix}.csv")
        for label_slug, subset in _split_by_label(summary_df).items():
            _write_csv(subset, dest / f"summary_{label_slug}.csv")
            for suffix, split_subset in _split_by_correct(subset).items():
                _write_csv(split_subset, dest / f"summary_{label_slug}_{suffix}.csv")


def export_aggregate(folder: Path, aggregate_df: pd.DataFrame, score_root: Path) -> None:
    if aggregate_df.empty:
        return
    dest = score_root / folder.name / "csv"
    _write_csv(aggregate_df, dest / "aggregate.csv")


def run_exports(general_root: Path, output_root: Path) -> None:
    token_root = output_root / "token"
    sparsity_root = output_root / "sparsity"
    confidence_root = output_root / "confidence"
    score_root = output_root / "score"

    for folder in sorted(p for p in general_root.iterdir() if p.is_dir()):
        token_df = _load_csv(folder / "tokens.csv")
        summary_df = _load_csv(folder / "summary.csv")
        aggregate_df = _load_csv(folder / "aggregate.csv")
        export_tokens(folder, token_df, token_root)
        export_summary(folder, summary_df, sparsity_root, confidence_root)
        export_aggregate(folder, aggregate_df, score_root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export derived analytics for token/sparsity/score/confidence visualisations.")
    parser.add_argument(
        "--general-root",
        type=Path,
        default=Path("outputs/analytics/general"),
        help="Directory containing base semantic outputs (tokens/summary/aggregate).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/analytics"),
        help="Analytics root where category-specific folders will be created.",
    )
    args = parser.parse_args(argv)

    run_exports(args.general_root.resolve(), args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
