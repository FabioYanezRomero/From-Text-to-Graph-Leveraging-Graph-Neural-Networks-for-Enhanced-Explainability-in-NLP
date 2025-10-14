"\"\"\"Visualisations for unique token counts and sparsity metrics.\"\"\""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/semantic/sparsity")


def _iter_summary_csvs(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _label_suffix(label: Optional[str]) -> str:
    if label is None:
        return ""
    safe = str(label).strip()
    if not safe or safe.lower() in {"nan", "none"}:
        return ""
    return f"_label_{safe.replace('/', '_').replace(' ', '_')}"


def _plot_hist(series: pd.Series, xlabel: str, title: str, output_path: Path, *, bins: int = 30, dpi: int = 300) -> Path:
    if series.empty:
        return output_path
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(series, bins=bins, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_scatter(frame: pd.DataFrame, title: str, output_path: Path, *, hue_col: Optional[str] = None, dpi: int = 300) -> Path:
    if frame.empty:
        return output_path
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_kwargs = dict(x="unique_token_count", y="sparsity", alpha=0.6, s=36, ax=ax)
    if hue_col and hue_col in frame.columns and frame[hue_col].notna().any():
        sns.scatterplot(data=frame, hue=frame[hue_col].astype(str), **plot_kwargs)
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        sns.scatterplot(data=frame, color="#C44E52", **plot_kwargs)
    ax.set_xlabel("Unique token count")
    ax.set_ylabel("Sparsity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_sparsity_visuals(
    summary_root: Path | str,
    output_root: Path | str | None = DEFAULT_OUTPUT_ROOT,
    *,
    pattern: str = "*summary.csv",
    bins: int = 30,
    dpi: int = 300,
) -> List[Path]:
    root_dir = Path(summary_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_summary_csvs(root_dir, pattern):
        frame = pd.read_csv(csv_path)
        if "unique_token_count" not in frame.columns:
            continue

        target_dir = csv_path.parent if out_dir is None else (out_dir / csv_path.relative_to(root_dir).parent)
        target_dir.mkdir(parents=True, exist_ok=True)

        unique_series = pd.to_numeric(frame["unique_token_count"], errors="coerce").dropna()
        if not unique_series.empty:
            produced.append(
                _plot_hist(
                    unique_series,
                    "Unique token count",
                    f"Unique tokens - {csv_path.stem}",
                    target_dir / f"{csv_path.stem}_unique_token_count_hist.png",
                    bins=bins,
                    dpi=dpi,
                )
            )

        if "sparsity" in frame.columns:
            sparsity_series = pd.to_numeric(frame["sparsity"], errors="coerce").dropna()
            if not sparsity_series.empty:
                produced.append(
                    _plot_hist(
                        sparsity_series,
                        "Sparsity",
                        f"Sparsity - {csv_path.stem}",
                        target_dir / f"{csv_path.stem}_sparsity_hist.png",
                        bins=bins,
                        dpi=dpi,
                    )
                )

            scatter_frame = frame.dropna(subset=["unique_token_count", "sparsity"])
            if not scatter_frame.empty:
                produced.append(
                    _plot_scatter(
                        scatter_frame,
                        f"Unique tokens vs Sparsity - {csv_path.stem}",
                        target_dir / f"{csv_path.stem}_unique_vs_sparsity.png",
                        hue_col="label" if "label" in scatter_frame.columns else None,
                        dpi=dpi,
                    )
                )

        groups = _group_aggregate(frame)
        for label, subset in groups.items():
            label_suffix = _label_suffix(label)
            if label_suffix == "":
                continue
            unique_subset = pd.to_numeric(subset["unique_token_count"], errors="coerce").dropna()
            if not unique_subset.empty:
                produced.append(
                    _plot_hist(
                        unique_subset,
                        "Unique token count",
                        f"Unique tokens - label {label}",
                        target_dir / f"{csv_path.stem}{label_suffix}_unique_token_count_hist.png",
                        bins=bins,
                        dpi=dpi,
                    )
                )
            if "sparsity" in subset.columns:
                sparsity_subset = pd.to_numeric(subset["sparsity"], errors="coerce").dropna()
                if not sparsity_subset.empty:
                    produced.append(
                        _plot_hist(
                            sparsity_subset,
                            "Sparsity",
                            f"Sparsity - label {label}",
                            target_dir / f"{csv_path.stem}{label_suffix}_sparsity_hist.png",
                            bins=bins,
                            dpi=dpi,
                        )
                    )
                scatter = subset.dropna(subset=["unique_token_count", "sparsity"])
                if not scatter.empty:
                    produced.append(
                        _plot_scatter(
                            scatter,
                            f"Unique tokens vs Sparsity - label {label}",
                            target_dir / f"{csv_path.stem}{label_suffix}_unique_vs_sparsity.png",
                            hue_col=None,
                            dpi=dpi,
                        )
                    )
    return produced


def _group_aggregate(frame: pd.DataFrame) -> Dict[Optional[str], pd.DataFrame]:
    if "label" not in frame.columns:
        return {}
    groups: Dict[Optional[str], pd.DataFrame] = {}
    for label, subset in frame.groupby("label", dropna=False):
        if pd.isna(label):
            continue
        groups[str(label)] = subset
    return groups
