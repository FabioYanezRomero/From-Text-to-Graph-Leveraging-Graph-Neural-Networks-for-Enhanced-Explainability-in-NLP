"\"\"\"Score-based semantic visualisations.\"\"\""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_DENSITY_ROOT = Path("outputs/analytics/score/density")
DEFAULT_DIFFERENCE_ROOT = Path("outputs/analytics/score/difference")
DEFAULT_RANKING_ROOT = Path("outputs/analytics/score/ranking")


def _iter_token_csvs(root: Path, pattern: str = "*tokens.csv") -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _load_tokens(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "score" not in frame.columns:
        raise ValueError(f"'score' column missing in {path}")
    return frame


def _group_by_label(frame: pd.DataFrame) -> Dict[Optional[str], pd.DataFrame]:
    if "label" not in frame.columns:
        return {None: frame}
    groups: Dict[Optional[str], pd.DataFrame] = {}
    for label, subset in frame.groupby("label", dropna=False):
        if pd.isna(label):
            groups[None] = subset
        else:
            groups[str(label)] = subset
    return groups


def _label_suffix(label: Optional[str]) -> str:
    if label is None:
        return ""
    safe = str(label).strip()
    if not safe or safe.lower() in {"nan", "none"}:
        return ""
    return f"_label_{safe.replace('/', '_').replace(' ', '_')}"


def _ensure_dir(base: Optional[Path], csv_path: Path, root: Path) -> Path:
    if base is None:
        return csv_path.parent
    target = base / csv_path.relative_to(root).parent
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_token_score_density(series_or_path, output_path: Path | None = None, *, title: Optional[str] = None, dpi: int = 300) -> Path | None:
    if isinstance(series_or_path, (str, Path)):
        csv_file = Path(series_or_path)
        frame = _load_tokens(csv_file)
        series = frame["score"].dropna().astype(float).clip(0.0, 1.0)
        if series.empty:
            return None
        output_file = Path(output_path) if output_path else csv_file.with_suffix(".png")
        out_title = title or csv_file.stem.replace("_", " ").title()
    else:
        series = pd.Series(series_or_path).dropna().astype(float).clip(0.0, 1.0)
        if series.empty:
            return None
        if output_path is None:
            raise ValueError("output_path must be provided when passing raw series")
        output_file = Path(output_path)
        out_title = title or output_file.stem.replace("_", " ").title()

    if series.empty:
        return None
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    grid = np.linspace(0.0, 1.0, 256)
    density = _kde(series.to_numpy(), grid)
    ax.plot(grid, density, color="#4C72B0", linewidth=1.8)
    ax.fill_between(grid, density, alpha=0.2, color="#4C72B0")
    ax.set_xlim(0.0, 1.0)
    ax.margins(x=0)
    ax.set_xlabel("Token score (scaled 0–1)")
    ax.set_ylabel("Density")
    ax.set_title(out_title)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def generate_token_score_densities(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_DENSITY_ROOT,
    *,
    pattern: str = "*tokens.csv",
    dpi: int = 300,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(root, pattern):
        frame = _load_tokens(csv_path)
        groups = _group_by_label(frame)
        for label, subset in groups.items():
            target_dir = _ensure_dir(out_dir, csv_path, root)
            output_path = target_dir / f"{csv_path.stem}{_label_suffix(label)}_score_density.png"
            result = plot_token_score_density(subset["score"], output_path=output_path, title=f"Score density{'' if label is None else f' (label {label})'}", dpi=dpi)
            if result is not None:
                produced.append(result)
    return produced


def _kde(values: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid)
    if bandwidth is None:
        std = values.std(ddof=1) if values.size > 1 else 0.0
        bandwidth = 1.06 * std * (values.size ** (-1 / 5)) if std > 0 else 0.05
        bandwidth = max(bandwidth, 0.02)
    diff = (grid[None, :] - values[:, None]) / bandwidth
    densities = np.exp(-0.5 * diff**2).sum(axis=0)
    densities /= (values.size * bandwidth * np.sqrt(2 * np.pi))
    return densities


def _group_correct_incorrect(paths: Iterable[Path]) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in paths:
        stem = path.stem
        if not stem.endswith("_tokens"):
            continue
        core = stem[: -len("_tokens")]
        if core.endswith("_correct"):
            groups[core[: -len("_correct")]]["correct"] = path
        elif core.endswith("_incorrect"):
            groups[core[: -len("_incorrect")]]["incorrect"] = path
    return groups


def plot_token_score_difference(
    correct_csv: Path | str,
    incorrect_csv: Path | str,
    output_path: Path | str | None = None,
    *,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path | None:
    correct_frame = _load_tokens(Path(correct_csv))
    incorrect_frame = _load_tokens(Path(incorrect_csv))

    correct_values = correct_frame["score"].dropna().astype(float).clip(0.0, 1.0).to_numpy()
    incorrect_values = incorrect_frame["score"].dropna().astype(float).clip(0.0, 1.0).to_numpy()
    if correct_values.size == 0 and incorrect_values.size == 0:
        return None

    grid = np.linspace(0.0, 1.0, 256)
    difference = _kde(correct_values, grid) - _kde(incorrect_values, grid)

    output_file = Path(output_path) if output_path else Path(correct_csv).with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, difference, color="#C44E52", linewidth=1.8)
    ax.axhline(0.0, color="#4C72B0", linestyle="--", linewidth=1.0)
    ax.fill_between(grid, difference, 0.0, where=difference >= 0, color="#C44E52", alpha=0.2)
    ax.fill_between(grid, difference, 0.0, where=difference < 0, color="#55A868", alpha=0.2)
    ax.set_xlabel("Token score (scaled 0–1)")
    ax.set_ylabel("Density difference (correct - incorrect)")
    ax.set_xlim(0.0, 1.0)
    ax.margins(x=0)
    ax.set_title(title or f"{Path(correct_csv).stem} score difference")
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def generate_token_score_differences(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_DIFFERENCE_ROOT,
    *,
    pattern: str = "*tokens.csv",
    dpi: int = 300,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_correct_incorrect(_iter_token_csvs(root, pattern))
    produced: List[Path] = []
    for key, subset in groups.items():
        correct_path = subset.get("correct")
        incorrect_path = subset.get("incorrect")
        if not correct_path or not incorrect_path:
            continue
        target_dir = _ensure_dir(out_dir, correct_path, root)
        if out_dir:
            target = target_dir / f"{key}_correct_vs_incorrect_score_diff.png"
        else:
            diff_dir = correct_path.parent / "score_differences"
            diff_dir.mkdir(parents=True, exist_ok=True)
            target = diff_dir / f"{key}_correct_vs_incorrect_score_diff.png"

        result = plot_token_score_difference(correct_path, incorrect_path, output_path=target, dpi=dpi, title=f"{key.replace('_', ' ').title()} Score (Correct - Incorrect)")
        if result is not None:
            produced.append(result)
    return produced


def _top_tokens_by_score(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    summary = frame.groupby("token")["score"].mean().sort_values(ascending=False).head(top_k)
    return pd.DataFrame({"token": summary.index, "avg_score": summary.values})


def _plot_score_ranking(summary: pd.DataFrame, title: str, output_path: Path, *, dpi: int = 300) -> Path:
    if summary.empty:
        return output_path
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(summary))))
    data = summary.iloc[::-1]
    ax.barh(data["token"], data["avg_score"], color="#C44E52", alpha=0.85)
    ax.set_xlabel("Average importance score")
    ax.set_ylabel("Token")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_token_score_ranking(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_RANKING_ROOT,
    *,
    pattern: str = "*tokens.csv",
    top_k: int = 30,
    dpi: int = 300,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(root, pattern):
        frame = _load_tokens(csv_path)
        groups = _group_by_label(frame)
        for label, subset in groups.items():
            summary = _top_tokens_by_score(subset, top_k=top_k)
            if summary.empty:
                continue
            target_dir = _ensure_dir(out_dir, csv_path, root)
            output_path = target_dir / f"{csv_path.stem}{_label_suffix(label)}_token_score_ranking.png"
            produced.append(_plot_score_ranking(summary, f"Average token score{'' if label is None else f' (label {label})'}", output_path, dpi=dpi))
    return produced
