"""High-level workflows to compute and visualise structural correlations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

from .structural_analysis import _ensure_safe_globals, _load_struct_config, run_graph_analytics
from src.visualisations import (
    generate_structural_correlation_heatmaps,
    generate_structural_difference_heatmaps,
)

DEFAULT_CONFIG_PATH = Path("configs/structural_analysis_config.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/structural")


def _normalise_token(value: str) -> str:
    token = value.lower().strip()
    token = token.replace("-", "_")
    return token


def _dataset_aliases(name: str) -> Tuple[str, ...]:
    normalised = _normalise_token(name)
    fragments = normalised.split("/")
    collapsed = normalised.replace("/", "_")
    if fragments:
        return tuple({normalised, collapsed, fragments[-1]})
    return (normalised, collapsed)


def _match_dataset(dataset: str, candidate: str) -> bool:
    target_aliases = set(_dataset_aliases(dataset))
    candidate_aliases = set(_dataset_aliases(candidate))
    return bool(target_aliases & candidate_aliases)


def _select_tasks(tasks: Sequence[Dict[str, object]], dataset: str, graph_type: str) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    graph_key = _normalise_token(graph_type)
    for task in tasks:
        task_dataset = str(task.get("dataset", ""))
        task_graph = _normalise_token(str(task.get("graph_type", "")))
        if not _match_dataset(dataset, task_dataset):
            continue
        if task_graph != graph_key:
            continue
        selected.append(task)
    return selected


def _resolve_output_dir(dataset_label: str, graph_type: str, output_root: Optional[Path]) -> Path:
    root = output_root or DEFAULT_OUTPUT_ROOT
    directory = root / f"{dataset_label.replace('/', '_')}_{graph_type}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_structural_visualisation(
    dataset: str,
    graph_type: str,
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_root: Optional[Path | str] = None,
    significance_threshold: Optional[float] = None,
    create_clustered: bool = True,
) -> Dict[str, object]:
    """Compute correlations and generate visualisations for a dataset/graph pair."""

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Structural config not found: {cfg_path}")

    tasks = _load_struct_config(cfg_path.resolve())
    matches = _select_tasks(tasks, dataset, graph_type)
    if not matches:
        raise ValueError(
            f"No structural tasks found for dataset='{dataset}' graph='{graph_type}'."
        )

    primary_dataset = str(matches[0].get("dataset", dataset))
    output_dir = _resolve_output_dir(primary_dataset, graph_type, Path(output_root) if output_root else None)

    _ensure_safe_globals()

    analytics_paths: List[Path] = []
    for task in matches:
        frame = task.get("frame")
        if frame is None:
            continue
        insight_like = SimpleNamespace(data=frame)
        subdir = output_dir
        analytics_path = run_graph_analytics(
            insight_like,
            subdir,
            backbone=task.get("backbone"),
            split=task.get("split"),
        )
        analytics_paths.append(analytics_path)

    heatmaps = generate_structural_correlation_heatmaps(output_dir)

    difference_outputs: List[Path] = []
    try:
        difference_outputs = generate_structural_difference_heatmaps(
            output_dir,
            significance_threshold=significance_threshold,
            create_clustered=create_clustered,
        )
    except FileNotFoundError:
        difference_outputs = []

    difference_heatmaps = [path for path in difference_outputs if Path(path).suffix.lower() == ".png"]
    difference_tables = [path for path in difference_outputs if Path(path).suffix.lower() == ".csv"]

    return {
        "output_dir": output_dir,
        "analytics_csv": analytics_paths,
        "heatmaps": heatmaps,
        "difference_heatmaps": difference_heatmaps,
        "difference_tables": difference_tables,
    }
