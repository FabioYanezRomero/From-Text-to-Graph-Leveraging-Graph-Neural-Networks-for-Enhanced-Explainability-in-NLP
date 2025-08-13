import os
import json
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from .compute_overlap import (
    LLM_RESULTS_PATH,
    GRAPH_VARIANTS,
    OUTPUT_DIR,
    ensure_output_dir,
    extract_llm_token_importance,
    extract_graph_node_importance,
    align_content_only,
)


def gini_coefficient(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, dtype=float))
    if np.allclose(x, 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.abs(np.asarray(p, dtype=float))
    s = p.sum()
    if s <= 0:
        return 0.0
    q = p / s
    q = np.clip(q, eps, 1.0)
    return float(-(q * np.log2(q)).sum())


def topk_mass(x: np.ndarray, k: int) -> float:
    x = np.abs(np.asarray(x, dtype=float))
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s == 0:
        return 0.0
    k = min(k, x.size)
    return float(np.sort(x)[-k:].sum() / s)


def summarize_vector(x: np.ndarray) -> Dict[str, float]:
    ax = np.abs(np.asarray(x, dtype=float))
    n = ax.size
    if n == 0:
        return {
            "length": 0,
            "mean_abs": 0.0,
            "std_abs": 0.0,
            "max_abs": 0.0,
            "gini": 0.0,
            "entropy_bits": 0.0,
            "top1_mass": 0.0,
            "top3_mass": 0.0,
            "top5_mass": 0.0,
        }
    return {
        "length": int(n),
        "mean_abs": float(ax.mean()),
        "std_abs": float(ax.std(ddof=0)),
        "max_abs": float(ax.max()),
        "gini": gini_coefficient(ax),
        "entropy_bits": entropy(ax),
        "top1_mass": topk_mass(ax, 1),
        "top3_mass": topk_mass(ax, 3),
        "top5_mass": topk_mass(ax, 5),
    }


def main():
    ensure_output_dir()

    llm = pickle.load(open(LLM_RESULTS_PATH, 'rb'))
    llm_by_index = {int(v['dataset_index']): v for v in llm.values()}

    outputs: Dict[str, Dict] = {}

    # Prepare LLM summaries (shared across variants; computed once per index)
    llm_rows: List[Dict] = []
    for idx, item in sorted(llm_by_index.items()):
        df: pd.DataFrame = item['result']
        num_tokens = int(item.get('num_tokens', 0))
        llm_vec = extract_llm_token_importance(df, num_tokens=num_tokens)
        llm_summary = summarize_vector(llm_vec)
        llm_rows.append({"dataset_index": idx, **llm_summary})
    llm_df = pd.DataFrame(llm_rows)
    llm_df.to_csv(os.path.join(OUTPUT_DIR, "informativeness_llm.csv"), index=False)
    outputs["llm_summary"] = {
        "count": int(len(llm_df)),
        "means": llm_df.drop(columns=["dataset_index"]).mean(numeric_only=True).to_dict(),
        "stds": llm_df.drop(columns=["dataset_index"]).std(numeric_only=True).to_dict(),
    }

    # For each graph variant, summarize node_importance
    for variant, paths in GRAPH_VARIANTS.items():
        gdata = pickle.load(open(paths["results"], 'rb'))
        grow_rows: List[Dict] = []
        for gkey, gitem in sorted(gdata.items(), key=lambda kv: int(kv[1]['dataset_index'])):
            idx = int(gitem['dataset_index'])
            node_scores = extract_graph_node_importance(gitem['node_importance'])
            node_scores_content = align_content_only(node_scores)
            gsum = summarize_vector(node_scores_content)
            grow_rows.append({"graph_key": gkey, "dataset_index": idx, **gsum})
        gdf = pd.DataFrame(grow_rows)
        gdf.to_csv(os.path.join(OUTPUT_DIR, f"informativeness_{variant}.csv"), index=False)
        outputs[f"graph_{variant}_summary"] = {
            "variant": variant,
            "count": int(len(gdf)),
            "means": gdf.drop(columns=["graph_key", "dataset_index"]).mean(numeric_only=True).to_dict(),
            "stds": gdf.drop(columns=["graph_key", "dataset_index"]).std(numeric_only=True).to_dict(),
        }

    with open(os.path.join(OUTPUT_DIR, "informativeness_overview.json"), 'w') as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    main()


