import os
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity


LLM_RESULTS_PATH = "/app/explanations/LLM/sst-2/tokenSHAP_results.pkl"
LLM_SUMMARY_PATH = "/app/explanations/LLM/sst-2/tokenSHAP_summary.json"

# Evaluate overlap for both graph variants to be thorough
GRAPH_VARIANTS = {
    "knn4": {
        "results": "/app/explanations/TransformerConv/knn4/sst-2/graphSVX_results_knn4.pkl",
        "summary": "/app/explanations/TransformerConv/knn4/sst-2/graphSVX_summary_knn4.json",
    },
    "fully_connected": {
        "results": "/app/explanations/TransformerConv/fully_connected/sst-2/graphSVX_results_fully_connected.pkl",
        "summary": "/app/explanations/TransformerConv/fully_connected/sst-2/graphSVX_summary_fully_connected.json",
    },
}

OUTPUT_DIR = "/app/src/Clean_Code/output/overlap"


def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def safe_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = np.sum(np.abs(values))
    if total == 0:
        return values
    return values / total


def safe_rank(values: np.ndarray) -> np.ndarray:
    # Ties: use average rank, higher value -> higher rank
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # handle ties by averaging ranks for equal values
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / counts
    return avg[inv]


def jaccard_at_k(indices_a: List[int], indices_b: List[int], k: int) -> float:
    set_a = set(indices_a[:k])
    set_b = set(indices_b[:k])
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def extract_llm_token_importance(df: pd.DataFrame, num_tokens: Optional[int] = None) -> np.ndarray:
    """
    Attempt to extract per-token attribution magnitudes from TokenSHAP results DataFrame.

    Expected columns (observed): ['Prompt', 'Response', 'Token_Indexes', 'Cosine_Similarity']
    Where 'Token_Indexes' is a tuple of included token indices for the coalition
    and 'Cosine_Similarity' measures prediction proximity for that coalition.

    Heuristic: approximate token contribution by averaging the coalition score
    differences when the token is included. This is a proxy if exact Shapley
    values are not explicitly stored. It preserves ranking trends across tokens.
    """
    if 'Token_Indexes' not in df.columns or 'Cosine_Similarity' not in df.columns:
        raise ValueError("Unexpected TokenSHAP DataFrame format. Required columns missing.")

    # Determine token universe from observed indices
    token_universe: set = set()
    for idxs in df['Token_Indexes']:
        try:
            token_universe.update(list(idxs))
        except Exception:
            # Try to parse from string representation
            if isinstance(idxs, str):
                s = idxs.strip().strip('()[]')
                if s:
                    token_universe.update(int(x) for x in s.split(',') if x.strip().isdigit())
    if num_tokens is None:
        num_tokens = max(token_universe) + 1 if token_universe else 0

    # Baseline: average score across all coalitions
    # Cosine_Similarity is higher ~ better alignment; use as utility
    baseline = float(df['Cosine_Similarity'].mean()) if len(df) else 0.0

    token_scores = np.zeros(num_tokens, dtype=float)
    token_counts = np.zeros(num_tokens, dtype=int)

    for _, row in df.iterrows():
        idxs = row['Token_Indexes']
        try:
            indices = list(idxs)
        except Exception:
            if isinstance(idxs, str):
                s = idxs.strip().strip('()[]')
                indices = [int(x) for x in s.split(',') if x.strip().isdigit()]
            else:
                indices = []
        score = float(row['Cosine_Similarity'])
        contribution = score - baseline
        for t in indices:
            if 0 <= t < num_tokens:
                token_scores[t] += contribution
                token_counts[t] += 1

    # Average contributions; unseen tokens remain 0
    for t in range(num_tokens):
        if token_counts[t] > 0:
            token_scores[t] /= token_counts[t]

    return token_scores


def extract_graph_node_importance(node_importance: np.ndarray) -> np.ndarray:
    return np.asarray(node_importance, dtype=float)


def align_content_only(values: np.ndarray) -> np.ndarray:
    # Exclude [CLS] and [SEP] assumed at positions 0 and last
    if values.size >= 2:
        return values[1:-1]
    return values.copy()


def compute_metrics(vec_a: np.ndarray, vec_b: np.ndarray) -> Dict[str, float]:
    # Ensure same length
    n = min(len(vec_a), len(vec_b))
    va = vec_a[:n]
    vb = vec_b[:n]

    # Replace NaNs
    va = np.nan_to_num(va, nan=0.0)
    vb = np.nan_to_num(vb, nan=0.0)

    # Rankings
    ra = safe_rank(va)
    rb = safe_rank(vb)

    # Correlations
    kt, kt_p = kendalltau(ra, rb)
    sp, sp_p = spearmanr(ra, rb)
    pr, pr_p = pearsonr(va, vb) if n >= 2 else (np.nan, np.nan)

    # Cosine
    cs = float(cosine_similarity(va.reshape(1, -1), vb.reshape(1, -1))[0][0]) if n > 0 else np.nan

    # Jaccard at top-k
    order_a = list(np.argsort(-va))
    order_b = list(np.argsort(-vb))
    j1 = jaccard_at_k(order_a, order_b, k=min(1, n))
    j3 = jaccard_at_k(order_a, order_b, k=min(3, n))
    j5 = jaccard_at_k(order_a, order_b, k=min(5, n))
    j10 = jaccard_at_k(order_a, order_b, k=min(10, n))

    # Distributional similarity (probabilities over absolute values)
    pa = safe_normalize(np.abs(va))
    pb = safe_normalize(np.abs(vb))
    try:
        jsd = float(jensenshannon(pa, pb, base=2.0))
    except Exception:
        jsd = np.nan

    return {
        "kendall_tau": float(kt) if kt is not None else np.nan,
        "kendall_tau_p": float(kt_p) if kt_p is not None else np.nan,
        "spearman_rho": float(sp) if sp is not None else np.nan,
        "spearman_rho_p": float(sp_p) if sp_p is not None else np.nan,
        "pearson_r": float(pr) if not np.isnan(pr) else np.nan,
        "pearson_r_p": float(pr_p) if not np.isnan(pr_p) else np.nan,
        "cosine": cs,
        "jaccard@1": j1,
        "jaccard@3": j3,
        "jaccard@5": j5,
        "jaccard@10": j10,
        "js_divergence": jsd,
        "length": int(n),
    }


def load_pickled(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main() -> None:
    ensure_output_dir()

    # Load LLM and map by dataset_index
    llm = load_pickled(LLM_RESULTS_PATH)
    # sentence_{i}: contains 'dataset_index'
    llm_by_index: Dict[int, Dict] = {}
    for key, item in llm.items():
        idx = int(item.get('dataset_index'))
        llm_by_index[idx] = item

    overall_summary: Dict[str, Dict] = {}

    for variant_name, paths in GRAPH_VARIANTS.items():
        graph = load_pickled(paths["results"])

        rows: List[Dict] = []
        for gkey, gitem in graph.items():
            ds_index = int(gitem.get('dataset_index'))
            if ds_index not in llm_by_index:
                continue

            # Extract LLM token scores
            llm_item = llm_by_index[ds_index]
            llm_df: pd.DataFrame = llm_item['result']
            num_tokens = int(llm_item.get('num_tokens', 0))
            llm_scores = extract_llm_token_importance(llm_df, num_tokens=num_tokens)

            # Extract graph node scores
            node_scores = extract_graph_node_importance(gitem['node_importance'])

            # Align content-only tokens/nodes
            llm_content = llm_scores  # TokenSHAP used include_special=False
            graph_content = align_content_only(node_scores)

            # Ensure same length by trimming to minimum
            n = min(len(llm_content), len(graph_content))
            llm_vec = llm_content[:n]
            graph_vec = graph_content[:n]

            metrics = compute_metrics(llm_vec, graph_vec)

            rows.append({
                "graph_key": gkey,
                "dataset_index": ds_index,
                "num_tokens": int(len(llm_vec)),
                **metrics,
            })

        # Save per-variant CSV and JSON summary
        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, f"overlap_{variant_name}.csv")
        df.to_csv(csv_path, index=False)

        summary = {
            "variant": variant_name,
            "count": int(len(df)),
            "means": df.drop(columns=["graph_key", "dataset_index", "length"]).mean(numeric_only=True).to_dict(),
            "stds": df.drop(columns=["graph_key", "dataset_index", "length"]).std(numeric_only=True).to_dict(),
        }
        json_path = os.path.join(OUTPUT_DIR, f"overlap_{variant_name}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        overall_summary[variant_name] = summary

    with open(os.path.join(OUTPUT_DIR, "overlap_summary_all_variants.json"), 'w') as f:
        json.dump(overall_summary, f, indent=2)


if __name__ == "__main__":
    main()


