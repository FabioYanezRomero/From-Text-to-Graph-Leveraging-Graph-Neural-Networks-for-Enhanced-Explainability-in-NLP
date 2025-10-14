#!/usr/bin/env python3
"""
Generate visualizations for constituency and syntactic graphs based on the skipgrams pattern.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualisations import (
    generate_sparsity_visuals,
    generate_confidence_threshold_visuals,
    generate_token_score_densities,
    generate_token_score_differences,
    generate_token_score_ranking,
    generate_token_frequency_charts,
    generate_embedding_importance_correlation,
    generate_embedding_neighborhoods,
)

def main():
    print("Generating visualizations for constituency and syntactic graphs...")

    # Define the analytics roots
    analytics_root = Path("outputs/analytics")
    general_root = analytics_root / "general"
    pyg_root = Path("outputs/pyg_graphs")

    # Generate visualizations for each dataset/graph type
    datasets_and_graphs = [
        ("SetFit_ag_news", "constituency"),
        ("SetFit_ag_news", "syntactic"),
        ("stanfordnlp_sst2", "constituency"),
        ("stanfordnlp_sst2", "syntactic"),
    ]

    for dataset, graph_type in datasets_and_graphs:
        print(f"\nProcessing {dataset}/{graph_type}...")

        dataset_graph = f"{dataset.replace('/', '_')}_{graph_type}"

        # 1. Sparsity visualizations
        summary_path = general_root / dataset_graph / "summary.csv"
        if summary_path.exists():
            print(f"  Generating sparsity visualizations...")
            generate_sparsity_visuals(
                summary_root=general_root / dataset_graph,
                output_root=analytics_root / "sparsity" / dataset_graph
            )

        # 2. Confidence visualizations
        if summary_path.exists():
            print(f"  Generating confidence visualizations...")
            generate_confidence_threshold_visuals(
                summary_root=general_root / dataset_graph,
                output_root=analytics_root / "confidence" / dataset_graph
            )

        # 3. Token score visualizations
        tokens_path = general_root / dataset_graph / "tokens.csv"
        if tokens_path.exists():
            print(f"  Generating token score visualizations...")
            generate_token_score_densities(
                tokens_root=general_root / dataset_graph,
                output_root=analytics_root / "score" / "density" / dataset_graph
            )
            generate_token_score_differences(
                tokens_root=general_root / dataset_graph,
                output_root=analytics_root / "score" / "difference" / dataset_graph
            )
            generate_token_score_ranking(
                tokens_root=general_root / dataset_graph,
                output_root=analytics_root / "score" / "ranking" / dataset_graph
            )

        # 4. Token frequency charts
        if tokens_path.exists():
            print(f"  Generating token frequency charts...")
            generate_token_frequency_charts(
                tokens_root=general_root / dataset_graph,
                output_root=analytics_root / "token" / "frequency" / dataset_graph
            )

        # 5. Embedding visualizations (only for constituency and syntactic)
        if tokens_path.exists() and pyg_root.exists():
            print(f"  Generating embedding correlation visualizations...")
            generate_embedding_importance_correlation(
                tokens_root=general_root / dataset_graph,
                pyg_root=pyg_root,
                output_root=analytics_root / "embedding" / "correlation" / dataset_graph
            )

            print(f"  Generating embedding neighborhoods visualizations...")
            generate_embedding_neighborhoods(
                tokens_root=general_root / dataset_graph,
                pyg_root=pyg_root,
                output_root=analytics_root / "embedding" / "neighborhoods" / dataset_graph
            )

    print("\nAll visualizations generated!")

if __name__ == "__main__":
    main()
