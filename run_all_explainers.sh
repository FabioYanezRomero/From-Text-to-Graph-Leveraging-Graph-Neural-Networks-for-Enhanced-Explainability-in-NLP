#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run all explainability methods with fair comparison mode
# ==============================================================================
# This script runs:
# - GraphSVX (GNN) on skipgrams and window graphs
# - SubgraphX (GNN) on constituency and syntactic graphs
# - TokenSHAP (LLM) on finetuned transformers
#
# All methods use --fair flag for 400 forward passes (default)
# Reduced from 2000 to make SubgraphX tractable while maintaining fairness
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_CMD="docker compose"
FAIR_FLAG="--fair"

# Optional: Override to disable fair mode
# FAIR_FLAG=""

# Optional: Set custom target forward passes (e.g., higher budget)
# FAIR_FLAG="--fair --target-forward-passes 1000"

# Optional: Limit graphs for testing (e.g., first 100 samples)
# Comment out to process all graphs
MAX_GRAPHS_FLAG="--max-graphs 100"

run_in_container() {
    local service=$1
    local use_gpu=$2
    local cmd=$3
    echo ""
    echo "===================================================================="
    echo "[${service}] ${cmd}"
    echo "===================================================================="
    # GPU access is configured in docker-compose.yml via deploy.resources.reservations
    ${COMPOSE_CMD} run --rm --entrypoint /bin/bash "${service}" -c "cd /app && ${cmd}"
}

run_graphsvx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    local cmd="python -m src.explain.gnn.graphsvx.main --dataset '${dataset}' --graph-type '${graph_type}' --backbone '${backbone}' --split '${split}' ${FAIR_FLAG} ${MAX_GRAPHS_FLAG}"
    run_in_container graphsvx true "${cmd}"
}

run_subgraphx() {
    local dataset=$1
    local backbone=$2
    local graph_type=$3
    local split=$4
    local cmd="python -m src.explain.gnn.subgraphx.main --dataset '${dataset}' --graph-type '${graph_type}' --backbone '${backbone}' --split '${split}' ${FAIR_FLAG} ${MAX_GRAPHS_FLAG}"
    run_in_container subgraphx true "${cmd}"
}

run_tokenshap() {
    local profile=$1
    local cmd="python -m src.explain.llm.main explain '${profile}' ${FAIR_FLAG} ${MAX_GRAPHS_FLAG/--max-graphs/--max-samples}"
    run_in_container tokenshap true "${cmd}"
}

main() {
    cd "${ROOT_DIR}" || exit 1

    echo ""
    echo "===================================================================="
    echo "Starting all explainability methods with fair comparison mode"
    echo "Target forward passes: 400 (default, optimized for SubgraphX)"
    if [[ -n "${MAX_GRAPHS_FLAG}" ]]; then
        echo "Test mode: Processing first 100 samples only"
        echo "To process all samples, comment out MAX_GRAPHS_FLAG in script"
    fi
    echo "===================================================================="
    echo ""

    # GNN explainers - GraphSVX (4 runs)
    echo ">>> Running GraphSVX on AG News and SST-2..."
    run_graphsvx "ag_news" "SetFit" "skipgrams" "test"
    run_graphsvx "ag_news" "SetFit" "window" "test"
    run_graphsvx "sst2" "stanfordnlp" "skipgrams" "validation"
    run_graphsvx "sst2" "stanfordnlp" "window" "validation"

    # GNN explainers - SubgraphX (4 runs)
    echo ">>> Running SubgraphX on AG News and SST-2..."
    run_subgraphx "ag_news" "SetFit" "constituency" "test"
    run_subgraphx "ag_news" "SetFit" "syntactic" "test"
    run_subgraphx "sst2" "stanfordnlp" "constituency" "validation"
    run_subgraphx "sst2" "stanfordnlp" "syntactic" "validation"

    # LLM explainers - TokenSHAP (2 runs)
    echo ">>> Running TokenSHAP on AG News and SST-2..."
    run_tokenshap "setfit/ag_news"
    run_tokenshap "stanfordnlp/sst2"

    echo ""
    echo "===================================================================="
    echo "✓ All explainability methods completed successfully!"
    echo "===================================================================="
    echo ""
    echo "Outputs generated:"
    echo "  - GNN insights: outputs/gnn_models/<backbone>/<dataset>/<graph_type>/<run_id>/explanations/"
    echo "  - LLM insights: outputs/insights/LLM/<backbone>/<dataset>/"
    echo ""
    echo "To analyze results, use:"
    echo "  python -m src.Insights.cli --graphsvx-root outputs/gnn_models --output-json outputs/insights/combined.json"
    echo ""
}

main "$@"