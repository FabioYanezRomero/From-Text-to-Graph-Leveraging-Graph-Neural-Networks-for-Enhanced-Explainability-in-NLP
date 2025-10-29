#!/usr/bin/env bash
set -euo pipefail

run_split_subgraphx() {
  local input_path="$1"
  local output_path="$2"
  shift 2
  local extra_args=("$@")

  echo "Splitting SubgraphX results:"
  echo "  input:  ${input_path}"
  echo "  output: ${output_path}"

  local cmd=(
    python -m src.explain.splitters.subgraphx
    --input "${input_path}"
    --output "${output_path}"
    --format pickle-raw
    --overwrite
  )
  if ((${#extra_args[@]})); then
    cmd+=("${extra_args[@]}")
  fi

  SPLIT_SUBGRAPHX_DEBUG=1 "${cmd[@]}"
}

run_split_graphsvx() {
  local input_path="$1"
  local output_path="$2"

  echo "Splitting graphsvx results:"
  echo "  input:  ${input_path}"
  echo "  output: ${output_path}"

  SPLIT_GRAPSVX_DEBUG=1 python -m src.explain.splitters.graphsvx \
    --input "${input_path}" \
    --output "${output_path}" \
    --format pickle \
    --overwrite

}


# # Syntactic shards
# for shard in 1 2 3; do
#   # AG News
#   run_split_subgraphx \
#     "/app/outputs/gnn_models/SetFit/ag_news/syntactic/explanations/subgraphx/SetFit_ag_news_syntactic_test_shard${shard}of3/results.pkl" \
#     "/app/outputs/gnn_models/SetFit/ag_news/syntactic/explanations/subgraphx/SetFit_ag_news_syntactic_test_shard${shard}of3/results_split_pickle" 

#   # SST-2
#   run_split_subgraphx \
#     "/app/outputs/gnn_models/stanfordnlp/sst2/syntactic/explanations/subgraphx/stanfordnlp_sst2_syntactic_validation_shard${shard}of3/results.pkl" \
#     "/app/outputs/gnn_models/stanfordnlp/sst2/syntactic/explanations/subgraphx/stanfordnlp_sst2_syntactic_validation_shard${shard}of3/results_split_pickle" 

# done


# Constituency shards
for shard in 1 2 3; do

  # AG News
  skip_args=()
  if [[ "${shard}" == "1" ]]; then
    skip_args+=(--skip 2157)
  fi
  run_split_subgraphx \
    "/app/outputs/gnn_models/SetFit/ag_news/constituency/explanations/subgraphx/SetFit_ag_news_constituency_test_shard${shard}of3/results.pkl" \
    "/app/outputs/gnn_models/SetFit/ag_news/constituency/explanations/subgraphx/SetFit_ag_news_constituency_test_shard${shard}of3/results_split_pickle" \
    "${skip_args[@]}"

  # SST-2
  run_split_subgraphx \
    "/app/outputs/gnn_models/stanfordnlp/sst2/constituency/explanations/subgraphx/stanfordnlp_sst2_constituency_validation_shard${shard}of3/results.pkl" \
    "/app/outputs/gnn_models/stanfordnlp/sst2/constituency/explanations/subgraphx/stanfordnlp_sst2_constituency_validation_shard${shard}of3/results_split_pickle"

done


# # Window shards
# for shard in 1 2 3; do

#   # AG News
#   run_split_graphsvx \
#     "/app/outputs/gnn_models/SetFit/ag_news/window/explanations/graphsvx/SetFit_ag_news_window_test_shard${shard}of3/results.pkl" \
#     "/app/outputs/gnn_models/SetFit/ag_news/window/explanations/graphsvx/SetFit_ag_news_window_test_shard${shard}of3/results_split_pickle" 

#   # SST-2
#   run_split_graphsvx \
#     "/app/outputs/gnn_models/stanfordnlp/sst2/window/explanations/graphsvx/stanfordnlp_sst2_window_validation_shard${shard}of3/results.pkl" \
#     "/app/outputs/gnn_models/stanfordnlp/sst2/window/explanations/graphsvx/stanfordnlp_sst2_window_validation_shard${shard}of3/results_split_pickle"
# done


# # Skipgram shards
# for shard in 1 2 3; do
  
#   # AG News
#   run_split_graphsvx \
#     "/app/outputs/gnn_models/SetFit/ag_news/skipgrams/explanations/graphsvx/SetFit_ag_news_skipgrams_test_shard${shard}of3/results.pkl" \
#     "/app/outputs/gnn_models/SetFit/ag_news/skipgrams/explanations/graphsvx/SetFit_ag_news_skipgrams_test_shard${shard}of3/results_split_pickle" 

#   # SST-2
#   run_split_graphsvx \
#     "/app/outputs/gnn_models/stanfordnlp/sst2/skipgrams/explanations/graphsvx/stanfordnlp_sst2_skipgrams_validation_shard${shard}of3/results.pkl" \
#     "/app/outputs/gnn_models/stanfordnlp/sst2/skipgrams/explanations/graphsvx/stanfordnlp_sst2_skipgrams_validation_shard${shard}of3/results_split_pickle"
# done
