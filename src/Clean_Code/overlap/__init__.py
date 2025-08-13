"""Overlap computation package for comparing TokenSHAP (LLM) and GraphSVX (GNN).

This package loads stored explanations from `/app/explanations/`, extracts per-token
and per-node attribution vectors, and computes overlap metrics per sentence.

Outputs are written to `/app/src/Clean_Code/output/overlap`.
"""


