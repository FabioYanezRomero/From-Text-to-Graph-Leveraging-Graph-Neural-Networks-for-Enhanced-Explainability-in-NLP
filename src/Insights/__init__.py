"""
Core explainability insights package.

This package hosts ingestion, metrics, and reporting utilities that unify
outputs from multiple graph explainability methods (e.g., GraphSVX, SubgraphX).

Modules:
    records: canonical dataclasses representing explanations and coalitions.
    readers: parsers for raw explanation artefacts (JSON/CSV bundles).
    metrics: reusable computations for faithfulness, sparsity, stability, etc.
    reporting: helpers to aggregate and persist analysis results.
    cli: simple entry point to drive the workflow from the command line.
"""

from .records import Coalition, ExplanationRecord, RelatedPrediction

__all__ = ["Coalition", "ExplanationRecord", "RelatedPrediction"]
