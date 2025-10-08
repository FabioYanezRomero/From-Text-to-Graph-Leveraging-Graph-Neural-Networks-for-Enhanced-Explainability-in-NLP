from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict

DEFAULT_BASE = os.environ.get("GRAPHTEXT_OUTPUT_DIR", "outputs")
METADATA_DIR = os.path.join(DEFAULT_BASE, "metadata")
METADATA_INDEX = os.path.join(METADATA_DIR, "index.json")


def _load_index() -> Dict[str, Any]:
    if not os.path.exists(METADATA_INDEX):
        return {"runs": []}
    try:
        with open(METADATA_INDEX, "r") as f:
            return json.load(f)
    except Exception:
        return {"runs": []}


def log_step(step: str, params: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    os.makedirs(METADATA_DIR, exist_ok=True)
    idx = _load_index()
    idx.setdefault("runs", [])
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "params": params,
        "outputs": outputs,
    }
    idx["runs"].append(entry)
    with open(METADATA_INDEX, "w") as f:
        json.dump(idx, f, indent=2)

