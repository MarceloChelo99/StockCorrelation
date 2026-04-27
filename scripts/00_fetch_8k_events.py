"""Compatibility wrapper for 8-K event metadata ingestion."""
from __future__ import annotations

import sys

import importlib.util
from pathlib import Path


def _load_wrapper_helper():
    path = Path(__file__).resolve().parent / "_run_moved.py"
    spec = importlib.util.spec_from_file_location("_stock_embeddings_run_moved", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load wrapper helper from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load = _load_wrapper_helper().load


_impl = load("pipeline/stage_01_ingest/fetch_8k_events.py", "script_fetch_8k_events")
main = _impl.main
parse_items = _impl.parse_items
value_at = _impl.value_at


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
