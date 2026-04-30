"""Compatibility wrapper for the stage 02 feature computation script."""
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


_impl = load("pipeline/stage_02_features/compute_features.py", "script_compute_features")
main = _impl.main


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
