"""Compatibility wrapper for the stage 05 evaluation script."""
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


_impl = load("pipeline/stage_05_evaluation/evaluate.py", "script_evaluate")
main = _impl.main
load_metadata = _impl.load_metadata


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: .venv/bin/python scripts/05_evaluate.py <config_name> <embeddings_path>")
    main(sys.argv[1], sys.argv[2])
