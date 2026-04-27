"""Compatibility wrapper for historical text feature streaming."""
from __future__ import annotations

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


_impl = load("historical_text/stream_features.py", "script_stream_historical_text")
main = _impl.main


if __name__ == "__main__":
    main()
