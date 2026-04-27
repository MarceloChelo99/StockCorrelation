"""Helpers for thin backward-compatible script wrappers."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def bootstrap():
    """Load sibling ``_bootstrap.py`` without assuming package import paths."""
    path = Path(__file__).resolve().parent / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_stock_embeddings_scripts_bootstrap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bootstrap module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(relative_path: str, module_name: str):
    """Load a moved implementation under ``scripts/``."""
    return bootstrap().load_script(relative_path, module_name)
