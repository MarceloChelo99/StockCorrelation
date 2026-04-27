"""Shared path setup for script entry points.

Import this module before importing project code from scripts. It is the only
place scripts should mutate ``sys.path``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MARKET_DATA_FETCHER_SRC = REPO_ROOT / "libraries" / "market_data_fetcher" / "src"

for path in (REPO_ROOT, MARKET_DATA_FETCHER_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_script(relative_path: str, module_name: str):
    """Load a script implementation by path relative to ``scripts/``."""
    path = Path(relative_path)
    if not path.is_absolute():
        path = REPO_ROOT / "scripts" / path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
