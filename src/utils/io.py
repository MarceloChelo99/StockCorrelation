"""Small IO helpers for JSON and parquet artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a small JSON artifact with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON dictionary artifact from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary in {path}.")
    return payload


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a dataframe to parquet with index disabled."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path
