"""Configuration loading helpers for experiment-based pipeline runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
EXPERIMENTS_DIR = CONFIG_DIR / "experiments"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries with override values winning."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(experiment: str | Path | None = None) -> dict[str, Any]:
    """Load the base config and apply an experiment override if requested."""
    base = _load_mapping(DEFAULT_CONFIG_PATH)
    if experiment is None:
        return base

    experiment_path = _resolve_experiment_path(experiment)
    override = _load_mapping(experiment_path)
    resolved = deep_merge(base, override)
    resolved.setdefault("experiment_name", experiment_path.stem)
    return resolved


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Persist a resolved config using JSON syntax that is valid YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _resolve_experiment_path(experiment: str | Path) -> Path:
    path = Path(experiment)
    if path.exists():
        return path
    candidate = EXPERIMENTS_DIR / f"{path.stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find experiment config {experiment!r}.")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must load to a dictionary.")
    return data
