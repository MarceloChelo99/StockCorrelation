"""Compatibility helpers for calling numbered script entry points from Python."""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def train_views(config: dict, experiment_dir: Path) -> None:
    """Call scripts/17_train_views.py without relying on numeric module imports."""
    module = load_script("17_train_views.py", "script_17_train_views")
    module.main(str(write_runtime_config(config, experiment_dir)), str(experiment_dir))


def fit_gmms(config: dict, experiment_dir: Path) -> None:
    """Call scripts/18_fit_gmms.py without relying on numeric module imports."""
    module = load_script("18_fit_gmms.py", "script_18_fit_gmms")
    config_path = write_runtime_config(config, experiment_dir)
    module.main(str(config_path), str(experiment_dir))


def interpret_views(config: dict, experiment_dir: Path, report_dir: Path) -> None:
    """Call scripts/19_interpret_views.py without relying on numeric module imports."""
    module = load_script("19_interpret_views.py", "script_19_interpret_views")
    config_path = write_runtime_config(config, experiment_dir)
    module.main(str(config_path), str(experiment_dir), str(report_dir))


def load_script(filename: str, module_name: str):
    """Load a numbered script file as a Python module."""
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_runtime_config(config: dict, experiment_dir: Path) -> Path:
    """Persist a resolved config for numbered scripts that expect config paths."""
    path = experiment_dir / "resolved_config_for_scripts.json"
    import json

    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path
