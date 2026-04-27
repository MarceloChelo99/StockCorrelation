"""Compatibility helpers for calling script entry points from Python."""
from __future__ import annotations

from pathlib import Path

from scripts import _bootstrap


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATHS = {
    "05_evaluate.py": "pipeline/stage_05_evaluation/evaluate.py",
    "09_covariance_slices.py": "analysis/covariance_slices.py",
    "17_train_views.py": "pipeline/stage_04_model/train_views.py",
    "18_fit_gmms.py": "pipeline/stage_04_model/fit_gmms.py",
    "19_interpret_views.py": "pipeline/stage_04_model/interpret_views.py",
}


def train_views(config: dict, experiment_dir: Path) -> None:
    """Call the view-training script without relying on file-path imports."""
    module = load_script("17_train_views.py", "script_17_train_views")
    module.main(str(write_runtime_config(config, experiment_dir)), str(experiment_dir))


def fit_gmms(config: dict, experiment_dir: Path) -> None:
    """Call the GMM-fitting script without relying on file-path imports."""
    module = load_script("18_fit_gmms.py", "script_18_fit_gmms")
    config_path = write_runtime_config(config, experiment_dir)
    module.main(str(config_path), str(experiment_dir))


def interpret_views(config: dict, experiment_dir: Path, report_dir: Path) -> None:
    """Call the view-interpretation script without relying on file-path imports."""
    module = load_script("19_interpret_views.py", "script_19_interpret_views")
    config_path = write_runtime_config(config, experiment_dir)
    module.main(str(config_path), str(experiment_dir), str(report_dir))


def load_script(filename: str, module_name: str):
    """Load a script file as a Python module."""
    relative_path = SCRIPT_PATHS.get(filename, filename)
    return _bootstrap.load_script(relative_path, module_name)


def write_runtime_config(config: dict, experiment_dir: Path) -> Path:
    """Persist a resolved config for numbered scripts that expect config paths."""
    path = experiment_dir / "resolved_config_for_scripts.json"
    import json

    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path
