"""Experiment directory lifecycle helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import save_config


@dataclass(frozen=True)
class Experiment:
    """Own one immutable experiment output directory."""

    name: str
    root: Path

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "embeddings.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def history_path(self) -> Path:
        return self.root / "training_history.json"

    @property
    def log_path(self) -> Path:
        return self.root / "log.txt"

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @classmethod
    def create(cls, base_dir: str | Path, name: str, config: dict) -> "Experiment":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(base_dir) / f"{timestamp}_{name}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "model").mkdir(exist_ok=False)
        (root / "figures").mkdir(exist_ok=False)
        save_config(config, root / "config.yaml")
        (root / "log.txt").touch()
        return cls(name=name, root=root)
