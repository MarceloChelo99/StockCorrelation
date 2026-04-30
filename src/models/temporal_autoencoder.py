"""Temporal-aware autoencoder for smoother business-view trajectories.

The architecture is intentionally identical to ``Autoencoder``. The difference
is the training loop: paired consecutive firm-month observations add an adaptive
smoothness penalty in embedding space.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.models.autoencoder import Autoencoder
from src.models.base import EmbeddingModel


class TemporalAutoencoder(Autoencoder):
    """Autoencoder trained with adaptive temporal smoothing outside forward()."""

    def save(self, path: str | Path) -> None:
        """Persist state dict and metadata with a temporal model name."""
        super().save(path)
        meta_path = Path(path) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["model_name"] = "temporal_autoencoder"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingModel:
        """Load a persisted temporal autoencoder."""
        return super().load(path)
