"""A lightweight PCA baseline that produces stock embeddings without torch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.base import EmbeddingModel


class PCAModel(EmbeddingModel):
    """Fit a low-dimensional linear projection with SVD."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: list[float] = []

    def fit(self, matrix: np.ndarray) -> dict[str, float]:
        if matrix.ndim != 2:
            raise ValueError("Training matrix must be two-dimensional.")
        if matrix.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {matrix.shape[1]}.")

        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        standardized = (matrix - self.mean_) / self.scale_
        _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
        components = vt[: self.embedding_dim]
        self.components_ = components

        denominator = np.square(singular_values).sum()
        if denominator > 0:
            top = np.square(singular_values[: self.embedding_dim]).sum()
            explained = float(top / denominator)
        else:
            explained = 0.0
        self.explained_variance_ratio_ = [explained]
        return {"explained_variance_ratio": explained}

    def forward(self, matrix: np.ndarray) -> np.ndarray:
        embeddings = self.encode(matrix)
        assert self.components_ is not None
        assert self.mean_ is not None
        assert self.scale_ is not None
        reconstructed_standardized = embeddings @ self.components_
        return reconstructed_standardized * self.scale_ + self.mean_

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise ValueError("Model must be fit before encode().")
        standardized = (matrix - self.mean_) / self.scale_
        return standardized @ self.components_.T

    def save(self, path: str | Path) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise ValueError("Cannot save an unfitted model.")
        np.savez(
            output_dir / "weights.npz",
            mean=self.mean_,
            scale=self.scale_,
            components=self.components_,
        )
        meta = {
            "model_name": "pca",
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "standardized": True,
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PCAModel":
        model_dir = Path(path)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        weights = np.load(model_dir / "weights.npz")
        model = cls(input_dim=int(meta["input_dim"]), embedding_dim=int(meta["embedding_dim"]))
        model.mean_ = weights["mean"]
        model.scale_ = weights["scale"] if "scale" in weights.files else np.ones_like(model.mean_)
        model.components_ = weights["components"]
        model.explained_variance_ratio_ = list(meta.get("explained_variance_ratio", []))
        return model
