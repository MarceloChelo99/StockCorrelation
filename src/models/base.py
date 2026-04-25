"""Abstract model interface for embedding generators."""
from __future__ import annotations

import types
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

try:
    from torch import nn  # type: ignore
except ModuleNotFoundError:
    class _FallbackModule:
        """Fallback stand-in when torch is not installed."""

    nn = types.SimpleNamespace(Module=_FallbackModule)


class EmbeddingModel(nn.Module, ABC):
    """Base class for embedding models used by the training stage."""

    @abstractmethod
    def fit(self, matrix: np.ndarray) -> dict[str, float]:
        """Fit the model on a numeric design matrix."""

    @abstractmethod
    def forward(self, matrix: np.ndarray) -> np.ndarray:
        """Reconstruct or transform an input matrix."""

    @abstractmethod
    def encode(self, matrix: np.ndarray) -> np.ndarray:
        """Project an input matrix into embedding space."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist weights and model metadata."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "EmbeddingModel":
        """Load a persisted model instance."""
