"""Explicit model registry used by the training stage."""
from __future__ import annotations

from src.models.autoencoder import Autoencoder
from src.models.pca import PCAModel


MODEL_REGISTRY = {
    "autoencoder": Autoencoder,
    "pca": PCAModel,
}
