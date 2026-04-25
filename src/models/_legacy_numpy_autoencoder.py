"""Deprecated NumPy autoencoder kept for reading older experiment artifacts.

New training should use ``src.models.autoencoder.Autoencoder``, which is a
PyTorch implementation. This module exists for one release cycle so prior
``weights.npz`` artifacts remain loadable and comparable in tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.base import EmbeddingModel


class Autoencoder(EmbeddingModel):
    """Legacy shallow nonlinear autoencoder with manual NumPy backpropagation."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 512,
        weight_decay: float = 0.00001,
        random_seed: int = 7,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.random_seed = int(random_seed)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: dict[str, np.ndarray] = {}

    def fit(self, matrix: np.ndarray) -> dict[str, float]:
        """Fit the autoencoder to reconstruct a standardized feature matrix."""
        if matrix.ndim != 2:
            raise ValueError("Training matrix must be two-dimensional.")
        if matrix.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {matrix.shape[1]}.")

        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        standardized = (matrix - self.mean_) / self.scale_
        self.weights_ = self._initial_weights()
        optimizer = _AdamOptimizer(self.weights_, learning_rate=self.learning_rate)

        initial_loss = self._loss(standardized)
        final_loss = initial_loss
        rng = np.random.default_rng(self.random_seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(standardized))
            epoch_losses: list[float] = []
            for start in range(0, len(order), self.batch_size):
                batch_index = order[start : start + self.batch_size]
                loss, gradients = self._loss_and_gradients(standardized[batch_index])
                optimizer.step(self.weights_, gradients)
                epoch_losses.append(loss)
            final_loss = float(np.mean(epoch_losses))

        return {
            "initial_reconstruction_mse": float(initial_loss),
            "final_reconstruction_mse": float(final_loss),
            "epochs": float(self.epochs),
        }

    def forward(self, matrix: np.ndarray) -> np.ndarray:
        """Reconstruct inputs in the original feature scale."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Model must be fit before forward().")
        standardized = (matrix - self.mean_) / self.scale_
        reconstruction, _ = self._forward_standardized(standardized)
        return reconstruction * self.scale_ + self.mean_

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        """Encode inputs into the bottleneck embedding layer."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Model must be fit before encode().")
        standardized = (matrix - self.mean_) / self.scale_
        _, cache = self._forward_standardized(standardized)
        return cache["embedding"]

    def save(self, path: str | Path) -> None:
        """Persist weights, normalization statistics, and model metadata."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.mean_ is None or self.scale_ is None or not self.weights_:
            raise ValueError("Cannot save an unfitted model.")

        np.savez(output_dir / "weights.npz", mean=self.mean_, scale=self.scale_, **self.weights_)
        meta = {
            "model_name": "legacy_numpy_autoencoder",
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "standardized": True,
            "deprecated": True,
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Autoencoder":
        """Load a saved legacy NumPy autoencoder model."""
        model_dir = Path(path)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        weights = np.load(model_dir / "weights.npz")
        model = cls(
            input_dim=int(meta["input_dim"]),
            embedding_dim=int(meta["embedding_dim"]),
            hidden_dim=int(meta["hidden_dim"]),
            learning_rate=float(meta["learning_rate"]),
            epochs=int(meta["epochs"]),
            batch_size=int(meta["batch_size"]),
            weight_decay=float(meta["weight_decay"]),
            random_seed=int(meta["random_seed"]),
        )
        model.mean_ = weights["mean"]
        model.scale_ = weights["scale"]
        model.weights_ = {name: weights[name] for name in _WEIGHT_NAMES}
        return model

    def _initial_weights(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.random_seed)
        shapes = {
            "w1": (self.input_dim, self.hidden_dim),
            "b1": (self.hidden_dim,),
            "w2": (self.hidden_dim, self.embedding_dim),
            "b2": (self.embedding_dim,),
            "w3": (self.embedding_dim, self.hidden_dim),
            "b3": (self.hidden_dim,),
            "w4": (self.hidden_dim, self.input_dim),
            "b4": (self.input_dim,),
        }
        weights: dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            if name.startswith("b"):
                weights[name] = np.zeros(shape)
            else:
                fan_in, fan_out = shape
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                weights[name] = rng.uniform(-limit, limit, size=shape)
        return weights

    def _forward_standardized(self, matrix: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        w = self.weights_
        hidden = np.tanh(matrix @ w["w1"] + w["b1"])
        embedding = np.tanh(hidden @ w["w2"] + w["b2"])
        decoder_hidden = np.tanh(embedding @ w["w3"] + w["b3"])
        reconstruction = decoder_hidden @ w["w4"] + w["b4"]
        cache = {
            "input": matrix,
            "hidden": hidden,
            "embedding": embedding,
            "decoder_hidden": decoder_hidden,
            "reconstruction": reconstruction,
        }
        return reconstruction, cache

    def _loss(self, matrix: np.ndarray) -> float:
        reconstruction, _ = self._forward_standardized(matrix)
        error = reconstruction - matrix
        return float(np.mean(np.square(error)))

    def _loss_and_gradients(self, matrix: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        reconstruction, cache = self._forward_standardized(matrix)
        error = reconstruction - matrix
        loss = float(np.mean(np.square(error)))
        output_grad = 2.0 * error / error.size

        w = self.weights_
        gradients: dict[str, np.ndarray] = {}
        gradients["w4"] = cache["decoder_hidden"].T @ output_grad + self.weight_decay * w["w4"]
        gradients["b4"] = output_grad.sum(axis=0)

        decoder_hidden_grad = output_grad @ w["w4"].T
        decoder_grad = decoder_hidden_grad * (1.0 - np.square(cache["decoder_hidden"]))
        gradients["w3"] = cache["embedding"].T @ decoder_grad + self.weight_decay * w["w3"]
        gradients["b3"] = decoder_grad.sum(axis=0)

        embedding_grad = decoder_grad @ w["w3"].T
        bottleneck_grad = embedding_grad * (1.0 - np.square(cache["embedding"]))
        gradients["w2"] = cache["hidden"].T @ bottleneck_grad + self.weight_decay * w["w2"]
        gradients["b2"] = bottleneck_grad.sum(axis=0)

        hidden_grad = bottleneck_grad @ w["w2"].T
        encoder_grad = hidden_grad * (1.0 - np.square(cache["hidden"]))
        gradients["w1"] = cache["input"].T @ encoder_grad + self.weight_decay * w["w1"]
        gradients["b1"] = encoder_grad.sum(axis=0)
        return loss, gradients


class _AdamOptimizer:
    """Small Adam implementation retained for legacy artifact comparisons."""

    def __init__(self, weights: dict[str, np.ndarray], learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.step_count = 0
        self.first = {name: np.zeros_like(value) for name, value in weights.items()}
        self.second = {name: np.zeros_like(value) for name, value in weights.items()}

    def step(self, weights: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> None:
        self.step_count += 1
        for name, gradient in gradients.items():
            self.first[name] = self.beta1 * self.first[name] + (1.0 - self.beta1) * gradient
            self.second[name] = self.beta2 * self.second[name] + (1.0 - self.beta2) * np.square(gradient)
            first_hat = self.first[name] / (1.0 - self.beta1**self.step_count)
            second_hat = self.second[name] / (1.0 - self.beta2**self.step_count)
            weights[name] -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)


_WEIGHT_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
