"""PyTorch autoencoder for reconstruction-trained stock embeddings.

This replaces the original hand-written NumPy autoencoder. The public research
interface remains ``fit`` / ``encode`` / ``save`` / ``load``, while the model is
now a real ``torch.nn.Module`` with multi-layer encoders, BatchNorm, ReLU, and
Dropout. ReLU is used instead of the legacy tanh hidden activations because it
trains more reliably for deeper ``hidden_dims`` configurations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import EmbeddingModel


class Autoencoder(EmbeddingModel):
    """Multi-layer PyTorch autoencoder with standardized reconstruction loss."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 512,
        weight_decay: float = 0.00001,
        random_seed: int = 7,
        train_val_split: float = 0.85,
        early_stopping_patience: int = 10,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = canonical_hidden_dims(hidden_dims, hidden_dim)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.random_seed = int(random_seed)
        self.train_val_split = float(train_val_split)
        self.early_stopping_patience = int(early_stopping_patience)
        self.device_name = device or "cpu"

        self.encoder = build_mlp(
            input_dim=self.input_dim,
            layer_dims=[*self.hidden_dims, self.embedding_dim],
            dropout=self.dropout,
            final_activation=False,
        )
        self.decoder = build_mlp(
            input_dim=self.embedding_dim,
            layer_dims=[*reversed(self.hidden_dims), self.input_dim],
            dropout=self.dropout,
            final_activation=False,
        )
        self.register_buffer("feature_mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(self.input_dim, dtype=torch.float32))
        self.is_fitted = False
        self.to(torch.device(self.device_name))

    def forward(self, matrix: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor] | np.ndarray:
        """Return standardized reconstruction and embedding for tensor inputs.

        For backward compatibility, NumPy inputs return reconstructed values in
        the original feature scale, matching the legacy public behavior.
        """
        if isinstance(matrix, np.ndarray):
            return self.reconstruct(matrix)
        embedding = self.encoder(matrix.float())
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def fit(self, matrix: np.ndarray) -> dict[str, Any]:
        """Fit the autoencoder using MSE reconstruction loss and early stopping."""
        clean = validate_matrix(matrix, self.input_dim)
        set_torch_seed(self.random_seed)
        self.to(torch.device(self.device_name))

        mean = clean.mean(axis=0).astype(np.float32)
        scale = clean.std(axis=0).astype(np.float32)
        scale[scale == 0.0] = 1.0
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=self.feature_mean.device))
        self.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=self.feature_scale.device))

        standardized = ((clean - mean) / scale).astype(np.float32)
        train_tensor, val_tensor = train_val_tensors(standardized, self.train_val_split, self.random_seed)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=max(2, self.batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        initial_train_loss = self._dataset_loss(train_tensor, loss_fn)
        initial_val_loss = self._dataset_loss(val_tensor, loss_fn)
        best_state = copy.deepcopy(self.state_dict())
        best_val_loss = initial_val_loss
        best_epoch = 0
        epochs_without_improvement = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            self.train()
            batch_losses: list[float] = []
            for (batch,) in train_loader:
                if len(batch) < 2 and has_batch_norm(self):
                    continue
                batch = batch.to(self.feature_mean.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = self(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else self._dataset_loss(train_tensor, loss_fn)
            val_loss = self._dataset_loss(val_tensor, loss_fn)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                break

        self.load_state_dict(best_state)
        self.is_fitted = True
        final_loss = self._dataset_loss(torch.as_tensor(standardized, dtype=torch.float32), loss_fn)
        return {
            "initial_reconstruction_mse": float(initial_val_loss),
            "initial_train_reconstruction_mse": float(initial_train_loss),
            "final_reconstruction_mse": float(final_loss),
            "best_val_reconstruction_mse": float(best_val_loss),
            "epochs": float(len(train_losses)),
            "epochs_trained": int(len(train_losses)),
            "best_epoch": int(best_epoch),
            "early_stopped": bool(len(train_losses) < self.epochs),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }

    def reconstruct(self, matrix: np.ndarray) -> np.ndarray:
        """Reconstruct inputs in the original feature scale."""
        standardized = self._standardize_numpy(matrix)
        reconstruction = self._reconstruct_standardized(standardized)
        mean = self.feature_mean.detach().cpu().numpy()
        scale = self.feature_scale.detach().cpu().numpy()
        return reconstruction * scale + mean

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        """Encode inputs into the bottleneck embedding layer."""
        standardized = self._standardize_numpy(matrix)
        self.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(standardized), max(1, self.batch_size)):
                batch = torch.as_tensor(
                    standardized[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.feature_mean.device,
                )
                embedding = self.encoder(batch)
                outputs.append(embedding.detach().cpu().numpy())
        return np.vstack(outputs) if outputs else np.empty((0, self.embedding_dim))

    def reconstruction_mse(self, matrix: np.ndarray) -> float:
        """Return standardized reconstruction MSE for a matrix."""
        standardized = self._standardize_numpy(matrix)
        reconstruction = self._reconstruct_standardized(standardized)
        return float(np.mean(np.square(reconstruction - standardized)))

    def save(self, path: str | Path) -> None:
        """Persist state dict and architecture metadata."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "weights.pt")
        meta = {
            "model_name": "autoencoder",
            "framework": "pytorch",
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "train_val_split": self.train_val_split,
            "early_stopping_patience": self.early_stopping_patience,
            "standardized": True,
            "normalization_buffers": ["feature_mean", "feature_scale"],
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingModel:
        """Load a PyTorch autoencoder, or a legacy NumPy artifact if needed."""
        model_dir = Path(path)
        weights_path = model_dir / "weights.pt"
        if not weights_path.exists() and (model_dir / "weights.npz").exists():
            from src.models._legacy_numpy_autoencoder import Autoencoder as LegacyAutoencoder

            return LegacyAutoencoder.load(model_dir)

        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        model = cls(
            input_dim=int(meta["input_dim"]),
            embedding_dim=int(meta["embedding_dim"]),
            hidden_dims=list(meta.get("hidden_dims") or [meta.get("hidden_dim", 64)]),
            dropout=float(meta.get("dropout", 0.0)),
            learning_rate=float(meta.get("learning_rate", 0.001)),
            epochs=int(meta.get("epochs", 40)),
            batch_size=int(meta.get("batch_size", 512)),
            weight_decay=float(meta.get("weight_decay", 0.00001)),
            random_seed=int(meta.get("random_seed", 7)),
            train_val_split=float(meta.get("train_val_split", 0.85)),
            early_stopping_patience=int(meta.get("early_stopping_patience", 10)),
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.is_fitted = True
        model.eval()
        return model

    def _standardize_numpy(self, matrix: np.ndarray) -> np.ndarray:
        clean = validate_matrix(matrix, self.input_dim)
        mean = self.feature_mean.detach().cpu().numpy()
        scale = self.feature_scale.detach().cpu().numpy()
        return ((clean - mean) / scale).astype(np.float32)

    def _reconstruct_standardized(self, standardized: np.ndarray) -> np.ndarray:
        self.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(standardized), max(1, self.batch_size)):
                batch = torch.as_tensor(
                    standardized[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.feature_mean.device,
                )
                reconstruction, _ = self(batch)
                outputs.append(reconstruction.detach().cpu().numpy())
        return np.vstack(outputs) if outputs else np.empty((0, self.input_dim))

    def _dataset_loss(self, tensor: torch.Tensor, loss_fn: nn.Module) -> float:
        self.eval()
        if len(tensor) == 0:
            return float("nan")
        with torch.no_grad():
            reconstruction, _ = self(tensor.to(self.feature_mean.device))
            loss = loss_fn(reconstruction, tensor.to(self.feature_mean.device))
        return float(loss.detach().cpu().item())


def canonical_hidden_dims(
    hidden_dims: list[int] | tuple[int, ...] | None,
    hidden_dim: int | None,
) -> list[int]:
    """Return the canonical hidden-dimension list, accepting legacy hidden_dim."""
    if hidden_dims is not None:
        values = [int(value) for value in hidden_dims]
    elif hidden_dim is not None:
        values = [int(hidden_dim)]
    else:
        values = [64]
    if not values or any(value <= 0 for value in values):
        raise ValueError("hidden_dims must contain positive integers.")
    return values


def build_mlp(input_dim: int, layer_dims: list[int], dropout: float, final_activation: bool) -> nn.Sequential:
    """Build a Linear/BatchNorm/ReLU/Dropout MLP with a linear final layer."""
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for index, width in enumerate(layer_dims):
        is_final = index == len(layer_dims) - 1
        layers.append(nn.Linear(previous, int(width)))
        if not is_final or final_activation:
            layers.append(nn.BatchNorm1d(int(width)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        previous = int(width)
    return nn.Sequential(*layers)


def validate_matrix(matrix: np.ndarray, input_dim: int) -> np.ndarray:
    """Validate and coerce an input design matrix."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Training matrix must be two-dimensional.")
    if array.shape[1] != int(input_dim):
        raise ValueError(f"Expected input_dim={input_dim}, got {array.shape[1]}.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def train_val_tensors(
    standardized: np.ndarray,
    train_val_split: float,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split standardized rows deterministically into train and validation tensors."""
    n_rows = len(standardized)
    if n_rows == 0:
        raise ValueError("Cannot train an autoencoder on zero rows.")
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(n_rows)
    if n_rows < 3:
        train_index = order
        val_index = order
    else:
        split = int(round(n_rows * float(train_val_split)))
        split = min(max(split, 2), n_rows - 1)
        train_index = order[:split]
        val_index = order[split:]
    return (
        torch.as_tensor(standardized[train_index], dtype=torch.float32),
        torch.as_tensor(standardized[val_index], dtype=torch.float32),
    )


def set_torch_seed(seed: int) -> None:
    """Seed PyTorch for deterministic CPU training."""
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(False)


def has_batch_norm(module: nn.Module) -> bool:
    """Return whether a module contains BatchNorm layers."""
    return any(isinstance(child, nn.BatchNorm1d) for child in module.modules())
