"""Training helpers for embedding models."""
from __future__ import annotations

from pathlib import Path
import copy
import inspect

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import has_batch_norm, set_torch_seed
from src.models.registry import MODEL_REGISTRY
from src.models.temporal_dataloader import build_consecutive_pairs, split_pair_dataset
from src.utils.io import write_json


def train_embedding_model(
    dataset: pd.DataFrame,
    config: dict,
    *,
    model_dir: str | Path,
    embeddings_path: str | Path,
    history_path: str | Path,
) -> tuple[object, pd.DataFrame, dict]:
    """Fit the configured model and write embeddings plus training metadata."""
    feature_columns = select_feature_columns(dataset, config)
    if not feature_columns:
        raise ValueError("No numeric feature columns were selected for training.")

    raw_matrix = dataset.loc[:, feature_columns].astype(float).to_numpy()
    affected_cells = int((~np.isfinite(raw_matrix)).sum())
    finite_rows = np.isfinite(raw_matrix).all(axis=1)
    rows_excluded_from_fit = 0
    matrix = raw_matrix.copy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    fit_complete_cases_only = bool(config["model"].get("fit_complete_cases_only", False))
    fit_matrix = matrix
    if fit_complete_cases_only:
        rows_excluded_from_fit = int((~finite_rows).sum())
        fit_matrix = matrix[finite_rows]
        if len(fit_matrix) == 0:
            raise ValueError("fit_complete_cases_only=True left zero rows for model fitting.")

    model_name = config["model"]["name"]
    model_cls = MODEL_REGISTRY[model_name]
    model = build_model(model_cls, matrix.shape[1], config)
    if model_name == "temporal_autoencoder":
        temporal_fit_dataset = dataset.loc[:, ["ticker", "date"]].copy()
        for index, column in enumerate(feature_columns):
            temporal_fit_dataset[column] = matrix[:, index]
        if fit_complete_cases_only:
            temporal_fit_dataset = temporal_fit_dataset.loc[finite_rows].reset_index(drop=True)
        max_pair_gap_days = int(config["model"].get("max_pair_gap_days", 45))
        pairs = build_consecutive_pairs(
            temporal_fit_dataset,
            feature_columns,
            max_gap_days=max_pair_gap_days,
        )
        train_pairs, val_pairs = split_pair_dataset(
            pairs,
            float(config["model"].get("train_val_split", 0.85)),
            int(config["random_seed"]),
        )
        history = train_temporal_autoencoder(model, train_pairs, val_pairs, config)
        history = {
            **history,
            "n_pairs": int(getattr(pairs, "n_pairs", len(pairs))),
            "n_pairs_train": int(len(train_pairs)),
            "n_pairs_val": int(len(val_pairs)),
            "n_observations_dropped_gap": int(getattr(pairs, "n_observations_dropped_gap", 0)),
            "n_pairs_dropped_non_finite": int(getattr(pairs, "n_pairs_dropped_non_finite", 0)),
            "max_pair_gap_days": max_pair_gap_days,
        }
    else:
        history = model.fit(fit_matrix)
    embeddings = model.encode(matrix)
    history = {
        **history,
        "fit_complete_cases_only": fit_complete_cases_only,
        "n_rows_fit": int(fit_matrix.shape[0]),
        "rows_excluded_from_fit_non_finite": rows_excluded_from_fit,
    }

    embedding_columns = [f"embedding_{index}" for index in range(embeddings.shape[1])]
    embeddings_frame = dataset.loc[:, ["ticker", "date"]].copy()
    for index, column in enumerate(embedding_columns):
        embeddings_frame[column] = embeddings[:, index]

    model.save(model_dir)
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    embeddings_frame.to_parquet(embeddings_path, index=False)
    if model_name == "temporal_autoencoder":
        diagnostics = temporal_embedding_diagnostics(
            embeddings_frame,
            max_gap_days=int(config["model"].get("max_pair_gap_days", 45)),
        )
        diagnostics = {
            "n_pairs_train": int(history["n_pairs_train"]),
            "n_pairs_val": int(history["n_pairs_val"]),
            "n_observations_dropped_gap": int(history["n_observations_dropped_gap"]),
            **diagnostics,
        }
        diagnostics_path = Path(history_path).with_name("temporal_diagnostics.json")
        write_json(diagnostics, diagnostics_path)
        history["temporal_diagnostics_path"] = str(diagnostics_path)
    write_json(
        {
            "model_name": model_name,
            "feature_columns": feature_columns,
            "n_rows": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "non_finite_cells_replaced_with_zero": affected_cells,
            **history,
        },
        history_path,
    )
    return model, embeddings_frame, history


def build_model(model_cls: type, input_dim: int, config: dict) -> object:
    """Instantiate a model with the config keys its constructor declares."""
    model_config = dict(config["model"])
    model_config["input_dim"] = input_dim
    model_config.pop("name", None)
    model_config.pop("exclude_columns", None)

    signature = inspect.signature(model_cls)
    accepted = set(signature.parameters)
    kwargs = {
        key: value
        for key, value in model_config.items()
        if key in accepted
    }
    if "random_seed" in accepted and "random_seed" not in kwargs:
        kwargs["random_seed"] = int(config["random_seed"])
    return model_cls(**kwargs)


def select_feature_columns(dataset: pd.DataFrame, config: dict) -> list[str]:
    """Select numeric training columns while excluding metadata fields."""
    excluded = set(config["model"].get("exclude_columns", []))
    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def train_temporal_autoencoder(
    model: object,
    train_pairs: TensorDataset,
    val_pairs: TensorDataset,
    config: dict,
) -> dict:
    """Train an autoencoder with adaptive temporal smoothing loss."""
    model_config = config["model"]
    lambda_temp = float(model_config.get("lambda_temp", 0.5))
    alpha = float(model_config.get("alpha", 1.0))
    random_seed = int(config.get("random_seed", getattr(model, "random_seed", 7)))
    set_torch_seed(random_seed)

    train_current = train_pairs.tensors[0].detach().cpu().numpy()
    if len(train_current) == 0:
        raise ValueError("Cannot train temporal autoencoder with zero training pairs.")
    mean = train_current.mean(axis=0).astype(np.float32)
    scale = train_current.std(axis=0).astype(np.float32)
    scale[scale == 0.0] = 1.0

    device = getattr(model, "feature_mean").device
    model.to(device)
    model.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=device))
    model.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=device))

    batch_size = max(2, int(getattr(model, "batch_size", 512)))
    train_loader = DataLoader(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=bool(has_batch_norm(model) and len(train_pairs) > batch_size),
        generator=torch.Generator().manual_seed(random_seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(getattr(model, "learning_rate", 0.001)),
        weight_decay=float(getattr(model, "weight_decay", 0.00001)),
    )
    loss_fn = nn.MSELoss()

    initial_train = _temporal_dataset_loss(model, train_pairs, loss_fn, lambda_temp, alpha)
    initial_val = _temporal_dataset_loss(model, val_pairs, loss_fn, lambda_temp, alpha)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float(initial_val["loss"])
    best_epoch = 0
    epochs_without_improvement = 0
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_recon: list[float] = []
    val_recon: list[float] = []
    train_smooth: list[float] = []
    val_smooth: list[float] = []

    for epoch in range(1, int(getattr(model, "epochs", 40)) + 1):
        model.train()
        batch_metrics: list[dict[str, float]] = []
        for x_t, x_prev in train_loader:
            if len(x_t) < 2 and has_batch_norm(model):
                continue
            x_t = x_t.to(device)
            x_prev = x_prev.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = _temporal_batch_loss(
                model,
                x_t,
                x_prev,
                loss_fn,
                lambda_temp=lambda_temp,
                alpha=alpha,
            )
            losses["loss_tensor"].backward()
            optimizer.step()
            batch_metrics.append(
                {
                    "loss": float(losses["loss_tensor"].detach().cpu().item()),
                    "recon": float(losses["reconstruction_tensor"].detach().cpu().item()),
                    "smooth": float(losses["smoothness_tensor"].detach().cpu().item()),
                }
            )

        if batch_metrics:
            train_epoch = {
                key: float(np.mean([metrics[key] for metrics in batch_metrics]))
                for key in ("loss", "recon", "smooth")
            }
        else:
            train_epoch = _temporal_dataset_loss(model, train_pairs, loss_fn, lambda_temp, alpha)
        val_epoch = _temporal_dataset_loss(model, val_pairs, loss_fn, lambda_temp, alpha)

        train_loss.append(float(train_epoch["loss"]))
        val_loss.append(float(val_epoch["loss"]))
        train_recon.append(float(train_epoch["recon"]))
        val_recon.append(float(val_epoch["recon"]))
        train_smooth.append(float(train_epoch["smooth"]))
        val_smooth.append(float(val_epoch["smooth"]))

        if val_epoch["loss"] < best_val_loss - 1e-8:
            best_val_loss = float(val_epoch["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(getattr(model, "early_stopping_patience", 10)):
            break

    model.load_state_dict(best_state)
    model.is_fitted = True
    final_train = _temporal_dataset_loss(model, train_pairs, loss_fn, lambda_temp, alpha)
    final_val = _temporal_dataset_loss(model, val_pairs, loss_fn, lambda_temp, alpha)
    return {
        "initial_temporal_loss": float(initial_val["loss"]),
        "initial_train_temporal_loss": float(initial_train["loss"]),
        "final_temporal_loss": float(final_val["loss"]),
        "final_train_temporal_loss": float(final_train["loss"]),
        "best_val_temporal_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs": float(len(train_loss)),
        "epochs_trained": int(len(train_loss)),
        "early_stopped": bool(len(train_loss) < int(getattr(model, "epochs", 40))),
        "temporal_lambda": lambda_temp,
        "temporal_alpha": alpha,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_reconstruction_loss": train_recon,
        "val_reconstruction_loss": val_recon,
        "train_smoothness_loss": train_smooth,
        "val_smoothness_loss": val_smooth,
    }


def _temporal_batch_loss(
    model: object,
    x_t: torch.Tensor,
    x_prev: torch.Tensor,
    loss_fn: nn.Module,
    *,
    lambda_temp: float,
    alpha: float,
) -> dict[str, torch.Tensor]:
    mean = model.feature_mean
    scale = model.feature_scale
    x_t_norm = (x_t - mean) / scale
    x_prev_norm = (x_prev - mean) / scale

    x_t_hat, z_t = model(x_t_norm)
    _, z_prev = model(x_prev_norm)
    reconstruction = loss_fn(x_t_hat, x_t_norm)
    feature_change = torch.linalg.norm(x_t_norm - x_prev_norm, dim=1)
    embedding_change = torch.linalg.norm(z_t - z_prev, dim=1)
    weights = torch.exp(-float(alpha) * feature_change)
    smoothness = torch.mean(weights * torch.square(embedding_change))
    loss = reconstruction + float(lambda_temp) * smoothness
    return {
        "loss_tensor": loss,
        "reconstruction_tensor": reconstruction,
        "smoothness_tensor": smoothness,
    }


def _temporal_dataset_loss(
    model: object,
    pairs: TensorDataset,
    loss_fn: nn.Module,
    lambda_temp: float,
    alpha: float,
) -> dict[str, float]:
    model.eval()
    if len(pairs) == 0:
        return {"loss": float("nan"), "recon": float("nan"), "smooth": float("nan")}

    device = model.feature_mean.device
    loader = DataLoader(pairs, batch_size=max(2, int(getattr(model, "batch_size", 512))), shuffle=False)
    totals = {"loss": 0.0, "recon": 0.0, "smooth": 0.0}
    n_rows = 0
    with torch.no_grad():
        for x_t, x_prev in loader:
            x_t = x_t.to(device)
            x_prev = x_prev.to(device)
            losses = _temporal_batch_loss(
                model,
                x_t,
                x_prev,
                loss_fn,
                lambda_temp=lambda_temp,
                alpha=alpha,
            )
            batch_n = int(len(x_t))
            totals["loss"] += float(losses["loss_tensor"].detach().cpu().item()) * batch_n
            totals["recon"] += float(losses["reconstruction_tensor"].detach().cpu().item()) * batch_n
            totals["smooth"] += float(losses["smoothness_tensor"].detach().cpu().item()) * batch_n
            n_rows += batch_n
    return {key: value / max(1, n_rows) for key, value in totals.items()}


def temporal_embedding_diagnostics(embeddings: pd.DataFrame, max_gap_days: int = 45) -> dict:
    """Summarize whether temporal embeddings are smooth but still mobile."""
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError("No embedding_ columns found for temporal diagnostics.")

    frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    global_scale = float(np.std(frame.loc[:, embedding_columns].to_numpy(dtype=float), axis=0).mean())
    firm_ratios: list[float] = []
    velocities: list[float] = []
    velocity_by_firm: dict[str, list[float]] = {}
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        values = ticker_frame.loc[:, embedding_columns].to_numpy(dtype=float)
        if len(values) > 1 and global_scale > 0.0:
            firm_ratios.append(float(np.std(values, axis=0).mean() / global_scale))
        dates = ticker_frame["date"].to_numpy(dtype="datetime64[ns]")
        firm_velocities: list[float] = []
        for index in range(1, len(values)):
            gap_days = int((dates[index] - dates[index - 1]) / np.timedelta64(1, "D"))
            if gap_days > int(max_gap_days):
                continue
            velocity = float(np.linalg.norm(values[index] - values[index - 1]))
            velocities.append(velocity)
            firm_velocities.append(velocity)
        if firm_velocities:
            velocity_by_firm[str(ticker)] = firm_velocities

    if velocities:
        velocity_array = np.asarray(velocities, dtype=float)
        distribution = {
            "p10": float(np.percentile(velocity_array, 10)),
            "p50": float(np.percentile(velocity_array, 50)),
            "p90": float(np.percentile(velocity_array, 90)),
            "max": float(np.max(velocity_array)),
        }
    else:
        distribution = {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}

    high_velocity = sorted(
        velocity_by_firm,
        key=lambda ticker: float(np.mean(velocity_by_firm[ticker])),
        reverse=True,
    )[:20]
    return {
        "within_firm_temporal_variance_ratio": float(np.mean(firm_ratios)) if firm_ratios else float("nan"),
        "embedding_velocity_distribution": distribution,
        "high_velocity_firms": high_velocity,
        "n_velocity_steps": int(len(velocities)),
    }
