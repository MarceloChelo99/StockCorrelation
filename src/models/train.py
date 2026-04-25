"""Training helpers for embedding models."""
from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
import pandas as pd

from src.models.registry import MODEL_REGISTRY
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
    matrix = raw_matrix.copy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    model_name = config["model"]["name"]
    model_cls = MODEL_REGISTRY[model_name]
    model = build_model(model_cls, matrix.shape[1], config)
    history = model.fit(matrix)
    embeddings = model.encode(matrix)

    embedding_columns = [f"embedding_{index}" for index in range(embeddings.shape[1])]
    embeddings_frame = dataset.loc[:, ["ticker", "date"]].copy()
    for index, column in enumerate(embedding_columns):
        embeddings_frame[column] = embeddings[:, index]

    model.save(model_dir)
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    embeddings_frame.to_parquet(embeddings_path, index=False)
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
