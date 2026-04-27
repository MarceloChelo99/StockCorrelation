from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd



from src.config import load_config
from src.experiment import Experiment
from src.models.registry import MODEL_REGISTRY
from src.models.train import build_model
from src.utils.io import write_json
from src.utils.logging import log
from src.utils.seed import set_seed


def main(config_name: str, experiment_dir: str | None = None) -> None:
    """Train or reuse view-specific autoencoders and write per-view embeddings."""
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    if experiment_dir is None:
        experiment = Experiment.create(config["paths"]["experiments_dir"], config["experiment_name"], config)
        root = experiment.root
    else:
        root = Path(experiment_dir)
        root.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(config["paths"]["dataset_dir"]) / f"{config['assembly']['dataset_name']}.parquet"
    dataset = pd.read_parquet(dataset_path)
    for view_name in active_view_names(config):
        view_config = config["views"][view_name]
        output_dir = root / "views" / view_name
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = output_dir / "embeddings.parquet"
        reuse_value = str(view_config.get("reuse_embeddings_path", "") or "")
        reuse_path = Path(reuse_value)
        if reuse_value and reuse_path.is_file():
            shutil.copy2(reuse_path, embeddings_path)
            write_json({"reused_from": str(reuse_path), "view": view_name}, output_dir / "training_history.json")
            log(f"Reused {view_name} embeddings from {reuse_path}.", tag="views")
            continue

        feature_columns = select_view_feature_columns(dataset, str(view_config["feature_pattern"]))
        if not feature_columns:
            raise ValueError(f"No features matched view {view_name} pattern {view_config['feature_pattern']!r}.")
        frame = dataset.loc[:, ["ticker", "date", *feature_columns]].copy()
        min_non_missing = int(view_config.get("min_non_missing_features", min(3, len(feature_columns))))
        non_missing = frame.loc[:, feature_columns].notna().sum(axis=1)
        frame = frame.loc[non_missing >= min_non_missing].copy()
        if frame.empty:
            raise ValueError(
                f"View {view_name} has no rows with at least {min_non_missing} non-missing features."
            )
        feature_frame = frame.loc[:, feature_columns].astype(float)
        min_feature_abs_sum = float(view_config.get("min_feature_abs_sum", 0.0))
        if min_feature_abs_sum > 0.0:
            signal = feature_frame.abs().sum(axis=1)
            frame = frame.loc[signal >= min_feature_abs_sum].copy()
            feature_frame = feature_frame.loc[signal >= min_feature_abs_sum].copy()
            if frame.empty:
                raise ValueError(
                    f"View {view_name} has no rows with feature signal >= {min_feature_abs_sum}."
                )
        missing_rate = float(feature_frame.isna().to_numpy().mean())
        medians = feature_frame.median(axis=0, skipna=True).fillna(0.0)
        feature_frame = feature_frame.fillna(medians)
        raw_matrix = feature_frame.to_numpy()
        affected_cells = int((~np.isfinite(raw_matrix)).sum())
        matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        model_name = str(view_config.get("model_name", config["model"].get("name", "autoencoder")))
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {model_name!r} for view {view_name!r}.")
        model_config = {
            **config,
            "model": {
                **config.get("model", {}),
                **view_config,
                "name": model_name,
                "embedding_dim": int(view_config["embedding_dim"]),
            },
        }
        model = build_model(MODEL_REGISTRY[model_name], matrix.shape[1], model_config)
        history = model.fit(matrix)
        embeddings = model.encode(matrix)
        output = frame.loc[:, ["ticker", "date"]].copy()
        for index in range(embeddings.shape[1]):
            output[f"embedding_{index}"] = embeddings[:, index]
        model.save(output_dir / "model")
        output.to_parquet(embeddings_path, index=False)
        write_json(
            {
                "view": view_name,
                "feature_columns": feature_columns,
                "min_non_missing_features": int(min_non_missing),
                "min_feature_abs_sum": min_feature_abs_sum,
                "input_missing_rate_before_imputation": missing_rate,
                "median_imputation": True,
                "model_name": model_name,
                "non_finite_cells_replaced_with_zero": affected_cells,
                **history,
            },
            output_dir / "training_history.json",
        )
        log(f"Trained {view_name} view with {len(output)} rows and {len(feature_columns)} features.", tag="views")


def select_view_feature_columns(dataset: pd.DataFrame, pattern: str) -> list[str]:
    """Select numeric feature columns matching a view regex."""
    regex = re.compile(pattern)
    numeric = dataset.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric if regex.search(column)]


def active_view_names(config: dict) -> list[str]:
    """Return views requested for this run."""
    return list(config.get("views_enabled", list(config["views"].keys())))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.pipeline.stage_04_model.train_views "
            "<config_name> [experiment_dir]"
        )
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
