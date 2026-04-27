from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.config import load_config
from src.experiment import Experiment
from src.models.train import train_embedding_model
from src.utils.logging import log
from src.utils.seed import set_seed


def main(config_name: str = "baseline") -> None:
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))

    dataset_path = Path(config["paths"]["dataset_dir"]) / f"{config['assembly']['dataset_name']}.parquet"
    dataset = pd.read_parquet(dataset_path)
    experiment = Experiment.create(
        config["paths"]["experiments_dir"],
        config["experiment_name"],
        config,
    )
    _, embeddings, history = train_embedding_model(
        dataset,
        config,
        model_dir=experiment.model_dir,
        embeddings_path=experiment.embeddings_path,
        history_path=experiment.history_path,
    )
    log(f"Training wrote {len(embeddings)} embeddings to {experiment.embeddings_path}.", tag="train")
    log(f"Training history: {history}.", tag="train", log_path=experiment.log_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
