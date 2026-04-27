from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.config import load_config
from src.db import FilingsDB
from src.evaluation.registry import EVALUATOR_REGISTRY
from src.utils.io import write_json
from src.utils.logging import log
from src.utils.metadata import load_ticker_metadata
from src.utils.seed import set_seed


def main(config_name: str = "baseline", embeddings_path: str | None = None) -> None:
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    if embeddings_path is None:
        raise ValueError("Pass an embeddings parquet path as the second argument.")

    db = FilingsDB.from_config(config)
    embeddings = pd.read_parquet(embeddings_path)
    metadata = load_metadata(db, config)
    output_dir = Path(embeddings_path).parent
    metrics: dict[str, dict] = {}

    for evaluator_name in config["evaluation"]["enabled"]:
        evaluator = EVALUATOR_REGISTRY[evaluator_name]
        log(f"Running evaluator {evaluator_name}.", tag="evaluation")
        metrics[evaluator_name] = evaluator.run(embeddings, metadata, config, db=db, output_dir=output_dir)

    write_json(metrics, Path(embeddings_path).with_name("metrics.json"))
    log("Evaluation complete.", tag="evaluation")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load ticker metadata and merge optional external sector metadata."""
    return load_ticker_metadata(db, config)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.pipeline.stage_05_evaluation.evaluate "
            "<config_name> <embeddings_path>"
        )
    main(sys.argv[1], sys.argv[2])
