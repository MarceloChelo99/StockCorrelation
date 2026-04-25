from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.db import FilingsDB
from src.evaluation.registry import EVALUATOR_REGISTRY
from src.utils.io import write_json
from src.utils.logging import log
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
    metadata = db.load_tickers()
    metadata_path = Path(config["paths"].get("metadata_path", ""))
    if metadata_path.exists():
        external = pd.read_parquet(metadata_path)
        if "ticker" not in external.columns:
            raise ValueError(f"Metadata file {metadata_path} must contain a ticker column.")
        external = external.copy()
        external["ticker"] = external["ticker"].astype(str).str.upper()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        duplicate_columns = [column for column in external.columns if column in metadata.columns and column != "ticker"]
        external = external.drop(columns=duplicate_columns)
        metadata = metadata.merge(external, on="ticker", how="left")
    return metadata


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: .venv/bin/python scripts/05_evaluate.py <config_name> <embeddings_path>")
    main(sys.argv[1], sys.argv[2])
