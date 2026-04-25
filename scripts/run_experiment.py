from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.db import FilingsDB
from src.experiment import Experiment
from src.features.assembly import assemble_dataset
from src.features.registry import build_active_producers
from src.models.train import train_embedding_model
from src.evaluation.registry import EVALUATOR_REGISTRY
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log
from src.utils.seed import set_seed
from scripts_compat import fit_gmms, interpret_views, train_views


def main(config_name: str = "baseline") -> None:
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    experiment = Experiment.create(
        config["paths"]["experiments_dir"],
        config["experiment_name"],
        config,
    )
    db = FilingsDB.from_config(config)
    producers = build_active_producers(config)

    feature_dir = ensure_dir(config["paths"]["feature_dir"])
    manifest: dict[str, dict[str, object]] = {}
    for feature_name in config["features"]["enabled"]:
        producer = producers[feature_name]
        log(f"Computing {feature_name}.", tag="run", log_path=experiment.log_path)
        frame = producer.run(db, config, feature_dir)
        manifest[feature_name] = {
            "rows": int(len(frame)),
            "path": str(producer.output_path(feature_dir)),
        }
    write_json(manifest, Path(feature_dir) / "feature_manifest.json")

    dataset_path = Path(config["paths"]["dataset_dir"]) / f"{config['assembly']['dataset_name']}.parquet"
    dataset = assemble_dataset(db, config, feature_dir=feature_dir, output_path=dataset_path)
    log(f"Assembled dataset rows: {len(dataset)}.", tag="run", log_path=experiment.log_path)

    _, embeddings, history = train_embedding_model(
        dataset,
        config,
        model_dir=experiment.model_dir,
        embeddings_path=experiment.embeddings_path,
        history_path=experiment.history_path,
    )
    log(
        f"Training complete with {len(embeddings)} embedding rows and history {history}.",
        tag="run",
        log_path=experiment.log_path,
    )

    if config.get("pipeline", {}).get("views", False):
        train_views(config, experiment.root)
    if config.get("pipeline", {}).get("gmm", False):
        fit_gmms(config, experiment.root)
        interpret_views(config, experiment.root, Path("report"))
    if config["evaluation"]["enabled"]:
        metadata = load_metadata(db, config)
        metrics = {}
        for evaluator_name in config["evaluation"]["enabled"]:
            evaluator = EVALUATOR_REGISTRY[evaluator_name]
            log(f"Running evaluator {evaluator_name}.", tag="run", log_path=experiment.log_path)
            metrics[evaluator_name] = evaluator.run(
                embeddings,
                metadata,
                config,
                db=db,
                output_dir=experiment.root,
            )
        write_json(metrics, experiment.root / "metrics.json")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load ticker metadata and merge optional external sector metadata."""
    metadata = db.load_tickers()
    metadata_path = Path(config["paths"].get("metadata_path", ""))
    if metadata_path.exists():
        external = pd.read_parquet(metadata_path)
        external["ticker"] = external["ticker"].astype(str).str.upper()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        duplicate_columns = [column for column in external.columns if column in metadata.columns and column != "ticker"]
        metadata = metadata.merge(external.drop(columns=duplicate_columns), on="ticker", how="left")
    return metadata


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
