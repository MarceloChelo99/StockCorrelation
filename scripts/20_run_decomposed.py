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
from src.experiment import Experiment
from src.features.assembly import assemble_dataset
from src.features.registry import build_active_producers
from src.models.train import train_embedding_model
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log
from src.utils.seed import set_seed

from scripts_compat import fit_gmms, interpret_views, train_views


def main(config_name: str = "config/experiments/decomposed.yaml") -> None:
    """Run the decomposed-similarity pipeline for one config."""
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    experiment = Experiment.create(config["paths"]["experiments_dir"], config["experiment_name"], config)
    db = FilingsDB.from_config(config)

    compute_features(db, config)
    dataset = assemble_dataset(
        db,
        config,
        feature_dir=config["paths"]["feature_dir"],
        output_path=Path(config["paths"]["dataset_dir"]) / f"{config['assembly']['dataset_name']}.parquet",
    )
    log(f"Assembled decomposed dataset with {len(dataset)} rows.", tag="decomposed")
    train_embedding_model(
        dataset,
        config,
        model_dir=experiment.model_dir,
        embeddings_path=experiment.embeddings_path,
        history_path=experiment.history_path,
    )
    if config.get("pipeline", {}).get("views", False):
        train_views(config, experiment.root)
    if config.get("pipeline", {}).get("gmm", False):
        fit_gmms(config, experiment.root)
        interpret_views(config, experiment.root, Path("report"))
    if config.get("pipeline", {}).get("multiview_eval", False) or config["evaluation"].get("enabled"):
        run_evaluations(config, db, experiment)
    write_decomposed_report_stubs()
    log(f"Decomposed run complete at {experiment.root}.", tag="decomposed")


def compute_features(db: FilingsDB, config: dict) -> None:
    """Compute configured feature groups."""
    output_dir = ensure_dir(config["paths"]["feature_dir"])
    producers = build_active_producers(config)
    manifest = {}
    for feature_name in config["features"]["enabled"]:
        producer = producers[feature_name]
        log(f"Computing feature group {feature_name}.", tag="decomposed")
        frame = producer.run(db, config, output_dir)
        manifest[feature_name] = {"rows": int(len(frame)), "path": str(producer.output_path(output_dir))}
    write_json(manifest, output_dir / "feature_manifest.json")


def run_evaluations(config: dict, db: FilingsDB, experiment: Experiment) -> None:
    """Run configured evaluators and write metrics."""
    embeddings = pd.read_parquet(experiment.embeddings_path)
    metadata = load_metadata(db, config)
    metrics = {}
    for evaluator_name in config["evaluation"]["enabled"]:
        evaluator = EVALUATOR_REGISTRY[evaluator_name]
        log(f"Running evaluator {evaluator_name}.", tag="decomposed")
        metrics[evaluator_name] = evaluator.run(
            embeddings,
            metadata,
            config,
            db=db,
            output_dir=experiment.root,
        )
    write_json(metrics, experiment.root / "metrics.json")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load ticker metadata and optional external GICS metadata."""
    metadata = db.load_tickers()
    metadata_path = Path(config["paths"].get("metadata_path", ""))
    if metadata_path.exists():
        external = pd.read_parquet(metadata_path)
        external["ticker"] = external["ticker"].astype(str).str.upper()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        duplicate_columns = [column for column in external.columns if column in metadata.columns and column != "ticker"]
        metadata = metadata.merge(external.drop(columns=duplicate_columns), on="ticker", how="left")
    return metadata


def write_decomposed_report_stubs() -> None:
    """Create decomposed report files without overwriting existing notes."""
    report_dir = ensure_dir("report")
    stubs = {
        "decomposed_similarity_thesis.md": "# Decomposed Similarity Thesis\n\nCompany similarity is decomposed into business, behavioral, growth/lifecycle, and network views.\n",
        "decomposed_similarity_results.md": "# Decomposed Similarity Results\n\nResults are organized by view and evaluation artifact.\n",
        "multiview_covariance_results.md": "# Multi-View Covariance Results\n\nHeadline full-universe, leave-one-view-out, and slice results belong here after the decomposed run.\n",
        "view_interpretation_guide.md": "# View Interpretation Guide\n\nManual labels and representative themes from per-view reports belong here.\n",
    }
    for filename, text in stubs.items():
        path = report_dir / filename
        if not path.exists():
            path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/experiments/decomposed.yaml")
