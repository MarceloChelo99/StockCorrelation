from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.clustering.gmm import fit_gmm_on_embeddings, gmm_meta
from src.config import load_config
from src.utils.io import write_json
from src.utils.logging import log


def main(config_name: str, experiment_dir: str) -> None:
    """Fit GMM soft themes for every view embedding."""
    config = load_config(config_name)
    root = Path(experiment_dir)
    settings = config.get("gmm", {})
    minimum = int(settings.get("min_components", 5))
    maximum = int(settings.get("max_components", 40))
    if bool(config.get("pipeline", {}).get("smoke_test", False)):
        minimum, maximum = 2, 5
    candidates = range(minimum, maximum + 1)
    for view_name in config.get("views_enabled", list(config["views"].keys())):
        view_dir = root / "views" / view_name
        embeddings = pd.read_parquet(view_dir / "embeddings.parquet")
        result = fit_gmm_on_embeddings(
            embeddings,
            candidate_components=candidates,
            random_state=int(config["random_seed"]),
        )
        result.loadings.to_parquet(view_dir / "loadings.parquet", index=False)
        write_json(gmm_meta(result), view_dir / "gmm_meta.json")
        log(f"Fit {view_name} GMM with {result.n_components} components.", tag="gmm")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.pipeline.stage_04_model.fit_gmms "
            "<config_name> <experiment_dir>"
        )
    main(sys.argv[1], sys.argv[2])
