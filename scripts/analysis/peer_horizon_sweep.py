from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.config import load_config
from src.db import FilingsDB
from src.evaluation.peers import PeersEvaluator
from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.metadata import load_ticker_metadata
from src.utils.seed import set_seed


DEFAULT_HORIZONS = [21, 63, 126, 252, 504]


def main(
    config_name: str,
    embeddings_path: str,
    output_path: str,
    horizons_csv: str = ",".join(str(value) for value in DEFAULT_HORIZONS),
) -> None:
    """Run peer-identification evaluation across multiple forward horizons."""
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    db = FilingsDB.from_config(config)
    metadata = load_metadata(db, config)
    embeddings = pd.read_parquet(embeddings_path)
    horizons = [int(value.strip()) for value in horizons_csv.split(",") if value.strip()]

    rows = []
    for forward_days in horizons:
        run_config = clone_config_with_horizon(config, forward_days)
        log(f"Running peer horizon {forward_days} trading days.", tag="peer-sweep")
        metrics = PeersEvaluator().run(
            embeddings,
            metadata,
            run_config,
            db=db,
            output_dir=None,
        )
        rows.append({"forward_days": forward_days, **metrics})

    output = Path(output_path)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    log(f"Wrote peer horizon sweep to {output}.", tag="peer-sweep")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load DB metadata and optional external GICS metadata."""
    return load_ticker_metadata(db, config)


def clone_config_with_horizon(config: dict, forward_days: int) -> dict:
    """Return a shallow-enough config clone with one peer horizon changed."""
    cloned = dict(config)
    cloned["evaluation"] = dict(config["evaluation"])
    cloned["evaluation"]["peers"] = dict(config["evaluation"].get("peers", {}))
    cloned["evaluation"]["peers"]["forward_days"] = int(forward_days)
    return cloned


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.analysis.peer_horizon_sweep "
            "<config_name> <embeddings_path> <output_path> [horizons_csv]"
        )
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4] if len(sys.argv) > 4 else ",".join(str(value) for value in DEFAULT_HORIZONS),
    )
