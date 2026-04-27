from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path



from src.config import load_config
from src.db import FilingsDB
from src.features.assembly import assemble_dataset
from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.seed import set_seed


def main(config_name: str = "baseline") -> None:
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))

    db = FilingsDB.from_config(config)
    dataset_dir = ensure_dir(config["paths"]["dataset_dir"])
    dataset_path = dataset_dir / f"{config['assembly']['dataset_name']}.parquet"
    frame = assemble_dataset(
        db,
        config,
        feature_dir=config["paths"]["feature_dir"],
        output_path=dataset_path,
    )
    log(f"Wrote assembled dataset with {len(frame)} rows to {dataset_path}.", tag="assembly")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
