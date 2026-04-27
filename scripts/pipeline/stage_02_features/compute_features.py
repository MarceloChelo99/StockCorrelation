from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path



from src.config import load_config
from src.db import FilingsDB
from src.features.registry import build_active_producers
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log
from src.utils.seed import set_seed


def main(config_name: str = "baseline") -> None:
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))

    db = FilingsDB.from_config(config)
    output_dir = ensure_dir(config["paths"]["feature_dir"])
    producers = build_active_producers(config)
    manifest: dict[str, dict[str, object]] = {}

    for feature_name in config["features"]["enabled"]:
        producer = producers[feature_name]
        log(f"Computing feature group {feature_name}.", tag="features")
        frame = producer.run(db, config, output_dir)
        manifest[feature_name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "path": str(producer.output_path(output_dir)),
        }

    write_json(manifest, output_dir / "feature_manifest.json")
    log("Feature computation complete.", tag="features")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
