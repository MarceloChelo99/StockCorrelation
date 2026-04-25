"""Compare peer-correlation performance across embedding views and horizons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.base import Evaluator
from src.evaluation.peers import PeersEvaluator
from src.utils.io import ensure_dir


class MultiViewPeerEvaluator(Evaluator):
    """Run the existing peer evaluator for each view across configured horizons."""

    name = "multiview_peers"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        if db is None:
            raise ValueError("Multi-view peer evaluation requires a FilingsDB instance.")
        if output_dir is None:
            raise ValueError("Multi-view peer evaluation requires experiment output_dir with view embeddings.")

        settings = config["evaluation"].get("multiview_peers", {})
        horizons = list(settings.get("horizons", [21, 63, 126, 252, 504]))
        root = Path(output_dir)
        rows = []
        for view_name in config.get("views_enabled", list(config["views"].keys())):
            embeddings_path = root / "views" / view_name / "embeddings.parquet"
            if not embeddings_path.exists():
                continue
            view_embeddings = pd.read_parquet(embeddings_path)
            for horizon in horizons:
                run_config = clone_peer_config(config, int(horizon))
                try:
                    metrics = PeersEvaluator().run(view_embeddings, metadata, run_config, db=db, output_dir=None)
                except ValueError as error:
                    rows.append(
                        {
                            "view": view_name,
                            "forward_days": int(horizon),
                            "status": "no_valid_observations",
                            "error": str(error),
                            "mean_corr_diff": np.nan,
                            "n_observations": 0,
                        }
                    )
                    continue
                metrics["status"] = "ok"
                metrics["error"] = ""
                rows.append({"view": view_name, "forward_days": int(horizon), **metrics})

        result = pd.DataFrame(rows)
        if result.empty:
            raise ValueError("No multi-view peer rows were produced.")
        if output_dir is not None:
            artifacts_dir = ensure_dir(root / "evaluation")
            result.to_csv(artifacts_dir / "multiview_peer_horizon_matrix.csv", index=False)
        valid = result.dropna(subset=["mean_corr_diff"]).copy()
        if valid.empty:
            return {
                "rows": int(len(result)),
                "valid_rows": 0,
                "best_view": "",
                "best_forward_days": 0,
                "best_mean_corr_diff": float("nan"),
            }
        best = valid.sort_values("mean_corr_diff", ascending=False).iloc[0]
        return {
            "rows": int(len(result)),
            "valid_rows": int(len(valid)),
            "best_view": str(best["view"]),
            "best_forward_days": int(best["forward_days"]),
            "best_mean_corr_diff": float(valid["mean_corr_diff"].max()),
        }


def clone_peer_config(config: dict, forward_days: int) -> dict:
    """Clone config with one peer horizon changed."""
    cloned = dict(config)
    cloned["evaluation"] = dict(config["evaluation"])
    cloned["evaluation"]["peers"] = dict(config["evaluation"].get("peers", {}))
    cloned["evaluation"]["peers"]["forward_days"] = int(forward_days)
    return cloned
