from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir
from src.utils.logging import log


def main(output_path: str = "report/experiment_comparison.csv") -> None:
    rows = []
    for metrics_path in sorted(Path("experiments").glob("*/metrics.json")):
        root = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training = read_json_if_exists(root / "training_history.json")
        model_meta = read_json_if_exists(root / "model" / "meta.json")
        config = read_json_if_exists(root / "config.yaml")

        clustering = metrics.get("clustering", {})
        peers = metrics.get("peers", {})
        covariance = metrics.get("covariance", {})
        covariance_methods = covariance.get("methods", {})
        covariance_comparisons = covariance.get("comparisons", {})
        embedding_prior = covariance_methods.get("embedding_prior", {})
        ledoit_wolf = covariance_methods.get("ledoit_wolf", {})
        sample = covariance_methods.get("sample", {})
        embedding_vs_lw = covariance_comparisons.get("embedding_prior_minus_ledoit_wolf", {})
        rows.append(
            {
                "run": root.name,
                "experiment_name": config.get("experiment_name", root.name),
                "model_name": model_meta.get("model_name"),
                "input_dim": model_meta.get("input_dim"),
                "embedding_dim": model_meta.get("embedding_dim"),
                "final_reconstruction_mse": training.get("final_reconstruction_mse"),
                "clustering_ari": clustering.get("ari"),
                "clustering_nmi": clustering.get("nmi"),
                "peer_embedding_corr": peers.get("mean_embedding_peer_corr"),
                "peer_benchmark_corr": peers.get("mean_benchmark_peer_corr"),
                "peer_corr_diff": peers.get("mean_corr_diff"),
                "peer_n": peers.get("n_observations"),
                "cov_embedding_annual_variance": embedding_prior.get("realized_annual_variance"),
                "cov_ledoit_wolf_annual_variance": ledoit_wolf.get("realized_annual_variance"),
                "cov_sample_annual_variance": sample.get("realized_annual_variance"),
                "cov_embedding_minus_lw_daily_var": embedding_vs_lw.get("mean_daily_variance_diff"),
                "cov_embedding_minus_lw_ci_low": embedding_vs_lw.get("ci_low"),
                "cov_embedding_minus_lw_ci_high": embedding_vs_lw.get("ci_high"),
                "path": str(root),
            }
        )

    if not rows:
        raise ValueError("No metrics.json files found under experiments/.")

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        ["peer_corr_diff", "clustering_nmi"],
        ascending=[False, False],
        na_position="last",
    )
    output = Path(output_path)
    ensure_dir(output.parent)
    comparison.to_csv(output, index=False)
    log(f"Wrote experiment comparison to {output}.", tag="compare")


def read_json_if_exists(path: Path) -> dict:
    """Read a JSON dict if present."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/experiment_comparison.csv")
