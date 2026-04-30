from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from scripts_compat import load_script
from src.config import load_config
from src.db import FilingsDB
from src.evaluation.multiview_covariance import MultiviewCovarianceEvaluator
from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.metadata import load_ticker_metadata
from src.utils.seed import set_seed


def main(config_name: str, experiment_dir: str, output_path: str) -> None:
    """Run multi-view covariance evaluation inside sector and liquidity slices."""
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    root = Path(experiment_dir)
    db = FilingsDB.from_config(config)
    embeddings = pd.read_parquet(root / "embeddings.parquet")
    metadata = load_metadata(db, config)
    prices = db.load_prices()

    slice_helpers = load_script("09_covariance_slices.py", "covariance_slices")
    slice_specs = slice_helpers.sector_slices(metadata, min_tickers=35)
    slice_specs.extend(slice_helpers.liquidity_slices(prices, min_tickers=60))

    rows = []
    for spec in slice_specs:
        slice_name = str(spec["slice"])
        tickers = set(spec["tickers"])
        sliced_embeddings = embeddings[embeddings["ticker"].astype(str).str.upper().isin(tickers)]
        run_config = multiview_slice_config(config, slice_name)
        log(f"Running multi-view covariance slice {slice_name} with {len(tickers)} tickers.", tag="mv-cov-slice")
        try:
            metrics = MultiviewCovarianceEvaluator().run(
                sliced_embeddings,
                metadata,
                run_config,
                db=db,
                output_dir=root,
            )
        except ValueError as error:
            rows.append(
                {
                    "slice_type": spec["slice_type"],
                    "slice": slice_name,
                    "n_slice_tickers": len(tickers),
                    "status": "skipped",
                    "error": str(error),
                }
            )
            continue

        methods = metrics["methods"]
        comparison = metrics["comparisons"].get("multiview_minus_ledoit_wolf", {})
        rows.append(
            {
                "slice_type": spec["slice_type"],
                "slice": slice_name,
                "n_slice_tickers": len(tickers),
                "status": "ok",
                "n_rebalance_dates": metrics["n_rebalance_dates"],
                "multiview_annual_variance": methods["multiview"]["realized_annual_variance"],
                "ledoit_wolf_annual_variance": methods["ledoit_wolf"]["realized_annual_variance"],
                "sample_annual_variance": methods["sample"]["realized_annual_variance"],
                "embedding_prior_annual_variance": methods["embedding_prior"]["realized_annual_variance"],
                "multiview_minus_lw_daily_var": comparison.get("mean_daily_variance_diff"),
                "multiview_minus_lw_ci_low": comparison.get("ci_low"),
                "multiview_minus_lw_ci_high": comparison.get("ci_high"),
                "multiview_sharpe": methods["multiview"]["realized_annual_sharpe"],
                "ledoit_wolf_sharpe": methods["ledoit_wolf"]["realized_annual_sharpe"],
                "mean_assets": methods["multiview"]["mean_assets"],
                "best_method": best_variance_method(methods),
            }
        )

    output = Path(output_path)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    log(f"Wrote multi-view covariance slice results to {output}.", tag="mv-cov-slice")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load DB metadata and optional external sector metadata."""
    return load_ticker_metadata(db, config)


def multiview_slice_config(config: dict, slice_name: str) -> dict:
    """Clone config with looser asset limits and slice-specific artifact names."""
    cloned = dict(config)
    cloned["evaluation"] = dict(config["evaluation"])
    cloned["evaluation"]["multiview_covariance"] = dict(config["evaluation"].get("multiview_covariance", {}))
    cloned["evaluation"]["multiview_covariance"]["min_assets"] = 25
    cloned["evaluation"]["multiview_covariance"]["max_assets"] = 80
    cloned["evaluation"]["multiview_covariance"]["bootstrap_samples"] = 200
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in slice_name).strip("_")
    cloned["evaluation"]["multiview_covariance"]["artifact_prefix"] = f"slice_{safe_name}"
    return cloned


def best_variance_method(methods: dict[str, dict[str, float]]) -> str:
    """Return the method with the lowest realized annual variance."""
    return min(methods, key=lambda name: methods[name]["realized_annual_variance"])


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.decomposed.multiview_covariance_slices "
            "<config_name> <experiment_dir> <output_path>"
        )
    main(sys.argv[1], sys.argv[2], sys.argv[3])
