from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.db import FilingsDB
from src.evaluation.covariance import CovarianceEvaluator
from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.seed import set_seed


def main(config_name: str, embeddings_path: str, output_path: str) -> None:
    """Run covariance evaluation inside sector and liquidity slices."""
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))
    db = FilingsDB.from_config(config)
    embeddings = pd.read_parquet(embeddings_path)
    metadata = load_metadata(db, config)
    prices = db.load_prices()

    slice_specs = sector_slices(metadata, min_tickers=35)
    slice_specs.extend(liquidity_slices(prices, min_tickers=60))

    rows = []
    for spec in slice_specs:
        slice_name = str(spec["slice"])
        tickers = set(spec["tickers"])
        sliced_embeddings = embeddings[embeddings["ticker"].astype(str).str.upper().isin(tickers)]
        run_config = covariance_slice_config(config)
        log(f"Running covariance slice {slice_name} with {len(tickers)} tickers.", tag="cov-slice")
        try:
            metrics = CovarianceEvaluator().run(
                sliced_embeddings,
                metadata,
                run_config,
                db=db,
                output_dir=None,
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
        comparison = metrics["comparisons"].get("embedding_prior_minus_ledoit_wolf", {})
        rows.append(
            {
                "slice_type": spec["slice_type"],
                "slice": slice_name,
                "n_slice_tickers": len(tickers),
                "status": "ok",
                "n_rebalance_dates": metrics["n_rebalance_dates"],
                "embedding_annual_variance": methods["embedding_prior"]["realized_annual_variance"],
                "ledoit_wolf_annual_variance": methods["ledoit_wolf"]["realized_annual_variance"],
                "sample_annual_variance": methods["sample"]["realized_annual_variance"],
                "embedding_minus_lw_daily_var": comparison.get("mean_daily_variance_diff"),
                "embedding_minus_lw_ci_low": comparison.get("ci_low"),
                "embedding_minus_lw_ci_high": comparison.get("ci_high"),
                "embedding_sharpe": methods["embedding_prior"]["realized_annual_sharpe"],
                "ledoit_wolf_sharpe": methods["ledoit_wolf"]["realized_annual_sharpe"],
                "mean_assets": methods["embedding_prior"]["mean_assets"],
            }
        )

    output = Path(output_path)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    log(f"Wrote covariance slice results to {output}.", tag="cov-slice")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load DB metadata and optional external sector metadata."""
    metadata = db.load_tickers()
    metadata_path = Path(config["paths"].get("metadata_path", ""))
    if metadata_path.exists():
        external = pd.read_parquet(metadata_path)
        external["ticker"] = external["ticker"].astype(str).str.upper()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        duplicate_columns = [column for column in external.columns if column in metadata.columns and column != "ticker"]
        metadata = metadata.merge(external.drop(columns=duplicate_columns), on="ticker", how="left")
    return metadata


def sector_slices(metadata: pd.DataFrame, *, min_tickers: int) -> list[dict[str, object]]:
    """Return sector slice ticker sets with enough constituents."""
    frame = metadata.loc[:, ["ticker", "gics_sector"]].dropna().copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    slices = []
    for sector, group in frame.groupby("gics_sector", sort=True):
        tickers = sorted(group["ticker"].unique().tolist())
        if len(tickers) >= min_tickers:
            slices.append({"slice_type": "sector", "slice": sector, "tickers": tickers})
    return slices


def liquidity_slices(prices: pd.DataFrame, *, min_tickers: int) -> list[dict[str, object]]:
    """Return low/mid/high liquidity ticker buckets from recent dollar volume."""
    frame = prices.loc[:, ["ticker", "date", "adj_close", "volume"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    max_date = frame["date"].max()
    recent = frame[frame["date"] >= max_date - pd.Timedelta(days=365)].copy()
    recent["dollar_volume"] = recent["adj_close"].astype(float) * recent["volume"].astype(float)
    liquidity = recent.groupby("ticker")["dollar_volume"].median().dropna().sort_values()
    labels = pd.qcut(liquidity.rank(method="first"), q=3, labels=["low_liquidity", "mid_liquidity", "high_liquidity"])

    slices = []
    for label in ["low_liquidity", "mid_liquidity", "high_liquidity"]:
        tickers = sorted(liquidity[labels == label].index.astype(str).tolist())
        if len(tickers) >= min_tickers:
            slices.append({"slice_type": "liquidity", "slice": label, "tickers": tickers})
    return slices


def covariance_slice_config(config: dict) -> dict:
    """Clone covariance config with looser asset limits for subgroup runs."""
    cloned = dict(config)
    cloned["evaluation"] = dict(config["evaluation"])
    cloned["evaluation"]["covariance"] = dict(config["evaluation"].get("covariance", {}))
    cloned["evaluation"]["covariance"]["min_assets"] = 25
    cloned["evaluation"]["covariance"]["max_assets"] = 80
    cloned["evaluation"]["covariance"]["bootstrap_samples"] = 200
    return cloned


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: .venv/bin/python scripts/09_covariance_slices.py "
            "<config_name> <embeddings_path> <output_path>"
        )
    main(sys.argv[1], sys.argv[2], sys.argv[3])
