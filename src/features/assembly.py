"""Model-ready dataset assembly from feature-group parquet files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.dates import as_of_merge, select_month_end_rows


def build_observation_panel(db, config: dict) -> pd.DataFrame:
    """Build the monthly ticker-date panel used as the assembly backbone."""
    prices = db.load_prices(
        date_from=config["data"]["start_date"],
        date_to=config["data"]["end_date"],
    )
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "date"])

    monthly = select_month_end_rows(prices.loc[:, ["ticker", "date"]])
    min_history_days = config["features"]["price"]["min_history_days"]
    observation_counts = (
        prices.sort_values(["ticker", "date"])
        .groupby("ticker")
        .cumcount()
        .add(1)
    )
    eligible = prices.loc[:, ["ticker", "date"]].copy()
    eligible["observation_count"] = observation_counts
    eligible = select_month_end_rows(eligible)
    eligible = eligible[eligible["observation_count"] >= min_history_days]
    monthly = monthly.merge(eligible.loc[:, ["ticker", "date"]], on=["ticker", "date"], how="inner")
    return monthly.sort_values(["ticker", "date"]).reset_index(drop=True)


def assemble_dataset(db, config: dict, *, feature_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Merge monthly panel rows with all required feature groups and ticker metadata."""
    feature_root = Path(feature_dir)
    panel = build_observation_panel(db, config)
    if panel.empty:
        raise ValueError("Observation panel is empty. Check price group names and date filters.")

    dataset = panel.copy()
    for feature_name in config["assembly"]["required_feature_groups"]:
        feature_path = feature_root / f"{feature_name}.parquet"
        if not feature_path.exists():
            raise FileNotFoundError(f"Required feature parquet missing: {feature_path}")
        feature_frame = pd.read_parquet(feature_path)
        feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
        if feature_frame.duplicated(["ticker", "date"]).any():
            raise ValueError(f"Feature group {feature_name!r} contains duplicate ticker-date rows.")
        before = len(dataset)
        dataset = as_of_merge(dataset, feature_frame, by=["ticker"])
        if len(dataset) != before:
            raise ValueError(f"Row count changed while merging feature group {feature_name!r}.")

    tickers = db.load_tickers()
    metadata_columns = ["ticker", *config["assembly"]["metadata_columns"]]
    existing_metadata_columns = [column for column in metadata_columns if column in tickers.columns]
    if existing_metadata_columns:
        dataset = dataset.merge(
            tickers.loc[:, existing_metadata_columns].drop_duplicates(subset=["ticker"]),
            on="ticker",
            how="left",
            validate="many_to_one",
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    return dataset
