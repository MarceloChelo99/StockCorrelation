"""Trailing volatility features sampled at month-end dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.utils.dates import select_month_end_rows


class PriceVolatilityProducer(FeatureProducer):
    """Compute trailing realized volatility over daily adjusted returns."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="price_volatility",
            key_columns=["ticker", "date"],
            columns=["price_vol_21d", "price_vol_63d", "price_vol_252d"],
            source="prices",
            description="Trailing annualized volatility from adjusted close returns.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        prices = db.load_prices(
            date_from=config["data"]["start_date"],
            date_to=config["data"]["end_date"],
        )
        if prices.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        windows = config["features"]["price"]["volatility_windows"]
        working = prices.loc[:, ["ticker", "date", "adj_close"]].copy()
        working = working.sort_values(["ticker", "date"]).reset_index(drop=True)
        working["return_1d"] = working.groupby("ticker")["adj_close"].pct_change()
        annualizer = np.sqrt(252.0)
        for window in windows:
            column = f"price_vol_{window}d"
            working[column] = (
                working.groupby("ticker")["return_1d"]
                .transform(lambda values: values.rolling(window).std())
                * annualizer
            )

        monthly = select_month_end_rows(working.loc[:, ["ticker", "date", *self.spec.columns]])
        return monthly.reset_index(drop=True)
