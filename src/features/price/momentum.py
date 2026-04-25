"""Trailing price momentum features sampled at month-end dates."""
from __future__ import annotations

import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.utils.dates import select_month_end_rows


class PriceMomentumProducer(FeatureProducer):
    """Compute trailing adjusted-close returns over several lookback windows."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="price_momentum",
            key_columns=["ticker", "date"],
            columns=["price_mom_21d", "price_mom_63d", "price_mom_252d"],
            source="prices",
            description="Trailing adjusted-close momentum over fixed windows.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        prices = db.load_prices(
            date_from=config["data"]["start_date"],
            date_to=config["data"]["end_date"],
        )
        if prices.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        windows = config["features"]["price"]["momentum_windows"]
        working = prices.loc[:, ["ticker", "date", "adj_close"]].copy()
        working = working.sort_values(["ticker", "date"]).reset_index(drop=True)
        for window in windows:
            column = f"price_mom_{window}d"
            working[column] = working.groupby("ticker")["adj_close"].pct_change(window)

        monthly = select_month_end_rows(working.loc[:, ["ticker", "date", *self.spec.columns]])
        return monthly.reset_index(drop=True)
