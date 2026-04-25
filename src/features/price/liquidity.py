"""Rolling liquidity features sampled at month-end dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.utils.dates import select_month_end_rows


class PriceLiquidityProducer(FeatureProducer):
    """Compute coarse liquidity features from volume and daily returns."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="price_liquidity",
            key_columns=["ticker", "date"],
            columns=[
                "price_dollar_volume_21d",
                "price_dollar_volume_63d",
                "price_amihud_21d",
                "price_amihud_63d",
            ],
            source="prices",
            description="Rolling average dollar volume and Amihud illiquidity.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        prices = db.load_prices(
            date_from=config["data"]["start_date"],
            date_to=config["data"]["end_date"],
        )
        if prices.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        windows = config["features"]["price"]["liquidity_windows"]
        working = prices.loc[:, ["ticker", "date", "adj_close", "volume"]].copy()
        working = working.sort_values(["ticker", "date"]).reset_index(drop=True)
        working["return_1d"] = working.groupby("ticker")["adj_close"].pct_change()
        working["dollar_volume"] = working["adj_close"] * working["volume"]
        working["amihud_daily"] = working["return_1d"].abs() / working["dollar_volume"].replace(0, np.nan)

        for window in windows:
            working[f"price_dollar_volume_{window}d"] = (
                working.groupby("ticker")["dollar_volume"]
                .transform(lambda values: values.rolling(window).mean())
            )
            working[f"price_amihud_{window}d"] = (
                working.groupby("ticker")["amihud_daily"]
                .transform(lambda values: values.rolling(window).mean())
            )

        monthly = select_month_end_rows(working.loc[:, ["ticker", "date", *self.spec.columns]])
        return monthly.reset_index(drop=True)
