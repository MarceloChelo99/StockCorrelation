from __future__ import annotations

import unittest

import pandas as pd

from src.applications.sector_relative_outlook import (
    completed_sector_predictions,
    sector_backtest_by_date,
    sector_outlook_backtest,
    sector_prediction_audit,
)


class SectorRelativeOutlookTests(unittest.TestCase):
    def test_sector_outlook_backtest_produces_walk_forward_predictions(self) -> None:
        prices = synthetic_prices()
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "AAB", "BBB", "BBC", "CCC", "CCD"],
                "gics_sector": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            }
        )
        valuation = synthetic_monthly_features(prices, metadata)

        result = sector_outlook_backtest(
            prices,
            metadata,
            valuation=valuation,
            growth=None,
            horizon_days=21,
            min_train_months=3,
            ridge_alpha=1.0,
        )

        self.assertFalse(result.panel.empty)
        self.assertFalse(result.predictions.empty)
        self.assertFalse(result.latest.empty)
        self.assertIn("mean_rank_ic", result.metrics)
        self.assertIn("valuation_sales_yield", result.panel.columns)
        self.assertEqual(set(result.latest["gics_sector"]), {"Alpha", "Beta", "Gamma"})
        self.assertIn("is_horizon_complete", result.predictions.columns)
        self.assertIn("training_latest_target_end_date", result.predictions.columns)

        training_end = pd.to_datetime(result.predictions["training_latest_target_end_date"])
        prediction_date = pd.to_datetime(result.predictions["date"])
        self.assertTrue((training_end < prediction_date).all())

    def test_sector_backtest_excludes_incomplete_forward_horizons(self) -> None:
        prices = synthetic_prices().iloc[:-10].copy()
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "AAB", "BBB", "BBC", "CCC", "CCD"],
                "gics_sector": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            }
        )

        result = sector_outlook_backtest(
            prices,
            metadata,
            valuation=synthetic_monthly_features(prices, metadata),
            growth=None,
            horizon_days=21,
            min_train_months=3,
            ridge_alpha=1.0,
        )

        completed = completed_sector_predictions(result.predictions)
        dated = sector_backtest_by_date(result.predictions)
        audit = sector_prediction_audit(result.predictions)

        self.assertFalse(completed.empty)
        self.assertTrue(completed["is_horizon_complete"].all())
        self.assertTrue((completed["future_days_available"] >= 21).all())
        self.assertLess(dated["date"].max(), result.predictions["date"].max())
        self.assertGreater(audit["n_unrealized_rows"], 0)
        self.assertEqual(audit["n_leakage_violations"], 0)


def synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=210)
    rows = []
    drifts = {
        "AAA": 0.0012,
        "AAB": 0.0010,
        "BBB": 0.0004,
        "BBC": 0.0003,
        "CCC": -0.0001,
        "CCD": -0.0002,
    }
    for ticker, drift in drifts.items():
        price = 100.0
        for index, date in enumerate(dates):
            price *= 1.0 + drift + 0.0005 * ((index % 5) - 2)
            rows.append({"ticker": ticker, "date": date, "adj_close": price})
    return pd.DataFrame(rows)


def synthetic_monthly_features(prices: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    monthly = prices.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly = monthly.loc[monthly.groupby(["ticker", "month"])["date"].transform("max").eq(monthly["date"])]
    monthly = monthly.merge(metadata, on="ticker", how="left")
    sector_sales_yield = {"Alpha": 0.3, "Beta": 0.8, "Gamma": 1.2}
    monthly["valuation_sales_yield"] = monthly["gics_sector"].map(sector_sales_yield)
    monthly["valuation_earnings_yield"] = monthly["valuation_sales_yield"] / 10.0
    monthly["valuation_book_to_market"] = monthly["valuation_sales_yield"] / 2.0
    return monthly.loc[:, ["ticker", "date", "valuation_sales_yield", "valuation_earnings_yield", "valuation_book_to_market"]]


if __name__ == "__main__":
    unittest.main()
