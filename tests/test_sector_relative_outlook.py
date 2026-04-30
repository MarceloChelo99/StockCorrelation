from __future__ import annotations

import unittest

import pandas as pd

from src.applications.sector_relative_outlook import (
    apply_theme_assignment_strategy,
    backtest_validity_label,
    completed_sector_predictions,
    default_theme_assignment_strategy,
    rotation_simulation_metrics,
    sector_and_market_returns,
    sector_backtest_by_date,
    sector_outlook_backtest,
    sector_prediction_audit,
    simulate_group_rotation,
    theme_and_market_returns,
)


class SectorRelativeOutlookTests(unittest.TestCase):
    def test_backtest_validity_labels_distinguish_survivorship_modes(self) -> None:
        self.assertEqual(
            backtest_validity_label("current", "gics"),
            "diagnostic_current_roster_survivorship_biased",
        )
        self.assertEqual(
            backtest_validity_label("date_added", "gics"),
            "date_added_current_roster_survivorship_limited",
        )
        self.assertEqual(
            backtest_validity_label("historical", "gics"),
            "historical_constituent_backtest",
        )
        self.assertEqual(
            backtest_validity_label("historical", "theme"),
            "historical_covered_theme_universe_backtest",
        )

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
        self.assertIn("future_group_return", result.predictions.columns)
        self.assertIn("future_market_return", result.predictions.columns)

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

    def test_sector_outlook_supports_alternative_predictor(self) -> None:
        prices = synthetic_prices()
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
            model_type="elastic_net",
        )

        self.assertFalse(result.predictions.empty)
        self.assertEqual(result.metrics["model_type"], "elastic_net")
        self.assertTrue(result.predictions["model_type"].eq("elastic_net").all())
        self.assertIn("weight_type", result.coefficients.columns)

    def test_sector_outlook_can_use_learned_theme_groups(self) -> None:
        prices = synthetic_prices()
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "AAB", "BBB", "BBC", "CCC", "CCD"],
                "gics_sector": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            }
        )
        loadings = synthetic_theme_loadings(prices)

        result = sector_outlook_backtest(
            prices,
            metadata,
            valuation=synthetic_monthly_features(prices, metadata),
            growth=None,
            group_loadings=loadings,
            group_mode="theme",
            group_view="synthetic",
            horizon_days=21,
            min_train_months=3,
            ridge_alpha=1.0,
        )

        self.assertFalse(result.panel.empty)
        self.assertFalse(result.predictions.empty)
        self.assertEqual(result.metrics["group_mode"], "theme")
        self.assertEqual(set(result.panel["gics_sector"].unique()), {"theme_0", "theme_1", "theme_2"})
        self.assertIn("valuation_sales_yield", result.panel.columns)
        self.assertEqual(result.metrics["theme_assignment"], "soft")

    def test_theme_assignment_auto_uses_hard_behavioral_and_soft_growth(self) -> None:
        self.assertEqual(default_theme_assignment_strategy("behavioral"), "hard_top1")
        self.assertEqual(default_theme_assignment_strategy("growth"), "soft")

        prices = synthetic_prices()
        loadings = synthetic_theme_loadings(prices)
        hard = apply_theme_assignment_strategy(loadings, "hard_top1")
        theme_columns = ["theme_0", "theme_1", "theme_2"]

        self.assertTrue(hard[theme_columns].isin([0.0, 1.0]).all().all())
        self.assertTrue((hard[theme_columns].sum(axis=1) == 1.0).all())
        self.assertEqual(hard.loc[hard["ticker"].eq("AAA"), "theme_0"].iloc[0], 1.0)

    def test_theme_outlook_can_use_direct_embedding_features(self) -> None:
        prices = synthetic_prices()
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "AAB", "BBB", "BBC", "CCC", "CCD"],
                "gics_sector": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            }
        )
        loadings = synthetic_theme_loadings(prices)
        embeddings = synthetic_view_embeddings(loadings)

        result = sector_outlook_backtest(
            prices,
            metadata,
            valuation=synthetic_monthly_features(prices, metadata),
            growth=None,
            group_loadings=loadings,
            group_embeddings=embeddings,
            group_mode="theme",
            group_view="financial",
            horizon_days=21,
            min_train_months=3,
            ridge_alpha=1.0,
            include_embedding_features=True,
        )

        self.assertTrue(result.metrics["include_embedding_features"])
        self.assertIn("group_embedding_0", result.panel.columns)
        self.assertIn("group_embedding_0", result.coefficients["feature"].unique())

    def test_learned_theme_returns_mask_one_stock_themes(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame(
            {
                "ticker": [*["AAA"] * 4, *["BBB"] * 4, *["CCC"] * 4],
                "date": [*dates, *dates, *dates],
                "adj_close": [100.0, 200.0, 400.0, 800.0, 100.0, 101.0, 102.0, 103.0, 100.0, 101.0, 102.0, 103.0],
            }
        )
        metadata = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "gics_sector": ["A", "B", "C"]})
        loadings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "date": [dates[0], dates[0], dates[0]],
                "theme_0": [1.0, 0.0, 0.0],
                "theme_1": [0.0, 1.0, 1.0],
            }
        )

        theme_returns, market_returns = theme_and_market_returns(
            prices,
            loadings,
            metadata=metadata,
            min_effective_members=2.0,
        )

        self.assertTrue(theme_returns["theme_0"].isna().all())
        self.assertTrue(theme_returns["theme_1"].notna().any())
        self.assertAlmostEqual(float(market_returns.iloc[0]), (1.0 + 0.01 + 0.01) / 3.0)

    def test_learned_theme_market_uses_only_tickers_with_loadings(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        prices = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "BBB", "NOLOAD", "NOLOAD"],
                "date": [dates[0], dates[1], dates[0], dates[1], dates[0], dates[1]],
                "adj_close": [100.0, 110.0, 100.0, 120.0, 100.0, 1000.0],
            }
        )
        metadata = pd.DataFrame(
            {"ticker": ["AAA", "BBB", "NOLOAD"], "gics_sector": ["A", "B", "C"]}
        )
        loadings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "date": [dates[0], dates[0]],
                "theme_0": [1.0, 1.0],
            }
        )

        _, market_returns = theme_and_market_returns(prices, loadings, metadata=metadata)

        self.assertAlmostEqual(float(market_returns.loc[dates[1]]), 0.15)

    def test_date_added_membership_filter_excludes_pre_index_returns(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {
                "ticker": ["AAA", *["AAB"] * 5, *["AAA"] * 4],
                "date": [dates[0], *dates, *dates[1:]],
                "adj_close": [100.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "AAB"],
                "gics_sector": ["Alpha", "Alpha"],
                "date_added": ["2024-01-01", "2024-01-04"],
            }
        )

        current_sector, _ = sector_and_market_returns(prices, metadata, membership_mode="current")
        filtered_sector, _ = sector_and_market_returns(prices, metadata, membership_mode="date_added")

        early_date = pd.Timestamp("2024-01-02")
        self.assertGreater(current_sector.loc[early_date, "Alpha"], 0.50)
        self.assertAlmostEqual(filtered_sector.loc[early_date, "Alpha"], 0.01)

    def test_historical_membership_filter_handles_deleted_constituents(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.DataFrame(
            {
                "ticker": [*["OLD"] * 6, *["NEW"] * 6],
                "date": [*dates, *dates],
                "adj_close": [
                    100.0,
                    150.0,
                    225.0,
                    450.0,
                    900.0,
                    1800.0,
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                ],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["OLD", "NEW"],
                "gics_sector": ["Alpha", "Alpha"],
                "date_added": ["2020-01-01", "2024-01-04"],
            }
        )
        membership = pd.DataFrame(
            {
                "ticker": ["OLD", "NEW"],
                "start_date": ["2020-01-01", "2024-01-04"],
                "end_date": ["2024-01-04", pd.NaT],
                "gics_sector": ["Alpha", "Alpha"],
            }
        )

        current_sector, _ = sector_and_market_returns(prices, metadata, membership_mode="current")
        historical_sector, _ = sector_and_market_returns(
            prices,
            metadata,
            membership_mode="historical",
            membership=membership,
        )

        pre_removal_date = pd.Timestamp("2024-01-02")
        post_removal_date = pd.Timestamp("2024-01-05")
        self.assertGreater(current_sector.loc[pre_removal_date, "Alpha"], 0.25)
        self.assertAlmostEqual(historical_sector.loc[pre_removal_date, "Alpha"], 0.50)
        self.assertGreater(current_sector.loc[post_removal_date, "Alpha"], 0.25)
        self.assertAlmostEqual(historical_sector.loc[post_removal_date, "Alpha"], 1.0 / 103.0, places=6)

    def test_rotation_simulation_uses_non_overlapping_completed_predictions(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "target_end_date": "2024-03-01",
                    "gics_sector": "Alpha",
                    "group_label": "Alpha",
                    "prediction_rank": 1,
                    "predicted_excess_return": 0.03,
                    "future_group_return": 0.06,
                    "future_market_return": 0.02,
                    "future_excess_return": 0.04,
                    "is_horizon_complete": True,
                },
                {
                    "date": "2024-01-31",
                    "target_end_date": "2024-03-01",
                    "gics_sector": "Beta",
                    "group_label": "Beta",
                    "prediction_rank": 2,
                    "predicted_excess_return": 0.01,
                    "future_group_return": 0.01,
                    "future_market_return": 0.02,
                    "future_excess_return": -0.01,
                    "is_horizon_complete": True,
                },
                {
                    "date": "2024-02-29",
                    "target_end_date": "2024-03-29",
                    "gics_sector": "Alpha",
                    "group_label": "Alpha",
                    "prediction_rank": 1,
                    "predicted_excess_return": 0.02,
                    "future_group_return": 0.05,
                    "future_market_return": 0.01,
                    "future_excess_return": 0.04,
                    "is_horizon_complete": True,
                },
                {
                    "date": "2024-03-29",
                    "target_end_date": "2024-04-30",
                    "gics_sector": "Beta",
                    "group_label": "Beta",
                    "prediction_rank": 1,
                    "predicted_excess_return": 0.04,
                    "future_group_return": 0.03,
                    "future_market_return": 0.01,
                    "future_excess_return": 0.02,
                    "is_horizon_complete": True,
                },
            ]
        )

        benchmark_returns = pd.Series(
            [0.10, -0.02],
            index=pd.to_datetime(["2024-03-01", "2024-04-30"]),
            name="spy_return",
        )
        simulation = simulate_group_rotation(
            predictions,
            starting_capital=1_000.0,
            top_n=1,
            benchmark_returns=benchmark_returns,
            benchmark_label="Regular S&P 500 (SPY)",
        )
        metrics = rotation_simulation_metrics(simulation, 1_000.0)

        self.assertEqual(len(simulation), 2)
        self.assertEqual(simulation.iloc[0]["selected_group_label"], "Alpha")
        self.assertEqual(simulation.iloc[1]["selected_group_label"], "Beta")
        self.assertAlmostEqual(simulation.iloc[0]["start_capital"], 1_000.0)
        self.assertAlmostEqual(simulation.iloc[0]["capital_change"], 60.0)
        self.assertAlmostEqual(simulation.iloc[-1]["capital"], 1_000.0 * 1.06 * 1.03)
        self.assertAlmostEqual(simulation.iloc[0]["benchmark_capital"], 1_000.0 * 1.10)
        self.assertAlmostEqual(simulation.iloc[-1]["benchmark_capital"], 1_000.0 * 1.10 * 0.98)
        self.assertGreater(metrics["ending_capital"], metrics["market_ending_capital"])
        self.assertAlmostEqual(metrics["benchmark_ending_capital"], 1_000.0 * 1.10 * 0.98)


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


def synthetic_theme_loadings(prices: pd.DataFrame) -> pd.DataFrame:
    monthly = prices.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly = monthly.loc[monthly.groupby(["ticker", "month"])["date"].transform("max").eq(monthly["date"])]
    rows = []
    for _, row in monthly.iterrows():
        ticker = row["ticker"]
        if ticker.startswith("AA"):
            values = [0.85, 0.10, 0.05]
        elif ticker.startswith("BB"):
            values = [0.10, 0.80, 0.10]
        else:
            values = [0.05, 0.15, 0.80]
        rows.append(
            {
                "ticker": ticker,
                "date": row["date"],
                "theme_0": values[0],
                "theme_1": values[1],
                "theme_2": values[2],
            }
        )
    return pd.DataFrame(rows)


def synthetic_view_embeddings(loadings: pd.DataFrame) -> pd.DataFrame:
    frame = loadings.loc[:, ["ticker", "date"]].copy()
    ticker_codes = {ticker: index for index, ticker in enumerate(sorted(frame["ticker"].unique()))}
    frame["embedding_0"] = frame["ticker"].map(ticker_codes).astype(float)
    frame["embedding_1"] = pd.to_datetime(frame["date"]).dt.month.astype(float)
    return frame


if __name__ == "__main__":
    unittest.main()
