from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.evaluation.peers import (
    PeersEvaluator,
    daily_returns_matrix,
    paired_metrics,
    peer_group_correlation,
)


class _FakeDB:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_prices(self) -> pd.DataFrame:
        return self.prices


class PeersEvaluatorTests(unittest.TestCase):
    def test_daily_returns_matrix_pivots_returns(self) -> None:
        prices = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "BBB"],
                "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
                "adj_close": [100.0, 110.0, 50.0, 55.0],
            }
        )
        returns = daily_returns_matrix(prices)

        self.assertAlmostEqual(float(returns.loc[pd.Timestamp("2025-01-02"), "AAA"]), 0.1)
        self.assertAlmostEqual(float(returns.loc[pd.Timestamp("2025-01-02"), "BBB"]), 0.1)

    def test_peer_group_correlation_uses_average_peer_return(self) -> None:
        return_window = pd.DataFrame(
            {
                "AAA": [0.01, 0.02, 0.03, 0.04],
                "BBB": [0.02, 0.04, 0.06, 0.08],
                "CCC": [0.01, 0.02, 0.03, 0.04],
            }
        )
        corr = peer_group_correlation(return_window["AAA"].to_numpy(), return_window, ["BBB", "CCC"])

        self.assertAlmostEqual(corr, 1.0)

    def test_paired_metrics_detects_positive_difference(self) -> None:
        metrics = paired_metrics(
            np.array([0.5, 0.6, 0.7]),
            np.array([0.1, 0.2, 0.3]),
        )

        self.assertGreater(metrics["mean_corr_diff"], 0)
        self.assertGreater(metrics["t_stat"], 0)

    def test_evaluator_runs_on_tiny_panel(self) -> None:
        dates = pd.date_range("2025-01-01", periods=35, freq="B")
        prices = []
        for ticker, offset in [("AAA", 1.0), ("BBB", 1.1), ("CCC", 2.0), ("DDD", 2.1)]:
            for index, date in enumerate(dates):
                prices.append({"ticker": ticker, "date": date, "adj_close": offset + index * 0.01})
        prices_frame = pd.DataFrame(prices)
        embeddings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "date": [dates[5]] * 4,
                "embedding_0": [0.0, 0.1, 10.0, 10.1],
                "embedding_1": [0.0, 0.1, 10.0, 10.1],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "gics_sub_industry": ["One", "One", "One", "One"],
            }
        )
        config = {
            "random_seed": 7,
            "evaluation": {
                "peers": {
                    "k": 1,
                    "forward_days": 5,
                    "max_observations": 4,
                    "benchmark_column": "gics_sub_industry",
                    "min_benchmark_pool": 1,
                    "random_seed": 7,
                }
            },
        }

        with TemporaryDirectory() as tmp_dir:
            metrics = PeersEvaluator().run(
                embeddings,
                metadata,
                config,
                db=_FakeDB(prices_frame),
                output_dir=tmp_dir,
            )

        self.assertEqual(metrics["n_observations"], 4)
        self.assertIn("mean_corr_diff", metrics)


if __name__ == "__main__":
    unittest.main()
