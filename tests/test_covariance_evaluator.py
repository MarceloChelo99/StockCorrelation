from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.applications.portfolio_optimization import minimum_variance_long_only
from src.evaluation.covariance import (
    CovarianceEvaluator,
    embedding_similarity_prior,
    ledoit_wolf_covariance,
    shrink_covariance_to_prior,
)


class _FakeDB:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_prices(self) -> pd.DataFrame:
        return self.prices


class CovarianceEvaluatorTests(unittest.TestCase):
    def test_minimum_variance_weights_are_long_only(self) -> None:
        covariance = np.array(
            [
                [0.10, 0.03, 0.02],
                [0.03, 0.20, 0.01],
                [0.02, 0.01, 0.15],
            ]
        )

        weights = minimum_variance_long_only(covariance)

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(weights >= 0.0))

    def test_embedding_similarity_prior_is_correlation_shaped(self) -> None:
        embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])

        prior = embedding_similarity_prior(embeddings)

        self.assertTrue(np.allclose(prior, prior.T))
        self.assertTrue(np.allclose(np.diag(prior), 1.0))
        self.assertGreater(float(prior[0, 1]), float(prior[0, 2]))

    def test_shrink_covariance_to_prior_preserves_sample_variances(self) -> None:
        sample = np.array([[0.04, 0.01], [0.01, 0.09]])
        prior_corr = np.array([[1.0, 0.5], [0.5, 1.0]])

        shrunk = shrink_covariance_to_prior(sample, prior_corr, alpha=0.5)

        self.assertTrue(np.allclose(np.diag(shrunk), np.diag(sample), atol=1e-8))

    def test_ledoit_wolf_covariance_tracks_population_covariance(self) -> None:
        rng = np.random.default_rng(37)
        population = np.array(
            [
                [0.040, 0.012, 0.004],
                [0.012, 0.090, 0.010],
                [0.004, 0.010, 0.025],
            ]
        )
        samples = rng.multivariate_normal(np.zeros(3), population, size=5000)

        estimate = ledoit_wolf_covariance(samples)

        self.assertTrue(np.allclose(estimate, population, rtol=0.12, atol=0.006))

    def test_evaluator_runs_on_tiny_panel(self) -> None:
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        prices = []
        for ticker, drift in [("AAA", 0.001), ("BBB", 0.0012), ("CCC", -0.0002), ("DDD", 0.0001)]:
            value = 100.0
            for index, date in enumerate(dates):
                value = value * (1.0 + drift + 0.0001 * ((index % 5) - 2))
                prices.append({"ticker": ticker, "date": date, "adj_close": value})
        prices_frame = pd.DataFrame(prices)
        embeddings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "date": [dates[45]] * 4,
                "embedding_0": [0.0, 0.1, 4.0, 4.1],
                "embedding_1": [0.0, 0.1, 4.0, 4.1],
            }
        )
        config = {
            "random_seed": 7,
            "evaluation": {
                "covariance": {
                    "lookback_days": 30,
                    "holding_days": 5,
                    "start_date": None,
                    "end_date": None,
                    "min_assets": 3,
                    "max_assets": None,
                    "embedding_alpha": 0.25,
                    "bootstrap_samples": 10,
                    "random_seed": 7,
                }
            },
        }

        with TemporaryDirectory() as tmp_dir:
            metrics = CovarianceEvaluator().run(
                embeddings,
                pd.DataFrame(),
                config,
                db=_FakeDB(prices_frame),
                output_dir=tmp_dir,
            )

        self.assertIn("methods", metrics)
        self.assertIn("ledoit_wolf", metrics["methods"])
        self.assertIn("legacy_constant_variance", metrics["methods"])
        self.assertIn("embedding_prior", metrics["methods"])
        self.assertGreater(metrics["n_rebalance_dates"], 0)


if __name__ == "__main__":
    unittest.main()
