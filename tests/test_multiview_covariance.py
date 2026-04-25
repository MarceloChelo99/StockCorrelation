from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.applications.multiview_covariance import build_theme_returns, estimate_multiview_covariance


class MultiViewCovarianceTests(unittest.TestCase):
    def test_build_theme_returns_uses_loading_weighted_average(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3)
        returns = pd.DataFrame({"AAA": [0.01, 0.02, 0.03], "BBB": [0.03, 0.02, 0.01]}, index=dates)
        loadings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "date": [dates[0], dates[0]],
                "theme_0": [1.0, 0.0],
                "theme_1": [0.0, 1.0],
            }
        )

        theme_returns = build_theme_returns(loadings, returns)

        self.assertAlmostEqual(float(theme_returns.loc[dates[0], "theme_0"]), 0.01)
        self.assertAlmostEqual(float(theme_returns.loc[dates[0], "theme_1"]), 0.03)

    def test_estimate_multiview_covariance_returns_psd_square_matrix(self) -> None:
        dates = pd.date_range("2025-01-01", periods=8)
        returns = pd.DataFrame(
            {
                "AAA": np.linspace(0.001, 0.008, 8),
                "BBB": np.linspace(0.002, 0.009, 8),
                "CCC": np.linspace(-0.001, 0.002, 8),
            },
            index=dates,
        )
        loadings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "date": [dates[0]] * 3,
                "theme_0": [0.9, 0.8, 0.1],
                "theme_1": [0.1, 0.2, 0.9],
            }
        )

        covariance = estimate_multiview_covariance({"business": loadings}, returns, str(dates[-1]), window_days=6)

        self.assertEqual(covariance.shape, (3, 3))
        self.assertTrue(np.linalg.eigvalsh(covariance.to_numpy()).min() >= -1e-10)


if __name__ == "__main__":
    unittest.main()
