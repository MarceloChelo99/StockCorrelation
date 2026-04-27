from __future__ import annotations

import unittest

import numpy as np

from src.applications._legacy_analytical_optimizer import minimum_variance_weights
from src.applications.portfolio_optimization import minimum_variance_long_only


class CvxpyOptimizerTests(unittest.TestCase):
    def test_cvxpy_minimum_variance_weights_are_long_only(self) -> None:
        covariance = np.array(
            [
                [0.10, 0.02, 0.01],
                [0.02, 0.20, 0.03],
                [0.01, 0.03, 0.15],
            ]
        )

        weights = minimum_variance_long_only(covariance)

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(np.all(weights >= -1e-9))

    def test_cvxpy_matches_legacy_analytical_solver_on_basic_case(self) -> None:
        covariance = np.array(
            [
                [0.08, 0.01, 0.02, 0.00],
                [0.01, 0.11, 0.01, 0.02],
                [0.02, 0.01, 0.09, 0.01],
                [0.00, 0.02, 0.01, 0.13],
            ]
        )

        cvxpy_weights = minimum_variance_long_only(covariance)
        legacy_weights = minimum_variance_weights(covariance)

        self.assertTrue(np.allclose(cvxpy_weights, legacy_weights, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
