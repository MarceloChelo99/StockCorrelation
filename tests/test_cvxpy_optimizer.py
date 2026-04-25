from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.multiview_covariance import cvxpy_minimum_variance_weights


class CvxpyOptimizerTests(unittest.TestCase):
    def test_cvxpy_minimum_variance_weights_are_long_only(self) -> None:
        covariance = np.array(
            [
                [0.10, 0.02, 0.01],
                [0.02, 0.20, 0.03],
                [0.01, 0.03, 0.15],
            ]
        )

        weights = cvxpy_minimum_variance_weights(covariance)

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(np.all(weights >= -1e-9))


if __name__ == "__main__":
    unittest.main()
