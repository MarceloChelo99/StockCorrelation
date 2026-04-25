from __future__ import annotations

import unittest

import pandas as pd

from src.clustering.gmm import fit_gmm_on_embeddings


class GMMClusteringTests(unittest.TestCase):
    def test_fit_gmm_returns_probabilities_that_sum_to_one(self) -> None:
        embeddings = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(12)],
                "date": [pd.Timestamp("2025-01-31")] * 12,
                "embedding_0": [0.0, 0.1, 0.2, 0.0, 0.2, 0.1, 5.0, 5.1, 5.2, 5.0, 5.2, 5.1],
                "embedding_1": [0.0, 0.2, 0.1, 0.1, 0.0, 0.2, 5.0, 5.2, 5.1, 5.1, 5.0, 5.2],
            }
        )

        result = fit_gmm_on_embeddings(embeddings, range(2, 4), random_state=7)
        theme_columns = [column for column in result.loadings.columns if column.startswith("theme_")]
        row_sums = result.loadings[theme_columns].sum(axis=1)

        self.assertTrue(((row_sums - 1.0).abs() < 1e-8).all())
        self.assertIn(result.n_components, {2, 3})


if __name__ == "__main__":
    unittest.main()
