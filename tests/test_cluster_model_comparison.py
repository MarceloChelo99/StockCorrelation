from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.applications.cluster_model_comparison import (
    automatic_dbscan_eps,
    cluster_cross_section,
    embedding_columns,
)


class ClusterModelComparisonTests(unittest.TestCase):
    def test_cluster_cross_section_compares_three_models(self) -> None:
        embeddings = synthetic_embeddings()
        metadata = pd.DataFrame(
            {
                "ticker": embeddings["ticker"].unique(),
                "gics_sector": ["Alpha"] * 10 + ["Beta"] * 10 + ["Gamma"] * 10,
            }
        )

        assignments, metrics = cluster_cross_section(
            embeddings,
            metadata,
            as_of_date="2024-01-31",
            n_clusters=3,
            random_state=7,
        )

        self.assertEqual(set(metrics["model"]), {"GMM", "k-means", "DBSCAN"})
        self.assertEqual(set(assignments["model"]), {"GMM", "k-means", "DBSCAN"})
        self.assertEqual(assignments["ticker"].nunique(), 30)
        self.assertEqual(len(assignments), 90)
        self.assertTrue(metrics["clusters_found"].ge(1).all())
        self.assertTrue(metrics["nmi_vs_gics"].notna().all())

    def test_embedding_columns_returns_embedding_features_only(self) -> None:
        frame = pd.DataFrame({"ticker": ["AAA"], "embedding_0": [1.0], "other": [2.0]})
        self.assertEqual(embedding_columns(frame), ["embedding_0"])

    def test_automatic_dbscan_eps_is_positive(self) -> None:
        matrix = np.array([[0.0, 0.0], [0.1, 0.2], [3.0, 3.0], [3.1, 3.1], [6.0, 6.0]])
        self.assertGreater(automatic_dbscan_eps(matrix, min_samples=2), 0.0)


def synthetic_embeddings() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    centers = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [-4.0, 4.0, -4.0]])
    rows = []
    for cluster_index, center in enumerate(centers):
        for item_index in range(10):
            vector = center + rng.normal(0.0, 0.25, size=3)
            rows.append(
                {
                    "ticker": f"T{cluster_index}{item_index:02d}",
                    "date": "2024-01-31",
                    "embedding_0": vector[0],
                    "embedding_1": vector[1],
                    "embedding_2": vector[2],
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
