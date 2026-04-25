from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.evaluation.clustering import (
    ClusteringEvaluator,
    adjusted_rand_index,
    kmeans,
    normalized_mutual_info,
)


class ClusteringEvaluatorTests(unittest.TestCase):
    def test_adjusted_rand_index_and_nmi_are_one_for_identical_assignments(self) -> None:
        labels = np.array(["a", "a", "b", "b"])
        clusters = np.array([1, 1, 0, 0])

        self.assertAlmostEqual(adjusted_rand_index(labels, clusters), 1.0)
        self.assertAlmostEqual(normalized_mutual_info(labels, clusters), 1.0)

    def test_kmeans_separates_simple_blobs(self) -> None:
        matrix = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [8.0, 8.0],
                [8.1, 8.0],
            ]
        )
        clusters, _, _ = kmeans(matrix, k=2, n_init=5, max_iter=20, random_seed=7)

        self.assertEqual(clusters[0], clusters[1])
        self.assertEqual(clusters[2], clusters[3])
        self.assertNotEqual(clusters[0], clusters[2])

    def test_evaluator_writes_artifacts(self) -> None:
        embeddings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "date": pd.to_datetime(["2025-12-31"] * 4),
                "embedding_0": [0.0, 0.1, 8.0, 8.1],
                "embedding_1": [0.0, 0.0, 8.0, 8.0],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "gics_sector": ["Tech", "Tech", "Energy", "Energy"],
                "title": ["A", "B", "C", "D"],
            }
        )
        config = {
            "random_seed": 7,
            "evaluation": {
                "clustering": {
                    "label_column": "gics_sector",
                    "k": 2,
                    "n_init": 5,
                    "max_iter": 20,
                    "top_disagreements": 5,
                }
            },
        }

        with TemporaryDirectory() as tmp_dir:
            metrics = ClusteringEvaluator().run(embeddings, metadata, config, output_dir=tmp_dir)

            self.assertEqual(metrics["n_tickers"], 4)
            self.assertAlmostEqual(metrics["ari"], 1.0)
            self.assertTrue((Path(tmp_dir) / "figures" / "clustering_projection.svg").exists())
            self.assertTrue((Path(tmp_dir) / "evaluation" / "clustering_cluster_label_crosstab.csv").exists())


if __name__ == "__main__":
    unittest.main()
