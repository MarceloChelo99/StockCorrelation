from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.evaluation.relationships import RelationshipGraphEvaluator, direct_connection_rate, relationship_graph


class RelationshipGraphEvaluatorTests(unittest.TestCase):
    def test_relationship_graph_is_undirected(self) -> None:
        relationships = pd.DataFrame(
            {
                "source_ticker": ["AAA"],
                "target_ticker": ["BBB"],
                "relationship_type": ["partner"],
                "confidence": [0.9],
            }
        )

        graph = relationship_graph(relationships, min_confidence=0.45, relationship_types=None)

        self.assertEqual(graph["AAA"], {"BBB"})
        self.assertEqual(graph["BBB"], {"AAA"})
        self.assertEqual(direct_connection_rate("AAA", ["BBB", "CCC"], graph), 0.5)

    def test_evaluator_runs_on_tiny_graph(self) -> None:
        embeddings = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                "date": [pd.Timestamp("2025-01-31")] * 6,
                "embedding_0": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
                "embedding_1": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                "gics_sub_industry": ["One", "One", "One", "Two", "Two", "Two"],
            }
        )
        relationships = pd.DataFrame(
            {
                "source_ticker": ["AAA", "AAA", "DDD", "DDD"],
                "target_ticker": ["BBB", "CCC", "EEE", "FFF"],
                "relationship_type": ["partner", "supplier", "partner", "supplier"],
                "confidence": [0.9, 0.9, 0.9, 0.9],
            }
        )
        config = {
            "random_seed": 7,
            "paths": {},
            "evaluation": {
                "relationships": {
                    "k": 1,
                    "benchmark_column": "gics_sub_industry",
                    "min_confidence": 0.45,
                    "relationship_types": ["partner", "supplier"],
                    "random_seed": 7,
                }
            },
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "relationships.parquet"
            relationships.to_parquet(path, index=False)
            config["evaluation"]["relationships"]["relationship_path"] = str(path)
            metrics = RelationshipGraphEvaluator().run(
                embeddings,
                metadata,
                config,
                output_dir=tmp_dir,
            )

        self.assertEqual(metrics["n_observations"], 6)
        self.assertIn("mean_embedding_direct_rate", metrics)


if __name__ == "__main__":
    unittest.main()
