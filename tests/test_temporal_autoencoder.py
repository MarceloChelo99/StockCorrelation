from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.models.temporal_dataloader import build_consecutive_pairs
from src.models.temporal_autoencoder import TemporalAutoencoder
from src.models.train import train_embedding_model


class TemporalAutoencoderTests(unittest.TestCase):
    def test_build_consecutive_pairs_skips_gaps_and_non_finite_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "B", "B", "C", "C"],
                "date": pd.to_datetime(
                    [
                        "2022-01-31",
                        "2022-02-28",
                        "2022-03-31",
                        "2022-01-31",
                        "2022-05-31",
                        "2022-01-31",
                        "2022-02-28",
                    ]
                ),
                "feature_0": [1.0, 1.1, 1.2, 2.0, 2.2, 3.0, np.nan],
                "feature_1": [0.0, 0.1, 0.2, 1.0, 1.1, 2.0, 2.1],
            }
        )

        pairs = build_consecutive_pairs(frame, ["feature_0", "feature_1"], max_gap_days=45)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs.tensors[0].shape, (2, 2))
        self.assertEqual(pairs.n_observations_dropped_gap, 1)  # type: ignore[attr-defined]
        self.assertEqual(pairs.n_pairs_dropped_non_finite, 1)  # type: ignore[attr-defined]

    def test_temporal_train_embedding_model_smoke(self) -> None:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2022-01-31", periods=8, freq="ME")
        rows = []
        for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
            level = float(ticker_index)
            for date_index, date in enumerate(dates):
                trend = date_index / 10.0
                rows.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "feature_0": level + trend,
                        "feature_1": level - trend,
                        "feature_2": np.sin(trend) + rng.normal(scale=0.01),
                        "feature_3": np.cos(trend) + rng.normal(scale=0.01),
                    }
                )
        dataset = pd.DataFrame(rows)
        config = {
            "random_seed": 7,
            "model": {
                "name": "temporal_autoencoder",
                "embedding_dim": 2,
                "hidden_dims": [6],
                "dropout": 0.0,
                "epochs": 4,
                "batch_size": 6,
                "learning_rate": 0.01,
                "train_val_split": 0.8,
                "early_stopping_patience": 4,
                "fit_complete_cases_only": True,
                "lambda_temp": 0.25,
                "alpha": 1.0,
                "max_pair_gap_days": 45,
                "exclude_columns": [],
            },
        }

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model, embeddings, history = train_embedding_model(
                dataset,
                config,
                model_dir=root / "model",
                embeddings_path=root / "embeddings.parquet",
                history_path=root / "training_history.json",
            )
            diagnostics = json.loads((root / "temporal_diagnostics.json").read_text())
            loaded = TemporalAutoencoder.load(root / "model")

        self.assertEqual(len(embeddings), len(dataset))
        self.assertEqual(embeddings.filter(like="embedding_").shape[1], 2)
        self.assertGreater(history["n_pairs_train"], 0)
        self.assertIn("train_smoothness_loss", history)
        self.assertGreater(diagnostics["n_velocity_steps"], 0)
        self.assertTrue(np.allclose(model.encode(dataset.filter(like="feature_").to_numpy()), loaded.encode(dataset.filter(like="feature_").to_numpy())))


if __name__ == "__main__":
    unittest.main()
