from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from src.models._legacy_numpy_autoencoder import Autoencoder as LegacyNumpyAutoencoder
from src.models.autoencoder import Autoencoder


class AutoencoderModelTests(unittest.TestCase):
    def test_autoencoder_training_reduces_reconstruction_loss(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(64, 6))
        matrix[:, 3:] = matrix[:, :3] * 0.5 + rng.normal(scale=0.05, size=(64, 3))

        model = Autoencoder(
            input_dim=6,
            embedding_dim=2,
            hidden_dims=[8],
            epochs=35,
            batch_size=16,
            learning_rate=0.01,
            early_stopping_patience=10,
            random_seed=7,
        )
        history = model.fit(matrix)

        self.assertLess(
            history["final_reconstruction_mse"],
            history["initial_reconstruction_mse"],
        )
        self.assertEqual(model.encode(matrix).shape, (64, 2))
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)

    def test_save_and_load_preserves_embeddings(self) -> None:
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(20, 4))
        model = Autoencoder(
            input_dim=4,
            embedding_dim=2,
            hidden_dims=[6],
            epochs=8,
            batch_size=10,
            random_seed=11,
        )
        model.fit(matrix)

        with TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            loaded = Autoencoder.load(tmp_dir)

        self.assertTrue(np.allclose(model.encode(matrix), loaded.encode(matrix)))

    def test_multilayer_architecture_trains_and_encodes(self) -> None:
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(80, 10))
        model = Autoencoder(
            input_dim=10,
            embedding_dim=3,
            hidden_dims=[12, 8, 6],
            dropout=0.05,
            epochs=12,
            batch_size=20,
            random_seed=17,
        )
        history = model.fit(matrix)
        embeddings = model.encode(matrix)

        self.assertEqual(embeddings.shape, (80, 3))
        self.assertLess(history["final_reconstruction_mse"], 2.0)

    def test_pytorch_reconstruction_is_close_to_legacy_baseline(self) -> None:
        rng = np.random.default_rng(23)
        matrix = rng.normal(size=(96, 8))
        matrix[:, 4:] = matrix[:, :4] * 0.4 + rng.normal(scale=0.05, size=(96, 4))
        train = matrix[:72]
        held_out = matrix[72:]

        legacy = LegacyNumpyAutoencoder(
            input_dim=8,
            embedding_dim=3,
            hidden_dim=12,
            epochs=45,
            batch_size=24,
            learning_rate=0.01,
            random_seed=23,
        )
        legacy.fit(train)
        legacy_mean = legacy.mean_
        legacy_scale = legacy.scale_
        assert legacy_mean is not None
        assert legacy_scale is not None
        legacy_standardized = (held_out - legacy_mean) / legacy_scale
        legacy_reconstruction = (legacy.forward(held_out) - legacy_mean) / legacy_scale
        legacy_mse = float(np.mean(np.square(legacy_reconstruction - legacy_standardized)))

        model = Autoencoder(
            input_dim=8,
            embedding_dim=3,
            hidden_dims=[12],
            epochs=80,
            batch_size=24,
            learning_rate=0.01,
            early_stopping_patience=20,
            random_seed=23,
        )
        model.fit(train)
        pytorch_mse = model.reconstruction_mse(held_out)

        self.assertLessEqual(pytorch_mse, legacy_mse * 1.2)

    def test_load_reads_legacy_numpy_artifact(self) -> None:
        rng = np.random.default_rng(31)
        matrix = rng.normal(size=(24, 5))
        legacy = LegacyNumpyAutoencoder(
            input_dim=5,
            embedding_dim=2,
            hidden_dim=7,
            epochs=4,
            batch_size=8,
            random_seed=31,
        )
        legacy.fit(matrix)

        with TemporaryDirectory() as tmp_dir:
            legacy.save(tmp_dir)
            loaded = Autoencoder.load(tmp_dir)

        self.assertTrue(np.allclose(legacy.encode(matrix), loaded.encode(matrix)))


if __name__ == "__main__":
    unittest.main()
