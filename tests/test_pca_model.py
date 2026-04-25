from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from src.models.pca import PCAModel


class PCAModelTests(unittest.TestCase):
    def test_pca_standardizes_features_before_encoding(self) -> None:
        matrix = np.array(
            [
                [1.0, 1000.0, 5.0],
                [2.0, 2000.0, 5.0],
                [3.0, 3000.0, 5.0],
                [4.0, 4000.0, 5.0],
            ]
        )
        model = PCAModel(input_dim=3, embedding_dim=2)
        model.fit(matrix)

        assert model.mean_ is not None
        assert model.scale_ is not None
        self.assertTrue(np.allclose(model.mean_, [2.5, 2500.0, 5.0]))
        self.assertTrue(np.allclose(model.scale_, [1.11803399, 1118.03398875, 1.0]))

        embeddings = model.encode(matrix)
        self.assertEqual(embeddings.shape, (4, 2))

    def test_save_and_load_preserves_scaling(self) -> None:
        matrix = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 40.0],
            ]
        )
        model = PCAModel(input_dim=2, embedding_dim=1)
        model.fit(matrix)

        with TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            loaded = PCAModel.load(tmp_dir)

        self.assertTrue(np.allclose(model.encode(matrix), loaded.encode(matrix)))


if __name__ == "__main__":
    unittest.main()
