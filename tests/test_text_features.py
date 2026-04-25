from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import pandas as pd

from src.features.text.common import compute_text_feature_frame
from src.features.text.business_description import BusinessTextProducer
from src.features.text.risk_factors import RiskFactorsTextProducer


class TextFeatureProducerTests(unittest.TestCase):
    def test_business_and_risk_producers_create_fixed_width_vectors(self) -> None:
        with TemporaryDirectory() as temp_dir:
            frame = pd.DataFrame(
                {
                    "ticker": ["AAA", "BBB"],
                    "accession_no": ["a1", "b1"],
                    "filing_date": pd.to_datetime(["2024-01-15", "2024-02-20"]),
                    "business_text": [
                        "Cloud software analytics platform recurring revenue customers",
                        "Industrial tools manufacturing distribution engineering systems",
                    ],
                    "risk_factors_text": [
                        "Competition cybersecurity regulation supplier concentration",
                        "Commodity inflation inventory labor disruption tariffs",
                    ],
                }
            )
            output_path = f"{temp_dir}/ten_k_sections.parquet"
            frame.to_parquet(output_path, index=False)

            config = {
                "paths": {"sections_dir": temp_dir},
                "features": {"text": {"method": "hashed", "min_token_length": 3}},
            }
            business = BusinessTextProducer(embedding_dim=8).compute(None, config)
            risk = RiskFactorsTextProducer(embedding_dim=8).compute(None, config)

            self.assertEqual(business.shape, (2, 10))
            self.assertEqual(risk.shape, (2, 10))
            self.assertTrue(business.filter(like="text_business_").abs().sum(axis=1).gt(0).all())
            self.assertTrue(risk.filter(like="text_risk_").abs().sum(axis=1).gt(0).all())

    def test_minilm_method_uses_sentence_embedding_vector_shape(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "filing_date": pd.to_datetime(["2024-01-15"]),
                "business_text": ["Cloud software analytics platform recurring revenue customers"],
            }
        )

        with patch(
            "src.features.text.common.minilm_text_vectors",
            return_value=np.array([[0.1, 0.2, 0.3]]),
        ):
            result = compute_text_feature_frame(
                frame,
                text_column="business_text",
                prefix="text_business",
                embedding_dim=3,
                min_token_length=3,
                method="minilm",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )

        self.assertEqual(result.shape, (1, 5))
        self.assertEqual(result.loc[0, "text_business_2"], 0.3)


if __name__ == "__main__":
    unittest.main()
