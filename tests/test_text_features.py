from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import pandas as pd

from src.features.text.common import compute_text_feature_frame
from src.features.text.business_description import BusinessTextProducer
from src.features.text.historical_text_features import HistoricalTextFeatureProducer
from src.features.text.risk_factors import RiskFactorsTextProducer
from src.filings.topics import topic_evidence_snippets


class TextFeatureProducerTests(unittest.TestCase):
    def test_topic_evidence_snippets_center_on_keywords(self) -> None:
        text = (
            "We sell industrial widgets globally. "
            "Our artificial intelligence platform improves supply chain logistics for customers. "
            "The remainder of the section discusses ordinary operations."
        )

        snippets = topic_evidence_snippets(text, max_snippets=2, window_chars=120)

        self.assertGreaterEqual(len(snippets), 1)
        self.assertEqual(snippets[0]["snippet_rank"], 1)
        self.assertIn("artificial intelligence", str(snippets[0]["evidence_snippet"]).lower())
        self.assertIn(str(snippets[0]["snippet_topic"]), {"ai", "supply_chain"})

    def test_topic_evidence_snippets_fallback_to_section_start(self) -> None:
        snippets = topic_evidence_snippets(
            "Plain business model discussion with no tracked keyword phrase.",
            max_snippets=1,
            window_chars=80,
        )

        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0]["snippet_topic"], "section_start")

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

    def test_historical_text_producer_builds_point_in_time_monthly_features(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings_path = root / "historical_section_embeddings.parquet"
            historical = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "filing_date": pd.Timestamp("2020-01-15"),
                        "section": "business",
                        "embedding_0": 1.0,
                        "embedding_1": 0.0,
                    },
                    {
                        "ticker": "AAA",
                        "filing_date": pd.Timestamp("2020-01-15"),
                        "section": "risk_factors",
                        "embedding_0": 0.0,
                        "embedding_1": 1.0,
                    },
                    {
                        "ticker": "AAA",
                        "filing_date": pd.Timestamp("2020-03-15"),
                        "section": "business",
                        "embedding_0": 5.0,
                        "embedding_1": 0.0,
                    },
                    {
                        "ticker": "AAA",
                        "filing_date": pd.Timestamp("2020-03-15"),
                        "section": "risk_factors",
                        "embedding_0": 0.0,
                        "embedding_1": 5.0,
                    },
                    {
                        "ticker": "BBB",
                        "filing_date": pd.Timestamp("2020-01-20"),
                        "section": "business",
                        "embedding_0": 0.5,
                        "embedding_1": 0.5,
                    },
                    {
                        "ticker": "BBB",
                        "filing_date": pd.Timestamp("2020-01-20"),
                        "section": "risk_factors",
                        "embedding_0": 0.25,
                        "embedding_1": 0.75,
                    },
                ]
            )
            historical.to_parquet(embeddings_path, index=False)
            db = TinyDB(
                pd.DataFrame(
                    {
                        "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
                        "date": pd.to_datetime(
                            ["2020-01-31", "2020-02-28", "2020-03-31", "2020-01-31", "2020-02-28", "2020-03-31"]
                        ),
                        "adj_close": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                        "volume": [100, 100, 100, 200, 200, 200],
                    }
                )
            )
            config = {
                "random_seed": 7,
                "data": {"start_date": "2020-01-01", "end_date": "2020-03-31"},
                "features": {"price": {"min_history_days": 1}},
            }
            producer = HistoricalTextFeatureProducer(
                target_dim=2,
                input_path=embeddings_path,
                pca_fit_end_date="2020-02-29",
            )
            output = producer.run(db, config, root)

            self.assertEqual(list(output.columns), ["ticker", "date", "text_hist_emb_0", "text_hist_emb_1"])
            self.assertFalse(output.duplicated(["ticker", "date"]).any())
            self.assertTrue((root / "text_historical.parquet").exists())
            self.assertTrue((root / "text_historical_pca.pkl").exists())
            self.assertTrue((root / "text_historical_pca_meta.json").exists())


class TinyDB:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_prices(self, *, ticker=None, date_from=None, date_to=None):
        frame = self.prices.copy()
        if date_from is not None:
            frame = frame[frame["date"] >= pd.Timestamp(date_from)]
        if date_to is not None:
            frame = frame[frame["date"] <= pd.Timestamp(date_to)]
        return frame.reset_index(drop=True)


if __name__ == "__main__":
    unittest.main()
