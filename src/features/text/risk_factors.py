"""Legacy latest-risk-section text features.

Superseded for new experiments by ``text_historical``, which uses historical
MiniLM section embeddings. This producer is retained to reproduce earlier runs.
"""
from __future__ import annotations

from src.features.base import FeatureProducer, FeatureSpec
from src.features.text.common import (
    compute_text_feature_frame,
    load_sections_frame,
    vector_column_names,
)


class RiskFactorsTextProducer(FeatureProducer):
    """Build hashed text vectors from parsed Item 1A risk-factor sections."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="text_risk",
            key_columns=["ticker", "date"],
            columns=vector_column_names("text_risk", self.embedding_dim),
            source="ten_k_sections",
            description="Hashed bag-of-words vectors from Item 1A risk-factor sections.",
        )

    def compute(self, db, config: dict):
        sections = load_sections_frame(config)
        return compute_text_feature_frame(
            sections,
            text_column="risk_factors_text",
            prefix="text_risk",
            embedding_dim=self.embedding_dim,
            min_token_length=int(config["features"]["text"]["min_token_length"]),
            method=config["features"]["text"].get("method", "hashed"),
            model_name=config["features"]["text"].get("model_name"),
            batch_size=int(config["features"]["text"].get("batch_size", 16)),
            max_chars=int(config["features"]["text"].get("max_chars", 12000)),
            normalize_embeddings=bool(config["features"]["text"].get("normalize_embeddings", True)),
        )
