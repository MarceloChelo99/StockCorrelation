"""Explicit registry of active feature producers."""
from __future__ import annotations

from src.features.events.item_frequency import EventItemFrequencyProducer
from src.features.fundamentals.growth_lifecycle import GrowthLifecycleProducer
from src.features.fundamentals.valuation import ValuationProducer
from src.features.graph.network_position import NetworkPositionProducer
from src.features.price.liquidity import PriceLiquidityProducer
from src.features.price.momentum import PriceMomentumProducer
from src.features.price.volatility import PriceVolatilityProducer
from src.features.text.business_description import BusinessTextProducer
from src.features.text.historical_text_features import HistoricalTextFeatureProducer
from src.features.text.risk_factors import RiskFactorsTextProducer


def build_active_producers(config: dict) -> dict[str, object]:
    """Build the active feature producer registry from config."""
    return {
        "price_volatility": PriceVolatilityProducer(),
        "price_momentum": PriceMomentumProducer(),
        "price_liquidity": PriceLiquidityProducer(),
        "event_item_frequency": EventItemFrequencyProducer(
            items=list(config["features"]["events"]["items"]),
            windows=list(config["features"]["events"]["windows"]),
        ),
        "growth_lifecycle": GrowthLifecycleProducer(),
        "valuation": ValuationProducer(),
        "network_position": NetworkPositionProducer(),
        "text_historical": HistoricalTextFeatureProducer(
            target_dim=int(config["features"]["text_historical"]["target_dim"]),
            input_path=config["features"]["text_historical"]["input_path"],
            aggregation=config["features"]["text_historical"].get("aggregation", "concat_business_risk"),
            business_sections=list(config["features"]["text_historical"].get("business_sections", ["business"])),
            risk_sections=list(config["features"]["text_historical"].get("risk_sections", ["risk_factors", "q_risk_factors"])),
            pca_fit_end_date=config["features"]["text_historical"].get("pca_fit_end_date"),
        ),
        "text_business": BusinessTextProducer(
            embedding_dim=int(config["features"]["text"]["business_embedding_dim"])
        ),
        "text_risk": RiskFactorsTextProducer(
            embedding_dim=int(config["features"]["text"]["risk_embedding_dim"])
        ),
    }
