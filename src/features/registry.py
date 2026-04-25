"""Explicit registry of active feature producers."""
from __future__ import annotations

from src.features.events.item_frequency import EventItemFrequencyProducer
from src.features.fundamentals.growth_lifecycle import GrowthLifecycleProducer
from src.features.graph.network_position import NetworkPositionProducer
from src.features.price.liquidity import PriceLiquidityProducer
from src.features.price.momentum import PriceMomentumProducer
from src.features.price.volatility import PriceVolatilityProducer
from src.features.text.business_description import BusinessTextProducer
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
        "network_position": NetworkPositionProducer(),
        "text_business": BusinessTextProducer(
            embedding_dim=int(config["features"]["text"]["business_embedding_dim"])
        ),
        "text_risk": RiskFactorsTextProducer(
            embedding_dim=int(config["features"]["text"]["risk_embedding_dim"])
        ),
    }
