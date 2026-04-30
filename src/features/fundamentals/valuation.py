"""Valuation features from point-in-time XBRL fundamentals and monthly prices.

This feature group captures whether a company looks expensive or cheap relative
to trailing fundamentals known as of each month. It complements the
growth/lifecycle view: two companies can have similar growth profiles but very
different market-implied prices for that growth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.features.fundamentals.growth_lifecycle import (
    annual_fundamental_panel,
    monthly_price_panel,
    winsorize_by_date,
)
from src.ingest.fundamentals import CONCEPT_GROUPS
from src.utils.dates import as_of_merge


VALUATION_COLUMNS = [
    "valuation_market_cap_log",
    "valuation_sales_yield",
    "valuation_earnings_yield",
    "valuation_operating_income_yield",
    "valuation_gross_profit_yield",
    "valuation_book_to_market",
    "valuation_debt_to_market",
    "valuation_cash_to_market",
    "valuation_ev_to_sales",
    "valuation_ev_to_operating_income",
    "valuation_ev_to_gross_profit",
    "valuation_free_cash_flow_yield",
    "valuation_ev_to_free_cash_flow",
    "valuation_shareholder_yield",
    "valuation_has_full_valuation",
]


class ValuationProducer(FeatureProducer):
    """Compute point-in-time valuation ratios from fundamentals and prices."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="valuation",
            key_columns=["ticker", "date"],
            columns=VALUATION_COLUMNS,
            source="sec_companyfacts",
            description="Market valuation features from SEC XBRL fundamentals and monthly adjusted prices.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        fundamentals_path = Path(config["paths"]["fundamentals_path"])
        if not fundamentals_path.exists():
            raise FileNotFoundError(f"Fundamentals parquet not found at {fundamentals_path}.")

        fundamentals = pd.read_parquet(fundamentals_path)
        if fundamentals.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        annual = annual_fundamental_panel(fundamentals)
        monthly = monthly_price_panel(db, config)
        if annual.empty or monthly.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        annual = ensure_fundamental_columns(annual)
        merged = as_of_merge(monthly, annual, by=["ticker"])
        valued = compute_valuation_ratios(merged)
        numeric_columns = [column for column in VALUATION_COLUMNS if column != "valuation_has_full_valuation"]
        valued = winsorize_by_date(valued, numeric_columns)
        valued = signed_log_transform(
            valued,
            [
                "valuation_ev_to_sales",
                "valuation_ev_to_operating_income",
                "valuation_ev_to_gross_profit",
                "valuation_ev_to_free_cash_flow",
            ],
        )
        valued["valuation_has_full_valuation"] = valued["valuation_has_full_valuation"].fillna(False).astype(bool)
        return valued.loc[:, self.spec.all_columns].reset_index(drop=True)


def ensure_fundamental_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure every canonical fundamental column exists before ratio math."""
    result = frame.copy()
    for column in CONCEPT_GROUPS:
        if column not in result.columns:
            result[column] = np.nan
    return result


def compute_valuation_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute valuation ratios from as-of fundamentals and monthly prices."""
    result = frame.copy()
    numeric_inputs = [
        "adj_close",
        "shares_outstanding",
        "revenue",
        "net_income",
        "operating_income",
        "gross_profit",
        "stockholders_equity",
        "long_term_debt",
        "cash_and_equivalents",
        "operating_cash_flow",
        "capex",
        "dividends",
        "buybacks",
    ]
    for column in numeric_inputs:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    market_cap = result["shares_outstanding"] * result["adj_close"]
    market_cap = market_cap.where(market_cap > 0.0)
    debt = result["long_term_debt"].clip(lower=0.0)
    cash = result["cash_and_equivalents"].clip(lower=0.0)
    enterprise_value = market_cap + debt.fillna(0.0) - cash.fillna(0.0)
    enterprise_value = enterprise_value.where(enterprise_value > 0.0)
    free_cash_flow = result["operating_cash_flow"] - result["capex"].abs()

    result["valuation_market_cap_log"] = np.log1p(market_cap)
    result["valuation_sales_yield"] = safe_divide(result["revenue"], market_cap)
    result["valuation_earnings_yield"] = safe_divide(result["net_income"], market_cap)
    result["valuation_operating_income_yield"] = safe_divide(result["operating_income"], market_cap)
    result["valuation_gross_profit_yield"] = safe_divide(result["gross_profit"], market_cap)
    result["valuation_book_to_market"] = safe_divide(result["stockholders_equity"], market_cap)
    result["valuation_debt_to_market"] = safe_divide(debt, market_cap)
    result["valuation_cash_to_market"] = safe_divide(cash, market_cap)
    result["valuation_ev_to_sales"] = safe_divide(enterprise_value, positive_denominator(result["revenue"]))
    result["valuation_ev_to_operating_income"] = safe_divide(
        enterprise_value,
        positive_denominator(result["operating_income"]),
    )
    result["valuation_ev_to_gross_profit"] = safe_divide(
        enterprise_value,
        positive_denominator(result["gross_profit"]),
    )
    result["valuation_free_cash_flow_yield"] = safe_divide(free_cash_flow, market_cap)
    result["valuation_ev_to_free_cash_flow"] = safe_divide(
        enterprise_value,
        positive_denominator(free_cash_flow),
    )

    shareholder_payout = result["dividends"].abs().fillna(0.0) + result["buybacks"].abs().fillna(0.0)
    result["valuation_shareholder_yield"] = safe_divide(shareholder_payout, market_cap)
    result["valuation_has_full_valuation"] = result[
        ["valuation_sales_yield", "valuation_earnings_yield", "valuation_book_to_market"]
    ].notna().all(axis=1)
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while preserving NaNs for invalid denominators."""
    clean_denominator = denominator.where(np.isfinite(denominator) & (denominator != 0.0))
    return numerator / clean_denominator


def positive_denominator(values: pd.Series) -> pd.Series:
    """Return values only where positive, for conventional valuation multiples."""
    return values.where(np.isfinite(values) & (values > 0.0))


def signed_log_transform(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compress skewed valuation multiples while preserving sign if present."""
    result = frame.copy()
    for column in columns:
        result[column] = np.sign(result[column]) * np.log1p(np.abs(result[column]))
    return result
