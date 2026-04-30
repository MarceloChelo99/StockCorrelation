"""Growth/lifecycle features from point-in-time XBRL fundamentals.

This view captures where a company appears to sit in its lifecycle. Young or
growth firms tend to have high revenue growth, lower or less stable margins,
higher investment intensity, and low payout. Mature firms tend to have slower
growth, steadier margins, lower investment intensity, and higher payout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.ingest.fundamentals import CONCEPT_GROUPS
from src.utils.dates import as_of_merge, select_month_end_rows


GROWTH_COLUMNS = [
    "growth_revenue_yoy_1y",
    "growth_revenue_cagr_3y",
    "growth_revenue_volatility",
    "growth_gross_margin",
    "growth_gross_margin_trend",
    "growth_operating_margin",
    "growth_operating_margin_trend",
    "growth_rd_intensity",
    "growth_capex_intensity",
    "growth_capex_intensity_trend",
    "growth_payout_dividend_yield",
    "growth_payout_buyback_yield",
    "growth_payout_total_yield",
    "growth_asset_growth_1y",
    "growth_leverage",
    "growth_size_log_assets",
    "growth_has_full_history",
]


class GrowthLifecycleProducer(FeatureProducer):
    """Compute point-in-time growth and lifecycle features from fundamentals."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="growth_lifecycle",
            key_columns=["ticker", "date"],
            columns=GROWTH_COLUMNS,
            source="sec_companyfacts",
            description="Lifecycle features from SEC XBRL companyfacts, as-of filing date.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        fundamentals_path = Path(config["paths"]["fundamentals_path"])
        if not fundamentals_path.exists():
            raise FileNotFoundError(f"Fundamentals parquet not found at {fundamentals_path}.")

        fundamentals = pd.read_parquet(fundamentals_path)
        if fundamentals.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        annual = annual_fundamental_panel(fundamentals)
        annual_features = compute_annual_growth_features(annual)
        monthly = monthly_price_panel(db, config)
        if monthly.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        merged = as_of_merge(monthly, annual_features, by=["ticker"])
        merged = add_market_cap_payout_features(merged, db, config)
        merged = winsorize_by_date(merged, [column for column in GROWTH_COLUMNS if column != "growth_has_full_history"])
        merged = log_transform_skewed(merged)
        merged["growth_has_full_history"] = merged["growth_has_full_history"].fillna(False).astype(bool)
        return merged.loc[:, self.spec.all_columns].reset_index(drop=True)


def annual_fundamental_panel(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize long-format companyfacts rows into annual wide rows."""
    frame = fundamentals.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "concept", "value", "filing_date", "fiscal_year"])
    # SEC companyfacts occasionally contains amended or malformed observations
    # whose period end is after the public filing date. Those rows cannot be
    # point-in-time safe, so exclude them before canonical concept selection.
    frame = frame[~(frame["end_date"].notna() & (frame["end_date"] > frame["filing_date"]))]
    frame = frame[frame["form"].isin(["10-K", "10-K/A", "20-F", "40-F"])]
    frame = frame[frame["fiscal_period"].isin(["FY", "CY"]) | frame["fiscal_period"].isna()]

    concept_to_canonical = {}
    concept_priority = {}
    for canonical, concepts in CONCEPT_GROUPS.items():
        for priority, concept in enumerate(concepts):
            concept_to_canonical[concept] = canonical
            concept_priority[concept] = priority
    frame["canonical"] = frame["concept"].map(concept_to_canonical)
    frame["priority"] = frame["concept"].map(concept_priority)
    frame = frame.dropna(subset=["canonical"])
    frame = frame.sort_values(["ticker", "fiscal_year", "canonical", "priority", "filing_date"])
    frame = frame.drop_duplicates(subset=["ticker", "fiscal_year", "canonical"], keep="last")

    wide = frame.pivot_table(
        index=["ticker", "fiscal_year"],
        columns="canonical",
        values="value",
        aggfunc="last",
    ).reset_index()
    dates = frame.groupby(["ticker", "fiscal_year"], as_index=False)["filing_date"].max()
    wide = wide.merge(dates, on=["ticker", "fiscal_year"], how="left")
    wide = wide.rename(columns={"filing_date": "date"})
    return wide.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_annual_growth_features(annual: pd.DataFrame) -> pd.DataFrame:
    """Compute annual lifecycle features before monthly as-of expansion."""
    frame = annual.copy()
    for column in CONCEPT_GROUPS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False)

    revenue = frame["revenue"].replace(0.0, np.nan)
    assets = frame["assets"].replace(0.0, np.nan)
    equity = frame["stockholders_equity"].replace(0.0, np.nan)

    frame["growth_revenue_yoy_1y"] = grouped["revenue"].pct_change(1)
    frame["growth_revenue_cagr_3y"] = grouped["revenue"].pct_change(3).add(1.0).pow(1.0 / 3.0).sub(1.0)
    frame["growth_revenue_volatility"] = grouped["growth_revenue_yoy_1y"].transform(lambda values: values.rolling(4, min_periods=2).std())
    frame["growth_gross_margin"] = frame["gross_profit"] / revenue
    frame["growth_gross_margin_trend"] = grouped["growth_gross_margin"].diff(3) / 3.0
    frame["growth_operating_margin"] = frame["operating_income"] / revenue
    frame["growth_operating_margin_trend"] = grouped["growth_operating_margin"].diff(3) / 3.0
    frame["growth_rd_intensity"] = frame["rd_expense"] / revenue
    frame["growth_capex_intensity"] = frame["capex"].abs() / revenue
    frame["growth_capex_intensity_trend"] = grouped["growth_capex_intensity"].diff(3) / 3.0
    frame["growth_asset_growth_1y"] = grouped["assets"].pct_change(1)
    frame["growth_leverage"] = frame["long_term_debt"] / equity
    frame["growth_size_log_assets"] = np.log1p(frame["assets"].clip(lower=0.0))
    frame["growth_has_full_history"] = grouped.cumcount() >= 3

    # Temporary revenue-normalized fallback; market-cap yields are added monthly when prices are present.
    frame["growth_payout_dividend_yield"] = frame["dividends"].abs() / revenue
    frame["growth_payout_buyback_yield"] = frame["buybacks"].abs() / revenue
    frame["growth_payout_total_yield"] = frame["growth_payout_dividend_yield"] + frame["growth_payout_buyback_yield"]

    return frame.loc[:, ["ticker", "date", "shares_outstanding", "dividends", "buybacks", *GROWTH_COLUMNS]]


def monthly_price_panel(db, config: dict) -> pd.DataFrame:
    """Return monthly ticker-date rows with adjusted close for payout yields."""
    prices = db.load_prices(date_from=config["data"]["start_date"], date_to=config["data"]["end_date"])
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "date", "adj_close"])
    prices = prices.loc[:, ["ticker", "date", "adj_close"]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    return select_month_end_rows(prices).reset_index(drop=True)


def add_market_cap_payout_features(frame: pd.DataFrame, db, config: dict) -> pd.DataFrame:
    """Use shares times price for payout yields when available."""
    result = frame.copy()
    market_cap = result["shares_outstanding"] * result["adj_close"]
    valid_market_cap = market_cap > 0
    dividend_yield = result["dividends"].abs() / market_cap.where(valid_market_cap)
    buyback_yield = result["buybacks"].abs() / market_cap.where(valid_market_cap)
    result.loc[valid_market_cap, "growth_payout_dividend_yield"] = dividend_yield[valid_market_cap]
    result.loc[valid_market_cap, "growth_payout_buyback_yield"] = buyback_yield[valid_market_cap]
    result["growth_payout_total_yield"] = (
        result["growth_payout_dividend_yield"].fillna(0.0)
        + result["growth_payout_buyback_yield"].fillna(0.0)
    )
    return result


def winsorize_by_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Winsorize numeric features cross-sectionally by date."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        lower = result.groupby("date")[column].transform(lambda values: values.quantile(0.01))
        upper = result.groupby("date")[column].transform(lambda values: values.quantile(0.99))
        result[column] = result[column].clip(lower=lower, upper=upper)
    return result


def log_transform_skewed(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply signed log transform to highly skewed non-ratio lifecycle features."""
    result = frame.copy()
    columns = [
        "growth_rd_intensity",
        "growth_capex_intensity",
        "growth_payout_dividend_yield",
        "growth_payout_buyback_yield",
        "growth_payout_total_yield",
        "growth_leverage",
        "growth_size_log_assets",
    ]
    for column in columns:
        result[column] = np.sign(result[column]) * np.log1p(np.abs(result[column]))
    return result
