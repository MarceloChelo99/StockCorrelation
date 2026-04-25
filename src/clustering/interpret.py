"""Interpret soft themes through top firms, feature profiles, and GICS overlap."""
from __future__ import annotations

import re

import pandas as pd


def top_firms_per_theme(loadings: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Return the top-loaded latest firms for each theme."""
    latest = latest_by_ticker(loadings)
    theme_columns = theme_cols(latest)
    rows = []
    for theme in theme_columns:
        top = latest.sort_values(theme, ascending=False).head(k)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append({"theme": theme, "rank": rank, "ticker": row["ticker"], "loading": float(row[theme])})
    return pd.DataFrame(rows)


def feature_profile_per_theme(loadings: pd.DataFrame, features: pd.DataFrame, view_config: dict) -> pd.DataFrame:
    """Compute loading-weighted feature means for each theme."""
    pattern = re.compile(view_config["feature_pattern"])
    feature_columns = [column for column in features.columns if pattern.search(column)]
    if not feature_columns:
        return pd.DataFrame(columns=["theme", "feature", "weighted_mean"])

    latest_loadings = latest_by_ticker(loadings)
    latest_features = latest_by_ticker(features)
    frame = latest_loadings.merge(latest_features.loc[:, ["ticker", *feature_columns]], on="ticker", how="inner")
    rows = []
    for theme in theme_cols(frame):
        weights = frame[theme].astype(float)
        denominator = float(weights.sum())
        if denominator == 0.0:
            continue
        for feature in feature_columns:
            value = float((frame[feature].fillna(0.0).astype(float) * weights).sum() / denominator)
            rows.append({"theme": theme, "feature": feature, "weighted_mean": value})
    return pd.DataFrame(rows)


def compare_themes_to_gics(loadings: pd.DataFrame, gics_sectors: pd.DataFrame) -> pd.DataFrame:
    """Return top GICS sector shares for each soft theme."""
    latest = latest_by_ticker(loadings)
    metadata = gics_sectors.loc[:, ["ticker", "gics_sector"]].dropna().copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    frame = latest.merge(metadata, on="ticker", how="inner")
    rows = []
    for theme in theme_cols(frame):
        totals = frame.groupby("gics_sector")[theme].sum().sort_values(ascending=False)
        denominator = float(totals.sum())
        for sector, value in totals.head(5).items():
            rows.append({"theme": theme, "gics_sector": sector, "soft_share": float(value / denominator if denominator else 0.0)})
    return pd.DataFrame(rows)


def latest_by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row for each ticker."""
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    return result.reset_index(drop=True)


def theme_cols(frame: pd.DataFrame) -> list[str]:
    """Return theme loading columns."""
    return [column for column in frame.columns if column.startswith("theme_")]
