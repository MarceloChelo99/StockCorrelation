"""Sector-relative return outlook built from transparent price and fundamental signals.

The target is sector excess return versus the equal-weight S&P 500 universe.
This is intentionally a small walk-forward ridge model, not a large neural net:
there are only 11 sectors and roughly monthly observations, so transparency and
leak avoidance matter more than model flexibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PRICE_FEATURE_COLUMNS = [
    "sector_excess_momentum_21d",
    "sector_excess_momentum_63d",
    "sector_excess_momentum_126d",
    "sector_excess_volatility_63d",
    "sector_excess_hit_rate_63d",
]

DEFAULT_FUNDAMENTAL_COLUMNS = [
    "valuation_sales_yield",
    "valuation_earnings_yield",
    "valuation_operating_income_yield",
    "valuation_gross_profit_yield",
    "valuation_book_to_market",
    "valuation_debt_to_market",
    "valuation_shareholder_yield",
    "growth_revenue_yoy_1y",
    "growth_revenue_cagr_3y",
    "growth_gross_margin",
    "growth_operating_margin",
    "growth_rd_intensity",
    "growth_capex_intensity",
    "growth_payout_total_yield",
    "growth_leverage",
]


@dataclass(frozen=True)
class SectorOutlookResult:
    """Backtest artifacts for sector-relative predictions."""

    panel: pd.DataFrame
    predictions: pd.DataFrame
    latest: pd.DataFrame
    metrics: dict[str, float | int | str]
    coefficients: pd.DataFrame


def sector_outlook_backtest(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    valuation: pd.DataFrame | None = None,
    growth: pd.DataFrame | None = None,
    horizon_days: int = 63,
    min_train_months: int = 36,
    ridge_alpha: float = 10.0,
) -> SectorOutlookResult:
    """Fit a walk-forward sector excess-return model and return diagnostics."""
    sector_returns, market_returns = sector_and_market_returns(prices, metadata)
    month_ends = observed_month_ends(sector_returns)
    panel = build_sector_feature_panel(
        sector_returns,
        market_returns,
        month_ends,
        metadata,
        valuation=valuation,
        growth=growth,
        horizon_days=horizon_days,
    )
    feature_columns = usable_feature_columns(panel)
    predictions, coefficients = walk_forward_ridge_predictions(
        panel,
        feature_columns,
        min_train_months=min_train_months,
        ridge_alpha=ridge_alpha,
    )
    latest = latest_sector_scores(panel, feature_columns, min_train_months=min_train_months, ridge_alpha=ridge_alpha)
    metrics = sector_prediction_metrics(predictions, horizon_days)
    return SectorOutlookResult(
        panel=panel,
        predictions=predictions,
        latest=latest,
        metrics=metrics,
        coefficients=coefficients,
    )


def sector_and_market_returns(prices: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return equal-weight sector returns and equal-weight market returns."""
    required = {"ticker", "date", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Prices are missing required columns: {sorted(missing)}")
    if "gics_sector" not in metadata.columns:
        raise ValueError("Metadata must include gics_sector.")

    prices = prices.loc[:, ["ticker", "date", "adj_close"]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["adj_close"].pct_change()

    labels = metadata.loc[:, ["ticker", "gics_sector"]].dropna().copy()
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels = labels.drop_duplicates("ticker", keep="last")
    returns = prices.merge(labels, on="ticker", how="inner")
    returns = returns.dropna(subset=["return", "gics_sector"])
    sector = (
        returns.groupby(["date", "gics_sector"], as_index=False)["return"]
        .mean()
        .pivot(index="date", columns="gics_sector", values="return")
        .sort_index()
    )
    market = returns.groupby("date")["return"].mean().sort_index()
    return sector, market


def observed_month_ends(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return observed month-end trading dates from a daily return frame."""
    dates = pd.Series(pd.to_datetime(frame.index), index=frame.index)
    month_end = dates.groupby(dates.dt.to_period("M")).transform("max")
    return pd.DatetimeIndex(dates[dates.eq(month_end)].values)


def build_sector_feature_panel(
    sector_returns: pd.DataFrame,
    market_returns: pd.Series,
    month_ends: pd.DatetimeIndex,
    metadata: pd.DataFrame,
    *,
    valuation: pd.DataFrame | None,
    growth: pd.DataFrame | None,
    horizon_days: int,
) -> pd.DataFrame:
    """Build monthly sector-date rows with features and future excess returns."""
    excess = sector_returns.sub(market_returns, axis=0)
    rows: list[dict[str, object]] = []
    for date in month_ends:
        history = excess.loc[excess.index <= date]
        future_sector = sector_returns.loc[sector_returns.index > date].head(int(horizon_days))
        future_market = market_returns.loc[market_returns.index > date].head(int(horizon_days))
        target_end_date = future_sector.index.max() if len(future_sector) else pd.NaT
        for sector in sector_returns.columns:
            row = {
                "date": pd.Timestamp(date),
                "gics_sector": sector,
                "target_end_date": target_end_date,
                "future_excess_return": future_excess_return(future_sector.get(sector), future_market),
            }
            for window in [21, 63, 126]:
                row[f"sector_excess_momentum_{window}d"] = cumulative_return(history[sector].tail(window))
            recent = history[sector].tail(63).dropna()
            row["sector_excess_volatility_63d"] = float(recent.std(ddof=1) * np.sqrt(252.0)) if len(recent) > 2 else np.nan
            row["sector_excess_hit_rate_63d"] = float((recent > 0.0).mean()) if len(recent) else np.nan
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel = attach_sector_fundamentals(panel, metadata, valuation, "valuation")
    panel = attach_sector_fundamentals(panel, metadata, growth, "growth")
    return panel.sort_values(["date", "gics_sector"]).reset_index(drop=True)


def attach_sector_fundamentals(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    features: pd.DataFrame | None,
    prefix: str,
) -> pd.DataFrame:
    """Add sector-mean monthly feature values for one feature group."""
    if features is None or features.empty:
        return panel
    feature_columns = [column for column in features.columns if column.startswith(f"{prefix}_")]
    feature_columns = [column for column in feature_columns if not column.endswith("_has_full_history")]
    feature_columns = [column for column in feature_columns if not column.endswith("_has_full_valuation")]
    if not feature_columns:
        return panel

    labels = metadata.loc[:, ["ticker", "gics_sector"]].dropna().copy()
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    frame = features.loc[:, ["ticker", "date", *feature_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.merge(labels.drop_duplicates("ticker", keep="last"), on="ticker", how="inner")
    sector_values = frame.groupby(["date", "gics_sector"], as_index=False)[feature_columns].mean()
    return panel.merge(sector_values, on=["date", "gics_sector"], how="left")


def future_excess_return(sector_returns: pd.Series | None, market_returns: pd.Series) -> float:
    """Return future sector cumulative return minus market cumulative return."""
    if sector_returns is None:
        return float("nan")
    frame = pd.concat([sector_returns.rename("sector"), market_returns.rename("market")], axis=1).dropna()
    if frame.empty:
        return float("nan")
    sector_total = cumulative_return(frame["sector"])
    market_total = cumulative_return(frame["market"])
    return float(sector_total - market_total)


def cumulative_return(values: pd.Series) -> float:
    """Compound a return series."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.prod(1.0 + clean.to_numpy()) - 1.0)


def usable_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Select model inputs with enough observed data to be useful."""
    candidates = [
        column
        for column in panel.columns
        if column in PRICE_FEATURE_COLUMNS or column in DEFAULT_FUNDAMENTAL_COLUMNS
    ]
    usable = []
    for column in candidates:
        values = pd.to_numeric(panel[column], errors="coerce")
        if values.notna().mean() >= 0.15 and values.nunique(dropna=True) > 1:
            usable.append(column)
    return usable


def walk_forward_ridge_predictions(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    min_train_months: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each month using only completed prior-horizon outcomes."""
    if not feature_columns:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    dates = sorted(panel["date"].dropna().unique())
    for date in dates:
        current_date = pd.Timestamp(date)
        train = panel[(panel["target_end_date"] < current_date) & panel["future_excess_return"].notna()].copy()
        if train["date"].nunique() < int(min_train_months):
            continue
        current = panel[panel["date"] == current_date].copy()
        if current.empty:
            continue
        model, means, scales, fitted_columns = fit_ridge(train, feature_columns, ridge_alpha)
        current["predicted_excess_return"] = predict_with_standardization(current, fitted_columns, means, scales, model)
        current["prediction_rank"] = current["predicted_excess_return"].rank(ascending=False, method="first")
        rows.append(current)
        for column, coefficient in zip(fitted_columns, model.coef_, strict=True):
            coefficient_rows.append(
                {
                    "date": current_date,
                    "feature": column,
                    "coefficient": float(coefficient),
                }
            )

    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients


def latest_sector_scores(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    min_train_months: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    """Fit on all completed outcomes and score the latest available sector rows."""
    if not feature_columns or panel.empty:
        return pd.DataFrame()
    latest_date = pd.Timestamp(panel["date"].max())
    train = panel[(panel["target_end_date"] < latest_date) & panel["future_excess_return"].notna()].copy()
    if train["date"].nunique() < int(min_train_months):
        return pd.DataFrame()
    latest = panel[panel["date"] == latest_date].copy()
    model, means, scales, fitted_columns = fit_ridge(train, feature_columns, ridge_alpha)
    latest["predicted_excess_return"] = predict_with_standardization(latest, fitted_columns, means, scales, model)
    latest["prediction_rank"] = latest["predicted_excess_return"].rank(ascending=False, method="first")
    latest["score_z"] = zscore(latest["predicted_excess_return"])
    return latest.sort_values("prediction_rank").reset_index(drop=True)


def fit_ridge(
    train: pd.DataFrame,
    feature_columns: list[str],
    ridge_alpha: float,
) -> tuple[Ridge, pd.Series, pd.Series, list[str]]:
    """Fit a standardized ridge model."""
    fitted_columns = [
        column
        for column in feature_columns
        if pd.to_numeric(train[column], errors="coerce").notna().mean() >= 0.15
    ]
    if not fitted_columns:
        raise ValueError("No usable sector outlook feature columns were available for training.")
    x = train.loc[:, fitted_columns].apply(pd.to_numeric, errors="coerce")
    means = x.mean()
    scales = x.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    standardized = ((x.fillna(means) - means) / scales).fillna(0.0)
    y = pd.to_numeric(train["future_excess_return"], errors="coerce")
    model = Ridge(alpha=float(ridge_alpha))
    model.fit(standardized.to_numpy(), y.to_numpy())
    return model, means, scales, fitted_columns


def predict_with_standardization(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    scales: pd.Series,
    model: Ridge,
) -> np.ndarray:
    """Predict with training means/scales."""
    x = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    standardized = ((x.fillna(means) - means) / scales).fillna(0.0)
    return model.predict(standardized.to_numpy())


def sector_prediction_metrics(predictions: pd.DataFrame, horizon_days: int) -> dict[str, float | int | str]:
    """Summarize walk-forward sector prediction quality."""
    if predictions.empty:
        return {
            "horizon_days": int(horizon_days),
            "n_prediction_rows": 0,
            "n_prediction_dates": 0,
        }
    dated = sector_backtest_by_date(predictions)
    if dated.empty:
        return {
            "horizon_days": int(horizon_days),
            "n_prediction_rows": int(len(predictions)),
            "n_prediction_dates": 0,
        }
    return {
        "horizon_days": int(horizon_days),
        "n_prediction_rows": int(len(predictions)),
        "n_prediction_dates": int(dated["date"].nunique()),
        "mean_rank_ic": float(dated["rank_ic"].mean()),
        "median_rank_ic": float(dated["rank_ic"].median()),
        "mean_top_minus_bottom": float(dated["top_minus_bottom"].mean()),
        "mean_top_bucket_excess": float(dated["top_bucket_excess"].mean()),
        "top_sector_hit_rate": float(dated["top_sector_hit"].mean()),
        "first_prediction_date": dated["date"].min().strftime("%Y-%m-%d"),
        "latest_prediction_date": dated["date"].max().strftime("%Y-%m-%d"),
    }


def sector_backtest_by_date(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return date-level rank and top-minus-bottom diagnostics."""
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for date, group in predictions.groupby("date", sort=True):
        valid = group.dropna(subset=["predicted_excess_return", "future_excess_return"])
        if len(valid) < 3:
            continue
        rank_ic = valid["predicted_excess_return"].corr(valid["future_excess_return"], method="spearman")
        top = valid.nsmallest(3, "prediction_rank")["future_excess_return"].mean()
        bottom = valid.nlargest(3, "prediction_rank")["future_excess_return"].mean()
        best_sector = valid.sort_values("predicted_excess_return", ascending=False).iloc[0]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "top_minus_bottom": float(top - bottom),
                "top_bucket_excess": float(top),
                "top_sector_hit": float(best_sector["future_excess_return"] > 0.0),
            }
        )
    return pd.DataFrame(rows)


def zscore(values: pd.Series) -> pd.Series:
    """Return cross-sectional z-scores."""
    numeric = pd.to_numeric(values, errors="coerce")
    scale = numeric.std(ddof=0)
    if pd.isna(scale) or scale == 0.0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / scale
