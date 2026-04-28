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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRICE_FEATURE_COLUMNS = [
    "sector_excess_momentum_21d",
    "sector_excess_momentum_63d",
    "sector_excess_momentum_126d",
    "sector_excess_volatility_63d",
    "sector_excess_hit_rate_63d",
]
GROUP_EMBEDDING_PREFIX = "group_embedding_"

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

SECTOR_MODEL_LABELS = {
    "ridge": "Ridge regression",
    "elastic_net": "Elastic Net",
    "huber": "Huber robust regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
}
THEME_ASSIGNMENT_LABELS = {
    "gics": "GICS sectors",
    "soft": "Mixed soft loadings",
    "hard_top1": "Hard top-1 theme",
    "auto": "Auto by view",
}
HARD_TOP1_DEFAULT_VIEWS = {"behavioral"}

MAX_ABS_DAILY_RETURN = 1.0


@dataclass(frozen=True)
class SectorOutlookResult:
    """Backtest artifacts for sector-relative predictions."""

    panel: pd.DataFrame
    predictions: pd.DataFrame
    latest: pd.DataFrame
    metrics: dict[str, float | int | str]
    coefficients: pd.DataFrame


@dataclass(frozen=True)
class FittedSectorModel:
    """A fitted sector model plus the preprocessing needed for prediction."""

    model_type: str
    model: object
    feature_columns: list[str]
    means: pd.Series | None = None
    scales: pd.Series | None = None


def sector_outlook_backtest(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    valuation: pd.DataFrame | None = None,
    growth: pd.DataFrame | None = None,
    group_loadings: pd.DataFrame | None = None,
    group_embeddings: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    group_mode: str = "gics",
    group_view: str | None = None,
    horizon_days: int = 63,
    min_train_months: int = 36,
    ridge_alpha: float = 10.0,
    model_type: str = "ridge",
    membership_mode: str = "current",
    theme_assignment: str = "soft",
    include_embedding_features: bool = False,
) -> SectorOutlookResult:
    """Fit a walk-forward sector excess-return model and return diagnostics."""
    group_mode = normalized_group_mode(group_mode)
    membership_mode = normalized_membership_mode(membership_mode)
    resolved_theme_assignment = "gics"
    effective_group_loadings = group_loadings
    if group_mode == "theme":
        resolved_theme_assignment = normalized_theme_assignment_strategy(theme_assignment, group_view)
        effective_group_loadings = apply_theme_assignment_strategy(group_loadings, resolved_theme_assignment)
    group_returns, market_returns = group_and_market_returns(
        prices,
        metadata,
        group_loadings=effective_group_loadings,
        group_mode=group_mode,
        membership_mode=membership_mode,
        membership=membership,
    )
    month_ends = observed_month_ends(group_returns)
    panel = build_sector_feature_panel(
        group_returns,
        market_returns,
        month_ends,
        metadata,
        valuation=valuation,
        growth=growth,
        group_loadings=effective_group_loadings,
        group_embeddings=group_embeddings if include_embedding_features else None,
        membership=membership,
        group_mode=group_mode,
        group_view=group_view,
        membership_mode=membership_mode,
        horizon_days=horizon_days,
    )
    feature_columns = usable_feature_columns(panel)
    model_type = normalized_model_type(model_type)
    predictions, coefficients = walk_forward_model_predictions(
        panel,
        feature_columns,
        min_train_months=min_train_months,
        ridge_alpha=ridge_alpha,
        model_type=model_type,
    )
    latest = latest_sector_scores(
        panel,
        feature_columns,
        min_train_months=min_train_months,
        ridge_alpha=ridge_alpha,
        model_type=model_type,
    )
    metrics = sector_prediction_metrics(predictions, horizon_days)
    metrics["model_type"] = model_type
    metrics["group_mode"] = group_mode
    metrics["membership_mode"] = membership_mode
    metrics["theme_assignment"] = resolved_theme_assignment
    metrics["include_embedding_features"] = bool(include_embedding_features)
    if group_view:
        metrics["group_view"] = str(group_view)
    return SectorOutlookResult(
        panel=panel,
        predictions=predictions,
        latest=latest,
        metrics=metrics,
        coefficients=coefficients,
    )


def sector_and_market_returns(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return equal-weight sector returns and equal-weight market returns."""
    all_returns = daily_price_returns(
        prices,
        metadata,
        membership_mode=membership_mode,
        membership=membership,
    )
    labels = ticker_label_frame(metadata, membership=membership).dropna(subset=["gics_sector"])
    if labels.empty:
        raise ValueError("Metadata or membership must include gics_sector.")
    returns = all_returns.merge(labels, on="ticker", how="inner")
    returns = returns.dropna(subset=["return", "gics_sector"])
    sector = (
        returns.groupby(["date", "gics_sector"], as_index=False)["return"]
        .mean()
        .pivot(index="date", columns="gics_sector", values="return")
        .sort_index()
    )
    market = all_returns.groupby("date")["return"].mean().sort_index()
    return sector, market


def daily_price_returns(
    prices: pd.DataFrame,
    metadata: pd.DataFrame | None,
    *,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return cleaned daily ticker returns after applying membership rules."""
    required = {"ticker", "date", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Prices are missing required columns: {sorted(missing)}")

    metadata = metadata if metadata is not None else pd.DataFrame()
    frame = prices.loc[:, ["ticker", "date", "adj_close"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = filter_frame_by_membership(frame, metadata, membership_mode, membership=membership)
    frame = frame.sort_values(["ticker", "date"])
    frame["return"] = frame.groupby("ticker")["adj_close"].pct_change()
    frame.loc[frame["return"].abs() > MAX_ABS_DAILY_RETURN, "return"] = np.nan
    return frame.dropna(subset=["return"]).loc[:, ["ticker", "date", "return"]].copy()


def group_and_market_returns(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    group_loadings: pd.DataFrame | None,
    group_mode: str,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return daily return series for either GICS sectors or learned themes."""
    group_mode = normalized_group_mode(group_mode)
    if group_mode == "gics":
        return sector_and_market_returns(prices, metadata, membership_mode, membership=membership)
    if group_loadings is None or group_loadings.empty:
        raise ValueError("Learned-theme outlook requires a loadings parquet from an experiment view.")
    return theme_and_market_returns(
        prices,
        group_loadings,
        metadata=metadata,
        membership_mode=membership_mode,
        membership=membership,
    )


def normalized_group_mode(group_mode: str) -> str:
    """Validate and normalize the grouping mode."""
    key = str(group_mode).strip().lower()
    aliases = {
        "sector": "gics",
        "sectors": "gics",
        "gics_sector": "gics",
        "gics": "gics",
        "theme": "theme",
        "themes": "theme",
        "cluster": "theme",
        "clusters": "theme",
        "learned": "theme",
    }
    if key not in aliases:
        raise ValueError("group_mode must be 'gics' or 'theme'.")
    return aliases[key]


def default_theme_assignment_strategy(group_view: str | None) -> str:
    """Return the assignment strategy supported by current empirical results."""
    view = str(group_view or "").strip().lower()
    return "hard_top1" if view in HARD_TOP1_DEFAULT_VIEWS else "soft"


def normalized_theme_assignment_strategy(theme_assignment: str, group_view: str | None = None) -> str:
    """Validate the learned-theme assignment strategy."""
    key = str(theme_assignment or "soft").strip().lower().replace("-", "_")
    aliases = {
        "mixed": "soft",
        "soft_loadings": "soft",
        "hard": "hard_top1",
        "top1": "hard_top1",
        "winner_take_all": "hard_top1",
        "automatic": "auto",
    }
    key = aliases.get(key, key)
    if key == "auto":
        return default_theme_assignment_strategy(group_view)
    if key not in {"soft", "hard_top1"}:
        raise ValueError("theme_assignment must be 'soft', 'hard_top1', or 'auto'.")
    return key


def apply_theme_assignment_strategy(loadings: pd.DataFrame | None, theme_assignment: str) -> pd.DataFrame | None:
    """Return loadings transformed to soft or hard theme membership."""
    if loadings is None:
        return None
    strategy = normalized_theme_assignment_strategy(theme_assignment)
    result = loadings.copy()
    if strategy == "soft":
        return result

    theme_columns = [column for column in result.columns if column.startswith("theme_")]
    if not theme_columns:
        raise ValueError("Learned-theme loadings must contain theme_* columns.")
    weights = result.loc[:, theme_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    valid_rows = weights.notna().any(axis=1).to_numpy()
    matrix = weights.fillna(-np.inf).to_numpy()
    winners = np.argmax(matrix, axis=1)
    hard = np.zeros_like(matrix, dtype=float)
    row_numbers = np.arange(len(matrix))
    hard[row_numbers[valid_rows], winners[valid_rows]] = 1.0
    result.loc[:, theme_columns] = hard
    return result


def normalized_membership_mode(membership_mode: str) -> str:
    """Validate and normalize how historical index membership is approximated."""
    key = str(membership_mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "current": "current",
        "current_roster": "current",
        "none": "current",
        "date_added": "date_added",
        "date_added_filter": "date_added",
        "pit": "date_added",
        "point_in_time": "date_added",
        "historical": "historical",
        "historical_constituents": "historical",
        "historical_members": "historical",
        "point_in_time_constituents": "historical",
        "full_history": "historical",
    }
    if key not in aliases:
        raise ValueError("membership_mode must be 'current', 'date_added', or 'historical'.")
    return aliases[key]


def filter_frame_by_membership(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    membership_mode: str,
    *,
    date_column: str = "date",
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop ticker-date rows that were not S&P 500 members at the time.

    ``date_added`` mode removes the strongest current-roster look-ahead effect,
    but still cannot add deleted historical constituents. ``historical`` mode
    uses a constituent interval table and can represent both additions and
    deletions when the corresponding prices and labels exist.
    """
    membership_mode = normalized_membership_mode(membership_mode)
    if membership_mode == "current":
        return frame.copy()
    if frame.empty:
        return frame.copy()
    if membership_mode == "historical":
        return filter_frame_by_membership_intervals(frame, membership, date_column=date_column)

    required = {"ticker", "date_added"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"date_added membership mode requires metadata columns: {sorted(missing)}")

    membership = metadata.loc[:, ["ticker", "date_added"]].copy()
    membership["ticker"] = membership["ticker"].astype(str).str.upper()
    membership["date_added"] = pd.to_datetime(membership["date_added"], errors="coerce")
    membership = membership.dropna(subset=["date_added"]).drop_duplicates("ticker", keep="last")

    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result[date_column] = pd.to_datetime(result[date_column])
    result = result.merge(membership, on="ticker", how="left")
    result = result[result["date_added"].notna() & (result[date_column] >= result["date_added"])].copy()
    return result.drop(columns=["date_added"])


def filter_frame_by_membership_intervals(
    frame: pd.DataFrame,
    membership: pd.DataFrame | None,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return rows whose ticker-date falls inside a historical S&P interval."""
    if membership is None or membership.empty:
        raise ValueError(
            "historical membership mode requires data/processed/metadata/sp500_membership_history.parquet."
        )
    required = {"ticker", "start_date", "end_date"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Historical membership table is missing columns: {sorted(missing)}")
    if "ticker" not in frame.columns or date_column not in frame.columns:
        raise ValueError(f"Frame must include ticker and {date_column!r} columns for membership filtering.")

    intervals = membership.loc[:, ["ticker", "start_date", "end_date"]].copy()
    intervals["ticker"] = intervals["ticker"].astype(str).str.upper()
    intervals["_membership_start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["_membership_end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    intervals = intervals.drop(columns=["start_date", "end_date"]).dropna(subset=["ticker"])
    intervals = intervals.drop_duplicates()
    if intervals.empty:
        return frame.iloc[0:0].copy()

    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result[date_column] = pd.to_datetime(result[date_column])
    row_id = "__membership_filter_row_id__"
    while row_id in result.columns:
        row_id = f"_{row_id}"
    result[row_id] = np.arange(len(result))

    merged = result.merge(intervals, on="ticker", how="inner")
    if merged.empty:
        return frame.iloc[0:0].copy()
    dates = pd.to_datetime(merged[date_column])
    start_ok = merged["_membership_start_date"].isna() | (dates >= merged["_membership_start_date"])
    end_ok = merged["_membership_end_date"].isna() | (dates < merged["_membership_end_date"])
    filtered = merged[start_ok & end_ok].copy()
    filtered = filtered.sort_values(row_id).drop_duplicates(row_id, keep="first")
    filtered = filtered.drop(columns=[row_id, "_membership_start_date", "_membership_end_date"])
    return filtered.loc[:, frame.columns]


def ticker_label_frame(metadata: pd.DataFrame, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one ticker-to-GICS label table, preferring current metadata labels."""
    pieces = []
    label_columns = ["ticker", "gics_sector", "gics_sub_industry", "company_name"]
    if metadata is not None and not metadata.empty and "ticker" in metadata.columns:
        columns = [column for column in label_columns if column in metadata.columns]
        if "gics_sector" in columns:
            current = metadata.loc[:, columns].copy()
            current["ticker"] = current["ticker"].astype(str).str.upper()
            current["_label_priority"] = 0
            pieces.append(current)
    if membership is not None and not membership.empty and "ticker" in membership.columns:
        columns = [column for column in label_columns if column in membership.columns]
        if "gics_sector" in columns:
            historical = membership.loc[:, columns].copy()
            historical["ticker"] = historical["ticker"].astype(str).str.upper()
            historical["_label_priority"] = 1
            pieces.append(historical)
    if not pieces:
        return pd.DataFrame(columns=label_columns)

    labels = pd.concat(pieces, ignore_index=True, sort=False)
    labels = labels.dropna(subset=["ticker"])
    labels = labels.sort_values(["ticker", "_label_priority"])
    labels = labels.drop_duplicates("ticker", keep="first")
    return labels.drop(columns=["_label_priority"])


def theme_and_market_returns(
    prices: pd.DataFrame,
    loadings: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return loading-weighted learned-theme returns and equal-weight market returns.

    Theme memberships may be soft GMM loadings or a hard top-1 transform. For
    each ticker-day return, we attach the latest available ticker theme loading
    as of that date, then compute each theme's return as a loading-weighted
    average of stock returns.
    """
    theme_columns = [column for column in loadings.columns if column.startswith("theme_")]
    if not theme_columns:
        raise ValueError("Learned-theme loadings must contain theme_* columns.")

    if normalized_membership_mode(membership_mode) != "current":
        if metadata is None:
            raise ValueError("Non-current membership modes require metadata for learned-theme returns.")
    returns = daily_price_returns(
        prices,
        metadata,
        membership_mode=membership_mode,
        membership=membership,
    )
    market = returns.groupby("date")["return"].mean().sort_index()

    loadings = loadings.loc[:, ["ticker", "date", *theme_columns]].copy()
    loadings["ticker"] = loadings["ticker"].astype(str).str.upper()
    loadings["date"] = pd.to_datetime(loadings["date"])
    loadings = loadings.sort_values(["ticker", "date"])

    pieces = []
    loading_tickers = set(loadings["ticker"].unique())
    for ticker, ticker_returns in returns.groupby("ticker", sort=False):
        if ticker not in loading_tickers:
            continue
        ticker_loadings = loadings[loadings["ticker"].eq(ticker)].drop(columns=["ticker"])
        if ticker_loadings.empty:
            continue
        merged = pd.merge_asof(
            ticker_returns.sort_values("date"),
            ticker_loadings.sort_values("date"),
            on="date",
            direction="backward",
        )
        merged = merged.dropna(subset=theme_columns, how="all")
        if not merged.empty:
            pieces.append(merged)
    if not pieces:
        raise ValueError("No price rows could be matched to learned-theme loadings.")

    merged_returns = pd.concat(pieces, ignore_index=True)
    weights = merged_returns.loc[:, theme_columns].astype(float)
    returns_vector = pd.to_numeric(merged_returns["return"], errors="coerce")
    weighted_returns = weights.mul(returns_vector, axis=0)
    valid_weights = weights.where(returns_vector.notna())
    numerator = weighted_returns.groupby(merged_returns["date"]).sum(min_count=1)
    denominator = valid_weights.groupby(merged_returns["date"]).sum(min_count=1).replace(0.0, np.nan)
    theme_returns = numerator / denominator
    theme_returns = theme_returns.sort_index()
    return theme_returns, market


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
    group_loadings: pd.DataFrame | None = None,
    group_embeddings: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    group_mode: str = "gics",
    group_view: str | None = None,
    membership_mode: str = "current",
    horizon_days: int,
) -> pd.DataFrame:
    """Build monthly sector-date rows with features and future excess returns."""
    group_mode = normalized_group_mode(group_mode)
    membership_mode = normalized_membership_mode(membership_mode)
    excess = sector_returns.sub(market_returns, axis=0)
    rows: list[dict[str, object]] = []
    for date in month_ends:
        history = excess.loc[excess.index <= date]
        future_sector = sector_returns.loc[sector_returns.index > date].head(int(horizon_days))
        future_market = market_returns.loc[market_returns.index > date].head(int(horizon_days))
        for sector in sector_returns.columns:
            if pd.isna(sector_returns.loc[date, sector]):
                continue
            (
                future_group_return,
                future_market_return,
                future_excess_return,
                future_days,
                target_end_date,
                is_complete,
            ) = future_return_summary(
                future_sector.get(sector),
                future_market,
                int(horizon_days),
            )
            row = {
                "date": pd.Timestamp(date),
                "gics_sector": sector,
                "group_id": sector,
                "group_label": sector,
                "group_mode": group_mode,
                "group_view": group_view or ("gics" if group_mode == "gics" else "theme"),
                "target_end_date": target_end_date,
                "future_days_available": int(future_days),
                "is_horizon_complete": bool(is_complete),
                "future_group_return": future_group_return,
                "future_market_return": future_market_return,
                "future_excess_return": future_excess_return,
            }
            for window in [21, 63, 126]:
                row[f"sector_excess_momentum_{window}d"] = cumulative_return(history[sector].tail(window))
            recent = history[sector].tail(63).dropna()
            row["sector_excess_volatility_63d"] = float(recent.std(ddof=1) * np.sqrt(252.0)) if len(recent) > 2 else np.nan
            row["sector_excess_hit_rate_63d"] = float((recent > 0.0).mean()) if len(recent) else np.nan
            rows.append(row)

    panel = pd.DataFrame(rows)
    if group_mode == "gics":
        panel = attach_sector_fundamentals(
            panel,
            metadata,
            valuation,
            "valuation",
            membership_mode=membership_mode,
            membership=membership,
        )
        panel = attach_sector_fundamentals(
            panel,
            metadata,
            growth,
            "growth",
            membership_mode=membership_mode,
            membership=membership,
        )
        panel = attach_sector_embeddings(
            panel,
            metadata,
            group_embeddings,
            membership_mode=membership_mode,
            membership=membership,
        )
    else:
        if group_loadings is None or group_loadings.empty:
            raise ValueError("Learned-theme feature panel requires non-empty group_loadings.")
        panel = attach_theme_fundamentals(
            panel,
            group_loadings,
            valuation,
            "valuation",
            metadata=metadata,
            membership_mode=membership_mode,
            membership=membership,
        )
        panel = attach_theme_fundamentals(
            panel,
            group_loadings,
            growth,
            "growth",
            metadata=metadata,
            membership_mode=membership_mode,
            membership=membership,
        )
        panel = attach_theme_embeddings(
            panel,
            group_loadings,
            group_embeddings,
            metadata=metadata,
            membership_mode=membership_mode,
            membership=membership,
        )
    return panel.sort_values(["date", "gics_sector"]).reset_index(drop=True)


def attach_sector_fundamentals(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    features: pd.DataFrame | None,
    prefix: str,
    *,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add sector-mean monthly feature values for one feature group."""
    if features is None or features.empty:
        return panel
    feature_columns = [column for column in features.columns if column.startswith(f"{prefix}_")]
    feature_columns = [column for column in feature_columns if not column.endswith("_has_full_history")]
    feature_columns = [column for column in feature_columns if not column.endswith("_has_full_valuation")]
    if not feature_columns:
        return panel

    labels = ticker_label_frame(metadata, membership=membership).dropna(subset=["gics_sector"])
    frame = features.loc[:, ["ticker", "date", *feature_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = filter_frame_by_membership(frame, metadata, membership_mode, membership=membership)
    frame = frame.merge(labels.drop_duplicates("ticker", keep="last"), on="ticker", how="inner")
    sector_values = frame.groupby(["date", "gics_sector"], as_index=False)[feature_columns].mean()
    return panel.merge(sector_values, on=["date", "gics_sector"], how="left")


def attach_sector_embeddings(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    embeddings: pd.DataFrame | None,
    *,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add sector-mean embedding coordinates as direct prediction features."""
    if embeddings is None or embeddings.empty or panel.empty:
        return panel
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not embedding_columns:
        return panel

    labels = ticker_label_frame(metadata, membership=membership).dropna(subset=["gics_sector"])
    frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = filter_frame_by_membership(frame, metadata, membership_mode, membership=membership)
    frame = frame.merge(labels.drop_duplicates("ticker", keep="last"), on="ticker", how="inner")
    rename = {column: embedding_feature_name(index) for index, column in enumerate(embedding_columns)}
    frame = frame.rename(columns=rename)
    output_columns = list(rename.values())
    values = frame.groupby(["date", "gics_sector"], as_index=False)[output_columns].mean()
    return panel.merge(values, on=["date", "gics_sector"], how="left")


def attach_theme_fundamentals(
    panel: pd.DataFrame,
    loadings: pd.DataFrame,
    features: pd.DataFrame | None,
    prefix: str,
    *,
    metadata: pd.DataFrame | None = None,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add loading-weighted feature averages for learned themes."""
    if features is None or features.empty or panel.empty:
        return panel
    feature_columns = [column for column in features.columns if column.startswith(f"{prefix}_")]
    feature_columns = [column for column in feature_columns if not column.endswith("_has_full_history")]
    feature_columns = [column for column in feature_columns if not column.endswith("_has_full_valuation")]
    if not feature_columns:
        return panel
    theme_columns = [column for column in loadings.columns if column.startswith("theme_")]
    if not theme_columns:
        return panel

    feature_frame = features.loc[:, ["ticker", "date", *feature_columns]].copy()
    feature_frame["ticker"] = feature_frame["ticker"].astype(str).str.upper()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"])
    feature_frame = feature_frame.sort_values(["ticker", "date"])

    loading_frame = loadings.loc[:, ["ticker", "date", *theme_columns]].copy()
    loading_frame["ticker"] = loading_frame["ticker"].astype(str).str.upper()
    loading_frame["date"] = pd.to_datetime(loading_frame["date"])
    loading_frame = loading_frame.sort_values(["ticker", "date"])

    # These feature producers and GMM loadings are both monthly point-in-time
    # panels. Exact-date joins keep the dashboard fast; the upstream feature
    # producers are responsible for carrying values forward as-of each month.
    merged = feature_frame.merge(loading_frame, on=["ticker", "date"], how="inner")
    if merged.empty:
        return panel
    if normalized_membership_mode(membership_mode) != "current":
        if metadata is None:
            raise ValueError("Non-current membership modes require metadata for learned-theme fundamentals.")
        merged = filter_frame_by_membership(merged, metadata, membership_mode, membership=membership)
        if merged.empty:
            return panel

    weights = merged.loc[:, theme_columns].astype(float)
    pieces = []
    for feature in feature_columns:
        values = pd.to_numeric(merged[feature], errors="coerce")
        valid_weights = weights.where(values.notna())
        numerator = weights.mul(values, axis=0).groupby(merged["date"]).sum(min_count=1)
        denominator = valid_weights.groupby(merged["date"]).sum(min_count=1).replace(0.0, np.nan)
        averaged = numerator / denominator
        long = averaged.reset_index().melt(
            id_vars="date",
            var_name="gics_sector",
            value_name=feature,
        )
        pieces.append(long)
    theme_values = pieces[0]
    for piece in pieces[1:]:
        theme_values = theme_values.merge(piece, on=["date", "gics_sector"], how="outer")
    return panel.merge(theme_values, on=["date", "gics_sector"], how="left")


def attach_theme_embeddings(
    panel: pd.DataFrame,
    loadings: pd.DataFrame,
    embeddings: pd.DataFrame | None,
    *,
    metadata: pd.DataFrame | None = None,
    membership_mode: str = "current",
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add loading-weighted embedding coordinates for learned themes."""
    if embeddings is None or embeddings.empty or panel.empty:
        return panel
    theme_columns = [column for column in loadings.columns if column.startswith("theme_")]
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not theme_columns or not embedding_columns:
        return panel

    loading_frame = loadings.loc[:, ["ticker", "date", *theme_columns]].copy()
    loading_frame["ticker"] = loading_frame["ticker"].astype(str).str.upper()
    loading_frame["date"] = pd.to_datetime(loading_frame["date"])

    embedding_frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].copy()
    embedding_frame["ticker"] = embedding_frame["ticker"].astype(str).str.upper()
    embedding_frame["date"] = pd.to_datetime(embedding_frame["date"])

    merged = embedding_frame.merge(loading_frame, on=["ticker", "date"], how="inner")
    if merged.empty:
        return panel
    if normalized_membership_mode(membership_mode) != "current":
        if metadata is None:
            raise ValueError("Non-current membership modes require metadata for learned-theme embedding features.")
        merged = filter_frame_by_membership(merged, metadata, membership_mode, membership=membership)
        if merged.empty:
            return panel

    weights = merged.loc[:, theme_columns].astype(float)
    pieces = []
    for index, embedding_column in enumerate(embedding_columns):
        output_column = embedding_feature_name(index)
        values = pd.to_numeric(merged[embedding_column], errors="coerce")
        valid_weights = weights.where(values.notna())
        numerator = weights.mul(values, axis=0).groupby(merged["date"]).sum(min_count=1)
        denominator = valid_weights.groupby(merged["date"]).sum(min_count=1).replace(0.0, np.nan)
        averaged = numerator / denominator
        long = averaged.reset_index().melt(
            id_vars="date",
            var_name="gics_sector",
            value_name=output_column,
        )
        pieces.append(long)
    if not pieces:
        return panel
    theme_embeddings = pieces[0]
    for piece in pieces[1:]:
        theme_embeddings = theme_embeddings.merge(piece, on=["date", "gics_sector"], how="outer")
    return panel.merge(theme_embeddings, on=["date", "gics_sector"], how="left")


def embedding_feature_name(index: int) -> str:
    """Return stable predictor feature names for aggregated embedding coordinates."""
    return f"{GROUP_EMBEDDING_PREFIX}{int(index)}"


def latest_ticker_snapshot(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Return each ticker's latest row available on or before a date."""
    snapshot = frame[frame["date"] <= as_of_date].copy()
    if snapshot.empty:
        return snapshot
    return snapshot.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).drop(columns=["date"])


def future_excess_return(sector_returns: pd.Series | None, market_returns: pd.Series) -> float:
    """Return future sector cumulative return minus market cumulative return."""
    value, _, _, _ = future_excess_return_summary(sector_returns, market_returns, min_days=1)
    return value


def future_excess_return_summary(
    sector_returns: pd.Series | None,
    market_returns: pd.Series,
    min_days: int,
) -> tuple[float, int, pd.Timestamp | pd.NaT, bool]:
    """Return future excess return plus completeness diagnostics.

    Backtests should only score rows where the full forward horizon completed.
    Current/latest rows may still receive model predictions, but their realized
    future return is left blank until enough trading days are available.
    """
    _, _, excess_return, days_available, target_end_date, is_complete = future_return_summary(
        sector_returns,
        market_returns,
        min_days,
    )
    return excess_return, days_available, target_end_date, is_complete


def future_return_summary(
    sector_returns: pd.Series | None,
    market_returns: pd.Series,
    min_days: int,
) -> tuple[float, float, float, int, pd.Timestamp | pd.NaT, bool]:
    """Return future group, market, and excess returns over a completed horizon."""
    if sector_returns is None:
        return float("nan"), float("nan"), float("nan"), 0, pd.NaT, False
    frame = pd.concat([sector_returns.rename("sector"), market_returns.rename("market")], axis=1).dropna()
    if frame.empty:
        return float("nan"), float("nan"), float("nan"), 0, pd.NaT, False
    days_available = int(len(frame))
    target_end_date = pd.Timestamp(frame.index.max())
    is_complete = days_available >= int(min_days)
    if not is_complete:
        return float("nan"), float("nan"), float("nan"), days_available, target_end_date, False
    sector_total = cumulative_return(frame["sector"])
    market_total = cumulative_return(frame["market"])
    return (
        float(sector_total),
        float(market_total),
        float(sector_total - market_total),
        days_available,
        target_end_date,
        True,
    )


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
        if column in PRICE_FEATURE_COLUMNS
        or column in DEFAULT_FUNDAMENTAL_COLUMNS
        or column.startswith(GROUP_EMBEDDING_PREFIX)
    ]
    usable = []
    for column in candidates:
        values = pd.to_numeric(panel[column], errors="coerce")
        if values.notna().mean() >= 0.15 and values.nunique(dropna=True) > 1:
            usable.append(column)
    return usable


def normalized_model_type(model_type: str) -> str:
    """Validate and normalize a sector model key."""
    key = str(model_type).strip().lower()
    if key not in SECTOR_MODEL_LABELS:
        raise ValueError(f"Unknown sector model {model_type!r}. Expected one of {sorted(SECTOR_MODEL_LABELS)}.")
    return key


def walk_forward_ridge_predictions(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    min_train_months: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each month using only completed prior-horizon outcomes."""
    return walk_forward_model_predictions(
        panel,
        feature_columns,
        min_train_months=min_train_months,
        ridge_alpha=ridge_alpha,
        model_type="ridge",
    )


def walk_forward_model_predictions(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    min_train_months: int,
    ridge_alpha: float,
    model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each month using one registered model and completed prior outcomes only."""
    if not feature_columns:
        return pd.DataFrame(), pd.DataFrame()

    model_type = normalized_model_type(model_type)
    rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    dates = sorted(panel["date"].dropna().unique())
    for date in dates:
        current_date = pd.Timestamp(date)
        train = panel[
            panel["is_horizon_complete"].fillna(False)
            & (panel["target_end_date"] < current_date)
            & panel["future_excess_return"].notna()
        ].copy()
        if train["date"].nunique() < int(min_train_months):
            continue
        current = panel[panel["date"] == current_date].copy()
        if current.empty:
            continue
        fitted = fit_sector_model(train, feature_columns, model_type, ridge_alpha)
        current["predicted_excess_return"] = predict_sector_model(current, fitted)
        current["prediction_rank"] = current["predicted_excess_return"].rank(ascending=False, method="first")
        current["training_rows"] = int(len(train))
        current["training_dates"] = int(train["date"].nunique())
        current["training_latest_target_end_date"] = pd.Timestamp(train["target_end_date"].max())
        current["model_type"] = model_type
        rows.append(current)
        weights = model_feature_weights(fitted)
        for column, coefficient in weights.items():
            coefficient_rows.append(
                {
                    "date": current_date,
                    "feature": column,
                    "coefficient": float(coefficient),
                    "model_type": model_type,
                    "weight_type": model_weight_type(fitted),
                    "training_rows": int(len(train)),
                    "training_dates": int(train["date"].nunique()),
                    "training_latest_target_end_date": pd.Timestamp(train["target_end_date"].max()),
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
    model_type: str = "ridge",
) -> pd.DataFrame:
    """Fit on all completed outcomes and score the latest available sector rows."""
    if not feature_columns or panel.empty:
        return pd.DataFrame()
    model_type = normalized_model_type(model_type)
    latest_date = pd.Timestamp(panel["date"].max())
    train = panel[
        panel["is_horizon_complete"].fillna(False)
        & (panel["target_end_date"] < latest_date)
        & panel["future_excess_return"].notna()
    ].copy()
    if train["date"].nunique() < int(min_train_months):
        return pd.DataFrame()
    latest = panel[panel["date"] == latest_date].copy()
    fitted = fit_sector_model(train, feature_columns, model_type, ridge_alpha)
    latest["predicted_excess_return"] = predict_sector_model(latest, fitted)
    latest["prediction_rank"] = latest["predicted_excess_return"].rank(ascending=False, method="first")
    latest["score_z"] = zscore(latest["predicted_excess_return"])
    latest["training_rows"] = int(len(train))
    latest["training_dates"] = int(train["date"].nunique())
    latest["training_latest_target_end_date"] = pd.Timestamp(train["target_end_date"].max())
    latest["model_type"] = model_type
    return latest.sort_values("prediction_rank").reset_index(drop=True)


def fit_sector_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    ridge_alpha: float,
) -> FittedSectorModel:
    """Fit one sector predictor with model-specific preprocessing."""
    model_type = normalized_model_type(model_type)
    if model_type == "ridge":
        model, means, scales, fitted_columns = fit_ridge(train, feature_columns, ridge_alpha)
        return FittedSectorModel(
            model_type=model_type,
            model=model,
            feature_columns=fitted_columns,
            means=means,
            scales=scales,
        )

    fitted_columns = [
        column
        for column in feature_columns
        if pd.to_numeric(train[column], errors="coerce").notna().mean() >= 0.15
    ]
    if not fitted_columns:
        raise ValueError("No usable sector outlook feature columns were available for training.")

    estimator = sector_estimator(model_type, ridge_alpha)
    steps = [("imputer", SimpleImputer(strategy="mean"))]
    if model_type in {"elastic_net", "huber"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    model = Pipeline(steps)
    x = train.loc[:, fitted_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train["future_excess_return"], errors="coerce")
    model.fit(x.to_numpy(), y.to_numpy())
    return FittedSectorModel(model_type=model_type, model=model, feature_columns=fitted_columns)


def sector_estimator(model_type: str, ridge_alpha: float) -> object:
    """Return an initialized sklearn estimator for one sector model key."""
    if model_type == "elastic_net":
        return ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=42)
    if model_type == "huber":
        return HuberRegressor(alpha=max(float(ridge_alpha) * 0.001, 0.0001), max_iter=1000)
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=42,
        )
    raise ValueError(f"Unknown sector model {model_type!r}.")


def predict_sector_model(frame: pd.DataFrame, fitted: FittedSectorModel) -> np.ndarray:
    """Predict sector excess return with the fitted model wrapper."""
    if fitted.model_type == "ridge":
        assert fitted.means is not None
        assert fitted.scales is not None
        return predict_with_standardization(frame, fitted.feature_columns, fitted.means, fitted.scales, fitted.model)
    x = frame.loc[:, fitted.feature_columns].apply(pd.to_numeric, errors="coerce")
    return fitted.model.predict(x.to_numpy())


def model_feature_weights(fitted: FittedSectorModel) -> pd.Series:
    """Return linear coefficients or tree importances indexed by feature name."""
    estimator = fitted.model
    if isinstance(fitted.model, Pipeline):
        estimator = fitted.model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        values = np.full(len(fitted.feature_columns), np.nan)
    return pd.Series(values, index=fitted.feature_columns)


def model_weight_type(fitted: FittedSectorModel) -> str:
    """Describe whether the displayed weights are coefficients or importances."""
    estimator = fitted.model
    if isinstance(fitted.model, Pipeline):
        estimator = fitted.model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        return "importance"
    return "coefficient"


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
        if "is_horizon_complete" in group.columns:
            group = group[group["is_horizon_complete"].fillna(False)]
        valid = group.dropna(subset=["predicted_excess_return", "future_excess_return"])
        if len(valid) < 3:
            continue
        rank_ic = valid["predicted_excess_return"].corr(valid["future_excess_return"], method="spearman")
        top = valid.nsmallest(3, "prediction_rank")["future_excess_return"].mean()
        bottom = valid.nlargest(3, "prediction_rank")["future_excess_return"].mean()
        best_sector = valid.sort_values("predicted_excess_return", ascending=False).iloc[0]
        realized_best = valid.sort_values("future_excess_return", ascending=False).iloc[0]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "target_end_date": pd.Timestamp(valid["target_end_date"].max()),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "top_minus_bottom": float(top - bottom),
                "top_bucket_excess": float(top),
                "top_sector_hit": float(best_sector["future_excess_return"] > 0.0),
                "predicted_top_sector": str(best_sector["gics_sector"]),
                "predicted_top_realized_excess": float(best_sector["future_excess_return"]),
                "realized_best_sector": str(realized_best["gics_sector"]),
                "realized_best_excess": float(realized_best["future_excess_return"]),
                "n_sectors": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def simulate_group_rotation(
    predictions: pd.DataFrame,
    *,
    starting_capital: float = 10_000.0,
    top_n: int = 1,
    transaction_cost_bps: float = 0.0,
    benchmark_returns: pd.Series | None = None,
    benchmark_label: str = "S&P 500 benchmark",
) -> pd.DataFrame:
    """Simulate rotating into the highest-ranked predicted groups.

    The simulation uses only completed historical walk-forward predictions.
    To avoid double-counting overlapping forward windows, it rebalances on the
    first available prediction date after the previous holding window ends.
    """
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive.")
    if int(top_n) < 1:
        raise ValueError("top_n must be at least 1.")
    if predictions.empty:
        return pd.DataFrame()

    frame = completed_sector_predictions(predictions)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if "future_group_return" not in frame.columns:
        if {"future_excess_return", "future_market_return"}.issubset(frame.columns):
            frame["future_group_return"] = frame["future_excess_return"] + frame["future_market_return"]
        else:
            raise ValueError("Rotation simulation requires future_group_return or future_excess_return + future_market_return.")
    if "future_market_return" not in frame.columns:
        raise ValueError("Rotation simulation requires future_market_return.")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_end_date"] = pd.to_datetime(frame["target_end_date"])
    frame["future_group_return"] = pd.to_numeric(frame["future_group_return"], errors="coerce")
    frame["future_market_return"] = pd.to_numeric(frame["future_market_return"], errors="coerce")
    frame["predicted_excess_return"] = pd.to_numeric(frame["predicted_excess_return"], errors="coerce")
    frame = frame.dropna(
        subset=[
            "date",
            "target_end_date",
            "prediction_rank",
            "future_group_return",
            "future_market_return",
            "predicted_excess_return",
        ]
    )
    if frame.empty:
        return pd.DataFrame()

    cost_rate = float(transaction_cost_bps) / 10_000.0
    capital = float(starting_capital)
    market_capital = float(starting_capital)
    benchmark_capital = float(starting_capital)
    benchmark_returns = clean_benchmark_returns(benchmark_returns)
    next_rebalance_date = pd.Timestamp.min
    previous_selection: tuple[str, ...] | None = None
    rows: list[dict[str, object]] = []

    for date, date_group in frame.groupby("date", sort=True):
        current_date = pd.Timestamp(date)
        if current_date < next_rebalance_date:
            continue
        candidates = date_group.sort_values(["prediction_rank", "gics_sector"]).head(int(top_n)).copy()
        if candidates.empty:
            continue
        selected_groups = tuple(str(value) for value in candidates["gics_sector"].tolist())
        selected_labels = selected_group_labels(candidates)
        target_end_date = pd.Timestamp(candidates["target_end_date"].max())
        period_group_return = float(candidates["future_group_return"].mean())
        period_market_return = float(candidates["future_market_return"].mean())
        predicted_excess_return = float(candidates["predicted_excess_return"].mean())
        turnover = 1.0 if previous_selection != selected_groups else 0.0
        transaction_cost = cost_rate * turnover
        net_period_return = period_group_return - transaction_cost
        start_capital = float(capital)
        start_market_capital = float(market_capital)
        start_benchmark_capital = float(benchmark_capital)
        period_benchmark_return = period_benchmark_cumulative_return(
            benchmark_returns,
            current_date,
            target_end_date,
        )
        capital *= 1.0 + net_period_return
        market_capital *= 1.0 + period_market_return
        if np.isfinite(period_benchmark_return):
            benchmark_capital *= 1.0 + period_benchmark_return
        rows.append(
            {
                "step": int(len(rows) + 1),
                "date": current_date,
                "target_end_date": target_end_date,
                "selected_group": " / ".join(selected_groups),
                "selected_group_label": " / ".join(selected_labels),
                "n_selected": int(len(candidates)),
                "predicted_excess_return": predicted_excess_return,
                "period_group_return": period_group_return,
                "period_market_return": period_market_return,
                "period_excess_return": period_group_return - period_market_return,
                "transaction_cost": transaction_cost,
                "net_period_return": net_period_return,
                "start_capital": start_capital,
                "start_market_capital": start_market_capital,
                "capital": float(capital),
                "market_capital": float(market_capital),
                "capital_change": float(capital - start_capital),
                "market_capital_change": float(market_capital - start_market_capital),
                "benchmark_label": benchmark_label,
                "period_benchmark_return": float(period_benchmark_return),
                "start_benchmark_capital": start_benchmark_capital,
                "benchmark_capital": float(benchmark_capital) if np.isfinite(period_benchmark_return) else np.nan,
                "benchmark_capital_change": (
                    float(benchmark_capital - start_benchmark_capital)
                    if np.isfinite(period_benchmark_return)
                    else np.nan
                ),
                "turnover": turnover,
            }
        )
        previous_selection = selected_groups
        next_rebalance_date = target_end_date

    return pd.DataFrame(rows)


def clean_benchmark_returns(benchmark_returns: pd.Series | None) -> pd.Series:
    """Return a clean dated benchmark return series."""
    if benchmark_returns is None or len(benchmark_returns) == 0:
        return pd.Series(dtype=float)
    series = pd.Series(benchmark_returns).copy()
    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    return series


def period_benchmark_cumulative_return(
    benchmark_returns: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Compound benchmark returns over one holding period."""
    if benchmark_returns.empty:
        return float("nan")
    period = benchmark_returns.loc[
        (benchmark_returns.index > pd.Timestamp(start_date))
        & (benchmark_returns.index <= pd.Timestamp(end_date))
    ]
    if period.empty:
        return float("nan")
    return cumulative_return(period)


def selected_group_labels(candidates: pd.DataFrame) -> list[str]:
    """Return readable labels for selected rotation groups."""
    label_column = "group_label" if "group_label" in candidates.columns else "gics_sector"
    return [str(value) for value in candidates[label_column].tolist()]


def rotation_simulation_metrics(simulation: pd.DataFrame, starting_capital: float) -> dict[str, float | int | str | None]:
    """Summarize a completed group-rotation capital simulation."""
    if simulation.empty:
        return {
            "n_rebalances": 0,
            "ending_capital": float(starting_capital),
            "market_ending_capital": float(starting_capital),
        }
    frame = simulation.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_end_date"] = pd.to_datetime(frame["target_end_date"])
    period_return_frame = frame.loc[:, ["net_period_return", "period_market_return"]].apply(pd.to_numeric, errors="coerce")
    period_return_frame = period_return_frame.dropna()
    period_returns = period_return_frame["net_period_return"]
    market_returns = period_return_frame["period_market_return"]
    ending_capital = float(frame["capital"].iloc[-1])
    market_ending_capital = float(frame["market_capital"].iloc[-1])
    total_return = ending_capital / float(starting_capital) - 1.0
    market_total_return = market_ending_capital / float(starting_capital) - 1.0
    benchmark_ending_capital = float("nan")
    benchmark_total_return = float("nan")
    excess_total_return_vs_benchmark = float("nan")
    if "benchmark_capital" in frame.columns and frame["benchmark_capital"].notna().any():
        benchmark_ending_capital = float(frame["benchmark_capital"].dropna().iloc[-1])
        benchmark_total_return = benchmark_ending_capital / float(starting_capital) - 1.0
        excess_total_return_vs_benchmark = total_return - benchmark_total_return
    years = max((frame["target_end_date"].max() - frame["date"].min()).days / 365.25, 1e-9)
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    market_annualized_return = (
        (1.0 + market_total_return) ** (1.0 / years) - 1.0 if market_total_return > -1.0 else -1.0
    )
    return {
        "n_rebalances": int(len(frame)),
        "first_rebalance_date": frame["date"].min().strftime("%Y-%m-%d"),
        "latest_exit_date": frame["target_end_date"].max().strftime("%Y-%m-%d"),
        "ending_capital": ending_capital,
        "market_ending_capital": market_ending_capital,
        "benchmark_ending_capital": benchmark_ending_capital,
        "total_return": float(total_return),
        "market_total_return": float(market_total_return),
        "benchmark_total_return": float(benchmark_total_return),
        "excess_total_return": float(total_return - market_total_return),
        "excess_total_return_vs_benchmark": float(excess_total_return_vs_benchmark),
        "annualized_return": float(annualized_return),
        "market_annualized_return": float(market_annualized_return),
        "period_win_rate": float((period_returns > market_returns).mean()) if len(period_return_frame) else float("nan"),
        "average_period_return": float(period_returns.mean()) if len(period_returns) else float("nan"),
        "max_drawdown": float(max_drawdown(frame["capital"])),
    }


def max_drawdown(values: pd.Series) -> float:
    """Return max drawdown for a capital curve."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    running_max = numeric.cummax()
    drawdowns = numeric / running_max - 1.0
    return float(drawdowns.min())


def completed_sector_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return only historical prediction rows whose forward horizon is complete."""
    if predictions.empty:
        return predictions.copy()
    frame = predictions.copy()
    if "is_horizon_complete" in frame.columns:
        frame = frame[frame["is_horizon_complete"].fillna(False)]
    return frame.dropna(subset=["predicted_excess_return", "future_excess_return"]).reset_index(drop=True)


def sector_prediction_audit(predictions: pd.DataFrame) -> dict[str, float | int | str | None]:
    """Summarize whether walk-forward sector predictions are historical and leak-free."""
    if predictions.empty:
        return {
            "n_prediction_rows": 0,
            "n_completed_rows": 0,
            "n_unrealized_rows": 0,
            "n_leakage_violations": 0,
        }
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "target_end_date" in frame.columns:
        frame["target_end_date"] = pd.to_datetime(frame["target_end_date"])
    if "is_horizon_complete" in frame.columns:
        complete_mask = frame["is_horizon_complete"].fillna(False)
    else:
        complete_mask = pd.Series(True, index=frame.index)
    complete_mask = (
        complete_mask
        & frame["predicted_excess_return"].notna()
        & frame["future_excess_return"].notna()
    )
    complete = frame[complete_mask].copy()
    violations = pd.DataFrame()
    if "training_latest_target_end_date" in frame.columns:
        training_end = pd.to_datetime(frame["training_latest_target_end_date"])
        violations = frame[training_end >= frame["date"]]
    dated = sector_backtest_by_date(frame)
    return {
        "n_prediction_rows": int(len(frame)),
        "n_prediction_dates": int(frame["date"].nunique()),
        "n_completed_rows": int(len(complete)),
        "n_completed_dates": int(complete["date"].nunique()) if not complete.empty else 0,
        "n_unrealized_rows": int(len(frame) - len(complete)),
        "n_unrealized_dates": int(frame["date"].nunique() - complete["date"].nunique()) if not complete.empty else int(frame["date"].nunique()),
        "n_leakage_violations": int(len(violations)),
        "first_completed_date": None if complete.empty else complete["date"].min().strftime("%Y-%m-%d"),
        "latest_completed_date": None if complete.empty else complete["date"].max().strftime("%Y-%m-%d"),
        "latest_prediction_date": frame["date"].max().strftime("%Y-%m-%d"),
        "latest_target_end_date": None if dated.empty else dated["target_end_date"].max().strftime("%Y-%m-%d"),
    }


def zscore(values: pd.Series) -> pd.Series:
    """Return cross-sectional z-scores."""
    numeric = pd.to_numeric(values, errors="coerce")
    scale = numeric.std(ddof=0)
    if pd.isna(scale) or scale == 0.0:
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / scale
