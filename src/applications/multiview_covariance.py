"""Multi-view factor-model covariance from soft theme loadings."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_theme_returns(loadings: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Build theme return series as loading-weighted averages of firm returns."""
    theme_columns = [column for column in loadings.columns if column.startswith("theme_")]
    if not theme_columns:
        raise ValueError("Loadings must contain theme_* columns.")
    latest = latest_loadings_for_return_columns(loadings, returns.columns.astype(str).tolist())
    weights = latest.loc[:, theme_columns].astype(float)
    weights.index = latest["ticker"].astype(str)
    common = [ticker for ticker in weights.index if ticker in returns.columns]
    if not common:
        return pd.DataFrame(index=returns.index)
    weights = weights.loc[common]
    denominator = weights.sum(axis=0).replace(0.0, np.nan)
    theme_returns = returns.loc[:, common].fillna(0.0).to_numpy() @ weights.to_numpy()
    theme_returns = theme_returns / denominator.to_numpy()
    return pd.DataFrame(theme_returns, index=returns.index, columns=theme_columns)


def estimate_factor_covariance(theme_returns: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Estimate ridge-regularized factor covariance from trailing theme returns."""
    window = theme_returns.tail(int(window_days)).dropna(axis=1, how="all").fillna(0.0)
    covariance = window.cov()
    if covariance.empty:
        return covariance
    epsilon = 1e-6 * float(np.trace(covariance.to_numpy())) / max(1, len(covariance))
    covariance = covariance + np.eye(len(covariance)) * epsilon
    return covariance


def estimate_multiview_covariance(
    loadings_per_view: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    as_of_date: str,
    window_days: int = 252,
    max_systematic_fraction: float = 0.8,
) -> pd.DataFrame:
    """Estimate firm covariance using stacked multi-view theme loadings."""
    as_of = pd.Timestamp(as_of_date)
    history = returns.loc[returns.index <= as_of].tail(int(window_days))
    if len(history) < 2:
        raise ValueError("Not enough return history for covariance estimation.")

    loadings = stacked_latest_loadings(loadings_per_view, as_of, history.columns.astype(str).tolist())
    tickers = loadings.index.tolist()
    if not tickers:
        raise ValueError("No loadings matched return columns.")

    history = history.loc[:, tickers].dropna(axis=1, how="any")
    loadings = loadings.loc[history.columns]
    theme_returns = build_stacked_theme_returns(loadings, history)
    factor_covariance = estimate_factor_covariance(theme_returns, window_days)
    loading_matrix = loadings.loc[:, factor_covariance.columns].to_numpy()
    systematic = loading_matrix @ factor_covariance.to_numpy() @ loading_matrix.T
    firm_variance = history.var(ddof=1).to_numpy()
    systematic = cap_systematic_variance(systematic, firm_variance, max_systematic_fraction)
    diagonal_residual = firm_variance - np.diag(systematic)
    floor = np.maximum(firm_variance * 1e-6, 1e-12)
    diagonal_residual = np.maximum(diagonal_residual, floor)
    covariance = systematic + np.diag(diagonal_residual)
    covariance = nearest_psd_with_jitter(covariance)
    return pd.DataFrame(covariance, index=history.columns, columns=history.columns)


def cap_systematic_variance(
    systematic: np.ndarray,
    firm_variance: np.ndarray,
    max_systematic_fraction: float,
) -> np.ndarray:
    """Scale factor covariance if stacked views over-explain typical firm variance."""
    if max_systematic_fraction <= 0.0:
        return np.zeros_like(systematic)
    positive = firm_variance > 0.0
    if not positive.any():
        return systematic
    ratios = np.diag(systematic)[positive] / firm_variance[positive]
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return systematic
    typical_ratio = float(np.nanmedian(ratios))
    if typical_ratio <= max_systematic_fraction:
        return systematic
    scale = float(max_systematic_fraction) / typical_ratio
    return systematic * scale


def stacked_latest_loadings(
    loadings_per_view: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    return_tickers: list[str],
) -> pd.DataFrame:
    """Stack each view's latest loadings horizontally with view-prefixed columns."""
    pieces = []
    for view_name, loadings in loadings_per_view.items():
        frame = loadings.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame["date"] <= as_of_date]
        frame = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        theme_columns = [column for column in frame.columns if column.startswith("theme_")]
        frame = frame.loc[:, ["ticker", *theme_columns]].set_index("ticker")
        frame = frame.rename(columns={column: f"{view_name}_{column}" for column in theme_columns})
        pieces.append(frame)
    stacked = pd.concat(pieces, axis=1).fillna(0.0)
    return_ticker_set = {ticker.upper() for ticker in return_tickers}
    stacked = stacked[stacked.index.isin(return_ticker_set)]
    return stacked


def latest_loadings_for_return_columns(loadings: pd.DataFrame, return_tickers: list[str]) -> pd.DataFrame:
    """Return latest loadings for tickers available in a returns matrix."""
    frame = loadings.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    return_ticker_set = {ticker.upper() for ticker in return_tickers}
    return frame[frame["ticker"].isin(return_ticker_set)].reset_index(drop=True)


def build_stacked_theme_returns(loadings: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Build theme returns from an already-stacked loadings matrix."""
    weights = loadings.astype(float)
    denominator = weights.sum(axis=0).replace(0.0, np.nan)
    theme_returns = returns.fillna(0.0).to_numpy() @ weights.to_numpy()
    theme_returns = theme_returns / denominator.to_numpy()
    return pd.DataFrame(theme_returns, index=returns.index, columns=weights.columns).fillna(0.0)


def nearest_psd_with_jitter(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and add diagonal jitter until the covariance is PSD."""
    covariance = (matrix + matrix.T) / 2.0
    min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if min_eigenvalue < 0.0:
        covariance = covariance + np.eye(len(covariance)) * (-min_eigenvalue + 1e-12)
    return covariance
