"""Rolling covariance evaluation using embedding-similarity shrinkage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.applications.portfolio_optimization import minimum_variance_long_only
from src.evaluation.base import Evaluator
from src.evaluation.peers import daily_returns_matrix
from src.utils.dates import month_end_dates
from src.utils.io import ensure_dir


class CovarianceEvaluator(Evaluator):
    """Compare embedding-prior covariance shrinkage with standard baselines."""

    name = "covariance"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        if db is None:
            raise ValueError("Covariance evaluation requires a FilingsDB instance.")

        settings = config["evaluation"].get("covariance", {})
        embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
        if not embedding_columns:
            raise ValueError("Covariance evaluation requires embedding_* columns.")

        returns = daily_returns_matrix(db.load_prices())
        rebalancing_dates = covariance_rebalance_dates(embeddings, returns, settings)
        if not rebalancing_dates:
            raise ValueError("No covariance rebalancing dates have enough forward return data.")

        rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        previous_weights: dict[str, pd.Series] = {}
        for rebalance_date in rebalancing_dates:
            window = covariance_window(
                embeddings,
                returns,
                embedding_columns,
                rebalance_date,
                settings,
            )
            if window is None:
                continue

            estimates = covariance_estimates(
                window.history_returns,
                window.embedding_matrix,
                settings,
            )
            for method, covariance in estimates.items():
                weights = minimum_variance_long_only(covariance)
                ticker_weights = pd.Series(weights, index=window.tickers)
                turnover = portfolio_turnover(previous_weights.get(method), ticker_weights)
                previous_weights[method] = ticker_weights

                realized = window.forward_returns.loc[:, window.tickers].to_numpy() @ weights
                for return_date, portfolio_return in zip(window.forward_returns.index, realized, strict=True):
                    rows.append(
                        {
                            "method": method,
                            "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                            "date": pd.Timestamp(return_date).strftime("%Y-%m-%d"),
                            "portfolio_return": float(portfolio_return),
                            "turnover": float(turnover),
                            "n_assets": int(len(window.tickers)),
                        }
                    )

                for ticker, weight in ticker_weights.items():
                    if weight > 0.0001:
                        weight_rows.append(
                            {
                                "method": method,
                                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                                "ticker": ticker,
                                "weight": float(weight),
                            }
                        )

        results = pd.DataFrame(rows)
        if results.empty:
            raise ValueError("Covariance evaluation produced no valid portfolio returns.")

        metrics = covariance_metrics(results, settings)
        if output_dir is not None:
            artifacts_dir = ensure_dir(Path(output_dir) / "evaluation")
            results.to_csv(artifacts_dir / "covariance_portfolio_returns.csv", index=False)
            pd.DataFrame(weight_rows).to_csv(artifacts_dir / "covariance_portfolio_weights.csv", index=False)
        return metrics


class CovarianceWindow:
    """Container for one rebalance date's assets, training returns, and test returns."""

    def __init__(
        self,
        tickers: list[str],
        history_returns: pd.DataFrame,
        forward_returns: pd.DataFrame,
        embedding_matrix: np.ndarray,
    ) -> None:
        self.tickers = tickers
        self.history_returns = history_returns
        self.forward_returns = forward_returns
        self.embedding_matrix = embedding_matrix


def covariance_rebalance_dates(
    embeddings: pd.DataFrame,
    returns: pd.DataFrame,
    settings: dict,
) -> list[pd.Timestamp]:
    """Return month-end embedding dates with enough trailing and forward returns."""
    lookback_days = int(settings.get("lookback_days", 252))
    holding_days = int(settings.get("holding_days", 21))
    start_date = settings.get("start_date")
    end_date = settings.get("end_date")

    frame = embeddings.loc[:, ["date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = month_end_dates(frame).drop_duplicates().sort_values()

    earliest = returns.index[min(lookback_days, len(returns.index) - 1)]
    latest = returns.index[-holding_days - 1]
    dates = [pd.Timestamp(date) for date in frame if earliest <= pd.Timestamp(date) <= latest]
    if start_date is not None:
        start = pd.Timestamp(start_date)
        dates = [date for date in dates if date >= start]
    if end_date is not None:
        end = pd.Timestamp(end_date)
        dates = [date for date in dates if date <= end]
    return dates


def covariance_window(
    embeddings: pd.DataFrame,
    returns: pd.DataFrame,
    embedding_columns: list[str],
    rebalance_date: pd.Timestamp,
    settings: dict,
) -> CovarianceWindow | None:
    """Build the usable asset panel for one covariance rebalance date."""
    lookback_days = int(settings.get("lookback_days", 252))
    holding_days = int(settings.get("holding_days", 21))
    min_assets = int(settings.get("min_assets", 25))
    max_assets = settings.get("max_assets", 150)

    history = returns.loc[returns.index <= rebalance_date].tail(lookback_days)
    forward = returns.loc[returns.index > rebalance_date].head(holding_days)
    if len(history) < lookback_days or len(forward) < holding_days:
        return None

    latest_embeddings = latest_embeddings_at(embeddings, embedding_columns, rebalance_date)
    candidate_tickers = [ticker for ticker in latest_embeddings.index if ticker in history.columns and ticker in forward.columns]
    if not candidate_tickers:
        return None

    complete_history = history.loc[:, candidate_tickers].notna().all(axis=0)
    complete_forward = forward.loc[:, candidate_tickers].notna().all(axis=0)
    tickers = [ticker for ticker in candidate_tickers if bool(complete_history[ticker]) and bool(complete_forward[ticker])]
    tickers = sorted(tickers)
    if max_assets is not None and len(tickers) > int(max_assets):
        tickers = stable_low_volatility_subset(history, tickers, int(max_assets))
    if len(tickers) < min_assets:
        return None

    embedding_matrix = latest_embeddings.loc[tickers, embedding_columns].astype(float).to_numpy()
    return CovarianceWindow(
        tickers=tickers,
        history_returns=history.loc[:, tickers],
        forward_returns=forward.loc[:, tickers],
        embedding_matrix=embedding_matrix,
    )


def latest_embeddings_at(
    embeddings: pd.DataFrame,
    embedding_columns: list[str],
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return each ticker's latest embedding available at or before a date."""
    frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] <= rebalance_date]
    frame = frame.dropna(subset=embedding_columns)
    frame = frame.sort_values(["ticker", "date"])
    frame = frame.groupby("ticker", as_index=False).tail(1)
    return frame.set_index("ticker")


def stable_low_volatility_subset(history: pd.DataFrame, tickers: list[str], max_assets: int) -> list[str]:
    """Choose a deterministic, liquid-ish subset by favoring lower realized volatility."""
    volatilities = history.loc[:, tickers].std(axis=0).sort_values(kind="mergesort")
    selected = sorted(volatilities.head(max_assets).index.astype(str).tolist())
    return selected


def covariance_estimates(
    history_returns: pd.DataFrame,
    embedding_matrix: np.ndarray,
    settings: dict,
) -> dict[str, np.ndarray]:
    """Build sample, standard Ledoit-Wolf, legacy shrinkage, and embedding-prior estimates."""
    matrix = history_returns.astype(float).to_numpy()
    sample = sample_covariance(matrix)
    ledoit = ledoit_wolf_covariance(matrix)
    legacy = _legacy_constant_variance_shrinkage(matrix)
    prior_alpha = float(settings.get("embedding_alpha", 0.25))
    prior_corr = embedding_similarity_prior(embedding_matrix)
    embedding_covariance = shrink_covariance_to_prior(sample, prior_corr, prior_alpha)
    return {
        "sample": sample,
        "ledoit_wolf": ledoit,
        "legacy_constant_variance": legacy,
        "embedding_prior": embedding_covariance,
    }


def sample_covariance(matrix: np.ndarray) -> np.ndarray:
    """Return a regularized sample covariance matrix."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(matrix) - 1)
    return regularize_covariance(covariance)


def ledoit_wolf_covariance(matrix: np.ndarray) -> np.ndarray:
    """Return sklearn's standard Ledoit-Wolf covariance estimate."""
    values = np.asarray(matrix, dtype=float)
    estimator = LedoitWolf().fit(values)
    return regularize_covariance(estimator.covariance_)


def _legacy_constant_variance_shrinkage(matrix: np.ndarray) -> np.ndarray:
    """Legacy local shrinkage toward constant variance, not standard Ledoit-Wolf.

    This is retained for migration comparisons because earlier reports labeled
    this estimator as Ledoit-Wolf. New headline benchmarks should use
    ``ledoit_wolf_covariance``, which delegates to ``sklearn.covariance.LedoitWolf``.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    n_observations, n_assets = centered.shape
    sample = centered.T @ centered / n_observations
    average_variance = float(np.trace(sample) / n_assets)
    target = np.eye(n_assets) * average_variance

    differences = []
    for row in centered:
        outer = np.outer(row, row)
        differences.append(float(np.square(outer - sample).sum()))
    phi = float(np.mean(differences))
    gamma = float(np.square(sample - target).sum())
    if gamma == 0.0:
        shrinkage = 1.0
    else:
        shrinkage = max(0.0, min(1.0, phi / (n_observations * gamma)))

    covariance = shrinkage * target + (1.0 - shrinkage) * sample
    return regularize_covariance(covariance)


def embedding_similarity_prior(embedding_matrix: np.ndarray) -> np.ndarray:
    """Convert embedding distances into a positive semi-definite correlation prior."""
    squared_distances = pairwise_squared_distances(embedding_matrix)
    off_diagonal = squared_distances[~np.eye(len(squared_distances), dtype=bool)]
    positive_distances = off_diagonal[off_diagonal > 0.0]
    scale = float(np.median(positive_distances)) if len(positive_distances) else 1.0
    if scale <= 0.0:
        scale = 1.0
    prior = np.exp(-squared_distances / scale)
    np.fill_diagonal(prior, 1.0)
    return prior


def pairwise_squared_distances(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.square(diff).sum(axis=2)


def shrink_covariance_to_prior(sample_covariance_matrix: np.ndarray, prior_corr: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink sample covariance toward a correlation prior with sample variances."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("embedding_alpha must be between 0 and 1.")
    variances = np.clip(np.diag(sample_covariance_matrix), 1e-12, None)
    standard_deviations = np.sqrt(variances)
    prior_covariance = prior_corr * np.outer(standard_deviations, standard_deviations)
    covariance = (1.0 - alpha) * sample_covariance_matrix + alpha * prior_covariance
    return regularize_covariance(covariance)


def regularize_covariance(covariance: np.ndarray) -> np.ndarray:
    """Symmetrize a covariance matrix and add tiny diagonal jitter."""
    matrix = np.asarray(covariance, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    average_variance = float(np.trace(matrix) / max(1, len(matrix)))
    jitter = max(average_variance * 1e-8, 1e-12)
    return matrix + np.eye(len(matrix)) * jitter


def minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the shared cvxpy optimizer."""
    return minimum_variance_long_only(covariance)


def portfolio_turnover(previous: pd.Series | None, current: pd.Series) -> float:
    """Return one-way turnover between consecutive rebalance weights."""
    if previous is None:
        return 0.0
    all_tickers = sorted(set(previous.index) | set(current.index))
    previous_aligned = previous.reindex(all_tickers).fillna(0.0)
    current_aligned = current.reindex(all_tickers).fillna(0.0)
    return float(0.5 * np.abs(current_aligned - previous_aligned).sum())


def covariance_metrics(results: pd.DataFrame, settings: dict) -> dict[str, object]:
    """Summarize realized portfolio returns and paired variance comparisons."""
    daily_periods = int(settings.get("annualization_days", 252))
    metrics: dict[str, object] = {
        "n_return_observations": int(len(results)),
        "n_rebalance_dates": int(results["rebalance_date"].nunique()),
        "methods": {},
    }
    for method, frame in results.groupby("method", sort=True):
        returns = frame["portfolio_return"].astype(float).to_numpy()
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        variance = float(np.var(returns, ddof=1)) if len(returns) > 1 else 0.0
        metrics["methods"][method] = {
            "realized_daily_variance": variance,
            "realized_annual_variance": float(variance * daily_periods),
            "realized_annual_sharpe": float(mean_return / std_return * np.sqrt(daily_periods)) if std_return > 0 else 0.0,
            "mean_daily_return": mean_return,
            "max_drawdown": max_drawdown(returns),
            "mean_turnover": float(frame.groupby("rebalance_date")["turnover"].first().mean()),
            "mean_assets": float(frame.groupby("rebalance_date")["n_assets"].first().mean()),
        }

    metrics["comparisons"] = paired_variance_comparisons(results, settings)
    return metrics


def max_drawdown(returns: np.ndarray) -> float:
    """Return maximum drawdown from a sequence of simple returns."""
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1.0
    return float(drawdowns.min())


def paired_variance_comparisons(results: pd.DataFrame, settings: dict) -> dict[str, dict[str, float]]:
    """Bootstrap differences in squared daily returns against Ledoit-Wolf."""
    rng = np.random.default_rng(int(settings.get("random_seed", 7)))
    bootstrap_samples = int(settings.get("bootstrap_samples", 500))
    pivot = results.pivot_table(index="date", columns="method", values="portfolio_return", aggfunc="mean")
    comparisons: dict[str, dict[str, float]] = {}
    if "ledoit_wolf" not in pivot.columns:
        return comparisons

    baseline_squared = np.square(pivot["ledoit_wolf"].to_numpy())
    for method in pivot.columns:
        if method == "ledoit_wolf":
            continue
        paired = pivot.loc[:, [method, "ledoit_wolf"]].dropna()
        if paired.empty:
            continue
        differences = np.square(paired[method].to_numpy()) - np.square(paired["ledoit_wolf"].to_numpy())
        bootstrapped = []
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, len(differences), size=len(differences))
            bootstrapped.append(float(differences[indices].mean()))
        comparisons[f"{method}_minus_ledoit_wolf"] = {
            "mean_daily_variance_diff": float(differences.mean()),
            "ci_low": float(np.quantile(bootstrapped, 0.025)),
            "ci_high": float(np.quantile(bootstrapped, 0.975)),
        }

    if len(baseline_squared) == 0:
        return comparisons
    return comparisons
