"""Evaluate multi-view factor covariance against standard covariance benchmarks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

from src.applications.multiview_covariance import estimate_multiview_covariance
from src.evaluation.base import Evaluator
from src.evaluation.covariance import (
    covariance_metrics,
    covariance_rebalance_dates,
    covariance_estimates,
    covariance_window,
    ledoit_wolf_covariance,
    minimum_variance_weights,
    portfolio_turnover,
    sample_covariance,
)
from src.evaluation.peers import daily_returns_matrix
from src.utils.io import ensure_dir


class MultiviewCovarianceEvaluator(Evaluator):
    """Rolling minimum-variance evaluation for multi-view factor covariance."""

    name = "multiview_covariance"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        if db is None:
            raise ValueError("Multi-view covariance evaluation requires a FilingsDB instance.")
        if output_dir is None:
            raise ValueError("Multi-view covariance evaluation requires an experiment output_dir with view loadings.")

        root = Path(output_dir)
        loadings_per_view = load_view_loadings(root, config)
        returns = daily_returns_matrix(db.load_prices())
        settings = config["evaluation"].get("multiview_covariance", config["evaluation"].get("covariance", {}))
        max_systematic_fraction = float(settings.get("max_systematic_fraction", 0.8))
        embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
        if not embedding_columns:
            raise ValueError("Multi-view covariance requires embedding_* columns for the single-view baseline.")
        rebalancing_dates = covariance_rebalance_dates(embeddings, returns, settings)

        rows = []
        previous_weights: dict[str, pd.Series] = {}
        for rebalance_date in rebalancing_dates:
            window = covariance_window(embeddings, returns, embedding_columns, rebalance_date, settings)
            if window is None:
                continue
            history = window.history_returns.loc[:, window.tickers]
            forward = window.forward_returns.loc[:, window.tickers]
            single_view_estimates = covariance_estimates(history, window.embedding_matrix, settings)
            estimates = {
                "sample": pd.DataFrame(sample_covariance(history.to_numpy()), index=window.tickers, columns=window.tickers),
                "ledoit_wolf": pd.DataFrame(ledoit_wolf_covariance(history.to_numpy()), index=window.tickers, columns=window.tickers),
                "embedding_prior": pd.DataFrame(single_view_estimates["embedding_prior"], index=window.tickers, columns=window.tickers),
                "multiview": estimate_multiview_covariance(
                    loadings_per_view,
                    returns.loc[:, window.tickers],
                    rebalance_date.strftime("%Y-%m-%d"),
                    window_days=int(settings.get("lookback_days", 252)),
                    max_systematic_fraction=max_systematic_fraction,
                ),
            }
            for dropped_view in loadings_per_view:
                subset = {name: value for name, value in loadings_per_view.items() if name != dropped_view}
                estimates[f"multiview_without_{dropped_view}"] = estimate_multiview_covariance(
                    subset,
                    returns.loc[:, window.tickers],
                    rebalance_date.strftime("%Y-%m-%d"),
                    window_days=int(settings.get("lookback_days", 252)),
                    max_systematic_fraction=max_systematic_fraction,
                )

            for method, covariance in estimates.items():
                covariance = covariance.loc[window.tickers, window.tickers]
                weights = cvxpy_minimum_variance_weights(covariance.to_numpy())
                ticker_weights = pd.Series(weights, index=window.tickers)
                turnover = portfolio_turnover(previous_weights.get(method), ticker_weights)
                previous_weights[method] = ticker_weights
                realized = forward.loc[:, window.tickers].to_numpy() @ weights
                for return_date, portfolio_return in zip(forward.index, realized, strict=True):
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

        results = pd.DataFrame(rows)
        if results.empty:
            raise ValueError("Multi-view covariance produced no portfolio returns.")
        metrics = covariance_metrics(results, settings)
        if output_dir is not None:
            artifacts_dir = ensure_dir(root / "evaluation")
            prefix = str(settings.get("artifact_prefix", "") or "")
            filename = f"{prefix}_multiview_covariance_returns.csv" if prefix else "multiview_covariance_returns.csv"
            results.to_csv(artifacts_dir / filename, index=False)
        return metrics


def load_view_loadings(root: Path, config: dict) -> dict[str, pd.DataFrame]:
    """Load per-view GMM loadings from an experiment directory."""
    loadings = {}
    for view_name in config.get("views_enabled", list(config["views"].keys())):
        path = root / "views" / view_name / "loadings.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing view loadings: {path}")
        loadings[view_name] = pd.read_parquet(path)
    return loadings


def cvxpy_minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """Solve long-only minimum variance with cvxpy, falling back if needed."""
    matrix = np.asarray(covariance, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    n_assets = len(matrix)
    weights = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(matrix)))
    problem = cp.Problem(objective, [weights >= 0.0, cp.sum(weights) == 1.0])
    for solver in ["CLARABEL", "OSQP", "SCS"]:
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError:
            continue
        if weights.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            result = np.asarray(weights.value, dtype=float)
            result = np.clip(result, 0.0, None)
            if result.sum() > 0.0:
                return result / result.sum()
    return minimum_variance_weights(matrix)
