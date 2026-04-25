"""Evaluate whether embedding-nearest peers have stronger forward co-movement."""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.evaluation.base import Evaluator
from src.utils.io import ensure_dir


class PeersEvaluator(Evaluator):
    """Compare embedding nearest neighbors with same-sub-industry random peers."""

    name = "peers"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        if db is None:
            raise ValueError("Peer evaluation requires a FilingsDB instance.")

        settings = config["evaluation"].get("peers", {})
        benchmark_column = settings.get("benchmark_column", "gics_sub_industry")
        if benchmark_column not in metadata.columns:
            raise ValueError(f"Peer evaluation requires metadata column {benchmark_column!r}.")

        embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
        if not embedding_columns:
            raise ValueError("Peer evaluation requires embedding_* columns.")

        k = int(settings.get("k", 5))
        forward_days = int(settings.get("forward_days", 21))
        max_observations = int(settings.get("max_observations", 5000))
        min_benchmark_pool = int(settings.get("min_benchmark_pool", k + 1))
        rng = np.random.default_rng(int(settings.get("random_seed", config["random_seed"])))

        prices = db.load_prices()
        returns = daily_returns_matrix(prices)
        observations = build_observation_frame(
            embeddings,
            metadata,
            embedding_columns,
            benchmark_column,
            returns,
            forward_days,
            max_observations,
            rng,
        )
        if observations.empty:
            raise ValueError("No peer-evaluation observations had enough metadata and forward returns.")

        rows: list[dict[str, object]] = []
        grouped = observations.groupby("date", sort=True)
        for date, date_frame in grouped:
            date_frame = date_frame.reset_index(drop=True)
            matrix = date_frame.loc[:, embedding_columns].astype(float).to_numpy()
            tickers = date_frame["ticker"].astype(str).to_numpy()
            benchmark_labels = date_frame[benchmark_column].astype(str).to_numpy()
            return_window = forward_return_window(returns, pd.Timestamp(date), forward_days)
            if return_window.empty:
                continue

            distances = pairwise_distances(matrix)
            for row_index, ticker in enumerate(tickers):
                if ticker not in return_window.columns:
                    continue
                target_returns = return_window[ticker].to_numpy()
                embedding_peer_indices = nearest_peer_indices(distances[row_index], row_index, k)
                embedding_peers = [tickers[index] for index in embedding_peer_indices if tickers[index] in return_window.columns]
                if len(embedding_peers) < k:
                    continue

                benchmark_pool = [
                    index
                    for index, label in enumerate(benchmark_labels)
                    if label == benchmark_labels[row_index] and index != row_index and tickers[index] in return_window.columns
                ]
                if len(benchmark_pool) < max(k, min_benchmark_pool):
                    continue
                benchmark_indices = rng.choice(benchmark_pool, size=k, replace=False)
                benchmark_peers = [tickers[index] for index in benchmark_indices]

                embedding_corr = peer_group_correlation(target_returns, return_window, embedding_peers)
                benchmark_corr = peer_group_correlation(target_returns, return_window, benchmark_peers)
                if np.isnan(embedding_corr) or np.isnan(benchmark_corr):
                    continue

                rows.append(
                    {
                        "ticker": ticker,
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        benchmark_column: benchmark_labels[row_index],
                        "embedding_peers": "|".join(embedding_peers),
                        "benchmark_peers": "|".join(benchmark_peers),
                        "embedding_peer_corr": float(embedding_corr),
                        "benchmark_peer_corr": float(benchmark_corr),
                        "corr_diff": float(embedding_corr - benchmark_corr),
                    }
                )

        results = pd.DataFrame(rows)
        if results.empty:
            raise ValueError("Peer evaluation produced no valid paired correlations.")

        metrics = paired_metrics(results["embedding_peer_corr"].to_numpy(), results["benchmark_peer_corr"].to_numpy())
        metrics.update(
            {
                "k": int(k),
                "forward_days": int(forward_days),
                "benchmark_column": benchmark_column,
                "n_observations": int(len(results)),
                "n_unique_tickers": int(results["ticker"].nunique()),
                "n_dates": int(results["date"].nunique()),
            }
        )

        if output_dir is not None:
            artifacts_dir = ensure_dir(Path(output_dir) / "evaluation")
            results.to_csv(artifacts_dir / "peer_identification_results.csv", index=False)
        return metrics


def daily_returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a date-by-ticker daily adjusted-return matrix."""
    required = {"ticker", "date", "adj_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Prices are missing columns: {sorted(missing)}")

    prices = prices.loc[:, ["ticker", "date", "adj_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["adj_close"].pct_change()
    returns = prices.pivot(index="date", columns="ticker", values="return").sort_index()
    return returns


def build_observation_frame(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    embedding_columns: list[str],
    benchmark_column: str,
    returns: pd.DataFrame,
    forward_days: int,
    max_observations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build candidate ticker-date rows with labels and enough forward returns."""
    frame = embeddings.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    max_start_date = returns.index[-forward_days - 1]
    frame = frame[frame["date"] <= max_start_date]

    metadata_subset = metadata.loc[:, ["ticker", benchmark_column]].dropna().copy()
    metadata_subset["ticker"] = metadata_subset["ticker"].astype(str).str.upper()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.merge(metadata_subset, on="ticker", how="inner")
    frame = frame.dropna(subset=[benchmark_column, *embedding_columns])
    frame = frame[frame["ticker"].isin(returns.columns)]

    if len(frame) > max_observations:
        date_sizes = frame.groupby("date").size()
        rows_per_date = max(1, int(date_sizes.median()))
        date_count = max(1, max_observations // rows_per_date)
        available_dates = date_sizes.index.to_numpy()
        selected_dates = rng.choice(
            available_dates,
            size=min(date_count, len(available_dates)),
            replace=False,
        )
        frame = frame[frame["date"].isin(selected_dates)]
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def forward_return_window(returns: pd.DataFrame, date: pd.Timestamp, forward_days: int) -> pd.DataFrame:
    """Return forward daily returns after an observation date."""
    future = returns.loc[returns.index > date]
    return future.head(forward_days)


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances."""
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.square(diff).sum(axis=2))


def nearest_peer_indices(distances: np.ndarray, own_index: int, k: int) -> list[int]:
    """Return nearest peer indices excluding the focal row."""
    order = np.argsort(distances)
    return [int(index) for index in order if int(index) != own_index][:k]


def peer_group_correlation(target_returns: np.ndarray, return_window: pd.DataFrame, peers: list[str]) -> float:
    """Correlate a ticker's forward returns with the average peer forward return."""
    peer_returns = return_window.loc[:, peers].mean(axis=1).to_numpy()
    valid = np.isfinite(target_returns) & np.isfinite(peer_returns)
    if valid.sum() < 3:
        return float("nan")
    target = target_returns[valid]
    peer = peer_returns[valid]
    if np.std(target) == 0.0 or np.std(peer) == 0.0:
        return float("nan")
    return float(np.corrcoef(target, peer)[0, 1])


def paired_metrics(embedding_corrs: np.ndarray, benchmark_corrs: np.ndarray) -> dict[str, float]:
    """Return paired comparison metrics for two correlation arrays."""
    differences = embedding_corrs - benchmark_corrs
    mean_diff = float(differences.mean())
    std_diff = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    if std_diff == 0.0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = float(mean_diff / (std_diff / np.sqrt(len(differences))))
        # Normal approximation is enough for this research-code smoke evaluation.
        p_value = float(2.0 * (1.0 - NormalDist().cdf(abs(t_stat))))

    return {
        "mean_embedding_peer_corr": float(embedding_corrs.mean()),
        "mean_benchmark_peer_corr": float(benchmark_corrs.mean()),
        "mean_corr_diff": mean_diff,
        "median_corr_diff": float(np.median(differences)),
        "t_stat": t_stat,
        "p_value_normal_approx": p_value,
        "diff_std": std_diff,
    }
