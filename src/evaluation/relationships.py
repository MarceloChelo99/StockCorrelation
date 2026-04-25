"""Evaluate whether embedding peers align with filing-derived relationship links."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.base import Evaluator
from src.evaluation.peers import nearest_peer_indices, paired_metrics, pairwise_distances
from src.utils.io import ensure_dir


class RelationshipGraphEvaluator(Evaluator):
    """Compare embedding peers, GICS peers, and random peers on graph proximity."""

    name = "relationships"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        settings = config["evaluation"].get("relationships", {})
        relationship_path = Path(settings.get("relationship_path", config["paths"].get("relationships_path", "")))
        if not relationship_path.exists():
            raise FileNotFoundError(f"Relationship graph not found at {relationship_path}.")

        benchmark_column = settings.get("benchmark_column", "gics_sub_industry")
        if benchmark_column not in metadata.columns:
            raise ValueError(f"Relationship evaluation requires metadata column {benchmark_column!r}.")

        embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
        if not embedding_columns:
            raise ValueError("Relationship evaluation requires embedding_* columns.")

        relationships = pd.read_parquet(relationship_path)
        graph = relationship_graph(
            relationships,
            min_confidence=float(settings.get("min_confidence", 0.45)),
            relationship_types=settings.get("relationship_types"),
        )
        if not graph:
            raise ValueError("Relationship graph has no usable edges.")

        k = int(settings.get("k", 5))
        rng = np.random.default_rng(int(settings.get("random_seed", config["random_seed"])))
        observations = latest_embedding_frame(embeddings, metadata, embedding_columns, benchmark_column)
        observations = observations[observations["ticker"].isin(graph.keys())].reset_index(drop=True)
        if len(observations) < k + 2:
            raise ValueError("Not enough tickers with both embeddings and relationship graph links.")

        matrix = observations.loc[:, embedding_columns].astype(float).to_numpy()
        tickers = observations["ticker"].astype(str).to_numpy()
        labels = observations[benchmark_column].astype(str).to_numpy()
        distances = pairwise_distances(matrix)

        rows = []
        all_indices = np.arange(len(tickers))
        for row_index, ticker in enumerate(tickers):
            embedding_indices = nearest_peer_indices(distances[row_index], row_index, k)
            benchmark_pool = [
                index
                for index, label in enumerate(labels)
                if label == labels[row_index] and index != row_index
            ]
            random_pool = [index for index in all_indices if index != row_index]
            if len(embedding_indices) < k or len(benchmark_pool) < k or len(random_pool) < k:
                continue

            benchmark_indices = rng.choice(benchmark_pool, size=k, replace=False).tolist()
            random_indices = rng.choice(random_pool, size=k, replace=False).tolist()
            embedding_peers = [tickers[index] for index in embedding_indices]
            benchmark_peers = [tickers[index] for index in benchmark_indices]
            random_peers = [tickers[index] for index in random_indices]

            rows.append(
                {
                    "ticker": ticker,
                    benchmark_column: labels[row_index],
                    "embedding_peers": "|".join(embedding_peers),
                    "benchmark_peers": "|".join(benchmark_peers),
                    "random_peers": "|".join(random_peers),
                    "embedding_direct_rate": direct_connection_rate(ticker, embedding_peers, graph),
                    "benchmark_direct_rate": direct_connection_rate(ticker, benchmark_peers, graph),
                    "random_direct_rate": direct_connection_rate(ticker, random_peers, graph),
                    "embedding_jaccard": mean_neighbor_jaccard(ticker, embedding_peers, graph),
                    "benchmark_jaccard": mean_neighbor_jaccard(ticker, benchmark_peers, graph),
                    "random_jaccard": mean_neighbor_jaccard(ticker, random_peers, graph),
                }
            )

        results = pd.DataFrame(rows)
        if results.empty:
            raise ValueError("Relationship graph evaluation produced no comparable rows.")

        metrics = relationship_metrics(results, k, benchmark_column)
        if output_dir is not None:
            artifacts_dir = ensure_dir(Path(output_dir) / "evaluation")
            results.to_csv(artifacts_dir / "relationship_graph_alignment.csv", index=False)
        return metrics


def relationship_graph(
    relationships: pd.DataFrame,
    *,
    min_confidence: float,
    relationship_types: list[str] | None,
) -> dict[str, set[str]]:
    """Build an undirected adjacency list from relationship rows."""
    required = {"source_ticker", "target_ticker", "relationship_type", "confidence"}
    missing = required - set(relationships.columns)
    if missing:
        raise ValueError(f"Relationships are missing columns: {sorted(missing)}")

    frame = relationships.copy()
    frame["source_ticker"] = frame["source_ticker"].astype(str).str.upper()
    frame["target_ticker"] = frame["target_ticker"].astype(str).str.upper()
    frame = frame[frame["confidence"].astype(float) >= min_confidence]
    if relationship_types:
        frame = frame[frame["relationship_type"].isin(relationship_types)]

    graph: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        source = str(row["source_ticker"])
        target = str(row["target_ticker"])
        if source == target:
            continue
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
    return graph


def latest_embedding_frame(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    embedding_columns: list[str],
    benchmark_column: str,
) -> pd.DataFrame:
    """Return one latest embedding row per ticker with benchmark metadata."""
    frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=embedding_columns)
    frame = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)

    meta = metadata.loc[:, ["ticker", benchmark_column]].dropna().copy()
    meta["ticker"] = meta["ticker"].astype(str).str.upper()
    frame = frame.merge(meta, on="ticker", how="inner")
    return frame.sort_values("ticker").reset_index(drop=True)


def direct_connection_rate(ticker: str, peers: list[str], graph: dict[str, set[str]]) -> float:
    """Return the share of peers directly connected to ticker in the graph."""
    neighbors = graph.get(ticker, set())
    if not peers:
        return 0.0
    return float(sum(peer in neighbors for peer in peers) / len(peers))


def mean_neighbor_jaccard(ticker: str, peers: list[str], graph: dict[str, set[str]]) -> float:
    """Return mean neighborhood-overlap Jaccard similarity to peers."""
    own_neighbors = graph.get(ticker, set())
    scores = []
    for peer in peers:
        peer_neighbors = graph.get(peer, set())
        union = own_neighbors | peer_neighbors
        if not union:
            scores.append(0.0)
        else:
            scores.append(len(own_neighbors & peer_neighbors) / len(union))
    return float(np.mean(scores)) if scores else 0.0


def relationship_metrics(results: pd.DataFrame, k: int, benchmark_column: str) -> dict[str, object]:
    """Summarize graph-alignment rates and paired differences."""
    direct_embedding = results["embedding_direct_rate"].to_numpy()
    direct_benchmark = results["benchmark_direct_rate"].to_numpy()
    direct_random = results["random_direct_rate"].to_numpy()
    jaccard_embedding = results["embedding_jaccard"].to_numpy()
    jaccard_benchmark = results["benchmark_jaccard"].to_numpy()

    direct_vs_benchmark = paired_metrics(direct_embedding, direct_benchmark)
    direct_vs_random = paired_metrics(direct_embedding, direct_random)
    jaccard_vs_benchmark = paired_metrics(jaccard_embedding, jaccard_benchmark)

    return {
        "k": int(k),
        "benchmark_column": benchmark_column,
        "n_observations": int(len(results)),
        "mean_embedding_direct_rate": float(direct_embedding.mean()),
        "mean_benchmark_direct_rate": float(direct_benchmark.mean()),
        "mean_random_direct_rate": float(direct_random.mean()),
        "mean_embedding_jaccard": float(jaccard_embedding.mean()),
        "mean_benchmark_jaccard": float(jaccard_benchmark.mean()),
        "mean_random_jaccard": float(results["random_jaccard"].mean()),
        "embedding_minus_benchmark_direct": direct_vs_benchmark,
        "embedding_minus_random_direct": direct_vs_random,
        "embedding_minus_benchmark_jaccard": jaccard_vs_benchmark,
    }
