"""Point-in-time relationship-graph position features sampled onto monthly rows.

For each monthly observation date, the graph includes only relationships whose
source filing date is known on or before that date. This makes the network view
honest through time: early dates contain only relationships disclosed by then,
rather than the latest graph projected backward.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.utils.dates import select_month_end_rows


NETWORK_COLUMNS = [
    "net_in_degree",
    "net_out_degree",
    "net_in_degree_customer",
    "net_in_degree_supplier",
    "net_in_degree_competitor",
    "net_pagerank",
    "net_betweenness",
    "net_clustering_coefficient",
    "net_community_id",
    "net_community_size",
]


class NetworkPositionProducer(FeatureProducer):
    """Compute point-in-time graph-position features from filing-derived relationships."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="network_position",
            key_columns=["ticker", "date"],
            columns=NETWORK_COLUMNS,
            source="relationships",
            description="Point-in-time network centrality and community features from extracted relationship graph.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        relationships_path = Path(config["paths"]["relationships_path"])
        if not relationships_path.exists():
            raise FileNotFoundError(f"Relationship graph not found at {relationships_path}.")
        relationships = pd.read_parquet(relationships_path)
        if "filing_date" not in relationships.columns:
            raise ValueError("Relationship features require filing_date for point-in-time graph construction.")
        relationships["filing_date"] = pd.to_datetime(relationships["filing_date"], errors="coerce")
        relationships = relationships.dropna(subset=["filing_date"])
        tickers = db.load_tickers().loc[:, ["ticker"]].copy()
        tickers["ticker"] = tickers["ticker"].astype(str).str.upper()

        prices = db.load_prices(date_from=config["data"]["start_date"], date_to=config["data"]["end_date"])
        monthly = select_month_end_rows(prices.loc[:, ["ticker", "date"]])
        monthly["ticker"] = monthly["ticker"].astype(str).str.upper()
        monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")

        feature_frames = []
        universe = tickers["ticker"].tolist()
        for date in sorted(monthly["date"].dropna().unique()):
            as_of_relationships = relationships[relationships["filing_date"] <= pd.Timestamp(date)]
            features = compute_network_features(as_of_relationships, universe, config)
            features["date"] = pd.Timestamp(date)
            feature_frames.append(features)

        if feature_frames:
            dynamic_features = pd.concat(feature_frames, ignore_index=True)
        else:
            dynamic_features = pd.DataFrame(columns=["ticker", "date", *NETWORK_COLUMNS])

        frame = monthly.merge(dynamic_features, on=["ticker", "date"], how="left")
        for column in NETWORK_COLUMNS:
            frame[column] = frame[column].fillna(0.0)
        return frame.loc[:, self.spec.all_columns].reset_index(drop=True)


def compute_network_features(relationships: pd.DataFrame, universe_tickers: list[str], config: dict) -> pd.DataFrame:
    """Compute directed degree plus undirected centrality/community features."""
    frame = relationships.copy()
    frame["source_ticker"] = frame["source_ticker"].astype(str).str.upper()
    frame["target_ticker"] = frame["target_ticker"].astype(str).str.upper()
    min_confidence = float(config["features"].get("graph", {}).get("min_confidence", 0.45))
    frame = frame[frame["confidence"].astype(float) >= min_confidence]

    directed = nx.DiGraph()
    directed.add_nodes_from(universe_tickers)
    for _, row in frame.iterrows():
        source = str(row["source_ticker"])
        target = str(row["target_ticker"])
        if source != target:
            directed.add_edge(source, target, relationship_type=row["relationship_type"])

    undirected = directed.to_undirected()
    pagerank = nx.pagerank(undirected) if undirected.number_of_edges() else {ticker: 0.0 for ticker in universe_tickers}
    k = min(100, max(1, undirected.number_of_nodes()))
    betweenness = nx.betweenness_centrality(undirected, k=k, seed=int(config["random_seed"])) if undirected.number_of_edges() else {}
    clustering = nx.clustering(undirected) if undirected.number_of_edges() else {}
    communities = community_membership(undirected, int(config["random_seed"]))

    rows = []
    for ticker in universe_tickers:
        incoming = frame[frame["target_ticker"] == ticker]
        rows.append(
            {
                "ticker": ticker,
                "net_in_degree": float(directed.in_degree(ticker)),
                "net_out_degree": float(directed.out_degree(ticker)),
                "net_in_degree_customer": float((incoming["relationship_type"] == "customer").sum()),
                "net_in_degree_supplier": float((incoming["relationship_type"] == "supplier").sum()),
                "net_in_degree_competitor": float((incoming["relationship_type"] == "competitor").sum()),
                "net_pagerank": float(pagerank.get(ticker, 0.0)),
                "net_betweenness": float(betweenness.get(ticker, 0.0)),
                "net_clustering_coefficient": float(clustering.get(ticker, 0.0)),
                "net_community_id": float(communities.get(ticker, (0, 0))[0]),
                "net_community_size": float(communities.get(ticker, (0, 0))[1]),
            }
        )
    return pd.DataFrame(rows)


def community_membership(graph: nx.Graph, random_seed: int) -> dict[str, tuple[int, int]]:
    """Return Louvain community id and size per node."""
    if graph.number_of_edges() == 0:
        return {str(node): (0, 0) for node in graph.nodes}
    communities = nx.algorithms.community.louvain_communities(graph, seed=random_seed)
    membership: dict[str, tuple[int, int]] = {}
    for community_id, community in enumerate(communities):
        for node in community:
            membership[str(node)] = (community_id, len(community))
    return membership
