"""Compare clustering choices on one embedding cross-section.

This module is intentionally small and educational: it lets the dashboard show
how GMM, k-means, and DBSCAN partition the same firm embeddings on the same date.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return embedding feature columns from an embeddings parquet."""
    return [column for column in frame.columns if column.startswith("embedding_")]


def cluster_cross_section(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster one firm-date cross-section with GMM, k-means, and DBSCAN.

    Input embeddings must contain ticker, date, and embedding_* columns. Output
    assignments are long-form with one row per ticker per algorithm. Metrics
    include internal geometry and alignment with GICS sectors when metadata is
    available.
    """
    columns = embedding_columns(embeddings)
    if not columns:
        raise ValueError("Embeddings must contain embedding_* columns.")

    frame = embeddings.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    selected_date = pd.Timestamp(as_of_date)
    frame = frame[frame["date"].eq(selected_date)].copy()
    if frame.empty:
        raise ValueError(f"No embeddings found for {selected_date.date()}.")
    if len(frame) < 5:
        raise ValueError("Need at least five firms to compare clustering models.")

    matrix = (
        frame.loc[:, columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    matrix = StandardScaler().fit_transform(matrix)
    cluster_count = max(2, min(int(n_clusters), len(frame) - 1))

    label_sets = {
        "GMM": gmm_labels(matrix, cluster_count, random_state),
        "k-means": kmeans_labels(matrix, cluster_count, random_state),
        "DBSCAN": dbscan_labels(matrix),
    }

    sectors = ticker_sector_labels(frame, metadata)
    assignments = []
    metric_rows = []
    for model_name, labels in label_sets.items():
        assignments.append(assignment_frame(frame, model_name, labels))
        metric_rows.append(clustering_metric_row(model_name, labels, matrix, sectors))

    return (
        pd.concat(assignments, ignore_index=True),
        pd.DataFrame(metric_rows).sort_values("model").reset_index(drop=True),
    )


def gmm_labels(matrix: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Return hard labels from a diagonal-covariance Gaussian mixture."""
    model = GaussianMixture(
        n_components=int(n_clusters),
        covariance_type="diag",
        random_state=int(random_state),
        n_init=5,
        max_iter=300,
        reg_covar=1e-6,
    )
    return model.fit_predict(matrix)


def kmeans_labels(matrix: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Return hard labels from k-means with the same target cluster count."""
    model = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=20)
    return model.fit_predict(matrix)


def dbscan_labels(matrix: np.ndarray) -> np.ndarray:
    """Return density-based labels using an automatically chosen epsilon."""
    min_samples = max(5, min(15, int(round(math.sqrt(len(matrix)) / 2.0))))
    eps = automatic_dbscan_eps(matrix, min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(matrix)


def automatic_dbscan_eps(matrix: np.ndarray, min_samples: int) -> float:
    """Choose a DBSCAN epsilon from k-nearest-neighbor distances."""
    n_neighbors = min(int(min_samples), len(matrix))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)
    kth_distances = np.sort(distances[:, -1])
    positive = kth_distances[np.isfinite(kth_distances) & (kth_distances > 0.0)]
    if len(positive) == 0:
        return 0.5
    return float(np.percentile(positive, 80))


def ticker_sector_labels(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
    """Return GICS sector labels aligned to the embedding frame."""
    if metadata.empty or "gics_sector" not in metadata.columns:
        return pd.Series(index=frame.index, dtype=object)
    meta = metadata.loc[:, ["ticker", "gics_sector"]].copy()
    meta["ticker"] = meta["ticker"].astype(str).str.upper()
    joined = frame.loc[:, ["ticker"]].merge(meta.drop_duplicates("ticker"), on="ticker", how="left")
    return joined["gics_sector"]


def assignment_frame(frame: pd.DataFrame, model_name: str, labels: np.ndarray) -> pd.DataFrame:
    """Return one long-form assignment table for a model."""
    result = frame.loc[:, ["ticker", "date"]].copy()
    result["model"] = model_name
    result["cluster_id"] = labels.astype(int)
    result["is_noise"] = result["cluster_id"].eq(-1)
    result["cluster_label"] = [
        f"{model_name} noise" if int(label) == -1 else f"{model_name} {int(label) + 1}"
        for label in result["cluster_id"]
    ]
    return result


def clustering_metric_row(
    model_name: str,
    labels: np.ndarray,
    matrix: np.ndarray,
    sectors: pd.Series,
) -> dict[str, float | int | str]:
    """Return compact diagnostics for one model's assignments."""
    labels = labels.astype(int)
    non_noise = labels != -1
    cluster_ids = sorted(set(labels[non_noise]))
    n_clusters = len(cluster_ids)
    noise_share = float(np.mean(labels == -1))
    counts = pd.Series(labels).value_counts(normalize=True)
    largest_cluster_share = float(counts.max()) if not counts.empty else float("nan")

    silhouette = np.nan
    if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
        silhouette = float(silhouette_score(matrix[non_noise], labels[non_noise]))

    valid_sector = sectors.notna().to_numpy()
    nmi = np.nan
    ari = np.nan
    if int(valid_sector.sum()) > 1:
        sector_values = sectors.loc[valid_sector].astype(str)
        label_values = pd.Series(labels[valid_sector]).astype(str)
        nmi = float(normalized_mutual_info_score(sector_values, label_values))
        ari = float(adjusted_rand_score(sector_values, label_values))

    return {
        "model": model_name,
        "clusters_found": int(n_clusters),
        "noise_share": noise_share,
        "largest_cluster_share": largest_cluster_share,
        "silhouette": silhouette,
        "nmi_vs_gics": nmi,
        "ari_vs_gics": ari,
        "interpretation": model_interpretation(model_name),
    }


def model_interpretation(model_name: str) -> str:
    """Return a one-line explanation for dashboard display."""
    explanations = {
        "GMM": "Soft elliptical themes; best match for mixed-membership company identities.",
        "k-means": "Hard spherical groups; simple baseline, but every firm must belong to exactly one group.",
        "DBSCAN": "Density islands plus noise; useful for outliers, often unstable in high-dimensional embeddings.",
    }
    return explanations.get(model_name, "")
