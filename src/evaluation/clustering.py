"""Evaluate whether embedding clusters align with sector metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.base import Evaluator
from src.utils.io import ensure_dir


class ClusteringEvaluator(Evaluator):
    """Cluster latest ticker embeddings and compare clusters to sector labels."""

    name = "clustering"

    def run(self, embeddings, metadata, config, db=None, output_dir=None) -> dict:
        settings = config["evaluation"].get("clustering", {})
        label_column = settings.get("label_column", "gics_sector")
        embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
        if not embedding_columns:
            raise ValueError("Clustering evaluation requires embedding_* columns.")
        if label_column not in metadata.columns:
            raise ValueError(f"Clustering evaluation requires metadata column {label_column!r}.")

        latest = latest_embeddings_by_ticker(embeddings, embedding_columns)
        frame = latest.merge(metadata, on="ticker", how="inner")
        frame = frame.dropna(subset=[label_column]).reset_index(drop=True)
        if frame.empty:
            raise ValueError("No embeddings matched labeled metadata rows.")

        matrix = frame.loc[:, embedding_columns].astype(float).to_numpy()
        labels = frame[label_column].astype(str).to_numpy()
        k = int(settings.get("k") or len(set(labels)))
        clusters, centroids, inertia = kmeans(
            matrix,
            k=k,
            n_init=int(settings.get("n_init", 20)),
            max_iter=int(settings.get("max_iter", 200)),
            random_seed=int(config["random_seed"]),
        )
        frame["cluster"] = clusters
        frame["cluster_distance"] = distances_to_assigned_centroid(matrix, clusters, centroids)

        metrics = {
            "label_column": label_column,
            "n_tickers": int(len(frame)),
            "n_labels": int(len(set(labels))),
            "k": int(k),
            "inertia": float(inertia),
            "ari": float(adjusted_rand_index(labels, clusters)),
            "nmi": float(normalized_mutual_info(labels, clusters)),
            "matched_metadata_rows": int(len(frame)),
            "embedding_columns": embedding_columns,
        }

        if output_dir is not None:
            self._write_artifacts(frame, matrix, labels, clusters, metrics, output_dir, settings)
        return metrics

    def _write_artifacts(
        self,
        frame: pd.DataFrame,
        matrix: np.ndarray,
        labels: np.ndarray,
        clusters: np.ndarray,
        metrics: dict,
        output_dir: str | Path,
        settings: dict,
    ) -> None:
        figures_dir = ensure_dir(Path(output_dir) / "figures")
        artifacts_dir = ensure_dir(Path(output_dir) / "evaluation")

        crosstab = pd.crosstab(frame["cluster"], frame[metrics["label_column"]])
        crosstab.to_csv(artifacts_dir / "clustering_cluster_label_crosstab.csv")

        majority_label = {
            cluster: group[metrics["label_column"]].value_counts().idxmax()
            for cluster, group in frame.groupby("cluster")
        }
        disagreements = frame.copy()
        disagreements["cluster_majority_label"] = disagreements["cluster"].map(majority_label)
        disagreements = disagreements[
            disagreements[metrics["label_column"]] != disagreements["cluster_majority_label"]
        ]
        disagreements = disagreements.sort_values("cluster_distance", ascending=False)
        top_n = int(settings.get("top_disagreements", 50))
        columns = [
            "ticker",
            "date",
            "title",
            "company_name",
            metrics["label_column"],
            "gics_sub_industry",
            "cluster",
            "cluster_majority_label",
            "cluster_distance",
        ]
        disagreements.loc[:, [column for column in columns if column in disagreements.columns]].head(top_n).to_csv(
            artifacts_dir / "clustering_disagreements.csv",
            index=False,
        )

        projection = pca_2d(matrix)
        write_projection_svg(
            projection,
            labels,
            frame["ticker"].astype(str).to_numpy(),
            figures_dir / "clustering_projection.svg",
        )


def latest_embeddings_by_ticker(embeddings: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    """Return the latest embedding row for each ticker."""
    required = {"ticker", "date", *embedding_columns}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(f"Embeddings are missing required columns: {sorted(missing)}")
    latest = embeddings.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    latest = latest.sort_values(["ticker", "date"])
    return latest.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)


def kmeans(
    matrix: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a small deterministic k-means implementation."""
    if k <= 1:
        raise ValueError("k must be greater than one.")
    if len(matrix) < k:
        raise ValueError("Cannot cluster fewer observations than k.")

    rng = np.random.default_rng(random_seed)
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        centroids = initialize_centroids(matrix, k, rng)
        labels = np.zeros(len(matrix), dtype=int)
        for _ in range(max_iter):
            distances = squared_distances(matrix, centroids)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = matrix[labels == cluster]
                if len(members) == 0:
                    farthest = np.argmax(distances.min(axis=1))
                    centroids[cluster] = matrix[farthest]
                else:
                    centroids[cluster] = members.mean(axis=0)
        inertia = float(squared_distances(matrix, centroids).min(axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    assert best_labels is not None
    assert best_centroids is not None
    return best_labels, best_centroids, best_inertia


def initialize_centroids(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose k-means++ style initial centroids."""
    centroids = np.empty((k, matrix.shape[1]), dtype=float)
    first_index = int(rng.integers(0, len(matrix)))
    centroids[0] = matrix[first_index]
    closest_distance = squared_distances(matrix, centroids[:1]).reshape(-1)
    for index in range(1, k):
        total = closest_distance.sum()
        if total == 0:
            choice = int(rng.integers(0, len(matrix)))
        else:
            probabilities = closest_distance / total
            choice = int(rng.choice(len(matrix), p=probabilities))
        centroids[index] = matrix[choice]
        closest_distance = np.minimum(closest_distance, squared_distances(matrix, centroids[index : index + 1]).reshape(-1))
    return centroids


def squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances from each row to each centroid."""
    diff = matrix[:, None, :] - centroids[None, :, :]
    return np.square(diff).sum(axis=2)


def distances_to_assigned_centroid(matrix: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return Euclidean distance from each row to its assigned centroid."""
    assigned = centroids[labels]
    return np.sqrt(np.square(matrix - assigned).sum(axis=1))


def adjusted_rand_index(labels: np.ndarray, clusters: np.ndarray) -> float:
    """Compute adjusted Rand index from label and cluster assignments."""
    table = contingency_table(labels, clusters)
    sum_comb = sum(comb2(value) for value in table.ravel())
    row_comb = sum(comb2(value) for value in table.sum(axis=1))
    col_comb = sum(comb2(value) for value in table.sum(axis=0))
    total_comb = comb2(table.sum())
    if total_comb == 0:
        return 0.0
    expected = row_comb * col_comb / total_comb
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if denominator == 0:
        return 0.0
    return float((sum_comb - expected) / denominator)


def normalized_mutual_info(labels: np.ndarray, clusters: np.ndarray) -> float:
    """Compute normalized mutual information using sqrt entropy normalization."""
    table = contingency_table(labels, clusters).astype(float)
    total = table.sum()
    if total == 0:
        return 0.0
    probabilities = table / total
    row_probabilities = probabilities.sum(axis=1)
    col_probabilities = probabilities.sum(axis=0)

    mutual_info = 0.0
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = probabilities[row, col]
            if value > 0:
                mutual_info += value * np.log(value / (row_probabilities[row] * col_probabilities[col]))

    label_entropy = entropy(row_probabilities)
    cluster_entropy = entropy(col_probabilities)
    denominator = np.sqrt(label_entropy * cluster_entropy)
    if denominator == 0:
        return 0.0
    return float(mutual_info / denominator)


def contingency_table(labels: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Build a contingency table for labels by clusters."""
    label_values = {value: index for index, value in enumerate(sorted(set(labels)))}
    cluster_values = {value: index for index, value in enumerate(sorted(set(clusters)))}
    table = np.zeros((len(label_values), len(cluster_values)), dtype=int)
    for label, cluster in zip(labels, clusters, strict=True):
        table[label_values[label], cluster_values[cluster]] += 1
    return table


def comb2(value: int | np.integer) -> float:
    """Return n choose 2."""
    number = float(value)
    return number * (number - 1.0) / 2.0


def entropy(probabilities: np.ndarray) -> float:
    """Compute entropy for a probability vector."""
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum())


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix to two dimensions with SVD."""
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    return centered @ components.T


def write_projection_svg(
    projection: np.ndarray,
    labels: np.ndarray,
    tickers: np.ndarray,
    output_path: str | Path,
) -> None:
    """Write a lightweight SVG scatterplot colored by labels."""
    output = Path(output_path)
    width = 960
    height = 720
    padding = 70
    x = scale_to_range(projection[:, 0], padding, width - padding)
    y = scale_to_range(projection[:, 1], height - padding, padding)
    label_values = sorted(set(labels))
    colors = color_palette(len(label_values))
    color_by_label = {label: colors[index] for index, label in enumerate(label_values)}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        '<text x="40" y="38" font-family="Georgia, serif" font-size="24" fill="#17211a">Embedding Clusters vs GICS Sector</text>',
        '<text x="40" y="62" font-family="Verdana, sans-serif" font-size="12" fill="#516057">Latest ticker embeddings projected to 2D with PCA</text>',
    ]
    for px, py, label, ticker in zip(x, y, labels, tickers, strict=True):
        color = color_by_label[label]
        lines.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.2" fill="{color}" fill-opacity="0.78">'
            f"<title>{ticker}: {label}</title></circle>"
        )

    legend_x = width - 300
    legend_y = 90
    for index, label in enumerate(label_values):
        y_pos = legend_y + index * 24
        lines.append(f'<circle cx="{legend_x}" cy="{y_pos}" r="5" fill="{color_by_label[label]}"/>')
        lines.append(
            f'<text x="{legend_x + 12}" y="{y_pos + 4}" font-family="Verdana, sans-serif" '
            f'font-size="12" fill="#26312a">{escape_svg(str(label))}</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def scale_to_range(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Scale numeric values into a plotting range."""
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return np.full_like(values, (minimum + maximum) / 2.0, dtype=float)
    return minimum + (values - low) * (maximum - minimum) / (high - low)


def color_palette(count: int) -> list[str]:
    """Return distinct colors for a modest number of labels."""
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#264653",
        "#e76f51",
    ]
    if count <= len(base):
        return base[:count]
    return [base[index % len(base)] for index in range(count)]


def escape_svg(value: str) -> str:
    """Escape text for SVG."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
