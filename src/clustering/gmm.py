"""Gaussian-mixture soft clustering for embedding views."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class GMMResult:
    """Fitted GMM plus firm-date theme loadings."""

    model: GaussianMixture
    loadings: pd.DataFrame
    hard_labels: pd.Series
    bic: float
    aic: float
    n_components: int
    bic_curve: dict[int, float]


def fit_gmm_on_embeddings(
    embeddings: pd.DataFrame,
    candidate_components: range,
    random_state: int,
) -> GMMResult:
    """Fit a Gaussian mixture over embedding rows and return soft theme loadings."""
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError("Embeddings must contain embedding_* columns.")

    frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].dropna().copy()
    matrix = frame.loc[:, embedding_columns].astype(float).to_numpy()
    if len(matrix) < 2:
        raise ValueError("Need at least two embedding rows to fit a GMM.")

    bic_curve: dict[int, float] = {}
    best_model: GaussianMixture | None = None
    best_bic = float("inf")
    for n_components in candidate_components:
        if n_components >= len(matrix):
            continue
        model = GaussianMixture(
            n_components=int(n_components),
            covariance_type="diag",
            random_state=int(random_state),
            n_init=3,
            max_iter=200,
            reg_covar=1e-6,
        )
        model.fit(matrix)
        bic = float(model.bic(matrix))
        bic_curve[int(n_components)] = bic
        if bic < best_bic:
            best_bic = bic
            best_model = model

    if best_model is None:
        raise ValueError("No valid GMM component count was available.")

    probabilities = best_model.predict_proba(matrix)
    loading_columns = [f"theme_{index}" for index in range(probabilities.shape[1])]
    loadings = frame.loc[:, ["ticker", "date"]].copy()
    for index, column in enumerate(loading_columns):
        loadings[column] = probabilities[:, index]
    labels = pd.Series(best_model.predict(matrix), index=frame.index, name="cluster")
    return GMMResult(
        model=best_model,
        loadings=loadings,
        hard_labels=labels,
        bic=float(best_model.bic(matrix)),
        aic=float(best_model.aic(matrix)),
        n_components=int(best_model.n_components),
        bic_curve=bic_curve,
    )


def gmm_meta(result: GMMResult) -> dict:
    """Return JSON-serializable metadata for a fitted GMM."""
    return {
        "n_components": int(result.n_components),
        "bic": float(result.bic),
        "aic": float(result.aic),
        "bic_curve": {str(key): float(value) for key, value in result.bic_curve.items()},
        "component_means": result.model.means_.tolist(),
        "covariance_type": result.model.covariance_type,
    }
