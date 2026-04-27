"""Deprecated analytical minimum-variance solver kept for regression tests.

Production evaluators use ``src.applications.portfolio_optimization`` so that
all portfolio construction follows one auditable cvxpy path.
"""
from __future__ import annotations

import numpy as np


def minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """Return long-only minimum-variance weights using a simple active-set solve."""
    n_assets = len(covariance)
    active = np.arange(n_assets)
    weights = np.zeros(n_assets)
    for _ in range(n_assets):
        sub_covariance = covariance[np.ix_(active, active)]
        ones = np.ones(len(active))
        try:
            raw = np.linalg.solve(sub_covariance, ones)
        except np.linalg.LinAlgError:
            raw = np.linalg.pinv(sub_covariance) @ ones
        if raw.sum() <= 0.0:
            raw = np.ones(len(active))
        active_weights = raw / raw.sum()
        if np.all(active_weights >= -1e-10):
            weights[active] = np.clip(active_weights, 0.0, None)
            weights = weights / weights.sum()
            return weights

        keep = active_weights > 1e-10
        if not np.any(keep):
            keep[np.argmax(active_weights)] = True
        active = active[keep]

    weights[active] = 1.0 / len(active)
    return weights
