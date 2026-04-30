"""Shared portfolio optimization helpers."""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.utils.logging import log


def minimum_variance_long_only(
    cov: np.ndarray,
    max_weight: float | None = None,
) -> np.ndarray:
    """Solve long-only minimum-variance portfolio weights with cvxpy.

    CLARABEL is the primary solver. SCS is used as the single fallback and is
    logged when needed. If both solvers fail, this function raises rather than
    silently switching to a different optimization method.
    """
    matrix = np.asarray(cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    matrix = (matrix + matrix.T) / 2.0

    n_assets = len(matrix)
    weights = cp.Variable(n_assets)
    constraints = [weights >= 0.0, cp.sum(weights) == 1.0]
    if max_weight is not None:
        constraints.append(weights <= float(max_weight))

    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(matrix)))
    problem = cp.Problem(objective, constraints)
    statuses: list[str] = []
    for solver in ("CLARABEL", "SCS"):
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError as exc:
            statuses.append(f"{solver}: {exc}")
            if solver == "CLARABEL":
                log("CLARABEL failed for minimum-variance solve; trying SCS.", tag="optimizer")
            continue

        statuses.append(f"{solver}: {problem.status}")
        if weights.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            if solver == "CLARABEL":
                log(f"CLARABEL returned status {problem.status}; trying SCS.", tag="optimizer")
            continue

        if solver == "SCS":
            log("Used SCS fallback for minimum-variance solve.", tag="optimizer")
        result = np.asarray(weights.value, dtype=float)
        result = np.clip(result, 0.0, None)
        if result.sum() <= 0.0:
            raise RuntimeError("Minimum-variance optimizer returned zero-sum weights.")
        return result / result.sum()

    raise RuntimeError(f"Minimum-variance optimization failed: {'; '.join(statuses)}")
