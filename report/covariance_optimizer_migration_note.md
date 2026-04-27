# Covariance Optimizer Migration Note

This note documents the consolidation change that routes covariance portfolio
construction through one shared cvxpy helper:

```text
src/applications/portfolio_optimization.py::minimum_variance_long_only
```

The old active-set analytical solver now lives in
`src/applications/_legacy_analytical_optimizer.py` and is used only for
regression tests.

## What changed

- Single-view covariance evaluation now uses the shared cvxpy helper.
- Multi-view covariance evaluation now uses the same helper.
- CLARABEL is the primary solver.
- SCS is the only fallback and is logged if used.
- There is no silent analytical fallback in production paths.

## Regression check

A unit test compares cvxpy weights to the legacy analytical solver on a simple
positive-definite covariance matrix, with tolerance `1e-5`.

I also reran the real `risk_embedding` covariance evaluation on
`experiments/20260423_232623_risk_embedding/embeddings.parquet`.

| Method | Previous Analytical Ann. Var. | Shared cvxpy Ann. Var. | Direction |
| --- | ---: | ---: | --- |
| `ledoit_wolf` | `0.009690` | `0.009625` | slightly lower |
| `sample` | `0.009781` | `0.009661` | slightly lower |
| `embedding_prior` | `0.009943` | `0.009996` | slightly higher |

The migration does not change the research conclusion: Ledoit-Wolf remains the
best full-universe covariance benchmark, sample covariance is close, and the
embedding-prior shrinkage estimator does not beat Ledoit-Wolf.

The exact realized variances moved slightly because the production optimizer is
now cvxpy rather than the older active-set approximation. That is an acceptable
auditing tradeoff: one standard optimization path is easier to reason about
than two subtly different solvers.
