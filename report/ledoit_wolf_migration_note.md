# Ledoit-Wolf Migration Note

This note documents the consolidation change that makes `sklearn.covariance.LedoitWolf`
the primary covariance benchmark in `src/evaluation/covariance.py`.

## What changed

- `ledoit_wolf_covariance()` now delegates to sklearn's standard `LedoitWolf`.
- The previous local estimator is retained as `_legacy_constant_variance_shrinkage()`.
- The covariance evaluator now reports both `ledoit_wolf` and
  `legacy_constant_variance` so old reports can be compared directly.
- The headline benchmark should be `ledoit_wolf`.

## Migration check

I reran the covariance evaluator on:

- Config: `risk_embedding`
- Embeddings: `experiments/20260423_232623_risk_embedding/embeddings.parquet`
- Rebalance dates: `26`
- Return observations: `2,184`

| Method | Realized Annual Variance | Annual Sharpe | Mean Turnover |
| --- | ---: | ---: | ---: |
| `ledoit_wolf` | `0.009625` | `1.407` | `0.224` |
| `legacy_constant_variance` | `0.009625` | `1.407` | `0.224` |
| `sample` | `0.009661` | `1.432` | `0.233` |
| `embedding_prior` | `0.009996` | `1.172` | `0.365` |

In this run, sklearn Ledoit-Wolf and the legacy constant-variance shrinker produce
the same realized portfolio path to numerical precision. On the first valid
rebalance window (`2024-01-31`, 150 assets), the relative Frobenius norm
difference between the covariance matrices was `2.87e-17`.

The table above uses the consolidated cvxpy optimizer. Before optimizer
consolidation, the analytical active-set path gave `ledoit_wolf=0.009690`,
`sample=0.009781`, and `embedding_prior=0.009943`. The small numerical shift is
from the optimizer migration, not the Ledoit-Wolf estimator migration. The
ranking and interpretation are unchanged.

## Interpretation

The previous headline result is unchanged numerically for this experiment:
Ledoit-Wolf remains the strongest full-universe covariance benchmark, sample
covariance is close behind, and the single-view embedding-prior estimator does
not beat Ledoit-Wolf.

The important improvement is auditability. Reviewers can now see that
`ledoit_wolf` comes from sklearn's published implementation, while the older
local shrinker is explicitly labeled as legacy rather than silently carrying the
benchmark name.
