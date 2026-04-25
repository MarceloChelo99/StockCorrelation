# Hybrid Covariance Policy

This is an in-sample routing summary from the covariance slice analysis, not an out-of-sample production estimator.
It asks where embedding-prior shrinkage realized lower variance than Ledoit-Wolf in the current evaluation window.

- Slices favoring embedding-prior shrinkage: `3`
- Slices favoring Ledoit-Wolf: `6`

Embedding-prior slices:

| Slice Type | Slice | LW Ann Var | Emb Ann Var | Relative Improvement |
| --- | --- | ---: | ---: | ---: |
| sector | Health Care | 0.01761 | 0.01720 | 2.32% |
| liquidity | mid_liquidity | 0.01282 | 0.01263 | 1.44% |
| sector | Consumer Discretionary | 0.01861 | 0.01851 | 0.49% |

Ledoit-Wolf slices:

| Slice Type | Slice | LW Ann Var | Emb Ann Var | Relative Improvement |
| --- | --- | ---: | ---: | ---: |
| sector | Financials | 0.01717 | 0.01856 | -8.10% |
| liquidity | low_liquidity | 0.01425 | 0.01535 | -7.68% |
| sector | Industrials | 0.01717 | 0.01827 | -6.41% |
| sector | Consumer Staples | 0.01510 | 0.01542 | -2.06% |
| liquidity | high_liquidity | 0.01062 | 0.01081 | -1.81% |
| sector | Information Technology | 0.02210 | 0.02235 | -1.12% |
