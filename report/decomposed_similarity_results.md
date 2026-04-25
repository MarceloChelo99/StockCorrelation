# Decomposed Similarity Results

Canonical run: `experiments/20260424_032154_decomposed`.

## View Coverage

| View | Embedding rows | Tickers | GMM themes | Notes |
| --- | ---: | ---: | ---: | --- |
| Business | 87,282 | 502 | 35 | Reused existing semantic embedding. |
| Behavioral | 87,282 | 502 | 40 | Reused existing risk embedding. |
| Growth/lifecycle | 83,380 | 499 | 40 | XBRL fundamentals with median imputation; missing rate before imputation was 34.5%. |
| Network | 87,282 | 502 | 37 | Static relationship-graph features repeated across dates. |

## Cross-View Redundancy

The view-comparison evaluator reports mean off-diagonal NMI of `0.248` and max off-diagonal NMI of `0.312`. That supports the decomposed-similarity thesis: the views are related, but not redundant.

| Pair | NMI |
| --- | ---: |
| Behavioral / Business | 0.312 |
| Business / Network | 0.302 |
| Behavioral / Network | 0.257 |
| Business / Growth | 0.214 |
| Behavioral / Growth | 0.208 |
| Growth / Network | 0.197 |

## Peer-Horizon Matrix

All views still trail GICS sub-industry peers on forward return correlation, but the ranking is informative. Behavioral similarity is the strongest return-co-movement view, with its best gap at 126 trading days (`-0.071`). Business similarity improves with horizon, from `-0.272` at 21 days to `-0.177` at 504 days. Growth also improves with horizon, from `-0.180` to `-0.132`.

This reinforces the earlier interpretation: embedding similarity captures economically meaningful structure, but GICS sub-industry remains a very strong benchmark for short- and medium-horizon return co-movement.

Detailed matrix: `experiments/20260424_032154_decomposed/evaluation/multiview_peer_horizon_matrix.csv`.
