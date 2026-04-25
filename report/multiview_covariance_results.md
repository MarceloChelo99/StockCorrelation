# Multi-View Covariance Results

Canonical run: `experiments/20260424_032154_decomposed`.

## Headline

The multi-view factor covariance estimator does not beat Ledoit-Wolf on the full universe. After adding systematic-variance calibration, the multi-view model realizes annual variance of `0.01522` versus Ledoit-Wolf at `0.00963` and sample covariance at `0.00966`.

| Method | Daily variance | Annual variance | Annual Sharpe | Mean turnover |
| --- | ---: | ---: | ---: | ---: |
| Sample | 0.00003834 | 0.00966 | 1.432 | 0.233 |
| Ledoit-Wolf | 0.00003820 | 0.00963 | 1.407 | 0.224 |
| Single-view embedding prior | 0.00004004 | 0.01009 | 1.233 | 0.220 |
| Multi-view factor model | 0.00006041 | 0.01522 | 0.276 | 0.332 |

Mean daily variance difference versus Ledoit-Wolf is `+0.00002165`, with bootstrap CI `[+0.00001439, +0.00003130]`. Lower is better, so this is a clear negative result.

## Leave-One-View-Out

Dropping growth improves the factor model most, but no ablation beats Ledoit-Wolf.

| Method | Annual variance | Difference vs Ledoit-Wolf |
| --- | ---: | ---: |
| Multi-view without growth | 0.01404 | +0.00001639 daily variance |
| Multi-view without business | 0.01482 | +0.00002024 daily variance |
| Multi-view full | 0.01522 | +0.00002165 daily variance |
| Multi-view without behavioral | 0.01540 | +0.00002258 daily variance |
| Multi-view without network | 0.01691 | +0.00002889 daily variance |

Interpretation: the current factor construction is too blunt. The views are different, but stacking all soft themes into one covariance model creates a risk estimator that is less stable than the statistical benchmark.

## Slice Analysis

The multi-view estimator also fails in slices: it wins `0 / 9` tested sector/liquidity slices versus Ledoit-Wolf. Best methods by slice are sample covariance in 4 slices, Ledoit-Wolf in 4 slices, and the older single-view embedding prior in 1 slice.

Detailed slice file: `experiments/20260424_032154_decomposed/evaluation/multiview_covariance_slices.csv`.

The important research implication is not “multi-view similarity is useless.” It is narrower: this direct soft-theme factor covariance model is not the right way to turn the decomposed views into portfolio covariance. A more promising next estimator is a hybrid router or ensemble: Ledoit-Wolf as the base, with embedding/network/fundamental views used only where they add stable incremental information.
