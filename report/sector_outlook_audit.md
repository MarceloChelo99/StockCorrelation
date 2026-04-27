# Sector Outlook Walk-Forward Audit

Generated: 2026-04-27

## Summary

The sector outlook model is a walk-forward ridge model for sector excess returns
versus the equal-weight S&P 500 universe. The dashboard now separates:

- **Current scores**: latest sector scores whose future returns may not be known yet.
- **Historical walk-forward**: only prediction dates whose full forward horizon completed.

This distinction matters because rows near the dataset end can have partial future
returns available. Those rows should not be included in historical backtest metrics.

## Bug Found And Fixed

The first dashboard implementation could include rows near the latest price date
whose forward horizon was only partially observed. The model itself still trained
walk-forward, but the historical metrics could accidentally score incomplete
outcomes.

Fix:

- `build_sector_feature_panel` now stores `future_days_available`.
- It marks each row with `is_horizon_complete`.
- `future_excess_return` is left blank unless the full horizon is complete.
- `sector_backtest_by_date` only scores completed rows.
- The dashboard labels current/latest rows as live/unrealized and excludes them
  from historical audit metrics.
- Each prediction row stores `training_latest_target_end_date`; the audit checks
  this is strictly before the prediction date.

## Local Audit Result

Configuration:

- Config: `decomposed_point_in_time`
- Horizon: `63` trading days
- Minimum training months: `36`
- Ridge alpha: `10.0`

Historical completed-window metrics:

| Metric | Value |
| --- | ---: |
| Completed prediction rows | 1,683 |
| Completed prediction dates | 153 |
| Unrealized rows excluded | 44 |
| Unrealized dates excluded | 4 |
| Leakage violations | 0 |
| First completed prediction date | 2013-04-30 |
| Latest completed prediction date | 2025-12-31 |
| Latest current score date | 2026-04-22 |

Performance over completed historical predictions:

| Metric | Value |
| --- | ---: |
| Mean rank IC | 0.132 |
| Median rank IC | 0.173 |
| Mean top-minus-bottom excess return | 1.38% |
| Mean top-bucket excess return | 0.28% |
| Top-sector hit rate | 60.1% |

Recent completed historical predictions:

| Prediction date | Horizon end | Rank IC | Top-bottom | Predicted top | Realized best |
| --- | --- | ---: | ---: | --- | --- |
| 2025-08-29 | 2025-11-28 | 0.500 | 8.55% | Health Care | Information Technology |
| 2025-09-30 | 2025-12-30 | 0.400 | 4.78% | Health Care | Health Care |
| 2025-10-31 | 2026-02-03 | -0.427 | -10.48% | Communication Services | Materials |
| 2025-11-28 | 2026-03-03 | -0.682 | -13.29% | Communication Services | Energy |
| 2025-12-31 | 2026-04-02 | -0.336 | -8.81% | Communication Services | Energy |

## Interpretation

The signal is positive on average over the full completed history, but it had a
rough recent stretch in late 2025. This is exactly why the dashboard now exposes
the historical date-by-date audit: the headline average is not enough; we need to
see when the walk-forward model helped and when it failed.

The current/latest sector score is still useful as an exploratory signal, but it
should not be counted as historical evidence until its full forward horizon has
completed.
