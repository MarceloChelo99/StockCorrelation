# Point-In-Time Audit

This report checks whether the current local artifacts expose the fields and invariants needed for point-in-time evaluation. It is a diagnostic, not a formal proof that no leakage exists.

## Summary

- Pass: `26`
- Warn: `2`
- Fail: `0`

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| as_of_merge invariant | pass | Synthetic merge never selected future feature rows. |
| feature artifact event_item_frequency.parquet | pass | rows=87,282, tickers=502, date_range=2010-12-31 to 2026-04-22, duplicate_keys=0 |
| fundamentals.parquet | warn | Missing ticker/date columns: ['date'] |
| feature artifact growth_lifecycle.parquet | pass | rows=92,887, tickers=503, date_range=2010-01-29 to 2026-04-22, duplicate_keys=0 |
| feature artifact network_position.parquet | pass | rows=92,887, tickers=503, date_range=2010-01-29 to 2026-04-22, duplicate_keys=0 |
| feature artifact price_liquidity.parquet | pass | rows=92,887, tickers=503, date_range=2010-01-29 to 2026-04-22, duplicate_keys=0 |
| feature artifact price_momentum.parquet | pass | rows=92,887, tickers=503, date_range=2010-01-29 to 2026-04-22, duplicate_keys=0 |
| feature artifact price_volatility.parquet | pass | rows=92,887, tickers=503, date_range=2010-01-29 to 2026-04-22, duplicate_keys=0 |
| feature artifact text_business.parquet | pass | rows=487, tickers=487, date_range=2025-05-09 to 2026-04-22, duplicate_keys=0 |
| feature artifact text_historical.parquet | pass | rows=87,282, tickers=502, date_range=2010-12-31 to 2026-04-22, duplicate_keys=0 |
| feature artifact text_risk.parquet | pass | rows=496, tickers=496, date_range=2025-05-09 to 2026-04-22, duplicate_keys=0 |
| data/processed/historical_text manifest | pass | forms=['10-K'], filings=7,498, embedding_rows=23,391, failures=1 |
| historical_section_embeddings.parquet | pass | rows=23,391, tickers=500, sections=4, filing_date_missing=0 |
| historical_section_topic_counts.parquet | pass | rows=23,567, tickers=500, sections=4, filing_date_missing=0 |
| data/processed/historical_text_10k_10q manifest | pass | forms=['10-K', '10-Q'], filings=29,352, embedding_rows=115,549, failures=2 |
| historical_section_embeddings.parquet | pass | rows=115,549, tickers=500, sections=11, filing_date_missing=0 |
| historical_section_topic_counts.parquet | pass | rows=125,359, tickers=500, sections=11, filing_date_missing=0 |
| relationships filing_date | pass | rows=676, missing_filing_date=0, date_range=2025-05-09 to 2026-04-22 |
| fundamentals filing lag fields | warn | rows=729,163, tickers=500, concepts=14, missing_filing_date=0, raw_end_date_after_filing=9, annual_panel_rows_after_pit_filter=7,466 |
| registered producer price_volatility | pass | source=prices, keys=['ticker', 'date'], feature_columns=3 |
| registered producer price_momentum | pass | source=prices, keys=['ticker', 'date'], feature_columns=3 |
| registered producer price_liquidity | pass | source=prices, keys=['ticker', 'date'], feature_columns=4 |
| registered producer event_item_frequency | pass | source=eight_k_events, keys=['ticker', 'date'], feature_columns=24 |
| registered producer growth_lifecycle | pass | source=sec_companyfacts, keys=['ticker', 'date'], feature_columns=17 |
| registered producer network_position | pass | source=relationships, keys=['ticker', 'date'], feature_columns=10 |
| registered producer text_historical | pass | source=historical_section_embeddings, keys=['ticker', 'date'], feature_columns=64 |
| registered producer text_business | pass | source=ten_k_sections, keys=['ticker', 'date'], feature_columns=32 |
| registered producer text_risk | pass | source=ten_k_sections, keys=['ticker', 'date'], feature_columns=32 |

## Known Limitations

- `report/theme_labels.csv` contains manually curated theme labels assigned with full-sample knowledge. Theme loadings may be point-in-time, but the human-readable labels are interpretive and full-sample.
- Dashboard PCA projections are fit on all available points for a stable visual map. The projection therefore knows the future geometry, even when the underlying firm-date loadings are point-in-time.
- GICS metadata is currently static over the sample and may not reflect historical sector or sub-industry classifications.
- Model retraining on a longer sample can change embeddings for earlier observations through learned weights and normalization. For strict PIT backtests, train models only on data available up to the training cutoff and freeze them for that evaluation window.
- The audit verifies artifact structure and selected invariants. It does not inspect every row-level source dependency in raw SEC filings.
