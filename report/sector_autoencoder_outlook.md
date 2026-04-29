# Sector Autoencoder Outlook Ablation

> Audit note: this report was generated with `expanding` sector-AE embeddings. That mode is now treated as diagnostic only because it refits a new autoencoder each month, so embedding axes can drift across dates. Rerun `scripts/analysis/sector_autoencoder_outlook.py` with the default `fixed_initial` mode before using these numbers in the presentation.

Fit mode: `expanding`

This analysis tests whether a sector-level autoencoder can compress the stocks inside each GICS sector into a useful sector-state embedding for predicting sector excess returns.

Construction:

- Aggregate company price, valuation, and growth/lifecycle features within each `(date, GICS sector)` using mean, median, and standard deviation.
- Train an autoencoder on those sector-month states.
- Compare Ridge walk-forward sector prediction using raw features, sector-AE embeddings only, and raw features plus sector-AE embeddings.

Sector-state rows: `2,156`
Sector-state features: `111`
Embedding rows: `1,903`

## Results

| feature_set | fit_mode | n_features | n_prediction_dates | mean_rank_ic | mean_top_minus_bottom | top_sector_hit_rate | ending_capital | spy_ending_capital | excess_total_return_vs_spy | max_drawdown | first_rebalance_date | latest_exit_date | mean_selected_embeddings | selected_embedding_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_baseline | expanding | 20 | 153 | 0.0983 | 0.0115 | 0.5425 | $68,674 | $54,701 | 1.3972 | -0.1633 | 2013-04-30 | 2026-02-03 |  |  |
| expanding_sector_ae_only | expanding | 8 | 153 | 0.0359 | 0.0009 | 0.5686 | $62,487 | $54,701 | 0.7786 | -0.2720 | 2013-04-30 | 2026-02-03 |  |  |
| raw_plus_expanding_sector_ae | expanding | 28 | 153 | 0.0870 | 0.0100 | 0.5033 | $68,094 | $54,701 | 1.3393 | -0.1633 | 2013-04-30 | 2026-02-03 |  |  |
| corr_filtered_expanding_sector_ae_only | expanding | 4 | 153 | 0.0642 | 0.0076 | 0.6275 | $69,521 | $54,701 | 1.4820 | -0.1765 | 2013-04-30 | 2026-02-03 | 2.98 | group_embedding_1:136, group_embedding_4:99, group_embedding_0:85, group_embedding_2:82, group_embedding_5:45, group_embedding_3:11, group_embedding_6:5, group_embedding_7:5 |
| raw_plus_corr_filtered_expanding_sector_ae | expanding | 24 | 153 | 0.0836 | 0.0084 | 0.5294 | $73,836 | $54,701 | 1.9134 | -0.1633 | 2013-04-30 | 2026-02-03 | 2.98 | group_embedding_1:136, group_embedding_4:99, group_embedding_0:85, group_embedding_2:82, group_embedding_5:45, group_embedding_3:11, group_embedding_6:5, group_embedding_7:5 |

## Embedding Correlation Diagnostics

This table is a full-sample diagnostic ranking of sector-AE dimensions by absolute Spearman correlation with future sector excess return. It is not used directly for the point-in-time filter.

| embedding | n | pearson_corr | spearman_corr | abs_spearman_corr |
| --- | --- | --- | --- | --- |
| group_embedding_1 | 1859 | -0.0829 | -0.0865 | 0.0865 |
| group_embedding_2 | 1859 | 0.0455 | 0.0643 | 0.0643 |
| group_embedding_0 | 1859 | 0.0636 | 0.0525 | 0.0525 |
| group_embedding_3 | 1859 | 0.0542 | 0.0434 | 0.0434 |
| group_embedding_4 | 1859 | 0.0196 | 0.0279 | 0.0279 |
| group_embedding_6 | 1859 | 0.0449 | 0.0157 | 0.0157 |
| group_embedding_7 | 1859 | 0.0221 | 0.0140 | 0.0140 |
| group_embedding_5 | 1859 | 0.0154 | 0.0046 | 0.0046 |

Point-in-time filter settings:

- Minimum absolute training-window Spearman correlation: `0.050`
- Maximum selected embedding dimensions per prediction date: `4`

## Interpretation

Best mean rank IC: `raw_baseline`.
Best ending capital: `raw_plus_corr_filtered_expanding_sector_ae`.

The sector-AE concept has some signal if the embedding-only model produces positive rank IC and a competitive rotation simulation. The key question is whether it improves the raw-feature baseline. The correlation-filtered rows test whether weak embedding dimensions should be dropped rather than handed to Ridge.

Current recommendation: treat this as an exploratory numerical-embedding ablation, not a presentation headline, unless the filtered or hybrid version consistently improves rank IC/top-bottom spread under point-in-time settings.
