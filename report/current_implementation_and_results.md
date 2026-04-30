# Current Implementation And Results Report

Generated: 2026-04-25  
Branch: `explore`

This report summarizes the current implementation state and the main research results from the local artifacts. It is meant for a technical reviewer who wants to understand what exists, how the pipeline is organized, what was tested, and what conclusions are currently supported.

## Executive Summary

The project is now an end-to-end stock similarity and embedding system. It ingests SEC-derived filing data, streams historical section embeddings, computes point-in-time features, assembles monthly panels, trains PyTorch embedding models, evaluates multiple notions of company similarity, and exposes the results in a Streamlit dashboard.

The strongest current result is business/semantic structure. Historical text embeddings recover meaningful company groupings, produce economically sensible cross-sector peers, and show strong thematic language movement, especially the post-2023 rise of AI and cybersecurity language.

The strongest current negative result is risk/covariance. Embedding-informed covariance estimators do not beat standard Ledoit-Wolf on the full universe. Some slice-level wins exist, but the direct multi-view factor covariance model is too unstable and should not be treated as a better full-universe risk estimator.

The most recent temporal autoencoder work improves trajectory smoothness and preserves clustering structure, but it does not yet pass event-detection or strong AI-drift validation. That is useful: the current temporal model is good for a cleaner dashboard, not yet good enough to claim reliable strategic-shift detection.

## Technology Stack

Runtime and core libraries are defined in `pyproject.toml`:

| Component | Current choice | Purpose |
| --- | --- | --- |
| Python | `>=3.14` | Project runtime |
| NumPy / pandas | `numpy>=2.4`, `pandas>=3.0` | Numeric and tabular data |
| PyArrow | `pyarrow>=22.0` | Parquet IO |
| PyYAML | `pyyaml>=6.0` | Config loading |
| PyTorch | `torch==2.10.0` | Autoencoder and temporal autoencoder models |
| sentence-transformers | `5.4.1` | MiniLM filing-section embeddings |
| scikit-learn | `>=1.8` | PCA, Ledoit-Wolf, clustering utilities where needed |
| networkx | `>=3.6` | Relationship graph and network-position features |
| cvxpy | `>=1.8` | Long-only minimum-variance portfolio optimization |
| Streamlit | `>=1.50` | Dashboard |

The codebase is still research code, but the main paths now use standard library choices: PyTorch for neural models, sklearn's `LedoitWolf` for the covariance benchmark, and one shared cvxpy portfolio optimizer.

## Current Pipeline Structure

The implementation now follows the ETL-style stage design more closely:

```text
scripts/
├── pipeline/
│   ├── stage_01_ingest/
│   ├── stage_02_features/
│   ├── stage_03_assembly/
│   ├── stage_04_model/
│   ├── stage_05_evaluation/
│   └── run_experiment.py
├── historical_text/
├── decomposed/
├── analysis/
├── audit/
├── data_setup/
└── temporal_validation/
```

Root-level wrappers remain for common commands such as:

```text
scripts/02_compute_features.py
scripts/03_assemble_dataset.py
scripts/04_train.py
scripts/05_evaluate.py
scripts/run_experiment.py
scripts/24_stream_historical_text_features.py
```

Path manipulation is centralized through `scripts/_bootstrap.py`, rather than repeated `sys.path` edits in every script.

## Implemented Data And Feature Layers

### Historical Filing Text Stream

The historical text pipeline now streams filing sections, computes topic counts, embeds sections with MiniLM, and stores compact artifacts rather than full raw filing text.

Current expanded artifact:

```text
data/processed/historical_text_10k_10q/
```

Current coverage:

| Artifact | Rows | Tickers | Size |
| --- | ---: | ---: | ---: |
| `historical_section_embeddings.parquet` | 115,549 | 500 | 277 MB |
| `historical_section_topic_counts.parquet` | 125,359 | 500 | 2.65 MB |
| `text_historical.parquet` first-class feature | 87,282 | 502 | 14.6 MB |
| `historical_text_business.parquet` assembled dataset | 87,282 | 502 | 14.7 MB |

The expanded text artifact includes `10-K` and `10-Q` filings since 2010:

| Form | Section embedding rows | Tickers |
| --- | ---: | ---: |
| `10-K` | 43,665 | 500 |
| `10-Q` | 71,884 | 500 |

Actual embedded sections in the current artifact:

| Section | Rows | Tickers | Median chars |
| --- | ---: | ---: | ---: |
| `q_mda` | 21,399 | 496 | 54,268 |
| `q_market_risk` | 19,532 | 475 | 3,001 |
| `q_legal_proceedings` | 15,773 | 458 | 520 |
| `q_risk_factors` | 15,180 | 492 | 914 |
| `mda` | 7,387 | 499 | 78,897 |
| `risk_factors` | 7,302 | 496 | 57,696 |
| `business` | 7,232 | 497 | 44,342 |
| `item_2_properties` | 7,130 | 494 | 1,780 |
| `item_3_legal_proceedings` | 6,576 | 495 | 3,856 |
| `item_7a_market_risk` | 6,568 | 472 | 4,549 |
| `item_1c_cybersecurity` | 1,470 | 486 | 6,973 |

The parser/manifest has support for additional section keys such as 8-K, proxy, and S-1 sections, but the completed current artifact is the 10-K + 10-Q run.

### First-Class Feature Producers

Active feature producers are explicit in `src/features/registry.py`:

| Feature group | Module | Purpose |
| --- | --- | --- |
| `price_volatility` | `src/features/price/volatility.py` | Rolling volatility features |
| `price_momentum` | `src/features/price/momentum.py` | Rolling momentum features |
| `price_liquidity` | `src/features/price/liquidity.py` | Volume/liquidity features |
| `event_item_frequency` | `src/features/events/item_frequency.py` | 8-K item-frequency windows |
| `growth_lifecycle` | `src/features/fundamentals/growth_lifecycle.py` | XBRL-derived growth, margins, leverage, payout |
| `network_position` | `src/features/graph/network_position.py` | Relationship graph centrality/community features |
| `text_historical` | `src/features/text/historical_text_features.py` | Point-in-time historical filing text embeddings |
| `text_business` | `src/features/text/business_description.py` | Legacy latest-filing hashed text features |
| `text_risk` | `src/features/text/risk_factors.py` | Legacy latest-filing hashed text features |

The new preferred business feature source is `text_historical`, not the older latest-filing `text_business` / `text_risk` producers.

### Fundamentals And Network Data

Fundamentals:

- Source: SEC companyfacts XBRL endpoint.
- Artifact: `data/processed/features/fundamentals.parquet`.
- Coverage: `729,163` rows, `500` tickers, `14` concepts.
- Growth/lifecycle monthly feature panel: `92,887` rows, `503` tickers.

Relationship graph:

- Artifact: `data/processed/relationships/relationships.parquet`.
- First-pass graph: `676` edges.
- Network features: `92,887` monthly rows, `503` tickers.
- Current limitation: network features are static across history because the graph is currently pooled from available relationship filings.

## Implemented Model Layer

The model layer is now standardized on PyTorch.

| Model | Registry key | Status |
| --- | --- | --- |
| PyTorch autoencoder | `autoencoder` | Main nonlinear embedding model |
| Temporal autoencoder | `temporal_autoencoder` | Experimental business-view trajectory model |
| PCA model | `pca` | Lightweight baseline |
| Legacy NumPy autoencoder | `_legacy_numpy_autoencoder.py` | Retained only for backward compatibility/regression |

Key model-layer changes:

- `Autoencoder` is now a real `torch.nn.Module`.
- It supports multi-layer `hidden_dims`.
- It uses `Linear -> BatchNorm1d -> ReLU -> Dropout` blocks.
- It uses Adam, MSE reconstruction loss, validation split, and early stopping.
- Normalization statistics are stored as model buffers and saved with weights.
- `scripts/04_train.py` and view-training paths go through the model registry.

The temporal autoencoder adds adaptive smoothness:

```text
loss = reconstruction_loss + lambda_temp * exp(-alpha * feature_change) * embedding_change^2
```

This penalizes month-to-month embedding movement when the underlying text features did not change much, while allowing movement when the input features genuinely changed.

## Implemented Evaluation Layer

Evaluators are explicit in `src/evaluation/registry.py`:

| Evaluator | Registry key | Purpose |
| --- | --- | --- |
| Clustering | `clustering` | K-means vs GICS labels, ARI/NMI |
| Peers | `peers` | Embedding nearest-neighbor return correlation vs GICS peers |
| Covariance | `covariance` | Sample, sklearn Ledoit-Wolf, legacy shrinkage, embedding-prior covariance |
| Relationship graph | `relationships` | Whether embedding peers share disclosed graph links |
| Multi-view covariance | `multiview_covariance` | Soft-theme factor covariance estimator |
| Multi-view peers | `multiview_peers` | View-by-horizon peer correlation matrix |
| View comparison | `view_comparison` | Cross-view NMI redundancy test |

Portfolio optimization is consolidated in:

```text
src/applications/portfolio_optimization.py
```

Both covariance evaluators use the same cvxpy helper with CLARABEL primary and SCS fallback.

## Dashboard Implementation

The Streamlit dashboard lives in:

```text
apps/raw_filing_browser/app.py
```

Current dashboard tabs:

| Tab | What it does |
| --- | --- |
| Filing Browser | Browse structured and raw SEC filing sections |
| Historical Text | Explore historical topic trends, company changes, and semantic movement maps |
| Similarity Shifts | Inspect theme/loadings changes and theme labels |
| Market Map | View stocks as points in a 2D plane and step through time |

Recent dashboard improvements:

- Uses the expanded `historical_text_10k_10q` artifact by default.
- Supports selecting historical text artifact directories.
- Adds historical semantic movement maps from section embeddings.
- Adds company-level topic delta summaries.
- Adds semantic drift leaderboards.
- Adds smoother market-map stepping instead of only full autoplay.
- Supports manual theme labels through `report/theme_labels.csv`.

Important dashboard caveat: PCA projections are fit over all available points for stable visual geometry. The underlying feature dates are point-in-time, but the visualization projection itself is not a strict backtest object.

## Main Results

### Historical Text And Thematic Language

The historical text data shows a strong AI-language regime shift after 2023.

Business-section AI score:

| Year | AI score |
| ---: | ---: |
| 2023 | 3.389 |
| 2024 | 5.693 |
| 2025 | 6.909 |
| 2026 partial | 7.112 |

Risk-factor AI score:

| Year | AI score |
| ---: | ---: |
| 2023 | 1.576 |
| 2024 | 6.803 |
| 2025 | 9.988 |
| 2026 partial | 11.220 |

Largest company-level AI-language increases include:

```text
EPAM, ADBE, ADP, AMZN, EFX, PANW, NVDA, CDNS, NOW, INTU,
ORCL, QCOM, MSFT, META, VRSK, ZBH, PAYX, SNPS, AXP, JKHY
```

Sector-level increases are strongest in Information Technology, but Communication Services, Industrials, Financials, Consumer Discretionary, Consumer Staples, Health Care, Real Estate, Energy, Utilities, and Materials all show increases. This supports the dashboard use case of finding AI adoption or AI-risk language outside the obvious tech sector.

### Embedding Quality: Business, Risk, Historical Text, Temporal

| Experiment | ARI vs GICS | NMI vs GICS | Embedding peer corr | GICS peer corr | Peer diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy semantic MiniLM text | 0.219 | 0.381 | 0.382 | 0.653 | -0.272 |
| Legacy risk/price embedding | 0.052 | 0.171 | 0.553 | 0.653 | -0.100 |
| Historical-text business AE | 0.255 | 0.433 | 0.527 | 0.653 | -0.127 |
| Temporal historical-text business AE | 0.269 | 0.412 | 0.534 | 0.653 | -0.120 |

Interpretation:

- Text/business embeddings align much better with GICS sectors than price/risk embeddings.
- Price/risk embeddings are better for short-horizon return co-movement.
- Historical text embeddings improved semantic clustering materially over the earlier MiniLM text-only baseline.
- The first temporal autoencoder run smooths trajectories and slightly improves ARI/peer diff, but NMI is a bit lower than the vanilla historical-text run.

### Qualitative Peer Examples

The semantic embedding continues to produce useful cross-sector peers:

| Seed | Embedding peers | Interpretation |
| --- | --- | --- |
| `META` | `WDAY`, `CRM`, `INTU`, `APP` | Platform/software economics not captured by Communication Services label |
| `ETN` | `NEE`, `AES`, `SRE` | Electrification/grid-exposure theme |
| `CBRE` | `ARES`, `BX`, `KKR`, `IVZ`, `IBKR` | Real estate services linked to capital-markets exposure |
| `SBUX` | `KDP`, `HSY`, `YUM`, `PEP`, `KHC` | Consumer brand, beverage, restaurant overlap |

These examples remain one of the strongest qualitative results: the embedding finds thematic structure that GICS does not express cleanly.

### Peer Horizon Sweep

The peer-correlation tests still trail GICS sub-industry peers, but the horizon pattern is meaningful.

Semantic peer gap vs GICS:

| Forward days | Mean diff |
| ---: | ---: |
| 21 | -0.272 |
| 63 | -0.201 |
| 126 | -0.197 |
| 252 | -0.189 |
| 504 | -0.177 |

Risk peer gap vs GICS:

| Forward days | Mean diff |
| ---: | ---: |
| 21 | -0.100 |
| 63 | -0.072 |
| 126 | -0.071 |
| 252 | -0.072 |
| 504 | -0.079 |

Interpretation:

- GICS remains a very strong peer benchmark for return co-movement.
- Semantic similarity becomes less bad at longer horizons, consistent with structural similarity rather than immediate trading similarity.
- Risk/price similarity is strongest at medium horizons and remains the best embedding view for return co-movement.

### Decomposed Similarity Views

Canonical decomposed run:

```text
experiments/20260424_032154_decomposed
```

View coverage:

| View | Embedding rows | Tickers | GMM themes | Notes |
| --- | ---: | ---: | ---: | --- |
| Business | 87,282 | 502 | 35 | Semantic/business view |
| Behavioral | 87,282 | 502 | 40 | Price/risk behavior view |
| Growth/lifecycle | 83,380 | 499 | 40 | Fundamentals view with median imputation |
| Network | 87,282 | 502 | 37 | Static relationship graph features |

Cross-view redundancy:

- Mean off-diagonal NMI: `0.248`
- Max off-diagonal NMI: `0.312`

Interpretation: the views are related but not redundant. This supports the thesis that company similarity is not one object; business, behavioral, growth, and network similarity capture different structures.

### Relationship Graph Evaluation

Current relationship graph:

- `676` edges.
- Main relationship types: competitor, agreement, customer, partner, supplier.

Latest historical-text business relationship comparison:

| Peer set | Direct link rate |
| --- | ---: |
| Historical-text embedding peers | 0.110 |
| Temporal embedding peers | 0.119 |
| GICS sub-industry peers | 0.173 |
| Random peers | 0.010 |

Interpretation:

- Embedding peers are far more connected than random peers.
- GICS sub-industry still has stronger direct relationship overlap.
- Relationship extraction is promising but currently sparse. Better 8-K Item 1.01 agreement extraction is likely the next major improvement.

### Covariance And Portfolio Results

The full-universe covariance result remains negative versus Ledoit-Wolf.

After migration to sklearn Ledoit-Wolf and the shared cvxpy optimizer:

| Method | Annual variance | Annual Sharpe | Mean turnover |
| --- | ---: | ---: | ---: |
| `ledoit_wolf` | 0.009625 | 1.407 | 0.224 |
| `sample` | 0.009661 | 1.432 | 0.233 |
| `embedding_prior` | 0.009996 | 1.172 | 0.365 |
| `legacy_constant_variance` | 0.009625 | 1.407 | 0.224 |

The ranking did not change after consolidation. Ledoit-Wolf remains the full-universe benchmark to beat.

Slice-level embedding-prior wins:

| Slice | Ledoit-Wolf annual variance | Embedding annual variance | Relative improvement |
| --- | ---: | ---: | ---: |
| Health Care | 0.01761 | 0.01720 | 2.32% |
| Mid-liquidity | 0.01282 | 0.01263 | 1.44% |
| Consumer Discretionary | 0.01861 | 0.01851 | 0.49% |

Multi-view factor covariance:

| Method | Annual variance | Annual Sharpe | Mean turnover |
| --- | ---: | ---: | ---: |
| Sample | 0.00966 | 1.432 | 0.233 |
| Ledoit-Wolf | 0.00963 | 1.407 | 0.224 |
| Single-view embedding prior | 0.01009 | 1.233 | 0.220 |
| Multi-view factor model | 0.01522 | 0.276 | 0.332 |

Interpretation:

- The direct multi-view factor covariance estimator is a clear negative result.
- The views are useful for interpretation, but stacking soft themes into one factor covariance model is too blunt.
- The better next risk-modeling direction is a conservative hybrid: Ledoit-Wolf baseline plus embedding-informed adjustments only in slices where repeated evidence supports them.

## Temporal Autoencoder Results

Temporal model implementation:

```text
src/models/temporal_autoencoder.py
src/models/temporal_dataloader.py
scripts/temporal_validation/
config/experiments/temporal_business_view.yaml
```

First trained temporal run:

```text
experiments/20260425_165422_temporal_business_view
```

Training diagnostics:

| Metric | Value |
| --- | ---: |
| Embedding rows | 87,282 |
| Fit rows | 84,048 |
| Consecutive pairs | 83,549 |
| Dropped time gaps | 0 |
| Dropped non-finite pairs | 0 |
| Validation temporal loss | 1.0040 -> 0.0790 |
| Within-firm temporal variance ratio | 0.408 |

Validation outcomes for the first run:

| Test | Result | Main number |
| --- | --- | ---: |
| AI drift | FAIL | AI/control displacement ratio about 1.5x, below 2x |
| Stability | PASS | Temporal median YoY movement 1.214 vs vanilla 1.387 |
| Event alignment | FAIL | 0% event pass rate |

Hyperparameter sweep:

- Grid: `lambda_temp = {0.1, 0.25, 0.5, 1.0, 2.0}`, `alpha = {0.5, 1.0, 2.0}`.
- Runs: `15`.
- Overall pass: `0 / 15`.
- Stability pass: `15 / 15`.
- NMI preservation pass: `15 / 15`.
- AI drift pass: `0 / 15`.
- Event alignment pass: `0 / 15`.

Best diagnostic candidate:

| lambda | alpha | AI ratio | Stability improvement | Event pass rate | NMI ratio to vanilla | NMI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 2.0 | 1.565 | 0.159 | 0.00% | 1.144 | 0.496 |

Interpretation:

- Temporal smoothing works for reducing noise in stable firms.
- It preserves or improves cross-sectional semantic structure in the sweep.
- It does not yet detect strategic events or strong AI pivots reliably.
- The issue is likely not just `lambda/alpha`; it is probably the input/evaluation setup. Monthly business-text embeddings are flat between filing updates, and many calendar events are better captured in 8-Ks or event-specific text than in annual/quarterly business sections.

## Point-In-Time Audit And Testing

Point-in-time audit:

```text
report/point_in_time_audit.md
```

Current audit summary:

- Pass: `26`
- Warn: `2`
- Fail: `0`

Warnings:

- `fundamentals.parquet` is raw long-format and does not have standard `ticker/date` panel columns.
- Raw fundamentals contain a small number of `end_date > filing_date` cases before downstream annual-panel PIT filtering.

Known PIT limitations:

- Theme labels are manually assigned using full-sample interpretation.
- Dashboard PCA projections are fit on all available points.
- GICS metadata is static across the sample.
- Strict PIT backtests require model training only on data available up to each training cutoff.

Test status:

```text
.venv/bin/python -m unittest discover tests
```

Latest result:

```text
Ran 56 tests
OK
```

## Current Open Technical Risks

The dashboard and reports are ahead of the strictest model-validation story. The dashboard is useful, but some visual projections use full-sample geometry and should not be described as backtests.

The temporal autoencoder improves smoothness but not event detection. It should remain experimental until event-aware input features or filing-date-based validation improve the AI/event tests.

The relationship graph is sparse and static. It is useful for cross-sectional validation but not yet a true historical supply-chain/network trajectory.

The covariance research should not continue by making the multi-view factor model more complicated by default. The empirical evidence points toward hybrid routing or local adjustments around Ledoit-Wolf, not replacing Ledoit-Wolf full-universe.

The worktree currently includes a large consolidation/script reorganization pass. Reviewers should look at the new ETL-style script layout and root-level compatibility wrappers together, not as independent changes.

## Reproduction Commands

Run the main historical-text business model:

```bash
.venv/bin/python scripts/04_train.py historical_text_business
```

Evaluate a trained embedding file:

```bash
.venv/bin/python scripts/05_evaluate.py historical_text_business experiments/<run>/embeddings.parquet
```

Run the temporal business model:

```bash
.venv/bin/python scripts/04_train.py temporal_business_view
```

Run temporal validation checks:

```bash
.venv/bin/python -m scripts.temporal_validation.ai_drift_test --embeddings experiments/<temporal_run>/embeddings.parquet
.venv/bin/python -m scripts.temporal_validation.stability_test --temporal-embeddings experiments/<temporal_run>/embeddings.parquet --vanilla-embeddings experiments/<vanilla_run>/embeddings.parquet
.venv/bin/python -m scripts.temporal_validation.event_alignment_test --embeddings experiments/<temporal_run>/embeddings.parquet
```

Run the temporal hyperparameter sweep:

```bash
.venv/bin/python -m scripts.temporal_validation.hyperparameter_sweep \
  --vanilla-embeddings experiments/20260425_153806_historical_text_business/embeddings.parquet \
  --vanilla-metrics experiments/20260425_153806_historical_text_business/metrics.json
```

Run the dashboard:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py
```

Run tests:

```bash
.venv/bin/python -m unittest discover tests
```

## Bottom Line

The project now has a solid implementation foundation and several honest findings:

- Business text embeddings work well for semantic structure.
- Risk/price embeddings work better for return co-movement.
- Historical text features materially improve the business-view data layer.
- Decomposed views are not redundant.
- Relationship graph validation is promising but sparse.
- Ledoit-Wolf remains the full-universe covariance benchmark.
- Temporal smoothing helps dashboard stability but does not yet solve strategic event detection.

The next highest-value implementation is not another small model tweak. It is making event-aware text, especially 8-K agreement/results text, part of the temporal input and then validating movement around filing dates as well as calendar event dates.
