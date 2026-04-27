# Stock Embeddings Project Specification And Class Insights

Generated: 2026-04-27  
Branch: `explore`

## 1. Executive Summary

This project builds interpretable stock/company embeddings for the S&P 500 universe. The core idea is that "similar companies" can mean several different things:

- Companies can be similar because they describe similar businesses.
- Stocks can be similar because their returns, volatility, and liquidity behave similarly.
- Firms can be similar because they are at similar lifecycle stages: high growth, mature, capital-intensive, highly levered, etc.
- Firms can be similar because they are connected through competitors, suppliers, customers, partners, or agreements.

The project therefore evolved into a **decomposed similarity framework** with four views:

| View | Main data | What it tries to capture |
| --- | --- | --- |
| Business / semantic | 10-K and 10-Q filing-section embeddings | What the company does, how it describes risks, strategy, products, markets |
| Behavioral / risk | Daily prices | How the stock trades: momentum, volatility, liquidity, co-movement |
| Growth / lifecycle | SEC XBRL fundamentals plus valuation features | Firm maturity, profitability, leverage, capital intensity, payout, market-implied valuation |
| Network | Filing-derived relationships | Disclosed customers, suppliers, competitors, partners, agreements |

The biggest current finding is **not** that one embedding beats every benchmark. It is that each view captures a different notion of similarity. Business embeddings produce meaningful cross-sector peer groups and thematic language shifts. Behavioral embeddings are stronger for return co-movement. Ledoit-Wolf remains the stronger full-universe covariance estimator. The dashboard is now the main exploratory surface for showing how these views differ.

## 2. Project Goal

The original goal was to learn a low-dimensional vector representation for each public company that captures:

- What the company does.
- How its stock behaves.
- What events and relationships it experiences.

The embeddings are evaluated through several downstream questions:

1. Do company embeddings align with GICS sectors or reveal meaningful cross-sector structure?
2. Do embedding-nearest peers have stronger forward return correlation than GICS peers?
3. Can embedding similarity improve covariance estimation relative to Ledoit-Wolf?
4. Do decomposed views reveal different kinds of similarity rather than duplicating one another?
5. Can historical filing language expose strategic shifts, such as AI adoption or cybersecurity risk?
6. Can sector-level features predict sector returns relative to the broad S&P 500 universe in a historical walk-forward setup?

The project is research code, so the priority order is:

1. Correctness of results.
2. Transparency of logic.
3. Iteration speed.
4. Reproducibility.
5. Maintainability.

## 3. Current Repository Architecture

Top-level structure:

```text
StockCorrelation/
├── apps/
│   └── raw_filing_browser/        # Streamlit dashboard
├── config/
│   ├── default.yaml               # Base config
│   └── experiments/               # Experiment overrides
├── data/
│   ├── raw_filing_corpora/        # Local SEC filing corpus + SQLite DB
│   └── processed/                 # Stable parquet/json artifacts
├── experiments/                   # Per-run model/evaluation outputs
├── libraries/
│   └── market_data_fetcher/       # Local SEC/database helper library
├── report/                        # Research notes, audits, result summaries
├── scripts/                       # ETL, analysis, historical text, audit workflows
├── src/                           # Reusable library code
└── tests/                         # Unit/smoke tests
```

The implementation follows an ETL-style pipeline:

```text
ingest -> features -> assembly -> model -> evaluation -> dashboard/report
```

Detailed stage layout:

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

Root-level compatibility wrappers still exist for common commands:

```text
scripts/02_compute_features.py
scripts/03_assemble_dataset.py
scripts/04_train.py
scripts/05_evaluate.py
scripts/run_experiment.py
scripts/24_stream_historical_text_features.py
```

## 4. Main Libraries And Why They Are Used

| Library | Purpose |
| --- | --- |
| Python 3.14 | Main runtime |
| pandas / NumPy | Tabular feature engineering, metrics, matrices |
| PyArrow | Parquet storage |
| PyYAML | Config loading |
| PyTorch 2.10 | Autoencoder and temporal autoencoder |
| sentence-transformers | MiniLM filing-section text embeddings |
| scikit-learn | PCA, Gaussian Mixture Models, clustering metrics, Ledoit-Wolf, ridge regression |
| networkx | Relationship graph features and communities |
| cvxpy | Long-only minimum-variance portfolio optimization |
| Streamlit | Dashboard |
| Altair | Dashboard charts |

## 5. Data Sources And Artifacts

### 5.1 Raw / Ingested Data

| Data | Source | Current use |
| --- | --- | --- |
| SEC 10-K and 10-Q filings | Local filing corpus / SEC-derived DB | Historical business, risk, MD&A, cybersecurity, legal, market-risk embeddings |
| SEC 8-K events | Local SEC event data | Event item frequency and future event-aware text work |
| Daily prices | SQLite DB via `FilingsDB` | Momentum, volatility, liquidity, return covariance, sector-relative prediction |
| SEC XBRL fundamentals | SEC companyfacts endpoint | Growth, margins, leverage, payout, valuation features |
| GICS metadata | Processed metadata parquet | Evaluation labels, sector grouping |
| Relationship graph | Extracted from filings | Network features and relationship validation |

### 5.2 Current Processed Artifacts

Key artifacts under `data/processed/`:

| Artifact | Rows / coverage | Purpose |
| --- | ---: | --- |
| `historical_text_10k_10q/historical_section_embeddings.parquet` | 115,549 rows, 500 tickers | MiniLM embeddings for historical filing sections |
| `historical_text_10k_10q/historical_section_topic_counts.parquet` | 125,359 rows, 500 tickers | Interpretable topic counts before embedding |
| `features/text_historical.parquet` | 87,282 rows, 502 tickers | PCA-reduced monthly text features |
| `features/fundamentals.parquet` | 729,163 rows, 500 tickers | Raw long-format XBRL concept values |
| `features/growth_lifecycle.parquet` | 92,887 rows, 503 tickers | Growth/maturity/fundamental features |
| `features/valuation.parquet` | local generated artifact | Valuation ratios based on fundamentals and prices |
| `features/network_position.parquet` | 92,887 rows, 503 tickers | Graph centrality/community features |
| `relationships/relationships.parquet` | 676 relationship rows | Sparse relationship graph |
| `datasets/historical_text_business.parquet` | 87,282 rows, 502 tickers | Model-ready business-view panel |

## 6. Filing Sections Currently Embedded

The completed expanded historical text run covers `10-K` and `10-Q` filings since 2010.

Current embedded sections:

| Section | Rows | Tickers | Interpretation |
| --- | ---: | ---: | --- |
| `business` | 7,232 | 497 | Products, markets, customers, operating model |
| `risk_factors` | 7,302 | 496 | Material risks and emerging threats |
| `mda` | 7,387 | 499 | Management discussion and operating narrative |
| `item_1c_cybersecurity` | 1,470 | 486 | Cybersecurity governance and cyber risk |
| `item_2_properties` | 7,130 | 494 | Physical footprint, assets, capacity |
| `item_3_legal_proceedings` | 6,576 | 495 | Litigation and legal exposure |
| `item_7a_market_risk` | 6,568 | 472 | Interest-rate, FX, commodity, market risks |
| `q_mda` | 21,399 | 496 | Quarterly management discussion |
| `q_market_risk` | 19,532 | 475 | Quarterly market risk |
| `q_legal_proceedings` | 15,773 | 458 | Quarterly legal proceedings |
| `q_risk_factors` | 15,180 | 492 | Quarterly risk updates |

The parser also has support for 8-K, proxy, and S-1 sections, but those are not yet the main completed embedding artifact.

## 7. Feature Producers

Feature producers live under `src/features/` and register explicitly in `src/features/registry.py`.

| Feature group | Module | Output meaning |
| --- | --- | --- |
| `price_volatility` | `src/features/price/volatility.py` | Realized volatility windows |
| `price_momentum` | `src/features/price/momentum.py` | Trailing return windows |
| `price_liquidity` | `src/features/price/liquidity.py` | Volume, dollar volume, liquidity proxies |
| `event_item_frequency` | `src/features/events/item_frequency.py` | 8-K item counts over rolling windows |
| `growth_lifecycle` | `src/features/fundamentals/growth_lifecycle.py` | Revenue growth, margins, R&D, capex, payout, leverage, size |
| `valuation` | `src/features/fundamentals/valuation.py` | Sales yield, earnings yield, book-to-market, EV ratios, FCF yield |
| `network_position` | `src/features/graph/network_position.py` | Degree, PageRank, betweenness, community |
| `text_historical` | `src/features/text/historical_text_features.py` | Point-in-time historical MiniLM/PCA text features |
| `text_business` | `src/features/text/business_description.py` | Legacy latest-filing hashed features |
| `text_risk` | `src/features/text/risk_factors.py` | Legacy latest-filing hashed features |

Feature conventions:

- Keys are `ticker` and `date` as regular columns.
- Columns are prefixed by source, such as `price_`, `growth_`, `valuation_`, `net_`, `text_hist_`.
- Point-in-time joins use filing dates and backward as-of merges.
- Feature artifacts are parquet files under `data/processed/features/`.

## 8. Model Layer

### 8.1 Sentence Embedding Model

Text sections are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Role:

- Converts each filing section into a dense semantic vector.
- Lets the model compare filing language by meaning rather than exact word overlap.
- Used for historical business/risk/MD&A/cybersecurity text.

Storage strategy:

- Run keyword/topic counts first.
- Run MiniLM embeddings section-by-section.
- Store compact embeddings and topic counts.
- Do not store every full filing text in processed artifacts.

### 8.2 PCA Text Compression

Historical section embeddings are aggregated into monthly firm-date rows, then reduced with PCA:

- Current output: `text_hist_emb_0` through `text_hist_emb_63`.
- Purpose: keep feature count manageable before downstream autoencoders.
- PCA object is persisted next to the feature parquet.

### 8.3 PyTorch Autoencoder

Main nonlinear embedding model:

```text
src/models/autoencoder.py
```

Architecture:

```text
Input standardized features
-> Linear
-> BatchNorm1d
-> ReLU
-> Dropout
-> hidden layers from config
-> embedding layer
-> mirrored decoder
-> reconstruction
```

Key properties:

- Implemented as `torch.nn.Module`.
- Supports configurable `hidden_dims`, for example `[256, 128]`.
- Uses Adam optimizer.
- Uses MSE reconstruction loss in standardized feature space.
- Uses validation split and early stopping.
- Stores feature means/stds as model buffers.
- Saves `weights.pt` plus `meta.json`.

Interpretation:

- The embedding is the bottleneck representation that best reconstructs the input features.
- For the business view, it means: "which companies need similar compressed coordinates to reconstruct their filing-language features?"
- It is not trained to predict returns directly.

### 8.4 Temporal Autoencoder

Experimental business-view model:

```text
src/models/temporal_autoencoder.py
src/models/temporal_dataloader.py
```

Loss:

```text
total_loss =
    reconstruction_loss
    + lambda_temp * exp(-alpha * feature_change) * embedding_change^2
```

Purpose:

- Make firm trajectories smoother over time.
- Penalize month-to-month embedding jumps when input features barely changed.
- Allow larger movement when filing text changed substantially.

Current status:

- Useful for dashboard smoothness.
- Passed stability validation.
- Did not pass AI-drift or event-alignment validation.
- Should not yet be claimed as a reliable event detector.

### 8.5 PCA Baseline

`PCAModel` is a lightweight baseline to compare against learned autoencoder embeddings.

### 8.6 Gaussian Mixture Models

Soft clustering per view:

```text
src/clustering/gmm.py
```

Purpose:

- Convert embeddings into soft theme memberships.
- Each firm-date row receives theme loadings that sum to 1.
- Different views can have different numbers of themes.

Interpretation:

- A company does not belong to only one hard cluster.
- It can be, for example, 45% enterprise-software theme, 25% platform theme, 15% digital-advertising theme, etc.

### 8.7 Sector-Relative Ridge Model

Dashboard sector outlook:

```text
src/applications/sector_relative_outlook.py
```

Model:

- Standardized ridge regression.
- Predicts sector excess return versus the equal-weight S&P 500 universe.
- Uses only prior completed outcomes for each historical prediction date.

Inputs:

- Sector excess momentum over 21, 63, and 126 trading days.
- Sector excess volatility and hit rate.
- Sector-average valuation and growth/lifecycle features.

Recent audit fix:

- Current/unrealized rows are now separated from historical completed rows.
- The historical walk-forward tab only scores predictions whose full forward horizon completed.

## 9. Decomposed View Architecture

Current four-view setup:

| View | Feature regex | Embedding dim | Typical use |
| --- | --- | ---: | --- |
| Business | `^text_hist_emb_` | 32 | Semantic company similarity and thematic movement |
| Behavioral | `^price_` | 16 | Return co-movement and trading behavior |
| Growth | `^(growth_|valuation_)` | 8 | Lifecycle, fundamentals, valuation |
| Network | `^net_` | 8 | Relationship graph position |

View training pipeline:

```text
assembled panel
-> select columns by view regex
-> train/reuse autoencoder per view
-> save embeddings per view
-> fit GMM per view
-> save loadings
-> dashboard + evaluations
```

## 10. Dashboard

Dashboard app:

```text
apps/raw_filing_browser/app.py
```

Run command:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py --server.port 8502
```

Current tabs:

| Tab | What to show |
| --- | --- |
| Filing Browser | Raw and structured SEC filing sections |
| Historical Text | AI/cloud/cyber/supply-chain/electrification topic trends |
| Similarity Shifts | Mixed theme/category loadings through time |
| Market Map | Stocks as points in a 2D embedding/loading projection |
| Sector Outlook | Current sector scores and historical walk-forward predictions |

Important dashboard caveats:

- PCA map geometry is fitted over available points for stable visualization, so the projection itself is not a strict backtest object.
- Current sector scores are live/unrealized until the forward horizon completes.
- Manual theme labels are being avoided in favor of dynamic labels from representative filing fragments.

## 11. Evaluation Methods

### 11.1 Clustering Against GICS

Question:

> Do embeddings recover sector structure?

Metrics:

- Adjusted Rand Index.
- Normalized Mutual Information.

Current result:

| Model | ARI vs GICS | NMI vs GICS |
| --- | ---: | ---: |
| Legacy semantic MiniLM text | 0.219 | 0.381 |
| Risk / price embedding | 0.052 | 0.171 |
| Historical-text business AE | 0.255 | 0.433 |
| Temporal historical-text business AE | 0.269 | 0.412 |

Interpretation:

- Business text embeddings recover meaningful semantic sector structure.
- Historical text improves over the earlier latest-filing text baseline.

### 11.2 Peer Identification

Question:

> Are embedding-nearest peers more return-correlated than GICS sub-industry peers?

Current honest result:

- GICS sub-industry still wins return co-movement.
- Behavioral/risk embeddings are closer than semantic embeddings for returns.
- Semantic gap narrows at longer horizons, which supports the idea that text similarity is more structural than short-term trading similarity.

Peer horizon pattern:

| Horizon | Semantic peer gap vs GICS | Risk peer gap vs GICS |
| ---: | ---: | ---: |
| 21 days | -0.272 | -0.100 |
| 63 days | -0.201 | -0.072 |
| 126 days | -0.197 | -0.071 |
| 252 days | -0.189 | -0.072 |
| 504 days | -0.177 | -0.079 |

### 11.3 Covariance Estimation

Question:

> Can embedding similarity improve portfolio covariance estimation?

Current result:

| Method | Annual variance | Annual Sharpe | Mean turnover |
| --- | ---: | ---: | ---: |
| Sample covariance | 0.00966 | 1.432 | 0.233 |
| sklearn Ledoit-Wolf | 0.00963 | 1.407 | 0.224 |
| Single-view embedding prior | 0.01009 | 1.233 | 0.220 |
| Multi-view factor model | 0.01522 | 0.276 | 0.332 |

Interpretation:

- Ledoit-Wolf remains the full-universe benchmark.
- The direct multi-view factor covariance estimator is a negative result.
- The better future direction is a conservative hybrid: Ledoit-Wolf as the base, embedding adjustments only where repeatedly helpful.

### 11.4 Relationship Graph Alignment

Question:

> Do embedding peers also share disclosed company relationships?

Current graph:

- 676 relationship rows.
- Relationship types: competitor, agreement, customer, partner, supplier.

Latest relationship comparison:

| Peer set | Direct link rate |
| --- | ---: |
| Historical-text embedding peers | 0.110 |
| Temporal embedding peers | 0.119 |
| GICS sub-industry peers | 0.173 |
| Random peers | 0.010 |

Interpretation:

- Embedding peers are far more connected than random peers.
- GICS still has stronger direct relationship overlap.
- Relationship extraction is promising but still sparse.

### 11.5 View Redundancy

Question:

> Are the four views actually different?

Cross-view NMI:

| Pair | NMI |
| --- | ---: |
| Behavioral / Business | 0.312 |
| Business / Network | 0.302 |
| Behavioral / Network | 0.257 |
| Business / Growth | 0.214 |
| Behavioral / Growth | 0.208 |
| Growth / Network | 0.197 |

Mean off-diagonal NMI: `0.248`.

Interpretation:

- Views are related, but not redundant.
- This supports the decomposed-similarity thesis.

### 11.6 Sector-Relative Prediction Audit

Question:

> Can transparent sector-level features predict sector performance relative to the S&P universe?

Model:

- Walk-forward ridge regression.
- 63-trading-day default horizon.
- Trains only on prior rows whose forward outcome had already completed.

Corrected audit:

| Metric | Value |
| --- | ---: |
| Completed prediction rows | 1,683 |
| Completed prediction dates | 153 |
| Unrealized rows excluded | 44 |
| Leakage violations | 0 |
| Historical window | 2013-04-30 to 2025-12-31 |
| Latest current score date | 2026-04-22 |
| Mean rank IC | 0.132 |
| Median rank IC | 0.173 |
| Mean top-minus-bottom excess return | 1.38% |
| Top-sector hit rate | 60.1% |

Interpretation:

- The signal is positive on average over completed historical windows.
- It had a rough late-2025 stretch, which is valuable to show because it makes the backtest honest.

## 12. Point-In-Time Correctness

Point-in-time audit:

```text
report/point_in_time_audit.md
```

Current summary:

- Pass: 26
- Warn: 2
- Fail: 0

Known limitations:

- GICS labels are static over the sample.
- Dashboard PCA projection uses all available points for stable visualization.
- Human-readable theme labels can reflect full-sample interpretation unless generated dynamically.
- Strict production-grade PIT would require model retraining only on data available up to each training cutoff.

## 13. Current Test Status

Latest full test run:

```text
Ran 62 tests
OK
```

Environment warnings from PyArrow/joblib appear during tests on this machine, but tests pass.

## 14. Specific Insights To Show The Class

This is the most important section for presentation. The goal is not to show every metric; it is to show a few vivid examples that make the project understandable.

### Insight 1: AI language exploded after 2023, especially in risk factors

Show this in dashboard:

```text
Historical Text -> Topic = AI -> market trend by section
```

Key numbers:

| Section | 2023 AI score | 2026 partial AI score |
| --- | ---: | ---: |
| Business | 3.389 | 7.112 |
| Risk factors | 1.576 | 11.220 |

Interpretation:

- Companies are not just saying they use AI in business descriptions.
- They increasingly discuss AI as a risk factor.
- This gives the dashboard a concrete narrative: AI became a market-wide strategic and risk disclosure theme.

Good class line:

> The model does not need to know what ChatGPT is. It sees the language of public-company disclosures shift after 2023.

### Insight 2: AI movement is not just a tech-sector story

Largest company-level AI-language increases include:

```text
EPAM, ADBE, ADP, AMZN, EFX, PANW, NVDA, CDNS, NOW, INTU,
ORCL, QCOM, MSFT, META, VRSK, ZBH, PAYX, SNPS, AXP, JKHY
```

Class examples:

| Company | Sector | Why interesting |
| --- | --- | --- |
| ADP | Industrials | AI language in payroll / HR / enterprise workflow context |
| EFX | Industrials | AI plus data/security/risk language |
| AXP | Financials | AI and cybersecurity language outside pure tech |
| ZBH | Health Care | AI language appearing in medical device / health context |
| PAYX | Industrials | AI in payroll and business-services workflow |

Interpretation:

- The dashboard can identify AI adoption/risk language outside the obvious technology names.
- This is a good class demo because it shows why text embeddings are useful: they reveal thematic diffusion.

### Insight 3: Semantic embeddings find peers GICS misses

Show this in dashboard:

```text
Similarity Shifts or Market Map -> business view -> latest date -> inspect peers
```

Examples:

| Seed | GICS label | Embedding-nearest peers | Interpretation |
| --- | --- | --- | --- |
| META | Communication Services | WDAY, CRM, INTU, XYZ, APP | Platform/software economics beyond communication-services label |
| ETN | Industrials | NEE, AES, SRE, NI, CMS | Electrification and grid exposure connecting equipment and utilities |
| CBRE | Real Estate | ARES, BX, IVZ, IBKR, KKR | Real-estate services linked to capital markets and asset management |
| SBUX | Restaurants | KDP, HSY, YUM, PEP, KHC | Consumer brand, beverage, restaurant, and CPG overlap |
| HUM | Managed Health Care | MCK, CAH, HSIC, COR, KVUE | Managed care language near distributors and care-delivery infrastructure |

Good class line:

> GICS asks, "What industry is this company in?" The embedding asks, "Which companies describe their business reality similarly?"

### Insight 4: Business similarity and stock co-movement are different

Use the peer horizon table:

| Horizon | Semantic peer gap vs GICS | Risk peer gap vs GICS |
| ---: | ---: | ---: |
| 21 days | -0.272 | -0.100 |
| 504 days | -0.177 | -0.079 |

Interpretation:

- GICS wins short-horizon return correlation.
- Risk/price embeddings are closer to return behavior.
- Semantic embeddings improve at longer horizons but do not beat GICS.

Good class line:

> A company can be semantically similar to another company without its stock moving with it next month.

### Insight 5: Decomposed views really are different

Show:

```text
View comparison / decomposed result table
```

Mean cross-view NMI is only `0.248`.

Interpretation:

- Business, behavioral, growth, and network views are not just duplicates.
- This supports the project thesis that company similarity is multidimensional.

Good class line:

> There is no single "true" similarity map. There are different maps depending on the economic question.

### Insight 6: Sector model has a real historical audit, not just a current score

Show this in dashboard:

```text
Sector Outlook -> Historical walk-forward
```

Corrected audit:

| Metric | Value |
| --- | ---: |
| Completed prediction dates | 153 |
| Leakage flags | 0 |
| Mean rank IC | 0.132 |
| Top-minus-bottom average | 1.38% |
| Top-sector hit rate | 60.1% |

Good success example:

| Prediction date | Horizon end | Predicted top | Realized best | Rank IC | Top-bottom |
| --- | --- | --- | --- | ---: | ---: |
| 2025-09-30 | 2025-12-30 | Health Care | Health Care | 0.400 | 4.78% |

Good failure example:

| Prediction date | Horizon end | Predicted top | Realized best | Rank IC | Top-bottom |
| --- | --- | --- | --- | ---: | ---: |
| 2025-11-28 | 2026-03-03 | Communication Services | Energy | -0.682 | -13.29% |

Interpretation:

- The average signal is positive.
- Recent late-2025 predictions failed badly.
- This is worth showing because honest backtests include both.

Good class line:

> The dashboard separates current predictions from completed historical predictions so we do not accidentally grade ourselves on the future.

### Insight 7: Covariance is a clean negative result

Show:

| Method | Annual variance |
| --- | ---: |
| Ledoit-Wolf | 0.00963 |
| Single-view embedding prior | 0.01009 |
| Multi-view factor model | 0.01522 |

Interpretation:

- The embeddings are useful for interpretation.
- They do not beat Ledoit-Wolf as a full-universe covariance estimator.
- This is a valuable negative result.

Good class line:

> The project found structure, but not every structure is useful for every financial task.

### Insight 8: Temporal smoothing works for stability, not event detection yet

Temporal autoencoder diagnostics:

| Test | Result | Interpretation |
| --- | --- | --- |
| Stability | PASS | Stable firms move less than under vanilla AE |
| NMI preservation | PASS | Cross-sectional structure is preserved |
| AI drift | FAIL | AI/control displacement ratio below strict 2x bar |
| Event alignment | FAIL | Named calendar events did not reliably create velocity spikes |

Interpretation:

- The temporal model helps make trajectories cleaner.
- It is not yet a reliable strategic-event detector.
- The likely fix is to add 8-K event text and validate around filing dates.

Good class line:

> A smoother map is not automatically a better event detector. We tested that directly and it failed.

## 15. Recommended Class Demo Flow

Use this order if presenting live:

1. Start with the thesis: "company similarity is not one object."
2. Show historical AI language trend in the Historical Text tab.
3. Show company-level AI movers, emphasizing non-tech examples.
4. Show semantic peer examples: META, ETN, CBRE, SBUX.
5. Show Market Map or Similarity Shifts to make embeddings visual.
6. Show Sector Outlook historical walk-forward tab, including one success and one failure date.
7. Close with the honest result table: semantic structure works; GICS wins co-movement; Ledoit-Wolf wins covariance; the dashboard is the discovery tool.

Suggested 60-second explanation:

> We built a system that turns SEC filings, prices, fundamentals, and company relationships into multiple embeddings. The key finding is that similarity decomposes. Text embeddings find what companies do and how their strategic language shifts. Price embeddings are better for return behavior. Fundamentals capture lifecycle. Network features capture disclosed relationships. The dashboard lets us watch these maps move through time. Some results are positive, like AI-language detection and semantic peer groups. Some are negative, like covariance versus Ledoit-Wolf. The point is an honest system for exploring company similarity, not a black-box stock picker.

## 16. Known Limitations

| Limitation | Why it matters |
| --- | --- |
| S&P 500 current-constituent bias | Universe may suffer survivorship bias |
| Static GICS labels | Historical sector classifications may differ |
| Static network graph | Current relationship graph is pooled, not fully historical |
| Dashboard PCA projection is full-sample | Good for visualization, not strict PIT backtest evidence |
| XBRL coverage still improving | Some valuation/cash-flow fields need refreshed fundamentals |
| 8-K text not fully embedded yet | Event detection likely needs more event-specific text |
| Temporal AE failed event validation | Should not be oversold as strategic shift detector |
| Covariance model underperformed | Similarity views are not automatically better risk models |

## 17. Near-Term Next Steps

Highest-value next work:

1. Refresh fundamentals ingestion so valuation features fill in better, especially cash-flow-derived ratios.
2. Add 8-K event text embeddings for material agreements, results, acquisitions, and strategic updates.
3. Validate temporal embedding velocity around filing dates as well as calendar event dates.
4. Keep Ledoit-Wolf as the covariance base and test embedding-informed adjustments only in slices where evidence repeats.
5. Continue reducing dashboard clutter by making each tab answer one question at a time.

## 18. Key Commands

Run dashboard:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py --server.port 8502
```

Run tests:

```bash
.venv/bin/python -m unittest discover tests
```

Run historical text streaming:

```bash
.venv/bin/python -m scripts.historical_text.stream_features \
  --since 2010-01-01 \
  --forms "10-K,10-Q" \
  --output-dir data/processed/historical_text_10k_10q \
  --resume
```

Run model training:

```bash
.venv/bin/python scripts/04_train.py historical_text_business
```

Run evaluation:

```bash
.venv/bin/python scripts/05_evaluate.py historical_text_business experiments/<run>/embeddings.parquet
```

## 19. Bottom Line

The current project has a clear, defensible story:

- It is an end-to-end representation-learning system for public companies.
- It uses multiple data views rather than pretending one similarity definition is enough.
- It produces meaningful semantic and thematic insights.
- It includes honest negative results where strong benchmarks still win.
- It has a dashboard that makes the results explorable for class discussion.

The best class takeaway is:

> We did not build a magic stock predictor. We built a map of company similarity, then tested which maps help with which financial questions.
