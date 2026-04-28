# Stock Embeddings: A Decomposed Similarity Framework For Public Companies

Generated: 2026-04-27  
Project branch: `explore`

## Abstract

This project builds low-dimensional representations, or embeddings, for public companies using SEC filings, stock-price behavior, fundamentals, and company relationship data. The goal is not to build a magic stock predictor. The goal is to learn different maps of company similarity and then test which maps are useful for different financial questions.

The main finding is that **company similarity is not one thing**. A company can be similar to another company because it describes a similar business, because its stock trades similarly, because it is at a similar lifecycle stage, or because it has similar customers, suppliers, competitors, or partners. These similarities overlap, but they are not the same.

The strongest current result is semantic: historical filing-text embeddings recover meaningful company structure, detect broad AI-language shifts after 2023, and produce economically interesting cross-sector peers that GICS classifications miss. The strongest negative result is covariance: embedding-based covariance methods do not beat Ledoit-Wolf on the full universe. The dashboard ties these results together by letting users explore company movement, thematic language trends, sector predictions, and historical walk-forward performance.

## 1. Project Motivation

Traditional stock classification systems, such as GICS sectors and sub-industries, are useful but limited. They assign each company to one primary industry bucket. That is convenient, but real companies often span several themes:

- Meta is officially Communication Services, but its business language overlaps with software platforms and AI infrastructure.
- Eaton is Industrials, but its language increasingly overlaps with utilities because of electrification and grid investment.
- Starbucks is Restaurants, but its brand and distribution language overlaps with beverages and consumer packaged goods.

This project asks:

> Can we build data-driven company embeddings that reveal useful similarity structures beyond fixed sector labels?

The project uses representation learning instead of direct return prediction. That matters because short-term stock returns are noisy and dominated by news, flows, rates, sentiment, and idiosyncratic shocks. SEC filings and fundamentals are better suited for learning **what companies are like** than for predicting tomorrow's return.

## 2. High-Level Thesis

The project's current thesis is:

> Company similarity is multidimensional. Business text, stock behavior, fundamentals, and relationship networks each define a different kind of similarity. Different downstream tasks need different similarity views.

The four current views are:

| View | Data source | Captures |
| --- | --- | --- |
| Business / semantic | Historical 10-K and 10-Q filing text | What companies say they do, their risks, strategy, products, markets |
| Behavioral / risk | Daily stock prices | Momentum, volatility, liquidity, return co-movement |
| Growth / lifecycle | SEC XBRL fundamentals and valuation ratios | Growth, margins, leverage, capital intensity, payout, maturity |
| Network | Filing-derived relationships | Competitors, suppliers, customers, partners, agreements |

## 3. Data Sources

The project uses four major data categories.

### 3.1 SEC Filing Text

The filing text pipeline parses historical SEC filings and extracts major sections.

Current completed historical artifact:

```text
data/processed/historical_text_10k_10q/
```

Current coverage:

| Artifact | Rows | Tickers |
| --- | ---: | ---: |
| Historical section embeddings | 115,549 | 500 |
| Historical topic-count rows | 125,359 | 500 |
| Monthly historical text features | 87,282 | 502 |

Embedded sections include:

| Filing type | Sections |
| --- | --- |
| 10-K | Business, Risk Factors, MD&A, Cybersecurity, Properties, Legal Proceedings, Market Risk |
| 10-Q | Quarterly MD&A, Market Risk, Legal Proceedings, Risk Factors |

The project also has parser support for additional 8-K, proxy, and S-1 sections, but the current completed embedding artifact is primarily 10-K plus 10-Q.

### 3.2 Stock Price Data

Daily adjusted close and volume data are used to compute:

- Volatility.
- Momentum.
- Liquidity.
- Return correlations.
- Covariance matrices.
- Sector-relative return outcomes.

### 3.3 SEC XBRL Fundamentals

SEC companyfacts data are used for:

- Revenue growth.
- Gross and operating margins.
- R&D intensity.
- Capex intensity.
- Leverage.
- Payout.
- Asset growth.
- Valuation ratios such as sales yield, earnings yield, book-to-market, and EV ratios.

Current raw fundamentals artifact:

```text
data/processed/features/fundamentals.parquet
```

Current coverage:

| Artifact | Rows | Tickers |
| --- | ---: | ---: |
| Raw fundamentals | 729,163 | 500 |
| Growth/lifecycle monthly features | 92,887 | 503 |

### 3.4 Relationship Graph

The relationship graph is extracted from filing text using conservative public-company name matching and rule-based context classification.

Current graph:

| Relationship type | Count |
| --- | ---: |
| Competitor | 252 |
| Agreement | 144 |
| Customer | 128 |
| Partner | 92 |
| Supplier | 60 |
| Total | 676 |

This graph is currently sparse and mostly static, but it provides a useful external check on whether semantic peers are economically related.

## 4. Pipeline Architecture

The codebase follows an ETL-style research pipeline:

```text
ingest -> features -> assembly -> model -> evaluation -> dashboard/report
```

The main stages are:

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | SEC filings, prices, metadata, fundamentals | SQLite DB and raw/processed artifacts |
| Features | DB and processed source files | Per-feature-group parquet files |
| Assembly | Feature parquets | Monthly model-ready panel |
| Model | Assembled panel | Model checkpoint and embeddings |
| Evaluation | Embeddings, returns, metadata | Metrics, figures, reports |
| Dashboard | Processed artifacts and reports | Interactive Streamlit app |

The most important design rule is that each stage writes stable artifacts. Later stages should not re-parse raw filings or reach backward around the feature contracts.

## 5. Feature Engineering

Feature producers are registered explicitly in:

```text
src/features/registry.py
```

Current feature groups:

| Feature group | Prefix | Purpose |
| --- | --- | --- |
| Historical text | `text_hist_` | Semantic filing-language representation |
| Price volatility | `price_vol_` | Rolling realized volatility |
| Price momentum | `price_mom_` | Trailing returns |
| Price liquidity | `price_liq_` | Volume, dollar volume, illiquidity |
| 8-K event frequency | `event_` | Rolling counts of event types |
| Growth lifecycle | `growth_` | Growth, margins, capex, payout, leverage |
| Valuation | `valuation_` | Market-implied valuation ratios |
| Network position | `net_` | Graph centrality and community |

Point-in-time correctness is central:

- Filing features use `filing_date`, not fiscal period end.
- Price features use only data available through the previous trading days.
- Sector backtests now separate completed historical predictions from current/unrealized scores.

## 6. Models Used And How They Work

This section is the most important for class presentation. The project uses several models, each for a different purpose.

## 6.1 Keyword Topic Counters

Before using embeddings, the project counts specific topic words in filing sections.

Examples of topic groups:

- AI.
- Cloud / compute.
- Cybersecurity.
- Supply chain.
- Electrification.

This is not a neural model. It is a transparent feature extractor.

For each section:

```text
topic_score = topic_mentions / section_words * 10,000
```

Why it matters:

- It gives interpretable evidence.
- It lets us show that AI language rose sharply after 2023.
- It helps label and explain embedding movement.

Class explanation:

> Keyword counts are the simple microscope. They do not understand language deeply, but they are easy to audit and good for showing where a theme appears.

## 6.2 MiniLM Sentence Transformer

The main text embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

MiniLM is a transformer model that converts text into a dense vector. Similar text should produce nearby vectors.

For each filing section:

```text
section text -> MiniLM -> 384-dimensional vector
```

The important idea is semantic similarity. Two sections do not need to use the exact same words to be close. If they describe similar products, risks, or business models, their embeddings should be closer.

Why MiniLM was used:

- It is fast enough for a solo project.
- It works well on general semantic similarity.
- It avoids training a huge language model from scratch.
- It produces compact vectors that can be stored instead of raw text.

Class explanation:

> MiniLM turns a filing paragraph into coordinates. Companies that describe similar businesses end up closer together in that coordinate space.

## 6.3 PCA Compression

The raw MiniLM vectors are still relatively wide. The project uses PCA to reduce the feature count.

PCA stands for Principal Component Analysis. It finds directions in the data with the most variance and projects the data onto those directions.

In this project:

```text
historical section embeddings -> monthly firm-date aggregation -> PCA -> 64 text features
```

Why PCA is useful:

- Reduces noise.
- Makes training faster.
- Keeps the assembled dataset smaller.
- Preserves the largest semantic variation in filing embeddings.

Class explanation:

> PCA is like rotating the coordinate system so the most important directions come first, then keeping only the top directions.

## 6.4 PyTorch Autoencoder

The main embedding model is a PyTorch autoencoder.

An autoencoder has two parts:

```text
input features -> encoder -> small embedding -> decoder -> reconstructed input
```

The model is trained to reconstruct the original feature vector. The embedding is the bottleneck representation the model learns because it cannot simply copy every input feature directly.

Mathematically:

```text
z = encoder(x)
x_hat = decoder(z)
loss = mean_squared_error(x_hat, standardized_x)
```

Where:

- `x` is the original feature vector.
- `z` is the learned embedding.
- `x_hat` is the reconstruction.
- The model minimizes reconstruction error.

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
-> reconstructed features
```

Why use an autoencoder:

- The project goal is representation learning, not direct return prediction.
- Reconstruction loss matches what the data can support.
- The embedding summarizes the company in fewer dimensions.
- Different feature views can each get their own autoencoder.

Important interpretation:

> The embedding does not mean "expected return." It means "compressed coordinates that preserve the information needed to reconstruct this company's features."

Class explanation:

> The autoencoder is like forcing the model to write a short summary of a company, then asking it to recreate the original feature profile from that summary. The summary vector is the embedding.

## 6.5 View-Specific Autoencoders

Instead of forcing all features into one embedding, the project trains separate embeddings for each view.

| View | Input features | Embedding dimension |
| --- | --- | ---: |
| Business | Historical text features | 32 |
| Behavioral | Price features | 16 |
| Growth | Growth and valuation features | 8 |
| Network | Graph features | 8 |

This lets the project ask:

- Which companies are semantically similar?
- Which stocks behave similarly?
- Which firms have similar fundamentals?
- Which firms occupy similar network positions?

Class explanation:

> We do not ask one map to answer every question. We build several maps, each from a different kind of evidence.

## 6.6 Temporal Autoencoder

The temporal autoencoder extends the business autoencoder by adding a smoothing penalty.

The goal is:

- Keep embeddings stable when a company's filings barely changed.
- Allow embeddings to move when filing language changed meaningfully.

Loss function:

```text
total_loss =
    reconstruction_loss
    + lambda_temp * exp(-alpha * feature_change) * embedding_change^2
```

Interpretation:

- If two consecutive company observations have similar features, movement is penalized.
- If the features changed a lot, the penalty becomes smaller.

Current validation:

| Test | Result | Interpretation |
| --- | --- | --- |
| Stability | Pass | Stable companies move less |
| NMI preservation | Pass | Cross-sectional structure is preserved |
| AI drift | Fail | AI movers did not move enough versus controls |
| Event alignment | Fail | Named events did not reliably create velocity spikes |

Class explanation:

> The temporal autoencoder made the map smoother, but we tested whether it detects real events and it does not pass that bar yet.

## 6.7 Gaussian Mixture Models For Soft Themes

After embeddings are trained, the project fits Gaussian Mixture Models, or GMMs.

A GMM is a soft clustering model. Unlike k-means, it does not force each company into exactly one cluster. It gives probabilities or loadings.

Example:

```text
Company A:
  Theme 1: 0.55
  Theme 2: 0.30
  Theme 3: 0.15
```

Why soft clustering matters:

- Companies often span multiple themes.
- It supports dashboard visualizations of mixed category membership.
- It lets us watch whether a company's theme mixture changes over time.

Class explanation:

> A company does not need to be only one thing. GMMs let it be partially in several themes at once.

## 6.8 Ledoit-Wolf Covariance Estimator

Ledoit-Wolf is a standard covariance shrinkage method. It improves raw sample covariance estimates by shrinking them toward a more stable target.

The project uses sklearn's standard implementation as the main benchmark.

Why it is important:

- Covariance estimation is hard when there are many stocks.
- Raw sample covariance can be noisy.
- Ledoit-Wolf is a strong, widely accepted baseline.

Result:

- Embedding-based covariance did not beat Ledoit-Wolf on the full universe.

Class explanation:

> Ledoit-Wolf is the tough benchmark for risk modeling. Our embeddings found meaningful company structure, but that structure did not beat the specialized covariance estimator.

## 6.9 Multi-View Factor Covariance Model

The project tested a multi-view factor covariance model built from soft theme loadings.

Idea:

```text
firm returns -> theme returns -> factor covariance -> firm covariance
```

Using loadings:

```text
Sigma = L F L' + D
```

Where:

- `L` is the firm-by-theme loading matrix.
- `F` is the covariance of theme returns.
- `D` is idiosyncratic firm variance.

Result:

| Method | Annual variance |
| --- | ---: |
| Ledoit-Wolf | 0.00963 |
| Single-view embedding prior | 0.01009 |
| Multi-view factor model | 0.01522 |

Interpretation:

- The direct factor construction was too blunt.
- This is a clean negative result.
- The next risk-modeling direction should be conservative hybrid adjustments, not replacing Ledoit-Wolf.

## 6.10 Ridge Regression Sector Outlook Model

The dashboard includes a sector-relative prediction model.

Model:

```text
standardized sector features -> ridge regression -> predicted sector excess return
```

Target:

```text
future sector return - future equal-weight S&P 500 return
```

Features include:

- Sector excess momentum.
- Sector excess volatility.
- Sector excess hit rate.
- Sector-average valuation ratios.
- Sector-average growth/lifecycle features.

Why ridge regression:

- There are only 11 sectors.
- The dataset is monthly, not huge.
- A simple, transparent model is more appropriate than a large neural network.
- Ridge regularization reduces overfitting by penalizing large coefficients.

Walk-forward training rule:

> For each historical prediction date, train only on prior rows whose forward outcome had already completed.

Recent corrected audit:

| Metric | Value |
| --- | ---: |
| Completed prediction dates | 153 |
| Leakage violations | 0 |
| Mean rank IC | 0.132 |
| Mean top-minus-bottom return | 1.38% |
| Top-sector hit rate | 60.1% |

Class explanation:

> The sector model is not trained on the future. The dashboard now separates historical completed predictions from current live scores.

## 7. Main Results

## 7.1 Business Text Embeddings Recover Meaningful Structure

Clustering against GICS:

| Model | ARI vs GICS | NMI vs GICS |
| --- | ---: | ---: |
| Legacy semantic MiniLM text | 0.219 | 0.381 |
| Risk / price embedding | 0.052 | 0.171 |
| Historical-text business AE | 0.255 | 0.433 |
| Temporal historical-text business AE | 0.269 | 0.412 |

Interpretation:

- Business text embeddings align much more with sector structure than price embeddings.
- Historical filing text improved semantic clustering versus the earlier text baseline.
- Price embeddings capture a different kind of similarity.

## 7.2 Semantic Peers Reveal Cross-Sector Structure

Examples:

| Seed | GICS label | Embedding-nearest peers | Interpretation |
| --- | --- | --- | --- |
| META | Communication Services | WDAY, CRM, INTU, XYZ, APP | Platform/software economics beyond communication-services label |
| ETN | Industrials | NEE, AES, SRE, NI, CMS | Electrification and grid exposure |
| CBRE | Real Estate | ARES, BX, IVZ, IBKR, KKR | Real-estate services linked to capital markets |
| SBUX | Restaurants | KDP, HSY, YUM, PEP, KHC | Consumer brand, beverage, restaurant, and CPG overlap |
| HUM | Managed Health Care | MCK, CAH, HSIC, COR, KVUE | Managed care near distributors and care delivery |

This is one of the best class-demo results because it is intuitive. The embedding finds company neighborhoods that make economic sense even when they cross GICS boundaries.

## 7.3 Filing Language Shows The AI Regime Shift

AI topic score by section:

| Section | 2023 AI score | 2026 partial AI score |
| --- | ---: | ---: |
| Business | 3.389 | 7.112 |
| Risk factors | 1.576 | 11.220 |

Largest company-level AI-language increases include:

```text
EPAM, ADBE, ADP, AMZN, EFX, PANW, NVDA, CDNS, NOW, INTU,
ORCL, QCOM, MSFT, META, VRSK, ZBH, PAYX, SNPS, AXP, JKHY
```

Interpretation:

- AI language rose sharply after 2023.
- The rise appears in both business descriptions and risk factors.
- The trend is not limited to obvious technology firms.

Class explanation:

> We can see AI move from a niche technology topic into a broad public-company disclosure theme.

## 7.4 Return Co-Movement Still Favors GICS

Peer-correlation gap versus GICS:

| Horizon | Semantic peer gap | Risk peer gap |
| ---: | ---: | ---: |
| 21 days | -0.272 | -0.100 |
| 63 days | -0.201 | -0.072 |
| 126 days | -0.197 | -0.071 |
| 252 days | -0.189 | -0.072 |
| 504 days | -0.177 | -0.079 |

Interpretation:

- GICS sub-industry peers still have stronger return co-movement.
- Risk embeddings are better for return behavior than semantic embeddings.
- Semantic embeddings improve at longer horizons but do not beat GICS.

This is an important honest result:

> Semantic similarity is not the same as short-term return correlation.

## 7.5 Decomposed Views Are Not Redundant

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

- The views are related, but not duplicates.
- This supports the decomposed-similarity thesis.

## 7.6 Relationship Graph Confirms Some Economic Link Signal

Direct relationship link rate:

| Peer set | Direct link rate |
| --- | ---: |
| Historical-text embedding peers | 0.110 |
| Temporal embedding peers | 0.119 |
| GICS sub-industry peers | 0.173 |
| Random peers | 0.010 |

Interpretation:

- Embedding peers are much more connected than random peers.
- GICS peers are still more directly connected.
- The relationship graph is promising but still sparse.

## 7.7 Sector Outlook Has A Positive Average Historical Signal

Corrected 63-day walk-forward audit:

| Metric | Value |
| --- | ---: |
| Completed prediction rows | 1,683 |
| Completed prediction dates | 153 |
| Unrealized rows excluded | 44 |
| Leakage violations | 0 |
| Mean rank IC | 0.132 |
| Median rank IC | 0.173 |
| Mean top-minus-bottom excess return | 1.38% |
| Top-sector hit rate | 60.1% |

Recent examples:

| Prediction date | Horizon end | Predicted top | Realized best | Rank IC | Top-bottom |
| --- | --- | --- | --- | ---: | ---: |
| 2025-09-30 | 2025-12-30 | Health Care | Health Care | 0.400 | 4.78% |
| 2025-11-28 | 2026-03-03 | Communication Services | Energy | -0.682 | -13.29% |

Interpretation:

- Average performance is positive over the completed historical window.
- Recent late-2025 performance was poor.
- This is a good class example because it shows both success and failure.

## 8. Dashboard

The Streamlit dashboard is the main presentation tool.

Run:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py --server.port 8502
```

Tabs:

| Tab | What to show |
| --- | --- |
| Filing Browser | Raw and structured SEC filings |
| Historical Text | Topic trends and company-level filing-language shifts |
| Similarity Shifts | Soft theme memberships and company movement |
| Market Map | Stocks as points in a 2D similarity plane |
| Sector Outlook | Current sector scores and historical walk-forward predictions |

Recommended class demo order:

1. Open with the thesis: company similarity is not one object.
2. Show Historical Text -> AI trend.
3. Show non-tech AI movers.
4. Show semantic peer examples like META, ETN, CBRE, SBUX.
5. Show Similarity Shifts or Market Map.
6. Show Sector Outlook -> Historical walk-forward.
7. Close with honest results: text works, GICS still wins co-movement, Ledoit-Wolf wins covariance.

## 9. Limitations

| Limitation | Why it matters |
| --- | --- |
| Current S&P 500 universe | Survivorship bias |
| Static GICS labels | Historical sector classifications may differ |
| Static relationship graph | Network view is not yet a true historical graph |
| Dashboard PCA projection is full-sample | Good for visualization, not strict PIT evidence |
| XBRL coverage still improving | Some valuation and cash-flow features need refresh |
| 8-K text not fully embedded yet | Event detection likely needs event-specific disclosures |
| Temporal AE failed event validation | It should not be sold as a reliable event detector |
| Covariance underperformed | Embedding structure does not automatically improve risk models |

## 10. What We Learned

The project teaches several important lessons:

1. Text embeddings are useful for understanding company identity.
2. Business similarity and stock co-movement are different.
3. Strong financial benchmarks are hard to beat.
4. Negative results are informative when the evaluation is honest.
5. Dashboards are valuable for representation-learning projects because the output is exploratory, not just a single score.

## 11. Next Steps

Highest-value next improvements:

1. Refresh fundamentals ingestion so valuation features have better coverage.
2. Add 8-K text embeddings for events such as material agreements, earnings/results, acquisitions, and strategic announcements.
3. Validate temporal movement around filing dates, not only calendar event dates.
4. Build dynamic labels from representative filing fragments.
5. Continue decluttering the dashboard so each tab answers one question clearly.

## 12. Conclusion

This project built an end-to-end stock embedding system from SEC filings, prices, fundamentals, and company relationships. The central result is not a single winning model. The central result is a clearer view of similarity:

- Business embeddings explain what companies are and how their language changes.
- Behavioral embeddings explain how stocks move.
- Growth embeddings explain lifecycle and valuation differences.
- Network embeddings explain disclosed economic relationships.

The best current class takeaway is:

> We did not build a black-box stock picker. We built several maps of company similarity, then tested which maps help with which financial questions.

