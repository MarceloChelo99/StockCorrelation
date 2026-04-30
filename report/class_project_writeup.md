# Stock Embeddings: Project Writeup

Generated: 2026-04-29

## Executive Summary

This project builds an end-to-end system for learning and exploring public-company similarity. The core idea is that "similar company" is not one single concept. Two firms can be similar because they describe similar businesses, because their stocks trade similarly, because they are at a similar financial lifecycle stage, or because they are connected through customers, suppliers, competitors, or partners.

The project turns SEC filings, market prices, fundamentals, and relationship evidence into company embeddings. These embeddings are then used for clustering, peer discovery, dashboard exploration, and sector-relative return prediction.

The strongest result is semantic. Historical filing-text embeddings recover meaningful business structure and produce cross-sector groupings that GICS sometimes misses. The clearest negative result is covariance estimation: embedding-based covariance estimators do not beat Ledoit-Wolf on the full universe. That negative result is useful because it shows that business similarity and return-risk similarity are related but not interchangeable.

The final dashboard is not meant to be a black-box trading tool. It is a research and presentation interface for exploring:

- How companies move through learned business and financial themes.
- Which companies look similar based on filing language and financial characteristics.
- What supply-chain or relationship links are visible in filings.
- Whether group-level features can produce a walk-forward sector or theme outlook.

## Research Question

The motivating question is:

> Can we learn company embeddings that capture economically meaningful similarity beyond traditional GICS sector labels?

The project evaluates that question in several ways:

- Do embedding clusters align with GICS sectors?
- Do embedding peers reveal sensible cross-sector relationships?
- Do embeddings improve peer return correlation relative to GICS peers?
- Do embeddings improve covariance estimation relative to Ledoit-Wolf?
- Can group-level features help predict sector or theme outperformance in a walk-forward setting?
- Can a dashboard make the learned structures interpretable enough to present and inspect?

The most important conceptual finding is that similarity decomposes. Filing text is useful for business identity. Price features are more useful for stock behavior. Fundamentals help describe financial lifecycle. Relationship extraction helps show who a company is economically connected to. No single view dominates every task.

## Data Sources

The project combines several public data sources.

| Source | Data Used | Purpose |
| --- | --- | --- |
| SEC 10-K and 10-Q filings | Business, Risk Factors, MD&A, Market Risk, Legal Proceedings, Cybersecurity, and quarterly sections | Company language, strategy, risks, and topic movement |
| SEC XBRL companyfacts | Revenue, gross profit, operating income, net income, assets, debt, capex, R&D, dividends, buybacks, shares | Financial growth, profitability, leverage, payout, and lifecycle features |
| Daily prices | Adjusted close, returns, volatility, momentum, liquidity | Stock behavior and forward-return evaluation |
| Filing-derived relationship graph | Supplier, customer, competitor, partner, and agreement mentions | Current-state network and disruption analysis |
| GICS metadata | Sector and sub-industry | Benchmark labels for clustering and peer comparison |
| S&P 500 membership history | Add/remove intervals where available | Reduces survivorship bias in historical backtests |

The current historical text artifact covers 10-K and 10-Q filings since 2010. The local file `data/processed/historical_text_10k_10q/historical_section_embeddings.parquet` currently contains about 115,654 embedded section rows across 500 tickers and 11 section types.

The monthly historical text feature panel, `data/processed/features/text_historical.parquet`, contains about 87,282 firm-month rows, 502 tickers, and 64 PCA-reduced text dimensions.

Valuation and growth feature panels each contain about 92,887 monthly rows across 503 tickers. The relationship graph currently contains 665 extracted relationship edges, so it is useful for exploration but still sparse.

## Data Engineering And Pipeline Design

The pipeline follows an ETL-style structure:

```text
ingest -> features -> assembly -> model -> evaluation -> dashboard
```

Each stage writes stable artifacts that the next stage consumes. This matters because the project uses large, messy data. Without intermediate artifacts, the work would become difficult to reproduce or audit.

The major stages are:

| Stage | What Happens | Main Artifacts |
| --- | --- | --- |
| Ingest | Pull filings, prices, metadata, fundamentals, and membership data | SQLite DB, raw filing shards, prices parquet |
| Feature computation | Convert raw inputs into per-feature-group parquet files | `data/processed/features/*.parquet` |
| Assembly | Merge features into monthly model-ready panels | `data/processed/datasets/*.parquet` |
| Model training | Train autoencoders and save embeddings | `experiments/<run>/model`, `embeddings.parquet` |
| Clustering | Fit GMM themes and save loadings | `experiments/<run>/views/<view>/loadings.parquet` |
| Evaluation | Compute clustering, peers, covariance, and sector-outlook metrics | `report/*.csv`, `report/*.md`, experiment metrics |
| Dashboard | Load artifacts and expose interactive analysis | Streamlit app |

Point-in-time correctness is a central design rule. Filing features use `filing_date`, not fiscal period end. Price features only use data available before the observation date. Sector prediction uses only completed historical forward horizons when training each walk-forward model.

## Text Processing And Section Embeddings

The project originally used parsed 10-K sections such as Business and Risk Factors. As the project evolved, the text layer expanded to historical 10-K and 10-Q sections. Rather than storing all full filing text as the main modeling input, the system streams filings, extracts sections, computes topic counts, embeds each section, and stores compact numeric artifacts.

This design was chosen for two reasons:

1. Full SEC filings are large and slow to repeatedly parse.
2. Section embeddings and topic counts are compact enough to cache and reuse.

The main text embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

MiniLM converts each filing section into a dense semantic vector. The intuition is that two sections with similar meaning should have vectors near each other, even if they use different exact words.

The project also keeps topic counts for interpretability. Keyword counts are not as semantically rich as embeddings, but they are useful for showing concrete language shifts, such as increases in AI, cloud, cybersecurity, supply-chain, or electrification language.

Important caveat: Item 1C Cybersecurity is a newer SEC disclosure section. It is useful for recent cybersecurity language, but it should not be treated as a stable historical feature back to 2010 because the disclosure rule did not exist for the full sample.

## Feature Views

The system separates company similarity into multiple views.

| View | Inputs | What It Captures | Dashboard Status |
| --- | --- | --- | --- |
| Business | Historical MiniLM filing text features | What companies say they do, risks, strategy, and language shifts | Main similarity view |
| Financial | Valuation, profitability, growth, leverage, payout, liquidity, momentum | Company lifecycle and financial profile | Main similarity view |
| Network | Filing-derived company relationships | Customers, suppliers, competitors, partners | Separate current-state network tab |
| Behavioral | Price volatility, momentum, liquidity | How stocks trade | Mostly hidden because clusters were noisy |
| Growth | XBRL growth/lifecycle subset | Growth versus maturity | Folded into financial view rather than shown alone |

This separation became one of the main research conclusions. A business embedding and a risk embedding are not the same thing. Business text is better for semantic structure. Price behavior is closer to return co-movement. Financial features help describe company state. Network data helps explain relationship exposure, but the current graph is sparse.

## Embedding Models

### PCA Baseline

Principal Component Analysis, or PCA, is the simplest dimensionality-reduction baseline. It linearly projects high-dimensional features into a smaller number of coordinates. PCA is useful because it is fast, transparent, and hard to overfit.

Its limitation is that it can only learn linear combinations. Company similarity may depend on nonlinear feature interactions, so PCA is a baseline rather than the main embedding model.

### PyTorch Autoencoder

The main embedding model is a PyTorch autoencoder.

```text
features -> encoder -> low-dimensional embedding -> decoder -> reconstructed features
```

The autoencoder is trained to reconstruct its own input. The compressed middle layer is the company embedding. This is a good fit for the project because the target is representation learning, not direct return prediction. SEC filings and fundamentals are better suited to describing companies than predicting short-term stock returns.

Why autoencoders were used:

- They are unsupervised, so they do not require noisy return labels.
- They can compress text, price, fundamentals, and network features.
- They can learn nonlinear combinations of features.
- The embedding is easy to reuse in clustering, peer search, and visualization.

The model layer now uses PyTorch, mini-batch training, Adam optimization, validation loss, early stopping, and saved normalization buffers. The older NumPy autoencoder is retained only for legacy comparison.

### Temporal Autoencoder

A temporal autoencoder was added experimentally for smoother company trajectories. It adds a penalty that keeps consecutive firm-month embeddings close when the underlying features barely changed, while allowing movement when the features changed substantially.

The training objective is:

```text
loss = reconstruction_loss + lambda_temp * adaptive_smoothness_loss
```

This is useful for dashboard trajectories, but it should be presented cautiously. The current temporal model improves smoothness, but it has not fully passed all event-detection validation tests.

## Clustering Models

After embeddings are created, the project clusters companies into themes.

Three clustering methods were compared:

| Model | Output | Strength | Weakness |
| --- | --- | --- | --- |
| GMM | Soft probabilities across themes | Allows mixed membership | More complex than k-means |
| k-means | One hard label per company | Simple and strong baseline | Too rigid for mixed businesses |
| DBSCAN | Dense clusters plus noise points | Can identify outliers | Collapsed most firms into one large cluster |

The current dashboard uses Gaussian Mixture Models, or GMMs, because public companies are often mixed businesses. For example, Amazon is not purely retail, purely cloud, or purely advertising. Soft GMM loadings allow one company to partially belong to multiple themes.

Latest business-view cluster comparison:

| Clustering model | Groups found | Noise share | Largest group share | NMI vs GICS | ARI vs GICS | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GMM | 10 | 0.0% | 25.9% | 0.451 | 0.314 | Best for soft mixed-membership themes |
| k-means | 10 | 0.0% | 14.4% | 0.446 | 0.322 | Similar hard alignment, simpler baseline |
| DBSCAN | 1 | 2.0% | 98.0% | 0.020 | 0.002 | Collapses most firms into one dense region |

GMM is not chosen because it crushes k-means on every metric. It is chosen because it gives a more useful output for the product: companies can belong to multiple themes at once.

## Evaluation Results

### Feature And Embedding Ablations

The first evaluation asks whether embeddings recover meaningful semantic structure relative to GICS.

| Feature / representation | ARI vs GICS | NMI vs GICS | Peer corr diff vs GICS |
| --- | ---: | ---: | ---: |
| 8-K item counts only | 0.022 | 0.091 | -0.262 |
| Price / risk features only | 0.052 | 0.171 | -0.100 |
| Hashed text + price baseline | 0.206 | 0.341 | -0.108 |
| MiniLM text only | 0.219 | 0.381 | -0.272 |
| Historical-text business autoencoder | 0.255 | 0.433 | -0.127 |

NMI and ARI measure alignment with GICS labels. Higher values mean the learned groups are more sector-like. The text embeddings perform best on semantic structure. This supports the main dashboard focus on business-language similarity.

The peer-correlation result is more mixed. GICS peers still beat embedding peers for short-horizon return co-movement. That does not mean the embeddings failed. It means GICS is very strong for return co-movement, while text embeddings capture a different kind of similarity: business identity and thematic structure.

### Qualitative Semantic Results

The most interesting semantic examples are cross-sector peer groups that make economic sense even though GICS separates them. Examples found during analysis include:

- META clustering near CRM, WDAY, INTU, and APP, reflecting platform/software/AI-advertising economics rather than strict GICS sector labels.
- Eaton grouping near utilities and electrification-exposed firms, reflecting the electrification theme.
- CBRE connecting with alternative asset managers, reflecting real-estate and capital-markets exposure.
- Starbucks grouping with beverage, restaurant, and consumer-brand firms.

These examples are important because they show the value of embeddings even when they do not beat GICS in return correlation. They reveal economically interpretable relationships that rigid sector labels can miss.

### Covariance Estimation

The covariance evaluation was intentionally difficult. It compared embedding-based covariance methods against Ledoit-Wolf, a strong standard shrinkage estimator.

Full-universe result:

| Method | Annual variance | Interpretation |
| --- | ---: | --- |
| Sample covariance | about 0.00966 | Naive baseline |
| Ledoit-Wolf | about 0.00963 | Best full-universe benchmark |
| Single-view embedding prior | about 0.01000 | Close, but worse than Ledoit-Wolf |
| Multi-view factor covariance | about 0.01522 | Clearly worse |

This is a negative result. Embedding similarity does not beat Ledoit-Wolf as a full-universe covariance estimator.

The practical conclusion is not that embeddings are useless for risk. It is that a direct embedding-to-covariance replacement is too blunt. A better future direction would be a hybrid estimator that uses Ledoit-Wolf as the base and only uses embedding information in slices where it repeatedly helps.

### Sector / Group Prediction

The sector outlook model is a walk-forward group excess-return prediction experiment. At each prediction date, the model trains only on completed historical outcomes and predicts future group excess return.

The default model is Ridge regression. Ridge was chosen because the panel is small, the features are correlated, and interpretability matters. More flexible models can produce larger simulated capital in one run, but they are easier to overfit and harder to explain.

Current 63-day walk-forward GICS-sector predictor comparison:

| Predictor | Mean rank IC | Top-bottom excess | Hit rate | Ending capital | Excess vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.098 | 1.15% | 54.3% | $68,674 | +139.7 pp |
| Elastic Net | 0.108 | 1.02% | 54.3% | $64,559 | +98.6 pp |
| Huber robust regression | 0.094 | 0.77% | 56.2% | $78,116 | +234.2 pp |
| Random Forest | 0.083 | 0.97% | 54.3% | $69,743 | +150.4 pp |
| Extra Trees | 0.078 | 0.77% | 58.2% | $50,152 | -45.5 pp |
| Gradient Boosting | 0.038 | 0.26% | 51.0% | $76,929 | +222.3 pp |

Metric definitions:

- Mean rank IC is the average Spearman correlation between predicted and realized group ranks. Positive means the model ranking has signal.
- Top-bottom excess is the realized excess return of favored groups minus lagging groups.
- Hit rate is how often the favored group performs well relative to the benchmark.
- Ending capital is a historical simulation starting from $10,000. It is useful, but it should not be treated as a trading guarantee.

The result is promising but should be presented carefully. Ridge gives a positive ranking signal and a clean explanation. Some other models show higher simulated capital, but those results need robustness checks because capital curves can be sensitive to a few sector calls, especially around bear markets and recoveries.

### Numerical Embedding Features For Prediction

The project also tested whether an autoencoder over structured sector-state variables could add predictive value to the sector outlook.

| Feature set | Mean rank IC | Hit rate | Ending capital | SPY ending capital |
| --- | ---: | ---: | ---: | ---: |
| Raw structured baseline | 0.098 | 54.2% | $68,674 | $54,701 |
| Sector AE only | 0.036 | 56.9% | $62,487 | $54,701 |
| Raw + sector AE | 0.087 | 50.3% | $68,094 | $54,701 |
| Correlation-filtered sector AE only | 0.064 | 62.7% | $69,521 | $54,701 |
| Raw + correlation-filtered sector AE | 0.084 | 52.9% | $73,836 | $54,701 |

This suggests that numerical embeddings can add information, especially when weak embedding dimensions are filtered. However, the raw structured baseline is still strong and easier to explain. For class, the best framing is that the financial embedding is an exploratory enhancement, not the core result.

## Detailed Model Inventory And Defense

This section is designed for questions from the professor or class. For every major model or model family, it states the input, transformation, output, metric interpretation, performance, and final status.

### Metric Dictionary

| Metric | Used For | What It Means | How To Read It |
| --- | --- | --- | --- |
| Reconstruction MSE | Autoencoders | Mean squared error between standardized input features and reconstructed features | Lower means the embedding preserved more input information, but values are not directly comparable across very different feature sets |
| ARI vs GICS | Clustering / embeddings | Adjusted Rand Index comparing learned groups to GICS sectors | Higher means better hard-label agreement; 0 is roughly chance |
| NMI vs GICS | Clustering / embeddings | Normalized Mutual Information comparing learned groups to GICS sectors | Higher means learned clusters share more information with GICS labels |
| Peer corr diff | Peer evaluation | Embedding-peer forward return correlation minus GICS-peer correlation | Positive would mean embedding peers beat the GICS benchmark; negative means GICS peers were more return-correlated |
| Noise share | DBSCAN / clustering | Fraction of points labeled as outliers/noise | High noise means many firms are not assigned to usable groups |
| Largest group share | Clustering | Fraction of firms in the largest cluster | Very high values mean the model collapsed most firms into one group |
| Mean rank IC | Sector prediction | Average Spearman correlation between predicted and realized group rankings | Positive means the model's ranking contains useful signal |
| Directional accuracy | Sector prediction | Share of group-date rows where predicted positive/negative excess return matches realized positive/negative excess return | Above 50% suggests directional signal |
| Positive precision | Sector prediction | Of groups predicted to outperform, the share that actually outperformed | Higher means fewer false positives |
| Positive recall | Sector prediction | Of groups that actually outperformed, the share the model identified | Higher means fewer missed winners |
| Positive F1 | Sector prediction | Harmonic mean of positive precision and recall | Balances false positives and false negatives |
| Top-bottom excess | Sector prediction | Realized return of top-ranked groups minus bottom-ranked groups | Positive means the ranking separates winners from losers |
| Ending capital | Sector rotation simulation | Value of a $10,000 historical rotation simulation | Easy to understand but sensitive to a small number of decisions |
| Annual variance | Covariance evaluation | Realized annualized portfolio variance | Lower is better for a risk estimator |
| Sharpe-like ratio | Portfolio simulation | Return divided by volatility | Higher is better, but should be read with drawdown and variance |

### Data Extraction And Text Representation Models

| Model / Method | Family | Input | What It Does | Output | Performance / Evidence | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| SEC section parser | Rule-based information extraction | SEC filing HTML/text | Finds named filing sections such as Business, Risk Factors, MD&A, Market Risk, Legal Proceedings, and quarterly equivalents | Section-level text rows with filing dates | Current historical text artifact has about 115,654 embedded section rows, 500 tickers, 11 section types | Used |
| Latest-filing-only parser | Rule-based baseline | Most recent available filing per company | Parses current filing sections but does not create real history | Static company text features | Useful early, but caused historical views to inherit current embeddings | Retired for temporal analysis |
| Topic keyword counts | Rule-based feature extraction | Parsed section text | Counts interpretable terms for AI, cloud, cybersecurity, supply chain, electrification, and related topics | Topic count columns per section | Useful for explaining language movement; not used as the main semantic embedding | Used for interpretability |
| MiniLM sentence transformer | Transformer language model | Filing section text | Converts each section into a dense semantic vector | 384-dimensional section embedding | MiniLM text-only AE: ARI 0.219, NMI 0.381. Historical-text business AE: ARI 0.255, NMI 0.433 | Main text representation |
| FinBERT | Transformer language model | Filing section text | Would create finance-specialized text embeddings | Dense text embeddings | Not run in final pipeline; MiniLM already produced strong semantic structure at lower cost | Considered but not used |
| Full raw filing storage | Storage design, not a model | Complete filing text | Would retain all original text for every filing | Large raw-text corpus | More auditable but much heavier and slower. Compact embeddings achieved the needed workflow | Not used as dashboard input |

The important distinction is that the parser and keyword counts are transparent but brittle. MiniLM is less transparent but much stronger for semantic similarity because it can connect similar language even when the exact words differ.

### Feature Engineering And View Models

| View / Feature Set | Family | Input | Transformation | Output | Performance / Evidence | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| Historical business text | Text embedding + PCA | Section-level MiniLM vectors by filing date | As-of merge to firm-month, aggregate sections, reduce to 64 PCA text dimensions | `text_hist_emb_*` monthly features | Best semantic structure in current project; business AE NMI about 0.433 | Main similarity view |
| Price / risk features | Rolling time-series features | Daily adjusted prices and volume | Compute momentum, volatility, liquidity windows using past data only | Price behavior features | Price-only AE had peer corr diff -0.100, better than text for return co-movement but weak semantic NMI 0.171 | Used in analysis, hidden from main dashboard |
| 8-K item counts | Event-count features | Parsed 8-K item numbers | Counts event item frequency over windows | Event count features | ARI 0.022, NMI 0.091, peer diff -0.262 | Tested, not emphasized |
| Valuation features | Structured accounting/market features | SEC fundamentals plus market cap | Compute yields, book-to-market, debt-to-market, shareholder yield | Monthly valuation columns | Used in financial view and sector prediction | Used |
| Growth/lifecycle features | Structured accounting features | Revenue, margins, capex, R&D, leverage, payout | Compute YoY growth, CAGR, margin levels/trends, payout, leverage, size | Monthly growth columns | Useful as part of financial profile; too narrow alone for cluster map | Folded into financial view |
| Relationship graph | Rule-based relationship extraction + graph features | Filing snippets that mention other firms | Classify supplier/customer/competitor/partner evidence and compute network features | Relationship edges and network-position features | About 665 extracted edges; useful current-state exploration but sparse historically | Separate network tab |

### Embedding And Dimensionality-Reduction Models

| Model | Family | Input | What It Does | Output | Reconstruction MSE | ARI | NMI | Peer Corr Diff | Final Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PCA text baseline | Linear unsupervised dimensionality reduction | 74 older text/price features | Projects features onto directions of maximum variance | 8-dimensional embedding | n/a | 0.141 | 0.305 | -0.127 | Baseline only |
| Autoencoder hashed text baseline | Neural unsupervised autoencoder | 74 older hashed text/price features | Compresses and reconstructs input through nonlinear hidden layers | 8-dimensional embedding | 0.375 | 0.206 | 0.341 | -0.108 | Superseded |
| MiniLM text autoencoder | Neural unsupervised autoencoder | 768 MiniLM text dimensions | Compresses semantic text into compact company coordinates | 8-dimensional embedding | 0.552 | 0.219 | 0.381 | -0.272 | Key semantic baseline |
| Price-only autoencoder | Neural unsupervised autoencoder | 10 price/risk features | Compresses stock behavior features | 8-dimensional embedding | 0.075 | 0.052 | 0.171 | -0.100 | Behavior diagnostic, not main dashboard |
| Price + MiniLM autoencoder | Neural unsupervised autoencoder | 778 price and MiniLM text features | Learns joint text/price embedding | 8-dimensional embedding | 0.553 | 0.242 | 0.368 | -0.140 | Tested, not as clean as separated views |
| 8-K counts autoencoder | Neural unsupervised autoencoder | 24 event-count features | Compresses event frequency patterns | 8-dimensional embedding | 0.068 | 0.022 | 0.091 | -0.262 | Weak; not final |
| Historical-text business autoencoder | Neural unsupervised autoencoder | 64 historical text PCA features | Learns point-in-time business embedding | 32-dimensional business embedding | not directly comparable | 0.255 | 0.433 | -0.127 | Main business view |
| Temporal business autoencoder | Neural autoencoder with temporal regularization | Consecutive firm-month historical text features | Reconstructs current input and penalizes unnecessary embedding jumps | Smoother 32-dimensional business embedding | validation temporal loss 0.079 | 0.269 | 0.412 | -0.120 | Experimental |

How the autoencoder data flows:

```text
firm-month feature row
-> standardization using training mean/std
-> encoder neural network
-> low-dimensional embedding
-> decoder neural network
-> reconstructed standardized feature row
```

The embedding is not a return forecast. It is a compressed coordinate system that preserves the original company feature profile.

Why the current business autoencoder is preferred: it has the best semantic structure metric among production-style views, produces interpretable peer groups, and feeds the dashboard cleanly. The temporal autoencoder is useful for smoother movement but not yet strong enough to claim reliable event detection.

Temporal validation details:

| Temporal Test | Result | Main Number | Interpretation |
| --- | --- | ---: | --- |
| AI drift | Failed strict bar | AI median displacement 2.521 vs control 1.681 | Directional, but only about 1.5x rather than required 2x |
| Stability | Passed | Temporal stable-firm displacement 1.214 vs vanilla 1.387 | Smoothing reduces noise for stable firms |
| Event alignment | Failed | 0% event pass rate | Named events did not reliably create embedding velocity spikes |
| Hyperparameter sweep | 0 / 15 full passes | Stability and NMI passed 15 / 15; AI and event tests 0 / 15 | Current text-only temporal setup smooths well but does not detect events reliably |

### Clustering Models

All clustering methods receive the same cross-section of firm embeddings for a selected date. They output theme or cluster assignments.

| Model | Family | Input | What It Does | Output | Groups | Noise | Largest Group | NMI vs GICS | ARI vs GICS | Final Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gaussian Mixture Model | Probabilistic soft clustering | Standardized company embeddings | Fits a mixture of Gaussian distributions and assigns membership probabilities | Theme loadings summing to 1 per company | 10 | 0.0% | 25.9% | 0.451 | 0.314 | Default |
| k-means | Centroid-based hard clustering | Standardized company embeddings | Finds k centroids and assigns each firm to nearest centroid | One hard cluster label per company | 10 | 0.0% | 14.4% | 0.446 | 0.322 | Baseline |
| DBSCAN | Density-based clustering | Standardized company embeddings | Finds dense regions and labels sparse points as noise | Cluster labels plus noise | 1 | 2.0% | 98.0% | 0.020 | 0.002 | Not used |

Why GMM is final: k-means is slightly competitive on hard GICS-alignment metrics, but GMM gives the soft loadings the product needs. A company can be partly in several themes. DBSCAN mostly collapsed the market into one cluster, making it unhelpful for the dashboard.

### Sector / Group Prediction Models

The sector prediction models receive one row per group per prediction date. Inputs include recent relative momentum, relative volatility, valuation averages, growth averages, and optionally group embedding features. The target is future group excess return versus the equal-weight S&P universe over the selected horizon.

Data flow:

```text
group-date features available at T
-> train model on completed past horizons only
-> predict future group excess return
-> rank groups
-> evaluate once forward horizon completes
```

| Model | Family | Input | What It Does | Output | Rank IC | Accuracy | Precision | Recall | F1 | Top-Bottom | Hit Rate | Ending Capital | Final Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ridge regression | Linear regularized regression | Group features | Fits linear coefficients with L2 penalty to reduce overfitting | Predicted excess return | 0.098 | 54.8% | 53.8% | 52.2% | 53.0% | 1.15% | 54.2% | $68,674 | Default |
| Elastic Net | Linear regularized regression | Group features | Combines L1 and L2 penalties, allowing some coefficient shrinkage toward zero | Predicted excess return | 0.108 | 54.8% | 53.7% | 53.0% | 53.4% | 1.02% | 54.2% | $64,559 | Tested |
| Huber robust regression | Robust linear regression | Group features | Fits linear model with reduced sensitivity to outlier errors | Predicted excess return | 0.094 | 53.5% | 52.6% | 45.4% | 48.7% | 0.77% | 56.2% | $78,116 | Tested |
| Random Forest | Tree ensemble / bagging | Group features | Averages many decision trees trained on bootstrapped samples | Predicted excess return | 0.083 | 52.5% | 51.4% | 47.3% | 49.3% | 0.97% | 54.2% | $69,743 | Tested |
| Extra Trees | Tree ensemble / randomized trees | Group features | Averages trees with more randomized splits | Predicted excess return | 0.078 | 53.4% | 52.4% | 48.7% | 50.4% | 0.77% | 58.2% | $50,152 | Tested |
| Gradient Boosting | Sequential tree ensemble / boosting | Group features | Builds shallow trees sequentially to correct prior errors | Predicted excess return | 0.038 | 50.5% | 48.9% | 36.0% | 41.5% | 0.26% | 51.0% | $76,929 | Tested |

Why Ridge is final: Elastic Net has slightly higher rank IC, and Huber / Gradient Boosting have higher ending capital in this sample. But ending capital can be dominated by a few historical sector calls. Ridge is easier to explain, has a positive rank signal, avoids excessive flexibility, and is more defensible for a small monthly panel.

### Numerical Sector-State Embedding Variants

These models tested whether an autoencoder over structured sector-state variables could improve the sector prediction model.

| Feature Set | Input | What It Outputs | Rank IC | Hit Rate | Ending Capital | SPY Ending Capital | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Raw structured baseline | Momentum, valuation, growth, volatility features | Direct Ridge inputs | 0.098 | 54.2% | $68,674 | $54,701 | Strong, transparent default |
| Sector AE only | Autoencoder embedding of structured sector features | Embedding dimensions only | 0.036 | 56.9% | $62,487 | $54,701 | Some signal, weaker rank IC |
| Raw + sector AE | Raw features plus all AE dimensions | Combined predictor inputs | 0.087 | 50.3% | $68,094 | $54,701 | Did not improve clearly |
| Correlation-filtered sector AE only | Only AE dimensions with stronger historical correlations | Filtered embedding inputs | 0.064 | 62.7% | $69,521 | $54,701 | Better hit rate, still less interpretable |
| Raw + correlation-filtered sector AE | Raw features plus selected AE dimensions | Combined predictor inputs | 0.084 | 52.9% | $73,836 | $54,701 | Highest capital in this feature-set test, but lower rank IC than raw baseline |

Conclusion: numerical embeddings can add signal, especially after filtering weak dimensions, but the raw structured model is simpler and more defensible. The financial embedding is useful as an exploratory enhancement, not as the core claim.

### Covariance And Portfolio Risk Models

These models receive historical return matrices and, for embedding methods, company similarity information. They output covariance matrices used by a long-only minimum-variance portfolio optimizer.

| Model | Family | Input | What It Does | Output | Annual Variance | Sharpe | Turnover | Final Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Sample covariance | Classical statistics | Trailing stock returns | Computes empirical covariance directly | Covariance matrix | 0.00966 | 1.432 | 0.233 | Baseline |
| Ledoit-Wolf | Shrinkage covariance estimator | Trailing stock returns | Shrinks noisy sample covariance toward a stable target | Covariance matrix | 0.00963 | 1.407 | 0.224 | Benchmark |
| Legacy constant-variance shrinkage | Shrinkage covariance estimator | Trailing stock returns | Older local shrinkage variant | Covariance matrix | 0.00963 in migration check | 1.407 | 0.224 | Legacy only |
| Single-view embedding prior | Similarity-informed shrinkage | Returns plus embedding similarity | Shrinks correlations toward an embedding-similarity prior | Covariance matrix | 0.01000 | 1.172 | 0.365 | Negative full-universe result |
| Multi-view factor covariance | Factor covariance model | Soft theme loadings and returns | Builds theme returns, estimates factor covariance, reconstructs firm covariance | Covariance matrix | 0.01522 | 0.276 | 0.332 | Negative result |

Why Ledoit-Wolf is final: covariance estimation is a specialized statistical problem. The embedding methods capture business structure, but that structure did not produce a better full-universe risk estimator. Ledoit-Wolf remains the appropriate benchmark and practical default.

### Relationship And Network Methods

The network tab uses relationship extraction rather than a learned neural model.

| Method | Family | Input | What It Does | Output | Performance / Evidence | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| Rule-based relationship extraction | Information extraction | Filing snippets with company mentions | Uses patterns and evidence snippets to classify supplier, customer, competitor, partner, or agreement links | Directed relationship edges | About 665 current edges; useful but sparse | Used for network tab |
| Network centrality / graph features | Graph analytics | Relationship edge list | Computes degree, PageRank, betweenness, clustering, communities | Network-position feature columns | Static across history in current implementation | Used cautiously |
| Supply-chain disruption walk | Graph traversal simulation | Directed supplier-customer edges | Walks upstream and downstream through strongest links | Affected-company path table and sector summaries | Good for explanation, not a complete supply-chain stress test | Dashboard feature |

The network methods are valuable because they provide evidence snippets. Their weakness is coverage. A missing edge does not mean two companies have no relationship; it only means the relationship was not captured in our filings and parser.

## Dashboard Product

The current dashboard is intentionally simplified to three main tabs:

| Tab | Purpose |
| --- | --- |
| Similarity Explorer | Explore business and financial similarity, theme loadings, company peers, and theme movement over time |
| Network | Inspect current-state filing-derived relationships and simulate supply-chain disruption paths |
| Sector Outlook | Run walk-forward group prediction, inspect current group scores, historical decisions, and rotation simulation |

Older tabs such as raw filing browser, historical text, and model comparison were removed or hidden from the main product because they cluttered the presentation. Their analysis remains in reports and artifacts, but the dashboard now focuses on the most demonstrable product features.

The dashboard is meant to help answer questions like:

- What themes does a company currently belong to?
- Which companies look similar based on business language or financial profile?
- Did the company's theme mix change over time?
- What filing fragment supports a theme label?
- Who appears upstream or downstream in the relationship graph?
- Which sectors or learned groups look favored in the current walk-forward outlook?

## Limitations

The project has several important limitations.

First, some historical data remains incomplete. S&P membership history reduces survivorship bias, but deleted constituents can still have weaker filing, price, or metadata coverage.

Second, relationship extraction is sparse and noisy. Filing-derived supplier/customer links are not a full commercial supply-chain database. The network tab should be read as evidence-backed exploration, not complete ground truth.

Third, GICS labels are mostly treated as static. Real sector classifications can change over time.

Fourth, some dashboard projections and labels are interpretive tools, not strict backtest evidence. The theme labels are dynamically generated from representative filing fragments or financial feature profiles, but they are not supervised truth.

Fifth, sector rotation results are historical simulations. They are useful for evaluating signal direction, but they should not be presented as a deployable trading strategy.

Finally, covariance results are negative in the full universe. Ledoit-Wolf remains the stronger risk benchmark.

## Conclusion

The project succeeds as a representation-learning and interpretability project. It builds a reproducible pipeline from SEC filings and market data to company embeddings, learned themes, evaluations, and a working dashboard.

The most important conclusion is that company similarity is multidimensional. Filing text captures what companies say they do. Price features capture how stocks behave. Financial features capture lifecycle and valuation. Relationship data captures disclosed economic links. These views are complementary, not interchangeable.

The strongest positive result is semantic structure: historical filing-text embeddings produce meaningful business groupings and cross-sector peers that GICS can miss. The clearest negative result is covariance: embeddings do not replace Ledoit-Wolf for full-universe risk estimation. The sector-outlook work shows a promising but carefully framed walk-forward ranking signal.

For a class presentation, the best story is:

1. We started with the question of what company similarity means.
2. We built an ETL pipeline from SEC filings, fundamentals, prices, and relationships.
3. We used MiniLM and autoencoders to turn messy company data into embeddings.
4. We used GMMs because companies can belong to multiple themes.
5. We evaluated honestly against GICS, GICS peers, Ledoit-Wolf, and S&P benchmarks.
6. We found that business embeddings are useful for semantic structure, not a universal stock-return predictor.
7. We turned the result into a dashboard for exploring company similarity, relationships, and group outlooks.

The project is therefore best framed not as "we built a stock picker," but as:

> We built a data-driven map of public companies and showed that learned similarity can reveal interpretable business structure beyond traditional sector labels.
