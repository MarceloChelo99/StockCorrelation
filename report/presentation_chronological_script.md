# Stock Embeddings Presentation Script

Purpose: this is a chronological speaking script for class. It is meant to flow naturally from data collection to preprocessing, embeddings, clustering, dashboard exploration, validation against GICS, and finally sector-relative prediction. It intentionally does not mention every implementation detail.

Suggested length: 15 to 18 minutes.

## One-Sentence Thesis

We built a pipeline that turns SEC filings, prices, fundamentals, and company relationships into company embeddings, then tested whether those embeddings reveal useful company similarity beyond traditional GICS sector labels.

## Presentation Flow

```text
Data sources
-> preprocessing and point-in-time feature construction
-> text and financial embeddings
-> autoencoder compression
-> soft similarity clusters
-> dashboard exploration
-> evaluation versus GICS
-> sector-relative prediction and S&P benchmark
-> honest limitations and next steps
```

## 1. Opening Hook

**Slide cue:** Title slide with project name and one dashboard screenshot.

**Speaker script:**

The starting question for this project was simple: what does it mean for two public companies to be similar?

The standard answer is usually GICS. GICS tells us Apple is Information Technology, Meta is Communication Services, Starbucks is Consumer Discretionary, and so on. That is useful, but it is also rigid. A company can be similar to another company because it sells similar products, because its stock behaves similarly, because it has the same customer base, because it is in the same stage of maturity, or because it is exposed to the same theme, like AI or electrification.

So our goal was not to build a black-box stock picker. Our goal was to build a data-driven map of companies. We wanted embeddings that summarize what companies are, how they behave, and how they change over time. Then we tested whether those embeddings give us useful structure beyond GICS.

**Transition:**

To build that map, the first challenge was data. We needed data that describes both the company and the stock.

## 2. Data Sources

**Slide cue:** Data-source table.

**Key points to show:**

| Source | What It Gives Us | Why It Matters |
| --- | --- | --- |
| SEC 10-K and 10-Q filings | Business descriptions, risk factors, MD&A, cybersecurity, market risk | What the company says it does and what risks it faces |
| SEC XBRL companyfacts | Revenue, margins, assets, debt, capex, payouts | Growth and lifecycle characteristics |
| Daily prices | Returns, volatility, momentum, liquidity | How the stock behaves |
| Filing-derived relationships | Competitors, customers, suppliers, partners | Economic network structure |
| GICS metadata | Sector and sub-industry labels | Benchmark classification system |
| S&P membership history | Which firms were in the index at the time | Reduces survivorship bias in backtests |

**Speaker script:**

The project uses several public data sources.

First, we use SEC filings. The key filings are 10-Ks and 10-Qs. These contain the business description, risk factors, management discussion and analysis, cybersecurity disclosures, market-risk sections, and other sections that describe the company in its own words.

Second, we use SEC XBRL fundamentals. These are structured accounting concepts like revenue, assets, operating income, long-term debt, capex, R&D, and share repurchases. These help us capture whether a company looks like a young growth company, a mature cash-flow company, a leveraged company, or something else.

Third, we use daily price data. Prices give us the behavioral side: momentum, volatility, liquidity, and future returns for evaluation.

Fourth, we extract a small relationship graph from filings. When a filing mentions competitors, customers, suppliers, partners, or material agreements, we can treat that as a company-to-company link. This graph is still sparse, but it gives an external check on whether the embeddings are finding real economic relationships.

Finally, we use GICS labels and S&P 500 membership history. GICS is our benchmark classification system. Membership history matters because a backtest that uses today's S&P 500 roster back in 2010 has survivorship bias. If we know a company will eventually survive into the S&P 500, that is future information.

**Challenge story:**

The hardest data issue was that SEC data is public but not clean. Filing sections are not perfectly standardized, company names change, tickers change, and old or deleted S&P names are harder to fetch. We handled this by saving stable intermediate artifacts, logging failures, using filing dates for point-in-time correctness, and separating current predictions from historical completed predictions.

**Transition:**

Once we had the raw data, the next problem was turning messy text and prices into clean features.

## 3. Preprocessing Layer

**Slide cue:** Pipeline diagram.

```text
raw filings and prices
-> parsed sections
-> section embeddings and topic counts
-> monthly point-in-time feature panel
-> model-ready dataset
```

**Speaker script:**

The preprocessing layer is where the project becomes auditable.

For filings, we parse sections like Business, Risk Factors, MD&A, Cybersecurity, Legal Proceedings, and Market Risk. We do not want to repeatedly store and reprocess giant raw filings forever, so the historical text pipeline streams filings, extracts sections, counts important topic words, runs the text embedder, and stores compact section-level embeddings.

This design matters because the dataset grows quickly. Full filing text is large, but section embeddings are small enough to store and reuse. It also means that if we later add more sections, we can process them through the same pipeline.

For prices, we compute rolling volatility, momentum, and liquidity features. For fundamentals, we compute growth and lifecycle features such as revenue growth, margins, capex intensity, leverage, and payout.

The key rule is point-in-time correctness. If the model is making a prediction at date `T`, it only gets features that would have been known before `T`. Filing features use the filing date, not the fiscal period end date. Price features only use past prices.

**Important implementation highlight:**

The pipeline is ETL-style:

```text
ingest -> features -> assembly -> model -> evaluation -> dashboard
```

Each stage writes artifacts that the next stage consumes. That keeps the project from becoming one giant notebook where nobody knows what data was used.

**Transition:**

After preprocessing, the next question is how to represent a company numerically. That is where embeddings come in.

## 4. Text Embeddings And Why We Used MiniLM

**Slide cue:** Example: filing section text becomes a vector.

```text
Business section text -> MiniLM -> 384-dimensional section vector
```

**Speaker script:**

For filing text, we use a sentence-transformer model called MiniLM. MiniLM converts text into a dense vector, where texts with similar meanings should land near each other.

The reason to use a text embedding model instead of only keyword counts is that companies can describe the same idea using different words. For example, two companies might both be talking about cloud infrastructure or AI-enabled software without using exactly the same phrase. A transformer embedding can capture some of that semantic similarity.

We still keep keyword counts because they are interpretable. For example, if we want to show that AI language increased after 2023, keyword counts are easier to explain than a neural embedding dimension. But for broad similarity, MiniLM gives a better semantic representation.

**Good class-friendly analogy:**

The keyword counter is like a transparent checklist. The embedding is like a semantic fingerprint.

**Metric comparison to show here:**

This is where we can justify why we moved beyond simple keyword or event-count features.

| Feature / representation | ARI vs GICS | NMI vs GICS | Peer corr diff vs GICS |
| --- | ---: | ---: | ---: |
| 8-K item counts only | 0.022 | 0.091 | -0.262 |
| Price / risk features only | 0.052 | 0.171 | -0.100 |
| Hashed text + price baseline | 0.206 | 0.341 | -0.108 |
| MiniLM text only | 0.219 | 0.381 | -0.272 |
| Historical-text business AE | 0.255 | 0.433 | -0.127 |

**How to explain this table:**

8-K counts by themselves were weak because a simple event count does not tell us enough about what the event actually meant. Price features were better for return co-movement, but not for sector-like business structure. MiniLM and historical filing text were the strongest for semantic structure, which is why the business-view dashboard focuses on text embeddings.

**Transition:**

MiniLM gives us useful section vectors, but they are still high-dimensional. We need a smaller company-level representation. That leads to the autoencoder.

## 5. Autoencoder: What It Does And Why We Used It

**Slide cue:** Autoencoder diagram.

```text
company features -> encoder -> small embedding -> decoder -> reconstructed features
```

**Speaker script:**

The main embedding model is a PyTorch autoencoder.

An autoencoder is trained to reconstruct its own input. The model has an encoder that compresses the company features into a small vector, and a decoder that tries to rebuild the original feature vector from that small vector.

The small vector in the middle is the embedding.

This is important because we are not asking the model to predict stock returns directly. That would be a noisy target for a one-week research project, and SEC filings are not designed to predict next month's return. Instead, we ask the model to learn a compressed representation of the company. Then we test whether that representation is useful downstream.

**Why this model choice is defensible:**

- It is unsupervised, so we do not need to invent noisy labels.
- It can combine text, price, fundamentals, and network features.
- It gives each company a low-dimensional coordinate.
- It is simple enough to explain and inspect.

**Line to say clearly:**

The embedding does not mean "expected return." It means "compressed coordinates that preserve the company's feature profile."

**Model comparison to show here:**

This table compares dimensionality-reduction and embedding variants. The metric is alignment with GICS, which is not the only goal, but it is a useful sanity check that the embeddings learned real business structure.

| Embedding model | What changed | ARI vs GICS | NMI vs GICS |
| --- | --- | ---: | ---: |
| PCA text baseline | Linear compression of text/price features | 0.141 | 0.305 |
| Autoencoder hashed text baseline | Nonlinear compression, older text features | 0.206 | 0.341 |
| MiniLM text autoencoder | Better semantic text representation | 0.219 | 0.381 |
| Historical-text business autoencoder | True historical filing text over time | 0.255 | 0.433 |
| Temporal historical-text autoencoder | Smoother trajectories | 0.269 | 0.412 |

**How to explain this table:**

PCA is a useful baseline because it is simple and linear. The autoencoder improves because it can learn nonlinear feature combinations. The historical-text autoencoder performs best on NMI because it uses richer point-in-time filing history. The temporal autoencoder makes trajectories smoother and slightly improves ARI, but it does not dominate every metric, so we should present it as experimental.

**Transition:**

Once companies have coordinates, we can ask which companies are near each other. That is where clustering comes in.

## 6. Similarity Clustering: GMMs, K-Means, And DBSCAN

**Slide cue:** Cluster-model comparison view in dashboard.

**Speaker script:**

After producing embeddings, we group companies in embedding space.

We compared clustering methods because different methods answer slightly different questions.

K-means gives hard groups. Every company belongs to exactly one cluster. It is simple and fast, and it is useful as a baseline.

Gaussian Mixture Models, or GMMs, give soft groups. A company can be 70% in one theme and 30% in another. That matches the real problem better because many companies are mixed. For example, Amazon is not only retail, not only cloud, and not only advertising. A soft assignment is more honest.

DBSCAN looks for dense regions and can mark outliers. It is useful for comparison, but in high-dimensional financial embeddings it can be unstable because the idea of density becomes harder to tune.

For the dashboard, GMMs are the most useful because they produce mixed theme loadings. For some views, especially behavioral stock behavior, hard groups can be easier to interpret. But for business and fundamentals, mixed membership is valuable.

**Cluster-model metric comparison to show here:**

Latest business-view cross-section, target `10` groups:

| Clustering model | Groups found | Noise share | Largest group share | NMI vs GICS | ARI vs GICS | Main read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GMM | 10 | 0.0% | 25.9% | 0.451 | 0.314 | Best for soft mixed-membership themes |
| k-means | 10 | 0.0% | 14.4% | 0.446 | 0.322 | Similar hard alignment, simpler baseline |
| DBSCAN | 1 | 2.0% | 98.0% | 0.020 | 0.002 | Collapses most firms into one dense region |

**How to explain this table:**

K-means and GMM are close on hard GICS-alignment metrics. That is good: GMM is not wildly inventing structure. We use GMM mainly because the output is more useful for the dashboard: companies can have partial membership in multiple themes. DBSCAN is useful conceptually for finding outliers, but in this embedding cross-section it mostly collapses the market into one large cluster, so it is not a good default for theme exploration.

**Why not only GICS:**

GICS gives one official label per company. Our clustering asks whether the data itself suggests a different similarity structure.

**Transition:**

The dashboard is where this becomes tangible. Instead of just showing metrics, we can show the actual company map.

## 7. Dashboard Demo

**Slide cue:** Open Streamlit dashboard.

Run command if needed:

```bash
poetry run streamlit run apps/raw_filing_browser/app.py --server.port 8502
```

**Demo path:**

1. Start with the market map.
2. Select the business view.
3. Show how companies cluster in 2D.
4. Search for a familiar ticker like `META`, `AAPL`, `SBUX`, or `ETN`.
5. Show nearest peers or theme membership.
6. Move through time using filing steps.
7. Show topic trends, especially AI language.
8. Briefly show cluster-model comparison if asked why GMM.

**Speaker script:**

This dashboard is the main way to interact with the embeddings.

Each point is a company. The position comes from the embedding view. The colors come from learned themes or clusters. The exact 2D projection is only a visualization, so we should not overinterpret every pixel, but it lets us see broad structure.

The most interesting part is that companies sometimes group in ways that are economically intuitive but not identical to GICS.

For example, Meta is officially Communication Services, but its business-language peers can include companies like Workday, Salesforce, Intuit, and AppLovin. That reflects platform, software, advertising, and AI economics that GICS does not express cleanly.

Eaton is Industrials, but it can move near utilities because of electrification and grid exposure.

Starbucks is Restaurants, but it often looks close to beverage and consumer-brand companies.

That is the main qualitative value of the embeddings: they let us explore company neighborhoods that are not limited to official sector labels.

**AI example script:**

We also use the filing text to track themes over time. AI language increased sharply after 2023, and not only in obvious technology companies. This lets us ask questions like: which companies are starting to describe themselves as AI-exposed, and is that change concentrated in one sector or spreading across the market?

**Transition:**

The dashboard is compelling, but we still need quantitative tests. So we compare the embeddings against GICS.

## 8. Evaluation Against GICS

**Slide cue:** Results table: ARI/NMI and peer examples.

**Speaker script:**

The first evaluation asks whether embedding clusters align with GICS sectors.

We use two clustering metrics:

- ARI, adjusted Rand index.
- NMI, normalized mutual information.

Both compare our learned clusters to GICS sector labels. Higher is better, but we do not expect a perfect match because the point is not to reproduce GICS exactly.

Our historical-text business autoencoder produced stronger sector alignment than the older text baseline and much stronger alignment than price-only embeddings.

Key result:

| Embedding | ARI vs GICS | NMI vs GICS |
| --- | ---: | ---: |
| Historical-text business autoencoder | 0.255 | 0.433 |
| Temporal historical-text business autoencoder | 0.269 | 0.412 |
| Legacy semantic MiniLM text | 0.219 | 0.381 |
| Risk / price embedding | 0.052 | 0.171 |

Interpretation: business text is much better at recovering industry-like structure than price behavior. That makes sense. Filing text describes what companies do; prices describe how stocks trade.

**Peer evaluation script:**

We also tested whether embedding-nearest peers have stronger future return correlation than GICS sub-industry peers.

Here, GICS still wins. That is an important honest result. GICS sub-industries are very strong for short-horizon return co-movement. Our semantic embeddings find meaningful business similarity, but that is not the same as short-term stock correlation.

The nuance is that semantic embeddings improve at longer horizons. The semantic peer gap versus GICS narrows from about `-0.272` at 21 days to `-0.177` at 504 days. That supports the idea that text captures structural similarity, not immediate trading similarity.

**Relationship graph script:**

We also checked whether embedding peers are connected in the filing-derived relationship graph. Embedding peers are much more connected than random peers, but GICS sub-industry peers are still more directly connected. So again, the embeddings have signal, but GICS remains a tough benchmark.

**Transition:**

So far, the embeddings help us understand similarity. The final question is whether that structure can help with a more direct market task: predicting which groups outperform.

## 9. Sector-Relative Prediction

**Slide cue:** Sector Outlook dashboard tab.

**Speaker script:**

The final dashboard section is a sector-relative prediction model.

This is intentionally not a giant neural net. We use a walk-forward ridge regression model because the problem is small: roughly monthly observations and a limited number of groups. A simpler linear model is easier to audit and less likely to overfit.

The target is:

```text
future group return - future equal-weight S&P universe return
```

In other words, we ask: over the next horizon, did this sector or theme outperform the broad universe?

The model uses features such as:

- Sector momentum.
- Sector excess volatility.
- Fundamental and valuation signals.
- Growth and lifecycle features.
- In learned-theme mode, group embedding features.

The important design choice is walk-forward validation. At each historical prediction date, the model is trained only on completed past outcomes, then predicts the next period. Current live scores are shown separately and are not counted as backtest evidence until the forward horizon completes.

**Predictor-model comparison to show here:**

These are 63-day walk-forward GICS-sector prediction results under the current historical-membership setup. The capital simulation starts with `$10,000` and rotates into the top predicted sector at each completed rebalance date.

| Predictor | Mean rank IC | Top-bottom excess | Hit rate | Ending capital | Excess vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.098 | 1.15% | 54.3% | `$68,674` | +139.7 pp |
| Elastic Net | 0.108 | 1.02% | 54.3% | `$64,559` | +98.6 pp |
| Huber robust regression | 0.094 | 0.77% | 56.2% | `$78,116` | +234.2 pp |
| Random Forest | 0.083 | 0.97% | 54.3% | `$69,743` | +150.4 pp |
| Extra Trees | 0.078 | 0.77% | 58.2% | `$50,152` | -45.5 pp |
| Gradient Boosting | 0.038 | 0.26% | 51.0% | `$76,929` | +222.3 pp |

**How to explain this table:**

Ridge is the default not because it wins every single number, but because it is transparent, stable, and appropriate for a small monthly sector panel. Huber and Gradient Boosting currently produce higher simulated ending capital, but those results need more robustness checks because capital curves can be sensitive to a small number of sector calls. Ridge gives us a cleaner baseline story for class: simple model, walk-forward validation, positive average ranking signal.

**Current result to present carefully:**

Using the same historical-membership setup as the model-comparison table, the 63-day Ridge sector model has a positive average signal:

| Metric | Value |
| --- | ---: |
| Mean rank IC | 0.098 |
| Mean top-minus-bottom excess return | 1.15% |
| Top-sector hit rate | 54.3% |
| Strategy ending value from `$10,000` | `$68,674` |
| SPY benchmark ending value from `$10,000` | `$54,701` |
| Leakage violations | 0 |

We also added a simple rotation simulation. Starting with `$10,000`, the strategy rotates into the best predicted group at each completed rebalance date.

For the current GICS-sector setup, the simulation beats the regular S&P 500 proxy:

| Setup | Strategy Ending Value | SPY Benchmark Ending Value | Excess vs SPY |
| --- | ---: | ---: | ---: |
| GICS sector rotation | `$68,674` | `$54,701` | `+139.7 percentage points` |

**Very important caveat to say:**

This result is promising, but it should not be oversold. The learned-theme versions do not currently beat SPY, and the historical S&P membership/download work is still being improved to reduce survivorship bias. So the honest statement is:

> The sector-level model shows a positive walk-forward signal and beats the SPY benchmark under the current tested setup. The theme-level prediction models are not there yet.

**Transition:**

That brings us to the overall interpretation of the project.

## 10. What Worked, What Did Not, And Why That Matters

**Slide cue:** Three-column summary.

**What worked:**

- Historical filing-text embeddings recover meaningful business structure.
- The dashboard makes company similarity interpretable.
- AI-language shifts are visible over time.
- Some cross-sector peer groups make economic sense beyond GICS.
- The sector-relative model has a positive historical walk-forward signal.

**What did not fully work:**

- Embedding peers do not beat GICS sub-industry peers for short-horizon return co-movement.
- Embedding-based covariance does not beat Ledoit-Wolf on the full universe.
- The temporal autoencoder smooths trajectories, but it does not yet reliably detect known strategic events.
- The relationship graph is promising but sparse.

**Speaker script:**

The main result is not that one model beats everything. The main result is that company similarity decomposes into different views.

Business text is good for understanding company identity and themes.

Price behavior is better for return co-movement.

Fundamentals describe lifecycle and valuation.

Relationship data can validate economic links, but our current graph is still sparse.

And for sector prediction, the simpler GICS-level model currently has the strongest performance story.

## 11. Limitations

**Slide cue:** Limitations slide.

**Speaker script:**

There are several limitations we want to be explicit about.

First, survivorship bias is a real concern. We are adding historical S&P membership and deleted-company prices, but this is an area where the backtest needs continued hardening.

Second, GICS labels are mostly static in our current metadata. Companies can change sectors over time.

Third, the dashboard's 2D projections are for visualization. The underlying data is point-in-time, but the projection geometry can use the full sample to keep the map stable.

Fourth, SEC filings update slowly. Annual and quarterly filings are good for structural changes, but they may miss event-driven changes that appear first in 8-Ks or news.

Fifth, covariance was a negative result. Ledoit-Wolf remains the better full-universe risk benchmark.

**Transition:**

Even with those limitations, the project gives us a strong foundation.

## 12. Closing

**Slide cue:** Final thesis slide.

**Speaker script:**

The project started with the question: can we build company embeddings that reveal useful similarity beyond fixed sector labels?

The answer is yes, with nuance.

The business-text embeddings recover meaningful structure and reveal cross-sector themes that GICS misses. The dashboard makes those structures explorable. The evaluations show that GICS is still very strong for return co-movement, and Ledoit-Wolf remains hard to beat for covariance. But the embeddings are valuable because they capture a different kind of similarity: company identity, language, strategy, and theme exposure.

The best final framing is:

> GICS is a useful official classification. Our embeddings are an exploratory map of company similarity. They do not replace GICS, but they reveal structure GICS cannot show.

The next steps would be to harden the historical universe, add more event-driven 8-K features, improve relationship extraction, and test whether the sector prediction signal survives more realistic transaction costs and membership assumptions.

## Short Version If Time Is Tight

If you only have 7 to 8 minutes, use this compressed flow:

1. We wanted to learn company similarity beyond GICS.
2. Data came from SEC filings, XBRL fundamentals, prices, relationship extraction, and S&P metadata.
3. We parsed filings, generated MiniLM text embeddings, built point-in-time monthly features, and trained autoencoders.
4. The autoencoder compresses each company into a low-dimensional vector that represents its feature profile.
5. GMM clustering gives soft themes, which are useful because companies can belong to multiple themes.
6. Dashboard demo: show company map, META/ETN/SBUX examples, AI-language movement.
7. Quant results: historical-text business AE gets `NMI = 0.433` vs GICS; GICS still wins return co-movement.
8. Sector prediction: walk-forward ridge model has positive rank IC and the current GICS-sector rotation beats SPY in the tested setup.
9. Honest conclusion: embeddings are useful for understanding similarity, not a universal replacement for GICS or Ledoit-Wolf.

## Lines To Avoid Or Say Carefully

Do not say:

- "The embeddings predict stock returns."
- "We beat the market with AI."
- "The dashboard proves companies are pivoting."
- "The learned themes replace GICS."

Better phrasing:

- "The embeddings capture company similarity."
- "The sector-level walk-forward model shows a positive historical signal under the current tested setup."
- "The dashboard helps us explore possible strategic movement, which we then need to validate."
- "The learned themes complement GICS by revealing alternative company neighborhoods."
