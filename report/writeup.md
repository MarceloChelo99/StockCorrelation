# Stock Embeddings Research Notes

## Thesis

Company similarity is not a single object.

Text-derived embeddings capture structural and thematic similarity: what companies do, who they serve, what risks they discuss, and what strategic exposures they share. Price-derived embeddings capture behavioral and trading similarity: volatility, momentum, liquidity, and medium-horizon co-movement. Relationship graphs capture economic-network similarity: who companies mention as customers, suppliers, competitors, partners, or contractual counterparties. These similarity notions overlap, but they are not interchangeable.

The core result is that the project found this split empirically. Semantic embeddings recover meaningful company structure and cross-sector peer groups that GICS flattens. Risk embeddings are more useful for return co-movement, but still do not beat specialized risk benchmarks in the full universe. The first relationship-graph extension shows semantic peers are more economically connected than random peers, though still less connected than same-GICS-sub-industry peers.

This is a finding, not just a modeling decision. The original joint framing was tested through ablations and downstream tasks; the evidence showed that text-dominant and price-dominant representations serve different uses.

- **Semantic embedding:** text-heavy MiniLM features compressed with an autoencoder. This is best for explaining what companies do and for sector-like clustering.
- **Risk embedding:** price-behavior features compressed with an autoencoder. This is best for return co-movement, especially at intermediate horizons.

## Best Runs So Far

| Use case | Best current run | Evidence |
| --- | --- | --- |
| Sector / semantic structure | `semantic_embedding` | Highest NMI against GICS sectors |
| Return-peer correlation | `risk_embedding` | Best, least-negative peer correlation difference versus GICS sub-industry |
| Covariance estimation | `risk_embedding` | Embedding-prior shrinkage is close but currently worse than Ledoit-Wolf |
| Relationship graph alignment | `semantic_embedding` | Semantic peers are more directly linked than random peers |
| Combined narrative | Two-track setup | Semantic text embeddings for structure, price embeddings for risk behavior |

## Interpretation

MiniLM text features improve the model's ability to recover semantic sector structure. This is visible in stronger clustering metrics against GICS labels.

Price features are better aligned with return co-movement. This makes sense because volatility, momentum, and liquidity describe traded behavior more directly than annual filing language.

Simple 8-K item-count features add essentially no useful standalone structure in the current setup. Their ablation has `NMI = 0.091`, close to random for the GICS comparison, and peer-correlation difference of `-0.262`. The 8-K track likely needs content embeddings or event-severity features rather than raw item frequencies.

The first relationship graph currently comes from 10-K text, not full 8-K agreement text. The existing 8-K artifact stores item metadata but not agreement content, so extending the graph to 8-K Item 1.01 requires fetching or storing 8-K body text.

Read together, the evaluations suggest that "similar company" and "similar stock" are not the same object. The semantic embedding identifies structural and thematic similarity: business models, customers, risk language, and strategic exposure. The GICS and Ledoit-Wolf benchmarks are better aligned with short-horizon return co-movement because they are either explicitly industry-defined or purely return-statistical.

## Qualitative Examples

The semantic embedding produces several economically sensible peer groups that are not just copies of GICS labels. Full examples are saved in `report/semantic_peer_examples.csv`, with a compact selected table in `report/semantic_peer_examples_selected.md`.

| Ticker | GICS label | Embedding-nearest peers | Interpretation |
| --- | --- | --- | --- |
| CBRE | Real Estate / Real Estate Services | ARES, BX, IVZ, IBKR, KKR | Real-estate services clusters with asset managers, reflecting capital-markets exposure beyond the real-estate label. |
| ETN | Industrials / Electrical Components & Equipment | NEE, AES, SRE, NI, CMS | Electrical-equipment language clusters with utilities, consistent with grid investment and electrification exposure. |
| SBUX | Consumer Discretionary / Restaurants | KDP, HSY, YUM, PEP, KHC | A restaurant brand clusters with beverages and packaged-food names, suggesting brand and consumer-demand similarity. |
| META | Communication Services / Interactive Media & Services | WDAY, CRM, INTU, XYZ, APP | Communication-services classification misses that the filing language also resembles software and app-platform businesses. |
| HUM | Health Care / Managed Health Care | MCK, CAH, HSIC, COR, KVUE | Managed care clusters with health-care distributors, reflecting reimbursement and care-delivery overlap. |

## Relationship Graph

A first filing-derived relationship graph is saved at `data/processed/relationships/relationships.parquet`, with a readable summary in `report/relationship_graph_summary.md`.

The extractor uses conservative public-company name matching over parsed 10-K sections, then classifies the local context with rule-based labels: customer, supplier, competitor, partner, and agreement. This first pass produced `676` relationship rows across `265` source tickers and `210` target tickers.

| Relationship type | Count |
| --- | ---: |
| competitor | 252 |
| agreement | 144 |
| customer | 128 |
| partner | 92 |
| supplier | 60 |

As an evaluation target, the graph gives an external economic-structure check. For each firm, we ask whether its semantic embedding-nearest peers are directly connected in the filing-derived graph more often than random peers or GICS sub-industry peers.

| Peer set | Direct link rate | Neighbor Jaccard |
| --- | ---: | ---: |
| Semantic embedding peers | 0.070 | 0.030 |
| GICS sub-industry peers | 0.173 | 0.076 |
| Random peers | 0.010 | 0.011 |

Interpretation: semantic peers contain real relationship-network signal above random, but GICS sub-industry peers remain stronger for direct disclosed links. This strengthens the main thesis rather than overturning it: semantic similarity, trading similarity, and relationship-network similarity are overlapping but distinct views of company relatedness.

## Benchmark Results

The 21-day peer-identification test is negative versus the benchmark. GICS sub-industry peers have higher forward return correlation than embedding-nearest peers in the short-horizon setup.

At longer horizons, the result remains unfavorable but narrows. This matters because it is the direction predicted by the thesis: structural similarity should become less mismatched as the holding period lengthens.

The full-universe covariance test is also negative. On the 2024-2026 rolling window, Ledoit-Wolf realized annual variance is about `0.00969`, sample covariance is about `0.00978`, and embedding-prior shrinkage is about `0.00994`. A light embedding-prior alpha nearly matches Ledoit-Wolf in a sensitivity check, but it still does not beat it in the full universe.

The covariance picture becomes mixed, not uniformly negative, once sliced by sub-universe. Embedding-prior shrinkage beats Ledoit-Wolf in Consumer Discretionary, Health Care, and mid-liquidity stocks, but underperforms in Financials, Industrials, low-liquidity stocks, and the full universe.

## Stronger Framing

We built multimodal stock embeddings from 10-K/8-K text and price behavior, then evaluated them on three downstream tasks. The embeddings recover meaningful semantic structure against GICS sectors and reveal economically sensible cross-sector peer groups such as `META` with software platforms, `ETN` with utilities and electrification names, `CBRE` with alternative asset managers, and `SBUX` with consumer packaged brands. On short-horizon return correlation and covariance estimation, the embeddings do not beat specialized benchmarks, suggesting that embedding similarity and return similarity capture different notions of company relatedness.

## Follow-Up Tests

The follow-up analyses make the interpretation sharper.

Peer-correlation horizon sweeps are saved in `report/peer_horizon_semantic.csv` and `report/peer_horizon_risk.csv`. The semantic embedding's gap versus GICS narrows as the forward window lengthens: from `-0.272` at 21 trading days to `-0.177` at 504 trading days. The risk embedding is strongest at shorter/intermediate horizons, improving from `-0.100` at 21 days to about `-0.071` around 126 days, then widening slightly to `-0.079` at 504 days. The result does not flip positive, but it supports the idea that semantic similarity is less mismatched at longer horizons than at one-month horizons.

The semantic gap improves by about `0.019` correlation-difference units per doubling of the forward horizon. The risk embedding has a weaker horizon slope and a U-shaped profile: noisy at 21 days, strongest around 63-126 days, and less distinctive again by 504 days. That pattern is consistent with price-behavior features capturing medium-term trading similarity rather than permanent structural similarity.

Covariance slice results are saved in `report/covariance_slices.csv`. The embedding-prior covariance estimator beats Ledoit-Wolf on realized variance in some slices, including Consumer Discretionary, Health Care, and mid-liquidity stocks, but underperforms in Financials, Industrials, low-liquidity stocks, and the full universe. This suggests the embedding prior may add information in specific sub-universes rather than as a universal replacement for Ledoit-Wolf.

A first hybrid-policy summary is saved in `report/hybrid_covariance_policy.md`. In the current in-sample slice analysis, embedding-prior shrinkage is preferred in 3 of 9 tested slices: Health Care (`2.32%` lower annual variance than Ledoit-Wolf), mid-liquidity stocks (`1.44%` lower), and Consumer Discretionary (`0.49%` lower). This is not yet an out-of-sample routing rule, but it gives a concrete next methodological direction.

Thematic factor tests are saved in `report/thematic_factor_tests.csv`. Equal-weighted semantic peer clusters retain meaningful residual behavior after regressing on the market and the seed company's GICS sector basket. For example, the `META` cluster has market-plus-sector `R^2` of `0.536`, leaving about `68%` of its volatility as residual; the `SBUX` cluster has `R^2` of `0.544`, leaving about `68%` residual volatility. These clusters are not merely sector baskets in disguise.

Residual correlations across these thematic clusters are saved in `report/theme_residual_correlation_summary.md`. The mean pairwise residual correlation is only `0.033`, and the mean absolute residual correlation is `0.158`. The largest residual correlation is `SBUX` with `CMG` at `0.777`, which is economically sensible because both clusters are consumer food/restaurant themes. This supports the view that the residuals are not just one omitted common factor; they contain theme-specific behavior.

Feature ablations are summarized in `report/feature_ablation_summary.csv`. MiniLM text is the main driver of semantic structure (`NMI = 0.381`), price features are the best current driver of return-peer correlation (`diff = -0.100`), and simple 8-K item counts are weak on their own (`NMI = 0.091`, peer diff `-0.262`).

## Next Research Direction

Keep two deliverables:

- `semantic_embedding`: for clustering, peer maps, and qualitative examples.
- `risk_embedding`: for peer-correlation and later covariance tests.

The next empirical question is whether semantic similarity can be converted from an explanatory representation into a stronger quantitative signal without losing interpretability.

Specific follow-ups:

- Build a hybrid covariance policy that uses Ledoit-Wolf by default and routes to embedding-prior shrinkage in sub-universes where the slice tests show improvement.
- Expand the relationship graph with 8-K Item 1.01 agreement text and better supplier/customer direction extraction.
- Use the horizon-sweep results to design longer-horizon peer tests with non-overlapping evaluation windows.
- Turn selected cross-sector semantic clusters into explicit thematic factor portfolios and compare them with standard style and sector factors.
- Replace simple 8-K item counts with 8-K content embeddings or event-severity features, since raw item frequencies are weak in the current ablation.
