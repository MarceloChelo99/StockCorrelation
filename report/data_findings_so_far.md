# Data Findings So Far

Generated: 2026-04-24 22:20 EDT

This report summarizes the project state from the artifacts currently in `data/processed/`, `experiments/`, and `report/`. The main read is that the project has moved from a simple stock-embedding experiment into a decomposed similarity system: text captures what companies do, prices capture how stocks behave, fundamentals capture lifecycle/maturity, and relationships capture disclosed economic links. Those views overlap, but they are not the same object.

## Executive Read

The strongest positive result is semantic structure. Text-derived embeddings recover meaningful business themes, produce sensible cross-sector peers, and identify real thematic shifts such as the rapid rise of AI language after 2023. The clearest dashboard value is therefore exploratory: a user can watch companies and sectors move through business-language, lifecycle, and trading-behavior maps over time.

The strongest risk-modeling result is more cautious. Price/risk embeddings are better than text embeddings for return co-movement, but GICS sub-industry remains a stronger peer benchmark. Ledoit-Wolf remains the full-universe covariance benchmark to beat. Embedding-prior covariance helps in some slices, but the direct multi-view factor covariance estimator does not beat Ledoit-Wolf.

The key research conclusion is not that embeddings failed. It is that company similarity decomposes into multiple useful meanings:

| Similarity view | What it captures best | Current status |
| --- | --- | --- |
| Business / semantic | What the company says it does, its risks, themes, and strategic exposures | Strong qualitative and clustering signal |
| Behavioral / risk | Volatility, momentum, liquidity, and return-like behavior | Best embedding view for peer return correlation |
| Growth / lifecycle | Revenue growth, margins, capex, leverage, payout, maturity | Useful but affected by fundamentals coverage and imputation |
| Network | Disclosed competitors, customers, suppliers, partners, agreements | Useful cross-sectional structure, currently mostly static |

## Data Inventory

The current completed historical text stream is compact and workable. It stores section-level keyword counts plus MiniLM embeddings, not full SEC filing text.

| Artifact | Current coverage | Notes |
| --- | ---: | --- |
| Historical 10-K filing index | 7,498 filings, 500 tickers | Since 2010-01-01 |
| Historical 10-K topic counts | 23,567 section rows | Business, risk factors, MD&A, cybersecurity |
| Historical 10-K embeddings | 23,391 section rows, 393 columns | About 62 MB on disk |
| Expanded 10-K + 10-Q stream | 4,401 filings, 76 tickers | Partial/in-progress expanded run |
| Expanded 10-K + 10-Q embeddings | 17,469 section rows | About 40 MB for 76 tickers |
| Fundamentals | 729,163 rows, 500 tickers | 14 XBRL concepts |
| Growth/lifecycle features | 92,887 rows, 503 tickers | Monthly panel |
| Network-position features | 92,887 rows, 503 tickers | Static graph features repeated through time |
| Relationship graph | 676 edges | First-pass 10-K-derived graph |
| 8-K event metadata | 32,459 rows, 503 tickers | Since 2024-01-01 |

The expanded historical parser now supports more sections and forms, including 10-K properties/legal/market-risk sections, 10-Q MD&A/legal/risk/market-risk sections, 8-K Item 1.01 and 2.02, proxy governance/compensation sections, and S-1 sections. The completed full-universe historical run is still the 10-K-only version; the broader form run is partial at this snapshot.

## Historical Text Findings

The historical 10-K text data shows a clean AI-language regime shift after 2023. In business sections, the AI score rises from `3.389` in 2023 to `5.693` in 2024 and `6.909` in 2025. In risk factors, the shift is even stronger: `1.576` in 2023, `6.803` in 2024, `9.988` in 2025, and `11.220` in partial 2026 data.

The largest company-level AI-language increases are concentrated in technology and tech-adjacent businesses, but not only in the official Information Technology sector. Top movers include `EPAM`, `ADBE`, `ADP`, `AMZN`, `EFX`, `PANW`, `NVDA`, `CDNS`, `NOW`, `INTU`, `ORCL`, `QCOM`, `MSFT`, and `META`.

At the sector level, Information Technology has the largest late-period AI score, but Communication Services, Industrials, Financials, Consumer Staples, Consumer Discretionary, and Health Care also show large increases. That is useful: the AI theme is not just an NVIDIA/Microsoft story in the data. It appears across companies that sell software, use automation, manage data, or describe AI as an operational/cyber risk.

Cybersecurity language also jumps sharply, especially in 2024 and later. Some of this is real risk exposure, but some is a disclosure artifact from the SEC cybersecurity rule and 10-K Item 1C. The dashboard should treat cybersecurity as both an economic theme and a reporting-regime theme.

## Embedding And View Findings

The semantic/text embedding is the best current representation for business structure. In the ablation summary, MiniLM text-only has `NMI = 0.381` against GICS sectors, the strongest sector-alignment result among the simple ablations. It also produces economically sensible cross-sector peer groups:

| Seed | Embedding peers | Interpretation |
| --- | --- | --- |
| `META` | `WDAY`, `CRM`, `INTU`, `APP` | Platform/software economics not captured by Communication Services label |
| `ETN` | Utilities such as `NEE`, `AES`, `SRE` | Electrification/grid-exposure theme |
| `CBRE` | `ARES`, `BX`, `KKR`, `IVZ`, `IBKR` | Real estate services linked to capital-markets exposure |
| `SBUX` | `KDP`, `HSY`, `YUM`, `PEP`, `KHC` | Consumer brand/beverage/restaurant overlap |

The behavioral/risk embedding is the best current embedding for return co-movement. It still trails GICS sub-industry peers, but by less than the semantic embedding does. At 21 trading days, the risk embedding peer-correlation gap versus GICS is `-0.100`, compared with `-0.272` for semantic peers.

The horizon sweep supports the thesis that text similarity is more structural than short-term trading similarity. The semantic peer-correlation gap narrows from `-0.272` at 21 days to `-0.177` at 504 days. It does not flip positive, but the direction is exactly what we would expect if text captures longer-horizon structural similarity rather than immediate stock co-movement.

The four-view decomposed system is not redundant. Cross-view NMI has mean off-diagonal `0.248` and max `0.312`, meaning the views overlap but are not copies of each other. That is important because the project thesis requires the views to carry distinct information.

## Relationship Graph Findings

The first relationship graph has `676` edges across `265` source tickers and `210` target tickers.

| Relationship type | Count |
| --- | ---: |
| Competitor | 252 |
| Agreement | 144 |
| Customer | 128 |
| Partner | 92 |
| Supplier | 60 |

Semantic peers are more connected in this graph than random peers, but less connected than GICS sub-industry peers:

| Peer set | Direct link rate | Neighbor Jaccard |
| --- | ---: | ---: |
| Semantic embedding peers | 0.070 | 0.030 |
| GICS sub-industry peers | 0.173 | 0.076 |
| Random peers | 0.010 | 0.011 |

This is a good, honest result. The semantic embedding contains relationship-network signal above random, but GICS remains stronger for direct disclosed links. The likely next improvement is not more clustering polish; it is better relationship data, especially 8-K Item 1.01 agreement text and cleaner supplier/customer direction extraction.

## Covariance And Portfolio Findings

The full-universe covariance result remains negative versus Ledoit-Wolf. In the original single-view covariance test, Ledoit-Wolf realized annual variance was around `0.00969`, sample covariance around `0.00978`, and embedding-prior shrinkage around `0.00994`.

The slice results are more interesting. Embedding-prior shrinkage beats Ledoit-Wolf in 3 of 9 tested slices:

| Slice | Ledoit-Wolf annual variance | Embedding annual variance | Relative improvement |
| --- | ---: | ---: | ---: |
| Health Care | 0.01761 | 0.01720 | 2.32% |
| Mid-liquidity | 0.01282 | 0.01263 | 1.44% |
| Consumer Discretionary | 0.01861 | 0.01851 | 0.49% |

The multi-view factor covariance estimator is not currently the answer. It realizes annual variance of `0.01522`, worse than Ledoit-Wolf at `0.00963` and sample covariance at `0.00966`. Leave-one-view-out ablations do not rescue it. The practical read is that the views are useful for interpretation and potentially for targeted shrinkage, but directly stacking all soft themes into one factor covariance model is too blunt and unstable.

The best methodological next step is likely a conservative hybrid: use Ledoit-Wolf as the base estimator, then let embedding/fundamental/network views adjust or route only where slice tests show stable incremental benefit.

## Dashboard Findings

The dashboard is most valuable as a visual research cockpit, not just a model-results viewer. The strongest dashboard use cases are:

1. Watch historical topic movement, especially AI, cloud/compute, cybersecurity, supply chain, and electrification language.
2. Compare a company’s nearest peers across views, because different views answer different questions.
3. Use the market map to see sector/theme pockets and whether groups move together through time.
4. Use behavioral and growth views for time movement; use business and network views carefully until the historical text and relationship graph are fully point-in-time.

The current view-dynamics report warns that older business-view movement is not meaningful in the original artifacts because the business embedding was effectively a current/latest embedding repeated across history. The new historical text stream fixes the raw ingredient problem by embedding filings through time, but the dashboard and view-training layer still need to consume the new historical embeddings fully before we interpret long-run business pivots like “who moved toward AI.”

## Scaling Note

The bottleneck for expanding to more stocks or more forms is compute time, not disk storage. The completed 10-K historical embeddings for 500 tickers take about `62 MB`. The partial 10-K + 10-Q run takes about `40 MB` for 76 tickers; a rough full-universe extrapolation is a few hundred MB, still reasonable on a laptop.

The expensive part is running the text embedder over many filings and sections. The current design is still the right direction because it streams filings, computes keyword counts first, stores compact outputs, and avoids saving full raw filing text. If we scale beyond the S&P 500 or add many form types, the main improvements should be batching, resumability, caching by accession/section hash, and optional GPU/cloud embedding jobs.

## Limitations To Keep Visible

The universe is current-S&P-500-style, so survivorship bias remains. GICS labels and company classifications can change through time. Cybersecurity language after 2023 is partly a regulation/disclosure artifact. The relationship graph is sparse and biased toward named public-company relationships. The network view is currently static, so it should not be treated as a historical movement signal. The expanded historical 10-K + 10-Q run is partial at this snapshot, not yet a full-universe result.

## Recommended Next Steps

The highest-value next step is to finish the expanded historical text stream and make the dashboard use those point-in-time embeddings for the business map. That directly supports the question we care about: which companies are truly shifting language and positioning toward AI, cloud, cybersecurity, supply chain, electrification, or other themes?

After that, the project should prioritize theme labeling and movement interpretation over another covariance-model attempt. The current dashboard will become much more useful once themes have stable human-readable names and each company’s movement can be summarized as “moved from mature hardware/manufacturing language toward AI/data-center language,” or similar.

For quantitative research, the next best risk-modeling path is not the multi-view factor model. It is a guarded hybrid policy: Ledoit-Wolf baseline, embedding-informed adjustments only in slices or pair groups where the data repeatedly shows improvement.
