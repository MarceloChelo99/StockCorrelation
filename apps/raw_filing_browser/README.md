# Stock Embeddings Dashboard

Minimal Streamlit app for:

- fetching a small SEC raw filing corpus with `DatabaseOperator`
- browsing stored corpora on disk
- previewing structured filing rows and raw submission text
- exploring historical 10-K language trends and embedding movement
- exploring how company soft similarity-category memberships change over time

Run it from the repo root:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py
```

The `Similarity Shifts` tab reads the latest decomposed experiment with per-view GMM loadings, such as:

```text
experiments/20260424_144322_decomposed_point_in_time
```

Use the `Similarity Shifts` controls to pick a company, similarity view, granularity, number of visible groups, and a simulation date. The `Back` / `Forward` buttons move through time by the selected step size.

Use the `Market Map` tab to view the whole S&P 500 as a moving 2D similarity map. Each point is one stock, colors represent dominant soft themes, highlighted tickers show labels and movement trails, and the `Back` / `Forward` controls move through time by the selected step increment.

Use the `Historical Text` tab to analyze the compact historical 10-K artifacts in:

```text
data/processed/historical_text
```

It shows market-wide topic trends, company timelines, largest topic increases, sector-year heatmaps, and a 2D semantic trail map from historical section embeddings. This is the tab to use for questions like whether a company is increasingly describing itself through AI, cloud, cybersecurity, supply-chain, or electrification language.

The compact historical stream supports more than 10-K annual sections. For example:

```bash
.venv/bin/python scripts/24_stream_historical_text_features.py \
  --since 2010-01-01 \
  --forms "10-K,10-Q" \
  --output-dir data/processed/historical_text_10k_10q
```

For non-XBRL forms such as 8-Ks, proxies, or S-1s, use SEC submissions discovery:

```bash
.venv/bin/python scripts/24_stream_historical_text_features.py \
  --since 2024-01-01 \
  --forms "8-K,8-K/A,DEF 14A,S-1,S-1/A" \
  --discovery-source submissions \
  --output-dir data/processed/historical_text_event_proxy_ipo
```

Theme names live in `report/theme_labels.csv`. The file is seeded with top-firm labels, and you can improve them manually from the `Label themes` expander in the dashboard.

For honest historical maps, use the point-in-time decomposed experiment. The `business` and `network` views only have dates where actual filing text or relationship evidence exists; they no longer project current structure backward across the full history.
