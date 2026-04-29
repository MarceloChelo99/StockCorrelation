# Stock Embeddings Dashboard

Minimal Streamlit app for:

- fetching a small SEC raw filing corpus with `DatabaseOperator`
- browsing stored corpora on disk
- previewing structured filing rows and raw submission text
- exploring historical 10-K / 10-Q language trends and embedding movement
- exploring how company soft similarity-category memberships change over time

Run it from the repo root:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py
```

The `Similarity Explorer` tab reads the latest decomposed experiment with per-view GMM loadings, such as:

```text
experiments/20260424_144322_decomposed_point_in_time
```

Use the `Similarity Explorer` controls to pick a similarity view, map date, and focus company. The same screen shows the company's current theme mix, nearest peers by full loading vector, relationship evidence, compact full-period theme shifts, and the full S&P 500 2D map. The `Back` / `Forward` buttons move one usable map date at a time. Growth/lifecycle is intentionally hidden from this map because it is more useful as a prediction feature than as a visual cluster.

Use the `Model Comparison` tab for the classroom/audit view. The main dashboard uses opinionated defaults, while this final tab compares embedding experiments, clustering algorithms such as GMM/k-means/DBSCAN, and sector-prediction model variants.

Use the `Historical Text` tab to analyze the expanded compact historical 10-K / 10-Q artifacts in:

```text
data/processed/historical_text_10k_10q
```

It shows market-wide topic trends, company timelines, largest topic increases, filing-evidence snippets, sector-year heatmaps, and a 2D semantic trail map from historical section embeddings. This is the tab to use for questions like whether a company is increasingly describing itself through AI, cloud, cybersecurity, supply-chain, or electrification language. Older 10-K-only artifacts are still selectable if they exist under `data/processed/historical_text`.

The compact historical stream supports more than 10-K annual sections. For example:

```bash
.venv/bin/python -m scripts.historical_text.stream_features \
  --since 2010-01-01 \
  --forms "10-K,10-Q" \
  --output-dir data/processed/historical_text_10k_10q
```

For non-XBRL forms such as 8-Ks, proxies, or S-1s, use SEC submissions discovery:

```bash
.venv/bin/python -m scripts.historical_text.stream_features \
  --since 2024-01-01 \
  --forms "8-K,8-K/A,DEF 14A,S-1,S-1/A" \
  --discovery-source submissions \
  --output-dir data/processed/historical_text_event_proxy_ipo
```

Theme names are generated dynamically. The dashboard does not read `report/theme_labels.csv`: for the business view, it compares each theme centroid with historical filing-section embeddings and uses a stable representative fragment; for other views, it falls back to computed sector-mix labels. Inspect the generated labels and evidence from the `Dynamic theme labels` expander, including representative ticker, form, section, topic scores, similarity score, optional compact evidence snippet, and SEC source filing link.

Older historical text artifacts may not have snippets because the first compact stream discarded raw section text immediately after embedding/count extraction. If the raw historical corpus has already been downloaded, backfill snippets locally without recomputing embeddings or hitting SEC again:

```bash
.venv/bin/python scripts/historical_text/backfill_snippets_from_raw_corpus.py --resume
```

For honest historical maps, use the point-in-time decomposed experiment. The `business` and `network` views only have dates where actual filing text or relationship evidence exists; they no longer project current structure backward across the full history.
