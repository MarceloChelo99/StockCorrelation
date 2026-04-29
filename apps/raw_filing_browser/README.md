# Stock Embeddings Dashboard

Streamlit dashboard for exploring the project outputs: learned company themes, relationship evidence, and walk-forward group return predictions.

## Run

From the repository root:

```bash
poetry run streamlit run apps/raw_filing_browser/app.py --server.port 8510
```

Or with the local virtual environment:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py --server.port 8510
```

Open:

```text
http://localhost:8510
```

## Expected Local Artifacts

The app is designed around the current retained artifacts:

```text
data/processed/
experiments/20260425_192352_decomposed_point_in_time/
experiments/20260425_165422_temporal_business_view/
experiments/20260428_financial_embedding_outlook/
report/experiment_comparison.csv
report/feature_ablation_summary.csv
report/sector_autoencoder_outlook.csv
```

If one of these is missing, the relevant tab may show a warning or reduced output.

## Tabs

### Similarity Explorer

Pick a company and inspect:

- Current business-language or financial theme mix
- Closest peers in the full theme-loading space
- Filing-backed dynamic labels for business themes and feature-profile labels for financial themes

The peer table is the trusted similarity evidence. The UI intentionally avoids the older 2D market map because it was harder to interpret than the direct theme and peer tables.
The network view is intentionally kept in the separate Network tab, where the relationship evidence is easier to read.

### Network

Current-state relationship evidence from SEC filings. The tab focuses on readable supplier/customer/competitor/partner evidence rather than historical network animation.

### Sector Outlook

Walk-forward group excess-return predictions. Current scores are separated from completed historical predictions. Treat the simulation as a research backtest, not a trading recommendation.

### Model Comparison

Static presentation-style summary of model choices, saved metrics, and caveats. This tab is intentionally not a live experiment runner.

## Theme Labels

Theme names are generated dynamically. The dashboard does not depend on manually curated labels.

For the business view, labels come from representative filing fragments selected by similarity to theme centroids. For the financial view, labels come from weighted valuation, growth, profitability, liquidity, and momentum feature profiles.

## Historical Text Artifacts

The primary historical text artifact is:

```text
data/processed/historical_text_10k_10q
```

The dedicated Historical Text tab is hidden in the current class-demo dashboard, but these artifacts still power business-view labels and central filing snippets in Similarity Explorer.

To regenerate compact historical text features:

```bash
.venv/bin/python -m scripts.historical_text.stream_features \
  --since 2010-01-01 \
  --forms "10-K,10-Q" \
  --output-dir data/processed/historical_text_10k_10q
```

To backfill evidence snippets from an already downloaded raw corpus:

```bash
.venv/bin/python scripts/historical_text/backfill_snippets_from_raw_corpus.py --resume
```
