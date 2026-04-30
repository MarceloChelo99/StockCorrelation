# StockCorrelation

Research dashboard and pipeline for learning interpretable stock/company embeddings from SEC filings, fundamentals, relationship disclosures, and market behavior.

The current class-demo workflow is centered on the Streamlit dashboard in `apps/raw_filing_browser/app.py`.

## Quick Start

From the repository root:

```bash
poetry install
poetry run streamlit run apps/raw_filing_browser/app.py --server.port 8510
```

Then open:

```text
http://localhost:8510
```

If you prefer the existing local virtual environment:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py --server.port 8510
```

## Requirements

- Python `3.14`
- Poetry
- Local project artifacts under `data/processed/`, `experiments/`, and `report/`

The main dependencies are declared in `pyproject.toml`: `pandas`, `pyarrow`, `torch`, `sentence-transformers`, `scikit-learn`, `cvxpy`, `networkx`, `streamlit`, and `umap-learn`.

## What To Use In The Dashboard

- `Similarity Explorer`: pick a company and inspect its business or financial theme mix, closest peers, movement summary, and dynamic label evidence.
- `Network`: current-state relationship evidence from filings, plus an upstream/downstream supply-chain disruption simulation.
- `Sector Outlook`: walk-forward group excess-return predictions, historical simulation, and feature importance/evidence.

Model-comparison material now lives in the written report/presentation instead of the product dashboard, so the app stays focused on exploration.

## Important Artifacts To Keep

The dashboard expects these current artifacts:

```text
data/processed/
experiments/20260425_192352_decomposed_point_in_time/
experiments/20260425_165422_temporal_business_view/
experiments/20260428_financial_embedding_outlook/
report/experiment_comparison.csv
report/feature_ablation_summary.csv
report/sector_autoencoder_outlook.csv
report/sector_autoencoder_embedding_correlations.csv
```

The final generated PowerPoint is here:

```text
report/presentations/stock_embeddings_current_state/output/output.pptx
```

## Pipeline Layout

The canonical pipeline scripts are organized by stage:

```text
scripts/pipeline/stage_01_ingest/
scripts/pipeline/stage_02_features/
scripts/pipeline/stage_03_assembly/
scripts/pipeline/stage_04_model/
scripts/pipeline/stage_05_evaluation/
```

Compatibility wrappers still exist at the old root paths, so both styles work.

## Common Commands

Compute features:

```bash
.venv/bin/python -m scripts.pipeline.stage_02_features.compute_features historical_text_business
```

Assemble a model-ready dataset:

```bash
.venv/bin/python -m scripts.pipeline.stage_03_assembly.assemble_dataset historical_text_business
```

Train a model:

```bash
.venv/bin/python -m scripts.pipeline.stage_04_model.train historical_text_business
```

Evaluate a trained model:

```bash
.venv/bin/python -m scripts.pipeline.stage_05_evaluation.evaluate historical_text_business experiments/<run>/embeddings.parquet
```

Run the decomposed view workflow:

```bash
.venv/bin/python -m scripts.decomposed.run_decomposed decomposed_point_in_time
```

## Historical Filing Text Artifacts

The separate Historical Text dashboard tab is hidden in the class-demo UI, but the compact historical 10-K / 10-Q artifacts still support business-view labels and filing evidence:

```text
data/processed/historical_text_10k_10q
```

To regenerate the compact text features from already available raw filings:

```bash
.venv/bin/python -m scripts.historical_text.stream_features \
  --since 2010-01-01 \
  --forms "10-K,10-Q" \
  --output-dir data/processed/historical_text_10k_10q
```

If snippets are missing but the raw corpus is already downloaded, backfill snippets without hitting SEC again:

```bash
.venv/bin/python scripts/historical_text/backfill_snippets_from_raw_corpus.py --resume
```

## Data Setup Commands

Fetch/update S&P membership history:

```bash
.venv/bin/python scripts/data_setup/fetch_sp500_membership_history.py
```

Import the S&P/GICS workbook if needed:

```bash
.venv/bin/python scripts/data_setup/import_sp500_gics_workbook.py /path/to/sp500_members_gics_2010_present.xlsx
```

Build market data for the historical universe:

```bash
.venv/bin/python scripts/data_setup/run_sp500_market_build.py --universe historical --skip-filings
```

## Notes For Reviewers

- Cached data lives under `data/processed/` and should not be deleted casually.
- Old experiment runs were removed to keep the project straightforward; summary metrics remain in `report/`.
- Theme labels in the dashboard are generated dynamically from filing fragments, financial feature profiles, or sector mixes. The dashboard does not rely on manually curated labels.
- Historical evaluation tries to use point-in-time data. Known limitations are documented in `report/point_in_time_audit.md`.
- Sector prediction outputs are walk-forward historical simulations, not trading recommendations.
