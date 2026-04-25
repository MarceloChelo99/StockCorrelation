# Project Structure For Code Review

Generated: 2026-04-25

This document explains the current codebase structure, data pipeline, library choices, scripts, dashboard, artifacts, and known design tradeoffs. It is intended as a reviewer handoff: enough context to suggest architectural, reliability, and maintainability improvements without needing to reverse-engineer the whole repository first.

## Project Purpose

The project builds stock/company embeddings for the S&P 500 universe using several views of similarity:

| View | Data source | Purpose |
| --- | --- | --- |
| Business / semantic | 10-K and historical filing text | Capture what companies do, business risks, and strategic themes |
| Behavioral / risk | Daily prices | Capture stock behavior: volatility, momentum, liquidity |
| Growth / lifecycle | SEC XBRL fundamentals | Capture maturity, growth, profitability, leverage, payout |
| Network | Filing-derived relationship graph | Capture disclosed competitors, customers, suppliers, partners, agreements |

The main research thesis is that company similarity is decomposed: the company that “does similar things” is not always the stock that “trades similarly.” The project supports both model/evaluation experiments and an interactive Streamlit dashboard for exploring trajectories and themes.

## Top-Level Layout

```text
StockCorrelation/
├── apps/
│   └── raw_filing_browser/        # Streamlit dashboard
├── config/
│   ├── default.yaml               # Base config, currently JSON-formatted YAML
│   └── experiments/               # Experiment overrides
├── data/
│   ├── raw/                       # Small failure logs / raw-ish files
│   ├── raw_filing_corpora/        # Local SEC filing corpora managed by market_data_fetcher
│   └── processed/                 # Stable parquet/json artifacts
├── experiments/                   # Immutable-ish experiment output directories
├── libraries/
│   └── market_data_fetcher/       # Local helper library for SEC/database ingestion
├── report/                        # Research outputs, CSVs, summaries, reviewer docs
├── scripts/                       # Thin pipeline and analysis entry points
├── src/                           # Reusable library code
├── tests/                         # Unit/smoke tests
├── pyproject.toml                 # Poetry project and dependencies
└── poetry.lock
```

The repository currently does not contain the original `PROJECT_SPEC.md`; the implementation has evolved directly in code, reports, configs, and scripts.

## Runtime And Dependency Choices

The project is configured through Poetry in `pyproject.toml`.

| Dependency | Current use |
| --- | --- |
| Python `>=3.14` | Runtime target |
| `numpy>=2.4` | Numeric model code, distances, covariance calculations |
| `pandas>=3.0` | Main table engine |
| `pyarrow>=22.0` | Parquet IO |
| `pyyaml>=6.0` | Config loading |
| `torch==2.10.0` | Installed and base interface imports `torch.nn`, but current autoencoder is NumPy |
| `sentence-transformers==5.4.1` | MiniLM filing-section embeddings |
| `networkx>=3.6` | Relationship graph features, centrality, Louvain communities |
| `scikit-learn>=1.8` | PCA, GMMs, Ledoit-Wolf, clustering metrics |
| `cvxpy>=1.8` | Minimum-variance portfolio optimization |
| `streamlit>=1.50` | Dashboard |
| `altair` | Used by dashboard but not listed explicitly in `pyproject.toml` |

Reviewer note: `altair` is imported by the dashboard and should probably be made an explicit dependency. Also, the current `Autoencoder` is a manual NumPy implementation even though PyTorch is installed; this is intentional for the current baseline but worth reviewing before adding more neural models.

## Configuration System

Config loading lives in `src/config.py`.

Important files:

```text
config/default.yaml
config/experiments/*.yaml
```

The loader deep-merges experiment configs into the default config. Lists are replaced, not concatenated. Resolved configs are copied into experiment directories.

Important config sections:

| Section | Purpose |
| --- | --- |
| `paths` | Locations for SQLite/raw corpora, processed features, datasets, experiments |
| `data` | Filing/prices group names and sample start/end |
| `features` | Enabled feature groups and parameters |
| `fundamentals` | SEC companyfacts ingest options |
| `assembly` | Dataset name and required feature groups |
| `model` | Model selection and generic model parameters |
| `views` | Four decomposed view definitions and feature regexes |
| `evaluation` | Evaluator-specific parameters |
| `pipeline` | Flags for view training, GMMs, and multi-view evaluation |

Current quirk: `config/default.yaml` is syntactically JSON inside a `.yaml` file. This works with YAML parsers, but a reviewer may reasonably suggest making it idiomatic YAML or renaming if consistency matters.

## Data Artifacts

Processed data is stored mainly as parquet files under `data/processed/`.

```text
data/processed/
├── datasets/             # Model-ready assembled panels
├── events/               # 8-K event metadata
├── features/             # Per-feature-group parquet files
├── historical_text/      # Compact historical 10-K topic counts + embeddings
├── historical_text_10k_10q/
├── metadata/             # S&P 500 metadata / GICS labels
├── relationships/        # Filing-derived relationship graph
└── sections/             # Parsed latest 10-K sections
```

Key current artifacts:

| Artifact | Role |
| --- | --- |
| `data/raw_filing_corpora/raw_filing_corpora.sqlite` | Local SQLite corpus managed by the ingestion helper library |
| `data/processed/metadata/sp500_gics.parquet` | Ticker metadata and GICS labels |
| `data/processed/sections/ten_k_sections.parquet` | Parsed latest 10-K sections |
| `data/processed/events/eight_k_events.parquet` | 8-K item metadata since 2024 |
| `data/processed/features/*.parquet` | Price, text, event, fundamentals, and graph feature groups |
| `data/processed/datasets/*.parquet` | Assembled model training panels |
| `data/processed/historical_text/*.parquet` | Full-universe compact historical 10-K features |
| `data/processed/historical_text_10k_10q/*.parquet` | In-progress expanded historical 10-K + 10-Q features |
| `data/processed/relationships/relationships.parquet` | Sparse company relationship graph |

Historical text storage is compact by design. The script streams a filing, parses sections, computes keyword topic counts, optionally embeds section text, and discards raw filing text.

## Source Package Layout

```text
src/
├── applications/         # Downstream application logic, currently multi-view covariance
├── clustering/           # GMM soft clustering and theme interpretation helpers
├── evaluation/           # Evaluators and registry
├── features/             # Feature producers and assembly
├── filings/              # SEC filing section parsing and keyword topics
├── ingest/               # SEC fundamentals ingestion
├── models/               # PCA, NumPy autoencoder, model training helper
├── relationships/        # Relationship extraction from filing text
├── utils/                # Dates, IO, logging, seed helpers
├── config.py             # Config loading
├── db.py                 # SQLite/raw corpus access wrapper
└── experiment.py         # Experiment directory lifecycle
```

### `src/db.py`

`FilingsDB` is the stable wrapper around the SQLite/raw filing corpus. It centralizes access to tickers, filings, prices, and metadata. Downstream scripts generally receive a config, instantiate `FilingsDB.from_config(config)`, and then load the relevant data.

### `src/features/`

Feature groups use an explicit producer interface:

```text
src/features/base.py
src/features/registry.py
```

Current producers:

| Producer key | Module | Output |
| --- | --- | --- |
| `price_volatility` | `features/price/volatility.py` | Rolling realized vol |
| `price_momentum` | `features/price/momentum.py` | Trailing returns |
| `price_liquidity` | `features/price/liquidity.py` | Volume/liquidity proxies |
| `event_item_frequency` | `features/events/item_frequency.py` | Rolling 8-K item counts |
| `growth_lifecycle` | `features/fundamentals/growth_lifecycle.py` | XBRL lifecycle features |
| `network_position` | `features/graph/network_position.py` | Relationship graph position features |
| `text_business` | `features/text/business_description.py` | Hashed latest business-section text features |
| `text_risk` | `features/text/risk_factors.py` | Hashed latest risk-section text features |

`src/features/assembly.py` builds the monthly ticker-date panel and merges feature groups point-in-time. The core convention is that keys are regular columns `ticker` and `date`, not indices.

### `src/filings/`

`src/filings/sections.py` parses SEC submission text into named sections. It currently supports:

| Form family | Parsed sections |
| --- | --- |
| 10-K original | `business`, `risk_factors`, `mda`, `item_1c_cybersecurity` |
| 10-K expanded | `item_2_properties`, `item_3_legal_proceedings`, `item_7a_market_risk` |
| 10-Q | `q_mda`, `q_market_risk`, `q_legal_proceedings`, `q_risk_factors` |
| 8-K | `eightk_item_1_01_material_agreement`, `eightk_item_2_02_results` |
| DEF 14A / proxy | `proxy_governance`, `proxy_directors`, `proxy_compensation`, `proxy_pay_vs_performance` |
| S-1 | `s1_summary`, `s1_risk_factors`, `s1_business`, `s1_mda` |

`src/filings/topics.py` contains transparent keyword counters for themes such as AI, cloud/compute, cybersecurity, supply chain, and electrification. These are used before embedding so the dashboard can show interpretable topic movement.

### `src/models/`

Current model registry:

```python
MODEL_REGISTRY = {
    "autoencoder": Autoencoder,
    "pca": PCAModel,
}
```

`PCAModel` is a lightweight SVD baseline.

`Autoencoder` is a shallow nonlinear manual NumPy model with tanh activations, Adam, feature standardization, save/load via `.npz` and `meta.json`. It does not currently use PyTorch despite the project dependency. The abstract base class imports `torch.nn.Module` when available, but the concrete models use NumPy arrays and `.fit()` rather than a PyTorch training loop.

`src/models/train.py` selects numeric feature columns, builds the configured model, trains it, saves the model, and writes `embeddings.parquet` plus `training_history.json`.

Reviewer note: this is a likely refactor point if the next work adds temporal autoencoders or richer architectures. There is a mismatch between the PyTorch-style `EmbeddingModel` base and the current NumPy implementation.

### `src/clustering/`

Soft clustering is implemented with Gaussian Mixture Models:

| Module | Purpose |
| --- | --- |
| `clustering/gmm.py` | Fit GMMs on per-view embeddings, select component count by BIC |
| `clustering/interpret.py` | Top firms per theme, feature profiles, GICS overlap |

These outputs feed the dashboard and the multi-view covariance experiment.

### `src/evaluation/`

Evaluators implement a common `Evaluator.run(...)` interface and are registered in `src/evaluation/registry.py`.

| Evaluator key | Purpose |
| --- | --- |
| `clustering` | K-means / sector alignment vs GICS |
| `peers` | Embedding-nearest peer return correlation vs GICS sub-industry benchmark |
| `covariance` | Single-view embedding-prior covariance shrinkage vs sample and Ledoit-Wolf |
| `relationships` | Whether embedding peers align with filing-derived relationship graph |
| `multiview_covariance` | Factor covariance using stacked soft theme loadings |
| `multiview_peers` | View-by-horizon peer-correlation matrix |
| `view_comparison` | Cross-view NMI between hard GMM assignments |

Portfolio optimization uses `cvxpy` for long-only minimum-variance weights.

### `src/relationships/`

`relationships/extract.py` does conservative public-company name matching and rule-based context classification into customer, supplier, competitor, partner, agreement, or generic relationships. It produces a sparse graph, not a complete supply-chain database.

### `src/utils/`

Small helpers for:

| Module | Purpose |
| --- | --- |
| `dates.py` | Month-end sampling and point-in-time/as-of merges |
| `io.py` | JSON/parquet helpers and directory creation |
| `logging.py` | Timestamped `print` logging |
| `seed.py` | Seed Python, NumPy, and torch if available |

## Scripts

Scripts are intentionally thin entry points, but the project has accumulated many analysis scripts. Most scripts manually add the repo root to `sys.path`; some also add `libraries/market_data_fetcher/src`.

### Core Pipeline Scripts

| Script | Purpose |
| --- | --- |
| `scripts/00_fetch_sp500_metadata.py` | Fetch current S&P 500/GICS metadata |
| `scripts/run_sp500_market_build.py` | Build local raw filing/price corpus with `market_data_fetcher` |
| `scripts/00_fetch_8k_events.py` | Fetch 8-K event metadata from SEC submissions |
| `scripts/01_parse_sections.py` | Parse latest 10-K sections from stored filings |
| `scripts/02_compute_features.py` | Run registered feature producers |
| `scripts/03_assemble_dataset.py` | Build model-ready monthly dataset |
| `scripts/04_train.py` | Train configured model and write embeddings |
| `scripts/05_evaluate.py` | Run configured evaluators |
| `scripts/run_experiment.py` | Orchestrate feature, assembly, training, evaluation, optional views/GMMs |
| `scripts/20_run_decomposed.py` | Full decomposed-view orchestration |

### Decomposed Similarity Scripts

| Script | Purpose |
| --- | --- |
| `scripts/16_ingest_fundamentals.py` | SEC XBRL companyfacts ingestion |
| `scripts/17_train_views.py` | Train/reuse one autoencoder per view |
| `scripts/18_fit_gmms.py` | Fit GMM soft themes per view |
| `scripts/19_interpret_views.py` | Generate theme interpretation reports |
| `scripts/21_multiview_covariance_slices.py` | Slice multi-view covariance by sector/liquidity |
| `scripts/22_analyze_theme_evolution.py` | Analyze theme dynamics and topic enrichment |

### Historical Text Scripts

| Script | Purpose |
| --- | --- |
| `scripts/23_fetch_historical_10k_filings.py` | Older raw historical 10-K streaming approach |
| `scripts/24_stream_historical_text_features.py` | Preferred compact historical text stream: parse, count topics, embed, discard raw text |
| `scripts/25_report_historical_text_trends.py` | Produce topic trend reports from compact historical features |

### Analysis / Reporting Scripts

| Script | Purpose |
| --- | --- |
| `scripts/06_compare_experiments.py` | Summarize experiment metrics |
| `scripts/07_qualitative_peers.py` | Generate qualitative nearest-peer examples |
| `scripts/08_peer_horizon_sweep.py` | Peer test across forward horizons |
| `scripts/09_covariance_slices.py` | Single-view covariance slice analysis |
| `scripts/10_thematic_factor_tests.py` | Regress semantic peer clusters against market/sector |
| `scripts/11_ablation_summary.py` | Summarize feature ablations |
| `scripts/12_theme_residual_correlations.py` | Residual correlation between thematic clusters |
| `scripts/13_hybrid_covariance_policy.py` | In-sample hybrid covariance routing summary |
| `scripts/14_extract_relationships.py` | Extract relationship graph from parsed sections |
| `scripts/15_relationship_graph_summary.py` | Summarize graph coverage and alignment |
| `scripts/run_database_smoke_test.py` | Small ingestion smoke test |

Reviewer note: many script names encode chronological research history. This is useful for exploration but may be worth grouping into subdirectories later: `scripts/pipeline/`, `scripts/analysis/`, `scripts/historical_text/`, `scripts/decomposed/`.

## Dashboard

The dashboard is in:

```text
apps/raw_filing_browser/app.py
apps/raw_filing_browser/README.md
```

Run command:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py
```

Dashboard tabs include:

| Tab | Purpose |
| --- | --- |
| Filing Browser | Browse local raw filing corpora and raw submission text |
| Historical Text | Topic trends, company timelines, largest topic increases, sector heatmaps, semantic trails |
| Similarity Shifts | View a selected company's soft-theme membership over time |
| Market Map | 2D map of all stocks through time, colored by dominant theme |

Theme names live in:

```text
report/theme_labels.csv
```

The dashboard uses cached parquet loads and `PCA` projections for 2D maps. It reads experiments, view loadings, historical text features, metadata, and theme labels from local artifacts.

Reviewer note: the dashboard is currently one large Streamlit file. It may benefit from extracting data-loading, charting, and state-management helpers into modules under `apps/raw_filing_browser/`.

## Experiment Outputs

Experiments are written to timestamped directories:

```text
experiments/<YYYYMMDD_HHMMSS_name>/
├── config.yaml
├── embeddings.parquet
├── metrics.json
├── training_history.json
├── model/
└── views/                  # for decomposed runs
```

Examples of important runs:

| Run family | Purpose |
| --- | --- |
| `baseline` | PCA/price baseline |
| `ae_*` | Feature ablations |
| `semantic_embedding` | Text-heavy semantic embedding |
| `risk_embedding` | Price-only behavioral/risk embedding |
| `decomposed` | Four-view setup |
| `decomposed_point_in_time` | Point-in-time decomposed view experiment for dashboard |

Reviewer note: experiment directories are mostly immutable by convention, not enforced. Some scripts also write reports directly to `report/`.

## Reports

Research reports and CSV summaries live in `report/`. Important examples:

| Report | Purpose |
| --- | --- |
| `writeup.md` | Main research notes |
| `data_findings_so_far.md` | Current findings summary |
| `historical_text_trends.md` | Historical filing-language trends |
| `decomposed_similarity_results.md` | Four-view decomposed results |
| `multiview_covariance_results.md` | Multi-view covariance result |
| `relationship_graph_summary.md` | Relationship graph summary |
| `view_cluster_dynamics_summary.md` | Movement/dynamics interpretation |
| `view_*_themes.md` | Theme interpretation by view |
| `similarity_fingerprints.md` | Peer examples per view |

## Tests

Tests are in `tests/` and use Python `unittest` style.

Current test files:

```text
test_autoencoder_model.py
test_clustering_evaluator.py
test_covariance_evaluator.py
test_cvxpy_optimizer.py
test_database_operator.py
test_event_features.py
test_filing_sections.py
test_gmm_clustering.py
test_multiview_covariance.py
test_pca_model.py
test_peers_evaluator.py
test_relationship_evaluator.py
test_relationship_extraction.py
test_stock_embeddings_scaffold.py
test_text_features.py
```

Common command:

```bash
.venv/bin/python -m unittest discover tests
```

Tests cover parsers, feature/evaluator logic, PCA/autoencoder behavior, covariance optimization, relationship extraction, and the basic scaffold.

## Current Pipeline Flow

At a high level:

```text
SEC / market data
    ↓
local raw filing corpus + metadata + prices
    ↓
section parsing / event metadata / fundamentals / relationships
    ↓
feature-group parquets
    ↓
assembled monthly dataset
    ↓
PCA or autoencoder embeddings
    ↓
evaluations and decomposed view models
    ↓
reports + dashboard
```

For historical text specifically:

```text
SEC filing URL
    ↓
download one filing
    ↓
parse supported sections
    ↓
compute keyword topic counts
    ↓
compute MiniLM section embeddings
    ↓
discard raw text
    ↓
write compact parquet rows
```

## Technical Deep Dive

### Data Model And Column Conventions

Most durable tabular artifacts are parquet files. The project intentionally keeps keys as regular columns rather than indices.

Common conventions:

| Convention | Current implementation |
| --- | --- |
| Firm key | `ticker`, usually uppercased |
| Date key | `date` for monthly panels, `filing_date` for SEC filings, `period_end` where available |
| Model features | Prefix by source, e.g. `price_vol_21d`, `growth_leverage`, `net_pagerank`, `text_business_*` |
| Embeddings | `embedding_0`, `embedding_1`, ... |
| Soft themes | `theme_0`, `theme_1`, ... |
| Metadata | `gics_sector`, `gics_sub_industry`, `company_name` when present; older artifacts may use `title` or `search_label` |

The project has two related but different text feature systems:

| Text system | Artifact | Purpose |
| --- | --- | --- |
| Latest-section feature producers | `data/processed/features/text_business.parquet`, `text_risk.parquet` | Simple hashed features from latest parsed 10-K sections for original model pipeline |
| Compact historical text stream | `data/processed/historical_text*/historical_section_*.parquet` | Section-level topic counts and MiniLM embeddings over historical filings for dashboard/time analysis |

Reviewer concern: the historical text stream is not yet a first-class `FeatureProducer` in the generic `02 → 03 → 04` pipeline. It currently feeds dashboard/report workflows more directly.

### Point-In-Time Mechanics

The primary point-in-time merge is in `src/utils/dates.py::as_of_merge`.

Mechanics:

1. Sort left panel and right feature frame by `ticker` and `date`.
2. For each ticker group, use `pd.merge_asof(..., direction="backward", allow_exact_matches=True)`.
3. Preserve original left row count and order.
4. Feature rows after the observation date are not eligible.

The monthly observation panel is built in `src/features/assembly.py::build_observation_panel`.

Mechanics:

1. Load daily prices for configured date range.
2. Select observed month-end trading rows per ticker.
3. Require at least `features.price.min_history_days`, currently `252`, before a ticker-date enters the panel.
4. Merge feature groups onto this panel with as-of joins.

Reviewer concerns:

- `allow_exact_matches=True` is appropriate for features whose `date` means “available by that date,” but it is dangerous if a feature `date` is fiscal period end rather than public filing date. Fundamentals and filing-derived features must use public `filing_date`.
- Any feature producer that emits `date` from fiscal period end would leak. Reviewers should check this carefully in fundamentals/text/network feature producers.
- Current metadata is mostly latest/static and not point-in-time, especially GICS labels.

### SQLite And Raw Filing Access

`src/db.py::FilingsDB` reads from:

```text
data/raw_filing_corpora/raw_filing_corpora.sqlite
```

Main tables expected by current code:

| Table | Used by |
| --- | --- |
| `tickers` | Universe metadata |
| `filings` | Filing index |
| `raw_filings` | Raw filing text index and/or shard pointers |
| `prices` | Daily adjusted prices and volume |

`load_raw_submission_text(accession_no)` first checks for inline `submission_text`. If not present, it resolves `raw_shard_path` and `raw_shard_row_number`, then reads the parquet shard row. This means raw filing storage can be externalized into shards without changing downstream parsers.

Reviewer concern: DB schema is implicit. There is no formal migration/schema definition in the repo.

### Feature Producer Contract

Feature producers inherit from `src/features/base.py::FeatureProducer`. Each exposes a `FeatureSpec` with:

```python
name: str
columns: list[str]
key_columns: list[str]
source: str
description: str
```

`FeatureProducer.run()` computes the frame, validates exact expected columns, writes parquet, and returns the frame.

Validation behavior:

- Missing declared columns raise.
- Extra columns raise.
- Duplicate key rows raise.
- Keys are normal columns.

Feature groups can be recomputed independently, and the assembly step is the only cross-source merge point.

Reviewer concern: the registry returns producer instances but does not encode dependencies, freshness, schema versions, or feature provenance beyond the manifest.

### Current Feature Engineering Details

| Feature family | Main mechanics |
| --- | --- |
| Price volatility | Daily adjusted-return rolling realized vol over windows from config, sampled at month end |
| Price momentum | Trailing adjusted-close returns over configured windows |
| Price liquidity | Volume/liquidity windows and return/volume proxies |
| 8-K event counts | Rolling counts by item number over trailing windows |
| Text business/risk | Hashed token vectors from parsed latest 10-K sections |
| Fundamentals growth/lifecycle | SEC companyfacts concepts, point-in-time by filing date, ratios/trends/winsorization |
| Network position | Relationship graph centrality/community features, now intended to use only disclosed edges available by date |

Fundamentals source concepts include revenue, gross profit, operating income/loss, net income/loss, assets, stockholders’ equity, long-term debt, R&D, capex, repurchases, dividends, and shares outstanding. The lifecycle interpretation is high growth/investment/low payout versus mature/stable/high payout.

Network features use `networkx`, including degree variants, PageRank, betweenness sampling, clustering coefficient, and Louvain communities.

### Historical Text Stream Details

Primary script:

```text
scripts/24_stream_historical_text_features.py
```

Important CLI arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--since` | `2010-01-01` | Earliest filing date |
| `--forms` | `10-K` | Comma-separated SEC form list |
| `--sections` | all supported sections | Parsed section keys to keep |
| `--discovery-source` | `auto` | `companyfacts`, `submissions`, or automatic selection |
| `--metadata-path` | `data/processed/metadata/sp500_gics.parquet` | Ticker universe |
| `--output-dir` | `data/processed/historical_text` | Destination artifact directory |
| `--max-sec-requests-per-second` | `6` | Rate limiter |
| `--checkpoint-every-filings` | `25` | Periodic persistence |
| `--resume` | false | Load existing output and skip completed accessions |
| `--no-embed` | false | Count topics only |
| `--model-name` | `sentence-transformers/all-MiniLM-L6-v2` | Text embedding model |
| `--embedding-batch-size` | `32` | Embedder batch size |
| `--max-chars` | `12000` | Truncate section text before embedding |
| `--min-embed-chars` | script default below parsed args | Skip tiny sections |

Discovery choices:

- `companyfacts` is used for `10-K` / `10-Q` because companyfacts exposes enough filing accession metadata for these periodic forms.
- `submissions` is required for non-companyfacts forms such as `8-K`, `DEF 14A`, and `S-1`.
- `auto` switches to submissions if any requested form is outside `{"10-K", "10-Q"}`.

The output schema is long-format by section:

```text
historical_filing_index.parquet:
  ticker, cik, accession_no, form, filing_date, period_end, source_url, downloaded_at

historical_section_topic_counts.parquet:
  ticker, cik, accession_no, form, filing_date, period_end,
  section, section_label, section_chars,
  topic_*_mentions, topic_*_score, topic_*_score_per_10k_words, ...

historical_section_embeddings.parquet:
  ticker, cik, accession_no, form, filing_date, period_end,
  section, section_label, section_chars,
  embedding_0 ... embedding_383
```

Current completed artifact:

- `data/processed/historical_text`: full-universe historical 10-K compact stream.

Current in-progress artifact:

- `data/processed/historical_text_10k_10q`: expanded 10-K + 10-Q stream.

Reviewer concerns:

- The stream writes state as whole parquet files at checkpoint time. This is simple but may become slow as artifacts grow.
- Accessions are skipped on resume, but section-level versioning is not tracked. If parser logic changes, old rows may not be invalidated automatically.
- Embedding truncation at `max_chars=12000` is pragmatic but can bias long sections.

### Filing Parser Details

The parser operates on raw SEC submission text. It extracts document blocks, selects the document matching form type, strips HTML/XBRL noise, then identifies item headings using regex patterns and scoring heuristics.

Supported dispatch:

```python
parse_filing_sections(submission_text, form_type)
```

Supported section groups:

| Form | Section keys |
| --- | --- |
| `10-K` | `business`, `risk_factors`, `mda`, `item_1c_cybersecurity`, `item_2_properties`, `item_3_legal_proceedings`, `item_7a_market_risk` |
| `10-Q` | `q_mda`, `q_market_risk`, `q_legal_proceedings`, `q_risk_factors` |
| `8-K` | `eightk_item_1_01_material_agreement`, `eightk_item_2_02_results` |
| `DEF 14A` | `proxy_governance`, `proxy_directors`, `proxy_compensation`, `proxy_pay_vs_performance` |
| `S-1` | `s1_summary`, `s1_risk_factors`, `s1_business`, `s1_mda` |

Reviewer concerns:

- This is heuristic parsing, not an SEC-standardized section API.
- Section heading scoring is intentionally permissive for 10-Q short sections.
- Tests cover representative synthetic and some real-ish cases, but real SEC filings have many edge cases.

### Model Training Details

Generic training entry:

```text
src/models/train.py::train_embedding_model
```

Steps:

1. Select numeric feature columns excluding configured metadata.
2. Convert to float matrix.
3. Replace NaN/Inf with zero.
4. Instantiate `MODEL_REGISTRY[config["model"]["name"]]`.
5. Call `.fit(matrix)`.
6. Call `.encode(matrix)`.
7. Write `embeddings.parquet`, `model/`, and `training_history.json`.

`Autoencoder` details:

| Property | Current behavior |
| --- | --- |
| Implementation | Manual NumPy backpropagation |
| Standardization | Store feature mean/std; zero std set to 1 |
| Encoder | `input → tanh(hidden) → tanh(embedding)` |
| Decoder | `embedding → tanh(hidden) → linear reconstruction` |
| Loss | Mean squared reconstruction error in standardized feature space |
| Optimizer | Small custom Adam |
| Save format | `weights.npz` plus `meta.json` |
| Hidden dims | Generic AE supports one `hidden_dim`; view configs may specify `hidden_dims` but only first value is used in `scripts/17_train_views.py` |

`PCAModel` details:

- Standardizes features.
- Uses SVD for a linear projection.
- Saves projection and normalization arrays.

Reviewer concerns:

- No train/validation split in the generic NumPy AE path.
- No early stopping in the generic AE path.
- No mini-batch validation metrics beyond initial/final reconstruction MSE.
- Feature imputation differs between generic training and view training.
- If the project moves to temporal AE, PyTorch would likely be cleaner than extending custom NumPy backprop.

### View-Specific Training Details

`scripts/17_train_views.py` trains/reuses one autoencoder per view.

Mechanics:

1. Load assembled dataset.
2. For each active view in `views_enabled`, select numeric columns matching `views.<view>.feature_pattern`.
3. Drop rows with fewer than `min_non_missing_features`.
4. Optionally drop rows with absolute feature signal below `min_feature_abs_sum`.
5. Median-impute missing feature values.
6. Train a view-specific `Autoencoder`.
7. Save to `experiments/<run>/views/<view>/`.

Configured views:

| View | Regex | Dim | Notes |
| --- | --- | ---: | --- |
| Business | `^(text_business|text_risk)_` | 32 | Semantic/text view |
| Behavioral | `^price_` | 16 | Can reuse risk embedding artifact |
| Growth | `^growth_` | 8 | Fundamentals lifecycle |
| Network | `^net_` | 8 | Graph-position features |

Reviewer concern: this script bypasses `MODEL_REGISTRY` and directly imports `Autoencoder`. It also translates `hidden_dims` to only the first `hidden_dim`.

### GMM Soft Clustering Details

`src/clustering/gmm.py::fit_gmm_on_embeddings` fits GMMs on embedding columns.

Mechanics:

1. Input frame contains `ticker`, `date`, and `embedding_*`.
2. Candidate component counts are tested, typically range 5 to 40.
3. Select `n_components` by minimum BIC.
4. Save soft membership probabilities as `theme_*` loadings.
5. Save hard labels, BIC/AIC, and metadata.

The soft loadings are used for:

- View interpretation.
- Dashboard theme membership over time.
- Multi-view covariance factor construction.

Reviewer concerns:

- GMM themes can be unstable under random seed/component count.
- BIC-selected component count may optimize density fit, not interpretability.
- Theme labels are manually curated in `report/theme_labels.csv`, not learned.

### Peer Evaluation Details

`src/evaluation/peers.py::PeersEvaluator`

Goal: test whether embedding-nearest peers have stronger forward return co-movement than benchmark peers.

Mechanics:

1. Compute daily adjusted returns matrix from prices.
2. Build candidate embedding observations with metadata and enough forward returns.
3. Optionally downsample observations to `max_observations`, currently `5000`.
4. For each observation date:
   - Compute Euclidean pairwise distances in embedding space.
   - For each ticker, select top-`k` nearest embedding peers.
   - Select `k` random firms from the same benchmark group, usually `gics_sub_industry`.
   - Compute correlation between target forward returns and average peer-group forward returns.
5. Report paired metrics: means, mean difference, median difference, t-statistic, normal-approx p-value.

Key config:

```text
evaluation.peers.k = 5
evaluation.peers.forward_days = 21
evaluation.peers.benchmark_column = gics_sub_industry
evaluation.peers.min_benchmark_pool = 6
```

Reviewer concerns:

- Random GICS peers introduce sampling noise; repeated trials or fixed multiple draws could stabilize.
- The t-test uses a normal approximation and does not cluster standard errors by ticker/date.
- Same-subindustry GICS is a very strong benchmark for short-horizon return co-movement.

### Covariance Evaluation Details

`src/evaluation/covariance.py::CovarianceEvaluator`

Goal: test whether embedding similarity improves covariance estimation.

Rolling protocol:

1. Choose eligible rebalance month-ends from embedding dates.
2. For each date:
   - Use trailing `lookback_days=252` daily returns.
   - Use forward `holding_days=21` daily returns.
   - Keep assets with complete history and forward returns.
   - If more than `max_assets=150`, choose deterministic low-volatility subset.
3. Estimate covariance matrices:
   - Sample covariance.
   - Ledoit-Wolf-style shrinkage toward constant-variance diagonal target.
   - Embedding-prior shrinkage.
4. Solve long-only minimum-variance portfolios.
5. Record realized forward portfolio returns and turnover.

Embedding-prior covariance:

```text
squared_distances_ij = ||z_i - z_j||^2
scale = median positive off-diagonal squared distance
prior_corr_ij = exp(-squared_distances_ij / scale)
prior_cov = prior_corr * outer(sample_std, sample_std)
cov = (1 - alpha) * sample_cov + alpha * prior_cov
```

Current default:

```text
embedding_alpha = 0.25
```

Minimum-variance optimizer:

- Single-view covariance evaluator uses an active-set analytical solve with pseudo-inverse fallback.
- Multi-view covariance evaluator uses `cvxpy` with solvers `CLARABEL`, `OSQP`, then `SCS`, falling back to the analytical solver.

Reviewer concerns:

- The Ledoit-Wolf implementation is local, not imported from sklearn, and should be reviewed against the standard estimator.
- Asset selection by low volatility when above `max_assets` may bias covariance evaluation.
- Bootstrap CIs are used in metrics, but dependence across overlapping holding periods may remain.

### Multi-View Covariance Details

`src/applications/multiview_covariance.py`

Intended model:

```text
L = stacked firm × theme loadings across views
r_theme = weighted average firm returns by theme loading
F = rolling covariance of theme returns
Σ = L F L' + D
```

Implementation details:

1. For each view, take latest loadings at or before `as_of_date`.
2. Prefix theme columns by view and concatenate horizontally.
3. Build theme returns with loading-weighted averages:
   ```text
   r_theme = returns @ weights / weights.sum(axis=0)
   ```
4. Estimate factor covariance over trailing window.
5. Compute systematic covariance `L F L'`.
6. Cap systematic variance with `max_systematic_fraction`, currently `0.8`.
7. Set diagonal residual `D` so diagonal roughly matches firm realized variance.
8. Add diagonal floor and PSD jitter if needed.

Evaluator benchmarks:

- Sample covariance.
- Ledoit-Wolf.
- Single-view embedding-prior covariance.
- Full multi-view factor covariance.
- Leave-one-view-out ablations.

Reviewer concerns:

- Theme returns use latest loadings for the whole return history window, not time-varying historical loadings within the window. This is simpler but loses intra-window loading changes.
- Stacked soft themes are likely collinear, making factor covariance unstable.
- Current result underperforms Ledoit-Wolf; the method is useful as an experimental baseline, not production.

### Relationship Graph Details

Relationship extraction pipeline:

```text
parsed 10-K sections
    ↓
public-company name matching against universe metadata
    ↓
context window around mention
    ↓
rule-based relationship type classifier
    ↓
relationships.parquet
```

Relationship schema includes source firm, target firm, relationship type, confidence, source accession/date/form, and context snippet.

Types:

```text
customer, supplier, competitor, partner, agreement, generic
```

Network feature producer uses the relationship graph to compute graph-position features. Recent implementation intent is point-in-time: only relationships with filing dates at or before the observation date should be included.

Reviewer concerns:

- Entity resolution is heuristic and conservative.
- Directionality for customer/supplier can be wrong or ambiguous from context.
- The graph is sparse and biased toward named large/public counterparties.

### Dashboard Technical Details

The Streamlit app currently combines:

- Raw filing browser.
- Historical text topic charts.
- Similarity membership trajectories.
- Whole-market 2D map.
- Theme labeling interface.

Dashboard data sources:

| Source | Used for |
| --- | --- |
| `data/raw_filing_corpora` | Raw filing browsing |
| `experiments/*/views/*/loadings.parquet` | Similarity shifts and market map |
| `report/theme_labels.csv` | Human-readable theme names |
| `data/processed/historical_text` | Historical text topic/semantic charts |
| `data/processed/metadata/sp500_gics.parquet` | Sector/company labels |

2D maps currently use PCA projections of theme loadings/embeddings for stable visualization. They are exploratory and should not be overinterpreted as preserving all high-dimensional distances.

Reviewer concern: dashboard state logic and analysis logic live together in one file, making it harder to test.

### Current Results Artifacts Relevant For ML Review

The key quantitative artifacts are:

| Artifact | What to review |
| --- | --- |
| `report/feature_ablation_summary.csv` | Which feature groups help clustering/peer correlation |
| `report/peer_horizon_semantic.csv` | Semantic peer test by horizon |
| `report/peer_horizon_risk.csv` | Risk peer test by horizon |
| `report/covariance_slices.csv` | Single-view covariance slice behavior |
| `report/multiview_covariance_results.md` | Multi-view covariance failure analysis |
| `report/view_cluster_dynamics/*.csv` | View dynamics and cluster movement |
| `report/historical_text_trends.md` | Topic drift evidence |
| `report/theme_labels.csv` | Manual theme naming layer |

### Reproducibility Notes

Current reproducibility mechanisms:

- Configs are merged and saved into experiment directories.
- Random seed is set through `src/utils/seed.py`.
- Parquet artifacts are persisted between stages.
- Experiment directories include embeddings, metrics, training history, and logs.

Remaining reproducibility gaps:

- External SEC/Wikipedia data can change.
- S&P 500 universe is current-roster based, not historical constituents.
- Some scripts write reports outside experiment directories.
- Historical text stream can resume, but parser/model version metadata is minimal.
- PyTorch/CUDA determinism is not relevant to current NumPy AE but will matter if temporal neural models are added.

## Known Strengths

- The codebase uses explicit registries for features, models, and evaluators.
- Most durable artifacts are parquet or JSON, which keeps inspection simple.
- Point-in-time merging exists in `src/utils/dates.py` and feature assembly.
- The dashboard makes the research tangible and exposes model outputs interactively.
- The tests cover several nontrivial pieces: filing parsing, covariance, GMMs, relationships.
- Historical text streaming avoids storing huge raw filings by default.

## Known Rough Edges / Review Targets

These are the areas where reviewer feedback would be especially useful:

1. Model interface mismatch: `EmbeddingModel` subclasses `torch.nn.Module`, but current models are NumPy `.fit()` models. Temporal/neural models will likely force a decision.
2. Autoencoder architecture: the current AE supports one hidden dimension, while view configs include `hidden_dims`; `scripts/17_train_views.py` handles view training separately rather than through the generic registry.
3. Script sprawl: scripts are useful but chronologically organized; grouping or adding a CLI could improve usability.
4. Dashboard size: `apps/raw_filing_browser/app.py` is a large single file and could be modularized.
5. Dependency declaration: `altair` is imported but not explicitly listed in `pyproject.toml`.
6. Config consistency: `default.yaml` is JSON-style YAML; valid but visually inconsistent.
7. Experiment immutability: outputs are conventionally immutable, not enforced.
8. Data scale/resume: historical embedding runs can be long; resumability exists through output artifacts but could be hardened with accession-level caching.
9. Raw/current versus point-in-time features: some older business/network artifacts projected current state backward; newer point-in-time work is correcting this, but reviewers should check assumptions carefully.
10. Local library path handling: many scripts manually modify `sys.path` for repo root and `libraries/market_data_fetcher/src`; packaging this cleanly would reduce fragility.
11. Historical text features are not yet integrated as first-class feature producers, which limits temporal model training.
12. Current statistical tests do not use clustered/bootstrap inference for peer correlations, so significance should be interpreted cautiously.
13. GMM theme identities can drift across experiments; theme labels are manually curated and not guaranteed stable after retraining.
14. The covariance pipeline mixes custom covariance estimators, custom active-set optimization, and cvxpy optimization across evaluators; standardizing would reduce audit burden.
15. There is no formal data catalog that records artifact schema, producing script, code version, config hash, and upstream dependencies.

## Suggested Reviewer Questions

- Should the project standardize around PyTorch now, before adding temporal autoencoders?
- Should scripts be reorganized into pipeline stages versus exploratory analyses?
- Is the current feature registry enough, or should it include output path/schema/version metadata more formally?
- Should historical text artifacts become first-class feature producers rather than dashboard/report-only artifacts?
- Is the dashboard better kept as a research cockpit, or should it be split into reusable backend modules plus Streamlit UI?
- Are point-in-time guarantees sufficiently visible and testable?
- Should experiment outputs be made immutable by code rather than convention?
- Should the local `market_data_fetcher` package become an installed editable dependency instead of being added via `sys.path`?
- Should peer-evaluation inference account for clustering by ticker/date and overlapping forward windows?
- Should Ledoit-Wolf use sklearn's implementation for auditability, with the local implementation retained only for learning/tests?
- Should GMM/BIC themes be replaced or supplemented with more stable clustering methods for dashboard continuity?
- Should the temporal-business view be trained on section-level MiniLM embeddings, aggregated monthly point-in-time, rather than the older hashed text features?
- Should text embeddings be cached by `(accession_no, section, parser_version, model_name, max_chars)` to make parser/model changes auditable?

## Reviewer Quickstart

Install/use the existing environment, then run:

```bash
.venv/bin/python -m unittest discover tests
```

Launch the dashboard:

```bash
.venv/bin/streamlit run apps/raw_filing_browser/app.py
```

Inspect the main configs:

```text
config/default.yaml
config/experiments/decomposed_point_in_time.yaml
config/experiments/semantic_embedding.yaml
config/experiments/risk_embedding.yaml
```

Inspect the main reports:

```text
report/writeup.md
report/data_findings_so_far.md
report/decomposed_similarity_results.md
report/historical_text_trends.md
report/multiview_covariance_results.md
```
