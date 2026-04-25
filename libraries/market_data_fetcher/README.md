# market-data-fetcher

Reusable market-data library for:

- Yahoo Finance price history via `price-fetcher`
- SEC public ticker directory lookup
- structured SEC filing metrics from `companyfacts`
- raw SEC filing submission text for autoencoder / document-model pipelines
- raw filing corpus persistence with parquet shards for submission text
- built-in request throttling plus a certifi-backed HTTPS opener for SEC access

The package is intentionally independent from `core` so new projects can depend
on it without pulling in the model, database, dashboard, and backtest stack.

Price-history support is optional at runtime. If `price-fetcher` is not available,
the SEC ticker, structured filing, raw filing, and corpus/database flows still work.

## Example

```python
from market_data_fetcher import MarketDataCollector, RawFilingDownloader

collector = MarketDataCollector(identity="Your Name you@example.com")

bundle = collector.collect_for_symbols(
    ["AAPL", "MSFT"],
    include_filings=True,
    include_raw_filings=True,
    include_price_history=True,
    filing_options={"since": "2020-01-01"},
    raw_filing_options={"since": "2020-01-01", "filings_per_form": 2},
    price_history_options={"start": "2020-01-01", "interval": "1d", "lookback_days": None},
)

print(bundle.filings.head())
print(bundle.raw_filings[["ticker", "accession_no", "submission_text_length"]].head())
print(bundle.price_history.head())
```

Raw filings are downloaded as the SEC's complete submission text file:

`https://www.sec.gov/Archives/edgar/data/<cik>/<accession_no_no_dashes>/<accession_no_no_dashes>.txt`

## Raw Filing Corpus

```python
from market_data_fetcher import RawFilingCorpusBuilder

builder = RawFilingCorpusBuilder(identity="Your Name you@example.com")

corpus, written = builder.build_and_persist_for_symbols(
    ["AAPL", "MSFT"],
    "data/raw_filing_corpus",
    format="parquet",
    filing_options={"since": "2020-01-01"},
    raw_filing_options={"since": "2020-01-01", "filings_per_form": 2},
)

print(corpus.summary())
print(written["raw_filings"])
```

## Database Operator

```python
from market_data_fetcher import DatabaseOperator

operator = DatabaseOperator(
    identity="Your Name you@example.com",
    storage_dir="data/raw_filing_corpora",
)

corpus, stored = operator.populate_for_symbols(
    ["AAPL", "MSFT"],
    group_name="Tech Smoke Test",
    filing_options={"since": "2023-01-01"},
    raw_filing_options={"since": "2023-01-01", "filings_per_form": 1},
)

print(stored.raw_filings_path)
print(operator.find_raw_filings("Tech Smoke Test", ticker="AAPL").head())
print(operator.latest_filings("Tech Smoke Test", raw=True, limit=2))

corpus, stored, sqlite_path = operator.populate_for_symbols_with_sqlite(
    ["AAPL", "MSFT"],
    group_name="Tech Smoke Test SQLite",
    format="jsonl",
    filing_options={"since": "2024-01-01", "filings_per_form": 1},
    raw_filing_options={"since": "2024-01-01", "filings_per_form": 1},
)

print(sqlite_path)
print(operator.find_sqlite_raw_filings("Tech Smoke Test SQLite", ticker="MSFT").head())
```

There is also a runnable smoke script at [run_database_smoke_test.py](/Users/marcelocastellanos/PycharmProjects/StockCorrelation/scripts/run_database_smoke_test.py)
that populates a small `AAPL`/`MSFT` corpus and prints the stored results.
