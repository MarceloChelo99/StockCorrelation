from __future__ import annotations

from scripts import _bootstrap  # noqa: F401
REPO_ROOT = _bootstrap.REPO_ROOT

import sys
import urllib.request
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd



from market_data_fetcher import DatabaseOperator
from market_data_fetcher.networking import urlopen


SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BUILD_DATE = date.today().isoformat()


def load_sp500_tickers() -> list[str]:
    request = urllib.request.Request(
        SP500_CONSTITUENTS_URL,
        headers={"User-Agent": "StockCorrelation/0.1 castellanosmarcelo1@gmail.com"},
    )
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("No tables found at the S&P 500 constituents page.")

    constituents = tables[0].copy()
    if "Symbol" not in constituents.columns:
        raise RuntimeError("The S&P 500 constituents table is missing the Symbol column.")

    symbols = (
        constituents["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return symbols


def main() -> None:
    operator = DatabaseOperator(
        identity="StockCorrelation/0.1 castellanosmarcelo1@gmail.com",
        storage_dir=REPO_ROOT / "data" / "raw_filing_corpora",
        default_format="parquet",
    )

    tickers = load_sp500_tickers()
    filings_group_name = f"S&P 500 Filings ({BUILD_DATE} roster)"
    prices_group_name = f"S&P 500 Prices Since 2010 ({BUILD_DATE} roster)"

    print(
        {
            "constituent_count": len(tickers),
            "sample_tickers": tickers[:10],
            "filings_group_name": filings_group_name,
            "prices_group_name": prices_group_name,
        },
        flush=True,
    )

    def report_progress(ticker: str, index: int, total: int) -> None:
        if index == 1 or index == total or index % 25 == 0:
            print(
                {
                    "stage": "filings_and_raw_filings",
                    "progress": f"{index}/{total}",
                    "ticker": ticker,
                },
                flush=True,
            )

    corpus = operator.corpus_builder.build_for_symbols(
        tickers,
        group_name=filings_group_name,
        filing_options={"since": "2024-01-01", "forms": ("10-K", "10-Q"), "filings_per_form": 1},
        raw_filing_options={"since": "2024-01-01", "forms": ("10-K", "10-Q"), "filings_per_form": 1},
        on_progress=report_progress,
    )
    stored = operator.persist_corpus(corpus, format="parquet")
    sqlite_path = operator.persist_corpus_to_sqlite(corpus, stored=stored)

    prices, sqlite_path = operator.populate_prices_for_symbols_with_sqlite(
        tickers,
        group_name=prices_group_name,
        start="2010-01-01",
        end=BUILD_DATE,
        sqlite_path=sqlite_path,
    )

    print("filings_summary", corpus.summary())
    print("stored_summary", stored.summary())
    print("sqlite_path", sqlite_path)
    print(
        "price_summary",
        {
            "rows": len(prices),
            "tickers": int(prices["ticker"].nunique()) if not prices.empty else 0,
            "date_min": None if prices.empty else str(prices["date"].min()),
            "date_max": None if prices.empty else str(prices["date"].max()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
