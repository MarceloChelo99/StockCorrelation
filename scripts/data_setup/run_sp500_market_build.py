from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401
REPO_ROOT = _bootstrap.REPO_ROOT

import argparse
import urllib.request
from datetime import date
from io import StringIO

import pandas as pd



from market_data_fetcher import DatabaseOperator, MarketDataCollector, RawFilingCorpusBuilder
from market_data_fetcher.networking import urlopen
from src.utils.io import ensure_dir
from src.utils.logging import log


SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BUILD_DATE = date.today().isoformat()
MEMBERSHIP_PATH = REPO_ROOT / "data" / "processed" / "metadata" / "sp500_membership_history.parquet"
HISTORICAL_FILINGS_GROUP_NAME = "S&P 500 Historical Filings Since 2010"
HISTORICAL_PRICES_GROUP_NAME = "S&P 500 Historical Prices Since 2010"
PRICE_FAILURES_PATH = REPO_ROOT / "data" / "raw" / "sp500_historical_price_failures.csv"


def parse_args() -> argparse.Namespace:
    """Parse market-build arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        choices=["historical", "current"],
        default="historical",
        help="historical uses the membership interval table; current uses today's Wikipedia roster.",
    )
    parser.add_argument("--membership-path", default=str(MEMBERSHIP_PATH))
    parser.add_argument("--start", default="2010-01-01", help="Earliest price date.")
    parser.add_argument("--end", default=BUILD_DATE, help="Latest price date.")
    parser.add_argument("--forms", default="10-K,10-Q", help="Comma-separated SEC forms for latest filing seed.")
    parser.add_argument(
        "--filings-per-form",
        type=int,
        default=1,
        help="Newest-N filings per form for the quick market seed. Use the historical text script for full history.",
    )
    parser.add_argument("--limit-tickers", type=int, default=None, help="Optional smoke-test ticker limit.")
    parser.add_argument("--skip-filings", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    return parser.parse_args()


def load_current_sp500_tickers() -> list[str]:
    """Return today's S&P 500 ticker list from Wikipedia."""
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


def load_historical_sp500_universe(path: str | Path) -> tuple[list[str], pd.DataFrame, dict[str, str]]:
    """Load all known historical S&P tickers plus trusted CIK mappings."""
    membership_path = Path(path)
    if not membership_path.exists():
        raise FileNotFoundError(
            f"Historical membership file not found at {membership_path}. Run "
            "`python scripts/data_setup/fetch_sp500_membership_history.py` first."
        )
    membership = pd.read_parquet(membership_path)
    required = {"ticker", "cik_str"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Historical membership file is missing columns: {sorted(missing)}")

    membership = membership.copy()
    membership["ticker"] = membership["ticker"].astype(str).str.upper()
    membership["cik_str"] = membership["cik_str"].where(membership["cik_str"].notna(), pd.NA)
    membership["cik_str"] = membership["cik_str"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)
    membership.loc[membership["cik_str"].str.contains("nan", case=False, na=True), "cik_str"] = pd.NA
    if "start_date" in membership.columns:
        membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce")
        membership = membership.sort_values(["ticker", "start_date"])
    ticker_index = membership.drop_duplicates("ticker", keep="last").reset_index(drop=True)
    all_tickers = sorted(ticker_index["ticker"].dropna().unique())
    cik_lookup = {
        str(row.ticker).upper(): str(row.cik_str).zfill(10)
        for row in ticker_index.dropna(subset=["cik_str"]).itertuples(index=False)
    }
    return all_tickers, ticker_index, cik_lookup


def parse_forms(value: str) -> tuple[str, ...]:
    """Parse a comma-separated form list."""
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def download_prices_resilient(
    operator: DatabaseOperator,
    tickers: list[str],
    *,
    group_name: str,
    start: str,
    end: str | None,
    sqlite_path: Path,
) -> tuple[pd.DataFrame, Path, pd.DataFrame]:
    """Download prices while logging per-ticker failures instead of aborting.

    Historical S&P universes contain acquired, delisted, renamed, and reused
    tickers. Yahoo often returns 404s for those. Those failures are expected
    data coverage gaps, not reasons to stop the whole market build.
    """
    downloader = operator.corpus_builder.collector.get_price_history_downloader()
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    total = len(tickers)
    for index, ticker in enumerate([str(ticker).upper() for ticker in tickers], start=1):
        if index == 1 or index == total or index % 25 == 0:
            log(f"Price ticker {index}/{total}: {ticker}", tag="sp500-prices")
        try:
            frame = downloader.download(ticker, start=start, end=end, lookback_days=None)
        except Exception as exc:  # noqa: BLE001
            failures.append({"ticker": ticker, "stage": "price_history", "error": str(exc)})
            continue
        if frame.empty:
            failures.append({"ticker": ticker, "stage": "price_history", "error": "empty_price_history"})
            continue
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not prices.empty:
        prices["ticker"] = prices["ticker"].astype(str).str.upper()
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    ensure_dir(PRICE_FAILURES_PATH.parent)
    failures_frame = pd.DataFrame(failures, columns=["ticker", "stage", "error"])
    failures_frame.to_csv(PRICE_FAILURES_PATH, index=False)
    db_path = operator.persist_prices_to_sqlite(prices, group_name=group_name, sqlite_path=sqlite_path)
    return prices, db_path, failures_frame


def main() -> None:
    args = parse_args()
    if args.skip_filings and args.skip_prices:
        raise ValueError("At least one of filings or prices must be enabled.")

    if args.universe == "historical":
        tickers, ticker_index, cik_lookup = load_historical_sp500_universe(args.membership_path)
        filings_group_name = HISTORICAL_FILINGS_GROUP_NAME
        prices_group_name = HISTORICAL_PRICES_GROUP_NAME
    else:
        tickers = load_current_sp500_tickers()
        ticker_index = pd.DataFrame({"ticker": tickers})
        cik_lookup = {}
        filings_group_name = f"S&P 500 Filings ({BUILD_DATE} roster)"
        prices_group_name = f"S&P 500 Prices Since 2010 ({BUILD_DATE} roster)"

    if args.limit_tickers is not None:
        tickers = tickers[: args.limit_tickers]
        ticker_index = ticker_index[ticker_index["ticker"].isin(tickers)].reset_index(drop=True)
        cik_lookup = {ticker: cik for ticker, cik in cik_lookup.items() if ticker in set(tickers)}

    collector = MarketDataCollector(
        identity="StockCorrelation/0.1 castellanosmarcelo1@gmail.com",
        cik_resolver=(lambda ticker: cik_lookup.get(str(ticker).upper())) if cik_lookup else None,
    )
    operator = DatabaseOperator(
        identity="StockCorrelation/0.1 castellanosmarcelo1@gmail.com",
        storage_dir=REPO_ROOT / "data" / "raw_filing_corpora",
        corpus_builder=RawFilingCorpusBuilder(collector=collector),
        default_format="parquet",
    )

    filing_tickers = [ticker for ticker in tickers if not cik_lookup or ticker in cik_lookup]

    print(
        {
            "universe": args.universe,
            "constituent_count": len(tickers),
            "filing_cik_resolved_count": len(filing_tickers),
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

    sqlite_path = REPO_ROOT / "data" / "raw_filing_corpora" / "raw_filing_corpora.sqlite"
    corpus = None
    stored = None
    if not args.skip_filings:
        forms = parse_forms(args.forms)
        corpus = operator.corpus_builder.build_for_symbols(
            filing_tickers,
            group_name=filings_group_name,
            filing_options={"since": args.start, "forms": forms, "filings_per_form": args.filings_per_form},
            raw_filing_options={"since": args.start, "forms": forms, "filings_per_form": args.filings_per_form},
            on_progress=report_progress,
        )
        if args.universe == "historical":
            corpus.ticker_index = ticker_index.copy()
        stored = operator.persist_corpus(corpus, format="parquet")
        sqlite_path = operator.persist_corpus_to_sqlite(corpus, stored=stored)

    prices = pd.DataFrame()
    price_failures = pd.DataFrame()
    if not args.skip_prices:
        prices, sqlite_path, price_failures = download_prices_resilient(
            operator,
            tickers,
            group_name=prices_group_name,
            start=args.start,
            end=args.end,
            sqlite_path=sqlite_path,
        )

    if corpus is not None:
        print("filings_summary", corpus.summary())
    if stored is not None:
        print("stored_summary", stored.summary())
    print("sqlite_path", sqlite_path)
    print(
        "price_summary",
        {
            "rows": len(prices),
            "tickers": int(prices["ticker"].nunique()) if not prices.empty else 0,
            "date_min": None if prices.empty else str(prices["date"].min()),
            "date_max": None if prices.empty else str(prices["date"].max()),
            "failures": int(len(price_failures)),
            "failures_path": str(PRICE_FAILURES_PATH),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
