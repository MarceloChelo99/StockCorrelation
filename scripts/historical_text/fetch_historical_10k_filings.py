"""Stream historical SEC 10-K raw filings for the current S&P 500 universe.

The original market build intentionally downloaded only the latest 10-K/10-Q
per company. That is enough for a current snapshot, but not enough to study
business pivots through time. This script downloads all annual filings since a
chosen date and writes raw filing text incrementally into parquet shards so the
run does not need to hold thousands of full SEC submissions in memory.
"""
from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd



from market_data_fetcher import (  # noqa: E402
    CompanyFactsParser,
    DatabaseOperator,
    EdgarCompanyFactsClient,
    RAW_FILING_DATASET_COLUMNS,
    RequestRateLimiter,
    RawFilingCorpus,
    RawFilingSubmissionClient,
)
from src.utils.logging import log  # noqa: E402


DEFAULT_GROUP_NAME = "S&P 500 Historical 10-K Filings Since 2010 (2026-04-24 roster)"
RAW_FILING_INDEX_COLUMNS = (
    "ticker",
    "cik",
    "accession_no",
    "form",
    "filing_date",
    "period_end",
    "submission_text_length",
    "source_url",
    "downloaded_at",
    "raw_shard_path",
    "raw_shard_row_number",
)


def main() -> None:
    """Download historical 10-Ks and register the resulting corpus in SQLite."""
    args = parse_args()
    group_name = args.group_name or DEFAULT_GROUP_NAME
    storage_dir = Path(args.storage_dir)
    dataset_dir = storage_dir / slugify(group_name)

    if dataset_dir.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{dataset_dir} already exists. Pass --resume to continue or --overwrite to replace it."
        )
    if args.overwrite and args.resume:
        raise ValueError("Choose only one of --overwrite or --resume.")
    if args.overwrite and dataset_dir.exists():
        import shutil

        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(group_name)
    shards_dir = dataset_dir / f"{stem}_raw_filing_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path, args.limit_tickers)
    ticker_index = metadata.loc[:, ["ticker", "cik_str", "company_name", "gics_sector", "gics_sub_industry"]].copy()
    ticker_index = ticker_index.rename(columns={"cik_str": "cik"})

    existing = load_existing_state(dataset_dir, stem) if args.resume else ExistingState.empty()
    existing_accessions = set(existing.raw_index["accession_no"].astype(str)) if not existing.raw_index.empty else set()
    next_shard_number = next_available_shard_number(shards_dir, stem)

    limiter = RequestRateLimiter(max_requests=args.max_sec_requests_per_second, period_seconds=1.0)
    facts_client = EdgarCompanyFactsClient(args.identity, rate_limiter=limiter)
    raw_client = RawFilingSubmissionClient(args.identity, rate_limiter=limiter)
    parser = CompanyFactsParser()

    filing_rows = existing.filings.to_dict("records")
    raw_index_rows = existing.raw_index.to_dict("records")
    failure_rows = existing.failures.to_dict("records")
    shard_buffer: list[dict[str, object]] = []
    shard_text_bytes = 0

    forms = tuple(parse_csv_arg(args.forms))
    log(
        f"Downloading {forms} since {args.since} for {len(metadata)} tickers into {dataset_dir}.",
        tag="historical-10k",
    )

    for ticker_index_number, record in enumerate(metadata.itertuples(index=False), start=1):
        ticker = str(record.ticker).upper()
        cik = str(record.cik_str).zfill(10)
        if ticker_index_number == 1 or ticker_index_number % 10 == 0 or ticker_index_number == len(metadata):
            log(f"Ticker {ticker_index_number}/{len(metadata)}: {ticker}", tag="historical-10k")

        try:
            facts_payload = facts_client.fetch(cik)
            filings = parser.parse(ticker, facts_payload)
            filings = filter_filings(filings, since=args.since, forms=forms, max_per_form=args.max_filings_per_form)
        except Exception as exc:  # noqa: BLE001
            failure_rows.append({"ticker": ticker, "stage": "filing_index", "error": str(exc)})
            continue

        if filings.empty:
            failure_rows.append({"ticker": ticker, "stage": "filing_index", "error": "No matching filings found."})
            continue

        filing_rows.extend(filings.to_dict("records"))
        for filing in filings.itertuples(index=False):
            accession_no = str(filing.accession_no)
            if accession_no in existing_accessions:
                continue
            try:
                submission_text, source_url = raw_client.fetch(cik, accession_no)
            except Exception as exc:  # noqa: BLE001
                failure_rows.append(
                    {
                        "ticker": ticker,
                        "stage": "raw_filing",
                        "accession_no": accession_no,
                        "error": str(exc),
                    }
                )
                continue

            raw_row = {
                "ticker": ticker,
                "cik": cik,
                "accession_no": accession_no,
                "form": str(filing.form),
                "filing_date": str(filing.filing_date),
                "period_end": str(filing.period_end),
                "submission_text": submission_text,
                "submission_text_length": len(submission_text),
                "source_url": source_url,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
            shard_buffer.append(raw_row)
            shard_text_bytes += len(submission_text.encode("utf-8", errors="ignore"))
            if len(shard_buffer) >= args.shard_size or shard_text_bytes >= args.max_shard_text_bytes:
                next_shard_number = flush_raw_shard(
                    shard_buffer,
                    raw_index_rows,
                    shards_dir,
                    stem,
                    next_shard_number,
                )
                write_progress_files(dataset_dir, stem, filing_rows, raw_index_rows, failure_rows, ticker_index)
                shard_buffer = []
                shard_text_bytes = 0

    if shard_buffer:
        next_shard_number = flush_raw_shard(
            shard_buffer,
            raw_index_rows,
            shards_dir,
            stem,
            next_shard_number,
        )

    write_progress_files(dataset_dir, stem, filing_rows, raw_index_rows, failure_rows, ticker_index)
    write_manifest(dataset_dir, stem, group_name, ticker_index, filing_rows, raw_index_rows, failure_rows)
    register_sqlite(
        group_name=group_name,
        storage_dir=storage_dir,
        sqlite_path=Path(args.sqlite_path),
        ticker_index=ticker_index,
        filings=pd.DataFrame(filing_rows).drop_duplicates("accession_no", keep="last"),
        raw_index=pd.DataFrame(raw_index_rows).drop_duplicates("accession_no", keep="last"),
        failures=pd.DataFrame(failure_rows),
    )
    log(
        f"Historical 10-K corpus ready: {len(raw_index_rows)} raw filings, {len(failure_rows)} failures.",
        tag="historical-10k",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2010-01-01", help="Earliest SEC filing date to include.")
    parser.add_argument("--forms", default="10-K", help="Comma-separated annual forms to include.")
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME, help="Stored corpus group name.")
    parser.add_argument("--metadata-path", default="data/processed/metadata/sp500_gics.parquet")
    parser.add_argument("--storage-dir", default="data/raw_filing_corpora")
    parser.add_argument("--sqlite-path", default="data/raw_filing_corpora/raw_filing_corpora.sqlite")
    parser.add_argument("--identity", default="StockCorrelation research castellanosmarcelo1@gmail.com")
    parser.add_argument("--limit-tickers", type=int, default=None, help="Optional smoke-test ticker limit.")
    parser.add_argument("--max-filings-per-form", type=int, default=None, help="Optional newest-N limit per form.")
    parser.add_argument("--shard-size", type=int, default=25, help="Number of raw filings per parquet text shard.")
    parser.add_argument(
        "--max-shard-text-bytes",
        type=int,
        default=750_000_000,
        help="Flush a shard before PyArrow's 2 GB string-array limit can be hit.",
    )
    parser.add_argument("--max-sec-requests-per-second", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild an existing corpus directory.")
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded accessions and append new shards.")
    return parser.parse_args()


def load_metadata(path: str, limit: int | None) -> pd.DataFrame:
    """Load the ticker universe and CIKs."""
    frame = pd.read_parquet(path)
    required = {"ticker", "cik_str"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["cik_str"] = frame["cik_str"].astype(str).str.zfill(10)
    frame = frame.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit).reset_index(drop=True)
    return frame


def parse_csv_arg(value: str) -> list[str]:
    """Parse a comma-separated command-line value."""
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def filter_filings(
    filings: pd.DataFrame,
    *,
    since: str,
    forms: tuple[str, ...],
    max_per_form: int | None,
) -> pd.DataFrame:
    """Filter parsed companyfacts rows to the desired annual filings."""
    if filings.empty:
        return filings
    frame = filings.copy()
    frame = frame[frame["form"].isin(forms)]
    frame = frame[frame["filing_date"] >= since]
    frame = frame.sort_values("filing_date", ascending=False, kind="stable")
    if max_per_form is not None:
        parts = [part.head(max_per_form) for _, part in frame.groupby("form", sort=False)]
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return frame.reset_index(drop=True)


def flush_raw_shard(
    shard_rows: list[dict[str, object]],
    raw_index_rows: list[dict[str, object]],
    shards_dir: Path,
    stem: str,
    shard_number: int,
) -> int:
    """Write one raw text shard and append matching raw index rows."""
    shard_name = f"{stem}_raw_filings_{shard_number:05d}.parquet"
    shard_path = shards_dir / shard_name
    shard_frame = pd.DataFrame(shard_rows)
    shard_frame = shard_frame.loc[:, list(RAW_FILING_DATASET_COLUMNS)]
    shard_frame.to_parquet(shard_path, index=False)
    relative_path = f"{shards_dir.name}/{shard_name}"
    for row_number, row in enumerate(shard_rows):
        raw_index = {column: row.get(column) for column in RAW_FILING_INDEX_COLUMNS if column in row}
        raw_index["raw_shard_path"] = relative_path
        raw_index["raw_shard_row_number"] = row_number
        raw_index_rows.append(raw_index)
    log(f"Wrote raw shard {shard_name} with {len(shard_rows)} filings.", tag="historical-10k")
    return shard_number + 1


def write_progress_files(
    dataset_dir: Path,
    stem: str,
    filing_rows: list[dict[str, object]],
    raw_index_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    ticker_index: pd.DataFrame,
) -> None:
    """Write resumable index files after each shard."""
    raw_index_path = dataset_dir / f"{stem}_raw_filing_index.jsonl"
    filings_path = dataset_dir / f"{stem}_filings.parquet"
    tickers_path = dataset_dir / f"{stem}_tickers.parquet"
    failures_path = dataset_dir / f"{stem}_failures.jsonl"

    if raw_index_rows:
        pd.DataFrame(raw_index_rows).drop_duplicates("accession_no", keep="last").to_json(
            raw_index_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
    else:
        raw_index_path.write_text("", encoding="utf-8")
    if filing_rows:
        pd.DataFrame(filing_rows).drop_duplicates("accession_no", keep="last").to_parquet(filings_path, index=False)
    ticker_index.to_parquet(tickers_path, index=False)
    if failure_rows:
        pd.DataFrame(failure_rows).to_json(failures_path, orient="records", lines=True, force_ascii=False)


def write_manifest(
    dataset_dir: Path,
    stem: str,
    group_name: str,
    ticker_index: pd.DataFrame,
    filing_rows: list[dict[str, object]],
    raw_index_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
) -> None:
    """Write the corpus manifest expected by DatabaseOperator."""
    failure_tickers = {str(row.get("ticker")) for row in failure_rows if row.get("ticker")}
    payload = {
        "group_name": group_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_format": "parquet",
        "dataset_columns": list(RAW_FILING_DATASET_COLUMNS),
        "raw_filing_index_columns": list(RAW_FILING_INDEX_COLUMNS),
        "raw_filing_storage": "parquet_shards",
        "include_supporting_data": True,
        "summary": {
            "tickers": int(ticker_index["ticker"].nunique()),
            "ticker_index_rows": int(len(ticker_index)),
            "filing_rows": int(pd.DataFrame(filing_rows)["accession_no"].nunique()) if filing_rows else 0,
            "raw_filing_rows": int(pd.DataFrame(raw_index_rows)["accession_no"].nunique()) if raw_index_rows else 0,
            "failed_tickers": int(len(failure_tickers)),
        },
    }
    (dataset_dir / f"{stem}_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def register_sqlite(
    *,
    group_name: str,
    storage_dir: Path,
    sqlite_path: Path,
    ticker_index: pd.DataFrame,
    filings: pd.DataFrame,
    raw_index: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    """Register the streamed corpus in the existing SQLite catalog."""
    operator = DatabaseOperator(storage_dir=storage_dir, sqlite_path=sqlite_path, default_format="parquet")
    failure_dict: dict[str, list[str]] = defaultdict(list)
    if not failures.empty:
        for row in failures.to_dict("records"):
            stage = str(row.get("stage", "unknown"))
            error = str(row.get("error", ""))
            failure_dict[str(row.get("ticker", ""))].append(f"{stage}: {error}")
    corpus = RawFilingCorpus(
        group_name=group_name,
        tickers=ticker_index["ticker"].astype(str).tolist(),
        ticker_index=ticker_index,
        filings=filings,
        raw_filings=raw_index,
        failures=dict(failure_dict),
    )
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        operator._ensure_sqlite_schema(connection)
        operator._delete_sqlite_group(connection, group_name)
        operator._write_sqlite_frame(connection, "tickers", ticker_index, group_name)
        operator._write_sqlite_frame(connection, "filings", filings, group_name)
        operator._write_sqlite_frame(connection, "raw_filings", raw_index, group_name)
        operator._write_sqlite_frame(connection, "failures", failures, group_name)
        operator._upsert_sqlite_corpus_metadata(connection, corpus)
        operator._create_sqlite_indexes(connection)


class ExistingState:
    """Existing streamed corpus state used for resume mode."""

    def __init__(self, filings: pd.DataFrame, raw_index: pd.DataFrame, failures: pd.DataFrame) -> None:
        self.filings = filings
        self.raw_index = raw_index
        self.failures = failures

    @classmethod
    def empty(cls) -> "ExistingState":
        return cls(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def load_existing_state(dataset_dir: Path, stem: str) -> ExistingState:
    """Load current index files for resume mode."""
    filings_path = dataset_dir / f"{stem}_filings.parquet"
    raw_index_path = dataset_dir / f"{stem}_raw_filing_index.jsonl"
    failures_path = dataset_dir / f"{stem}_failures.jsonl"
    filings = pd.read_parquet(filings_path) if filings_path.exists() else pd.DataFrame()
    raw_index = (
        pd.read_json(raw_index_path, orient="records", lines=True)
        if raw_index_path.exists()
        else pd.DataFrame()
    )
    failures = (
        pd.read_json(failures_path, orient="records", lines=True)
        if failures_path.exists()
        else pd.DataFrame()
    )
    return ExistingState(filings, raw_index, failures)


def next_available_shard_number(shards_dir: Path, stem: str) -> int:
    """Return the next raw shard number for append/resume mode."""
    numbers = []
    pattern = re_compile_shard_pattern(stem)
    for path in shards_dir.glob(f"{stem}_raw_filings_*.parquet"):
        match = pattern.search(path.name)
        if match is not None:
            numbers.append(int(match.group(1)))
    return (max(numbers) + 1) if numbers else 1


def re_compile_shard_pattern(stem: str):
    """Compile the shard-number regex lazily to keep imports small."""
    import re

    return re.compile(rf"^{re.escape(stem)}_raw_filings_(\d+)\.parquet$")


def slugify(value: str) -> str:
    """Match the storage slug used by DatabaseOperator."""
    safe = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    return safe.strip("_") or "raw_filing_corpus"


if __name__ == "__main__":
    main()
