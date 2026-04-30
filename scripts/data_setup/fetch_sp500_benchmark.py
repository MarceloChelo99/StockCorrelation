"""Download a regular S&P 500 benchmark proxy for dashboard simulations.

The sector/theme rotation tab already compares the strategy against the
equal-weight universe used by the project. This script adds a separate
cap-weight S&P proxy using SPY so the dashboard can answer the simpler
question most viewers ask first: did the rotation beat the regular S&P 500?
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401

from market_data_fetcher import PriceHistoryDownloader
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log


REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "benchmarks" / "spy_benchmark.parquet"


def parse_args() -> argparse.Namespace:
    """Parse benchmark download arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="Benchmark ETF/proxy ticker. Default: SPY.")
    parser.add_argument("--start", default="2010-01-01", help="Earliest benchmark date.")
    parser.add_argument("--end", default=date.today().isoformat(), help="Latest benchmark date.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output parquet path.")
    return parser.parse_args()


def resolved_path(path: str | Path) -> Path:
    """Resolve repo-relative paths without requiring callers to type the repo root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def main() -> None:
    """Download the benchmark series and write parquet plus a small manifest."""
    args = parse_args()
    output_path = resolved_path(args.output)
    ticker = str(args.ticker).strip().upper()
    if not ticker:
        raise ValueError("ticker must be non-empty.")

    log(
        f"Downloading {ticker} benchmark prices from {args.start} to {args.end}.",
        tag="sp500-benchmark",
    )
    downloader = PriceHistoryDownloader()
    frame = downloader.download(
        ticker,
        start=args.start,
        end=args.end,
        lookback_days=None,
    )
    if frame.empty:
        raise RuntimeError(f"No benchmark prices returned for {ticker}.")

    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    ensure_dir(output_path.parent)
    frame.to_parquet(output_path, index=False)
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(
        {
            "ticker": ticker,
            "start": args.start,
            "end": args.end,
            "output": str(output_path),
            "n_rows": int(len(frame)),
            "first_date": frame["date"].min().strftime("%Y-%m-%d"),
            "latest_date": frame["date"].max().strftime("%Y-%m-%d"),
            "columns": list(frame.columns),
            "note": "SPY is used as a practical cap-weight S&P 500 total-return proxy.",
        },
        manifest_path,
    )
    log(f"Wrote {len(frame):,} rows to {output_path}.", tag="sp500-benchmark")
    log(f"Wrote manifest to {manifest_path}.", tag="sp500-benchmark")


if __name__ == "__main__":
    main()
