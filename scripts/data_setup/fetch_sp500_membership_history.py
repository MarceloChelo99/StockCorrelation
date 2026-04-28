"""Build a point-in-time-ish S&P 500 membership interval file from Wikipedia.

The current metadata table only tells us who is in the index today. This script
also parses the public "Selected changes" table so backtests can exclude firms
after deletion and include deleted firms when their historical prices are
available locally.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import ssl
import sys
import urllib.request
from urllib.error import URLError

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log


SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_OUTPUT = Path("data/processed/metadata/sp500_membership_history.parquet")
DEFAULT_HISTORY_START = "1957-03-04"
SEC_TICKER_CACHE_PATH = Path("company_tickers_cache.json")
LEGAL_SUFFIX_TOKENS = {
    "co",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "plc",
    "company",
    "the",
    "new",
    "holdings",
    "holding",
    "group",
}


def download_html(url: str) -> tuple[str, bool]:
    """Download HTML, falling back for local certificate-store issues."""
    request = urllib.request.Request(url, headers={"User-Agent": "stock-embeddings-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8"), True
    except URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            return response.read().decode("utf-8"), False


def normalize_ticker(value: object) -> str | None:
    """Normalize Wikipedia tickers to the project ticker convention."""
    if pd.isna(value):
        return None
    ticker = str(value).strip().upper().replace(".", "-")
    if not ticker or ticker.lower() == "nan":
        return None
    return ticker


def normalized_name_tokens(value: object) -> set[str]:
    """Return conservative company-name tokens for ticker/CIK matching."""
    if pd.isna(value):
        return set()
    text = "".join(character.lower() if character.isalnum() else " " for character in str(value))
    return {token for token in text.split() if len(token) > 1 and token not in LEGAL_SUFFIX_TOKENS}


def token_overlap(left: object, right: object) -> float:
    """Return a simple token-overlap score between two company names."""
    left_tokens = normalized_name_tokens(left)
    right_tokens = normalized_name_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_sec_ticker_cache(path: Path = SEC_TICKER_CACHE_PATH) -> dict[str, dict[str, str]]:
    """Load SEC ticker-to-CIK cache if available."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    raw = payload.get("data", payload) if isinstance(payload, dict) else {}
    lookup: dict[str, dict[str, str]] = {}
    for entry in raw.values():
        ticker = normalize_ticker(entry.get("ticker")) if isinstance(entry, dict) else None
        if not ticker:
            continue
        lookup[ticker] = {
            "cik_str": str(entry.get("cik_str")).zfill(10),
            "sec_title": str(entry.get("title", "")),
        }
    return lookup


def resolve_cik_from_sec_cache(ticker: str, company_name: object, lookup: dict[str, dict[str, str]]) -> str | None:
    """Resolve CIK only when ticker and company-name evidence agree.

    Deleted tickers can be reused, so ticker-only matching is too dangerous for
    historical constituents. A conservative token check avoids silently mapping
    an old S&P company to a different modern issuer with the same ticker.
    """
    entry = lookup.get(ticker)
    if not entry:
        return None
    if token_overlap(company_name, entry.get("sec_title")) < 0.35:
        return None
    return entry["cik_str"]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten possible MultiIndex table columns from pandas.read_html."""
    output = frame.copy()
    if not isinstance(output.columns, pd.MultiIndex):
        return output
    columns = []
    for column in output.columns:
        parts = [
            str(part).strip()
            for part in column
            if str(part).strip() and not str(part).startswith("Unnamed")
        ]
        if len(parts) >= 2 and parts[0] == parts[-1]:
            columns.append(parts[0])
        else:
            columns.append("_".join(parts))
    output.columns = columns
    return output


def current_constituents(table: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    """Return normalized current-constituent metadata from the Wikipedia table."""
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Current S&P 500 table is missing columns: {sorted(missing)}")
    frame = table.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Date added": "date_added",
            "CIK": "cik_str",
        }
    ).copy()
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["date_added"] = pd.to_datetime(frame.get("date_added"), errors="coerce")
    frame["source_url"] = SOURCE_URL
    frame["source_fetched_at"] = fetched_at
    if "cik_str" in frame.columns:
        frame["cik_str"] = frame["cik_str"].astype(str).str.zfill(10)
    keep = [
        "ticker",
        "company_name",
        "gics_sector",
        "gics_sub_industry",
        "date_added",
        "cik_str",
        "source_url",
        "source_fetched_at",
    ]
    return frame.loc[:, [column for column in keep if column in frame.columns]].dropna(subset=["ticker"])


def selected_changes(table: pd.DataFrame) -> pd.DataFrame:
    """Return normalized added/removed ticker events from the changes table."""
    frame = flatten_columns(table)
    rename = {
        "Effective Date": "effective_date",
        "Effective Date_Effective Date": "effective_date",
        "Added_Ticker": "added_ticker",
        "Added_Security": "added_security",
        "Removed_Ticker": "removed_ticker",
        "Removed_Security": "removed_security",
        "Reason": "reason",
        "Reason_Reason": "reason",
    }
    frame = frame.rename(columns={column: rename.get(column, column) for column in frame.columns})
    required = {"effective_date", "added_ticker", "removed_ticker"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"S&P 500 changes table is missing columns: {sorted(missing)}")
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce")
    frame["added_ticker"] = frame["added_ticker"].map(normalize_ticker)
    frame["removed_ticker"] = frame["removed_ticker"].map(normalize_ticker)
    return frame.dropna(subset=["effective_date"]).reset_index(drop=True)


def build_events(current: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Build one add/remove event table from current metadata and changes."""
    rows = []
    for _, row in current.iterrows():
        if pd.notna(row.get("date_added")):
            rows.append(
                {
                    "ticker": row["ticker"],
                    "date": pd.Timestamp(row["date_added"]),
                    "action": "add",
                    "company_name": row.get("company_name"),
                    "reason": "Current constituent date_added",
                }
            )
    for _, row in changes.iterrows():
        if row.get("added_ticker"):
            rows.append(
                {
                    "ticker": row["added_ticker"],
                    "date": pd.Timestamp(row["effective_date"]),
                    "action": "add",
                    "company_name": row.get("added_security"),
                    "reason": row.get("reason"),
                }
            )
        if row.get("removed_ticker"):
            rows.append(
                {
                    "ticker": row["removed_ticker"],
                    "date": pd.Timestamp(row["effective_date"]),
                    "action": "remove",
                    "company_name": row.get("removed_security"),
                    "reason": row.get("reason"),
                }
            )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.dropna(subset=["ticker", "date", "action"])
    events = events.sort_values(["ticker", "date", "action"])
    return events.drop_duplicates(["ticker", "date", "action"], keep="first").reset_index(drop=True)


def build_membership_intervals(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    history_start: str = DEFAULT_HISTORY_START,
    sec_cik_lookup: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Convert add/remove events into ticker membership intervals."""
    events = build_events(current, changes)
    sec_cik_lookup = sec_cik_lookup or {}
    current_tickers = set(current["ticker"].dropna())
    current_meta = current.drop_duplicates("ticker", keep="last").set_index("ticker")
    company_names = {}
    for frame, ticker_column, name_column in [
        (current, "ticker", "company_name"),
        (changes, "added_ticker", "added_security"),
        (changes, "removed_ticker", "removed_security"),
    ]:
        if ticker_column not in frame.columns or name_column not in frame.columns:
            continue
        for _, row in frame.dropna(subset=[ticker_column]).iterrows():
            ticker = row[ticker_column]
            if pd.notna(row.get(name_column)):
                company_names.setdefault(ticker, row[name_column])

    start_floor = pd.Timestamp(history_start)
    intervals = []
    all_tickers = sorted(set(events["ticker"].unique()).union(current_tickers))
    for ticker in all_tickers:
        ticker_events = events[events["ticker"].eq(ticker)].sort_values("date")
        is_active = False
        start_date = pd.NaT
        start_source = ""
        last_reason = ""
        for _, event in ticker_events.iterrows():
            action = event["action"]
            event_date = pd.Timestamp(event["date"])
            if action == "add":
                if not is_active:
                    is_active = True
                    start_date = event_date
                    start_source = str(event.get("reason") or "add_event")
                last_reason = str(event.get("reason") or "")
            elif action == "remove":
                if not is_active:
                    is_active = True
                    start_date = start_floor
                    start_source = "unknown_prior_to_first_observed_removal"
                intervals.append(
                    {
                        "ticker": ticker,
                        "company_name": company_names.get(ticker),
                        "start_date": start_date,
                        "end_date": event_date,
                        "is_current": False,
                        "start_source": start_source,
                        "end_source": str(event.get("reason") or "remove_event"),
                    }
                )
                is_active = False
                start_date = pd.NaT
                start_source = ""
                last_reason = str(event.get("reason") or "")
        if is_active or ticker in current_tickers:
            if pd.isna(start_date):
                start_date = current_meta.loc[ticker, "date_added"] if ticker in current_meta.index else start_floor
                if pd.isna(start_date):
                    start_date = start_floor
                start_source = "current_table"
            intervals.append(
                {
                    "ticker": ticker,
                    "company_name": company_names.get(ticker),
                    "start_date": pd.Timestamp(start_date),
                    "end_date": pd.NaT,
                    "is_current": ticker in current_tickers,
                    "start_source": start_source or last_reason or "add_event",
                    "end_source": "",
                }
            )

    output = pd.DataFrame(intervals)
    if output.empty:
        raise ValueError("No S&P 500 membership intervals were built.")
    output = output.merge(
        current.loc[:, [column for column in ["ticker", "gics_sector", "gics_sub_industry", "cik_str"] if column in current.columns]],
        on="ticker",
        how="left",
    )
    missing_cik = output["cik_str"].isna() if "cik_str" in output.columns else pd.Series(True, index=output.index)
    resolved = []
    for _, row in output.loc[missing_cik].iterrows():
        resolved.append(resolve_cik_from_sec_cache(str(row["ticker"]), row.get("company_name"), sec_cik_lookup))
    if "cik_str" not in output.columns:
        output["cik_str"] = pd.NA
    output.loc[missing_cik, "cik_str"] = resolved
    output["cik_resolution_source"] = np_where_notna(output["cik_str"], "wikipedia_or_sec_cache", "")
    output["source_url"] = SOURCE_URL
    output["source_fetched_at"] = current["source_fetched_at"].iloc[0]
    output = output.sort_values(["ticker", "start_date", "end_date"]).reset_index(drop=True)
    return output


def np_where_notna(series: pd.Series, true_value: str, false_value: str) -> list[str]:
    """Small helper to avoid pulling NumPy into this script."""
    return [true_value if pd.notna(value) and str(value) != "" else false_value for value in series]


def main(output_path: str = str(DEFAULT_OUTPUT), history_start: str = DEFAULT_HISTORY_START) -> None:
    """Fetch Wikipedia tables and write historical constituent intervals."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    html_text, ssl_verified = download_html(SOURCE_URL)
    tables = pd.read_html(StringIO(html_text))
    if len(tables) < 2:
        raise ValueError(f"Expected current constituents and changes tables at {SOURCE_URL}.")

    current = current_constituents(tables[0], fetched_at)
    changes = selected_changes(tables[1])
    sec_cik_lookup = load_sec_ticker_cache()
    membership = build_membership_intervals(
        current,
        changes,
        history_start=history_start,
        sec_cik_lookup=sec_cik_lookup,
    )

    output = Path(output_path)
    ensure_dir(output.parent)
    membership.to_parquet(output, index=False)
    ticker_summary = membership.groupby("ticker", as_index=False).agg(
        is_current=("is_current", "any"),
        has_cik=("cik_str", lambda values: values.notna().any()),
    )
    manifest = {
        "rows": int(len(membership)),
        "unique_tickers": int(membership["ticker"].nunique()),
        "current_tickers": int(membership["is_current"].sum()),
        "deleted_or_historical_tickers": int((~membership["is_current"]).sum()),
        "unique_current_tickers": int(ticker_summary["is_current"].sum()),
        "unique_historical_only_tickers": int((~ticker_summary["is_current"]).sum()),
        "tickers_with_cik": int(ticker_summary["has_cik"].sum()),
        "historical_only_tickers_with_cik": int(
            ticker_summary.loc[~ticker_summary["is_current"], "has_cik"].sum()
        ),
        "source_url": SOURCE_URL,
        "output_path": str(output),
        "history_start": str(history_start),
        "ssl_verified": bool(ssl_verified),
        "source_fetched_at": fetched_at,
        "note": (
            "Removed constituents before the first observed add event use history_start as their interval start. "
            "Deleted firms still require local price data and sector labels before they affect sector-level backtests."
        ),
    }
    write_json(manifest, output.with_suffix(".manifest.json"))
    log(
        f"Wrote {len(membership):,} S&P membership intervals for "
        f"{membership['ticker'].nunique():,} tickers to {output}.",
        tag="membership",
    )


if __name__ == "__main__":
    output_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_OUTPUT)
    history_start_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_HISTORY_START
    main(output_arg, history_start_arg)
