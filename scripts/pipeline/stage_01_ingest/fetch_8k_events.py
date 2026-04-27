from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import json
from pathlib import Path
import sys
import urllib.request

import pandas as pd



from market_data_fetcher.networking import RequestRateLimiter, urlopen

from src.config import load_config
from src.db import FilingsDB
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log


SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
USER_AGENT = "StockCorrelation/0.1 castellanosmarcelo1@gmail.com"


def main(config_name: str = "baseline") -> None:
    config = load_config(config_name)
    db = FilingsDB.from_config(config)
    tickers = db.load_tickers()
    events_path = Path(config["paths"]["events_path"])
    ensure_dir(events_path.parent)

    settings = config["features"]["events"]
    forms = {form.upper() for form in settings["forms"]}
    since = str(settings["since"])
    limiter = RequestRateLimiter(max_requests=8, period_seconds=1.0)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total = len(tickers)
    for index, record in tickers.iterrows():
        ticker = str(record["ticker"]).upper()
        cik = str(record["cik_str"]).zfill(10)
        if index == 0 or index + 1 == total or (index + 1) % 50 == 0:
            log(f"Fetching 8-K event metadata for {ticker} ({index + 1}/{total}).", tag="events")
        try:
            rows.extend(fetch_ticker_events(ticker, cik, forms=forms, since=since, limiter=limiter))
        except Exception as exc:
            failures.append({"ticker": ticker, "cik": cik, "error": str(exc)})

    events = pd.DataFrame(rows)
    columns = ["ticker", "cik_str", "accession_no", "form", "filing_date", "report_date", "item"]
    if events.empty:
        events = pd.DataFrame(columns=columns)
    else:
        events = events.loc[:, columns].sort_values(["ticker", "filing_date", "accession_no", "item"])
    events.to_parquet(events_path, index=False)
    write_json(
        {
            "rows": int(len(events)),
            "tickers": int(events["ticker"].nunique()) if not events.empty else 0,
            "failures": int(len(failures)),
            "source": "SEC submissions recent filings endpoint",
            "since": since,
            "forms": sorted(forms),
            "output_path": str(events_path),
        },
        events_path.with_suffix(".manifest.json"),
    )
    if failures:
        pd.DataFrame(failures).to_json(events_path.with_suffix(".failures.jsonl"), orient="records", lines=True)
    log(f"Wrote {len(events)} 8-K event item rows to {events_path}.", tag="events")


def fetch_ticker_events(
    ticker: str,
    cik: str,
    *,
    forms: set[str],
    since: str,
    limiter: RequestRateLimiter,
) -> list[dict[str, object]]:
    """Fetch recent 8-K item metadata for one ticker from SEC submissions JSON."""
    url = SUBMISSIONS_URL.format(cik=cik)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    limiter.wait()
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    recent = payload.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    items_list = recent.get("items", [])

    rows: list[dict[str, object]] = []
    for position, form in enumerate(forms_list):
        form_text = str(form).upper()
        filing_date = value_at(filing_dates, position)
        if form_text not in forms or not filing_date or filing_date < since:
            continue

        accession_no = value_at(accessions, position)
        report_date = value_at(report_dates, position)
        items = parse_items(value_at(items_list, position))
        for item in items:
            rows.append(
                {
                    "ticker": ticker,
                    "cik_str": cik,
                    "accession_no": accession_no,
                    "form": form_text,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "item": item,
                }
            )
    return rows


def parse_items(value: object) -> list[str]:
    """Parse SEC 8-K item strings like '5.02,9.01' into normalized item codes."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    items = []
    for part in text.replace(";", ",").split(","):
        item = part.strip()
        if item:
            items.append(item)
    return items


def value_at(values: list, position: int) -> object:
    """Return a list value if present."""
    if position >= len(values):
        return ""
    return values[position]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
