"""SEC XBRL companyfacts ingestion for point-in-time fundamentals."""
from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

CONCEPT_GROUPS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "stockholders_equity": ["StockholdersEquity"],
    "long_term_debt": ["LongTermDebt"],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends": ["PaymentsOfDividends"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
}


@dataclass(frozen=True)
class FundamentalsIngestResult:
    """Paths and coverage summary from a fundamentals ingestion run."""

    output_path: Path
    failures_path: Path
    rows: int
    revenue_coverage: float


def ingest_fundamentals(
    tickers: pd.DataFrame,
    *,
    output_path: str | Path,
    failures_path: str | Path,
    user_agent: str,
    request_delay_seconds: float = 0.12,
    max_retries: int = 4,
    smoke_limit: int | None = None,
    checkpoint_every: int = 25,
) -> FundamentalsIngestResult:
    """Fetch SEC companyfacts for a ticker universe and write long-format fundamentals."""
    required = {"ticker", "cik_str"}
    missing = required - set(tickers.columns)
    if missing:
        raise ValueError(f"Tickers are missing columns: {sorted(missing)}")
    if not user_agent or "@" not in user_agent:
        raise ValueError("SEC requests require a user agent containing contact information.")

    universe = tickers.loc[:, ["ticker", "cik_str"]].dropna().copy()
    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    universe["cik_str"] = universe["cik_str"].astype(str).str.zfill(10)
    universe = universe.drop_duplicates(subset=["cik_str"]).sort_values("ticker")
    if smoke_limit is not None:
        universe = universe.head(int(smoke_limit))

    output = Path(output_path)
    failure_output = Path(failures_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    failure_output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = existing_rows(output)
    failures: list[dict[str, object]] = existing_failures(failure_output)
    completed_ciks = {str(row["cik"]).zfill(10) for row in rows}
    completed_ciks.update(str(row["cik"]).zfill(10) for row in failures)

    pending = universe[~universe["cik_str"].isin(completed_ciks)].reset_index(drop=True)
    for index, company in pending.iterrows():
        ticker = str(company["ticker"])
        cik = str(company["cik_str"])
        try:
            payload = fetch_companyfacts(cik, user_agent=user_agent, max_retries=max_retries)
            rows.extend(extract_companyfacts_rows(payload, ticker=ticker, cik=cik))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            failures.append({"ticker": ticker, "cik": cik, "error": repr(error)})
        time.sleep(request_delay_seconds)
        if (index + 1) % checkpoint_every == 0:
            write_checkpoint(rows, failures, output, failure_output)
            print(f"[fundamentals] processed {index + 1}/{len(pending)} pending companies", flush=True)

    write_checkpoint(rows, failures, output, failure_output)
    frame = pd.DataFrame(rows, columns=fundamentals_columns())
    revenue_ciks = set(frame.loc[frame["concept"].isin(CONCEPT_GROUPS["revenue"]), "cik"])
    revenue_coverage = len(revenue_ciks) / max(1, len(universe))
    return FundamentalsIngestResult(
        output_path=output,
        failures_path=failure_output,
        rows=int(len(frame)),
        revenue_coverage=float(revenue_coverage),
    )


def fetch_companyfacts(cik: str, *, user_agent: str, max_retries: int) -> dict:
    """Fetch one companyfacts JSON payload with exponential backoff."""
    url = COMPANYFACTS_URL.format(cik=str(cik).zfill(10))
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
        "Host": "data.sec.gov",
    }
    last_error: Exception | None = None
    ssl_context = ssl._create_unverified_context()
    for attempt in range(max_retries + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30, context=ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504}:
                raise
        except URLError as error:
            last_error = error
        time.sleep(min(8.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def extract_companyfacts_rows(payload: dict, *, ticker: str, cik: str) -> list[dict[str, object]]:
    """Extract configured companyfacts concepts across SEC taxonomies."""
    rows: list[dict[str, object]] = []
    wanted_concepts = {concept for concepts in CONCEPT_GROUPS.values() for concept in concepts}
    facts_by_taxonomy = payload.get("facts", {})
    for facts in facts_by_taxonomy.values():
        if not isinstance(facts, dict):
            continue
        for concept, concept_payload in facts.items():
            if concept not in wanted_concepts:
                continue
            units = concept_payload.get("units", {})
            for unit, observations in units.items():
                for observation in observations:
                    value = observation.get("val")
                    filing_date = observation.get("filed")
                    end_date = observation.get("end")
                    if value is None or filing_date is None or end_date is None:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "concept": concept,
                            "unit": unit,
                            "value": value,
                            "start_date": observation.get("start"),
                            "end_date": end_date,
                            "filing_date": filing_date,
                            "form": observation.get("form"),
                            "fiscal_period": observation.get("fp"),
                            "fiscal_year": observation.get("fy"),
                        }
                    )
    return rows


def fundamentals_columns() -> list[str]:
    """Return stable long-format fundamentals columns."""
    return [
        "ticker",
        "cik",
        "concept",
        "unit",
        "value",
        "start_date",
        "end_date",
        "filing_date",
        "form",
        "fiscal_period",
        "fiscal_year",
    ]


def existing_rows(output_path: Path) -> list[dict[str, object]]:
    """Load checkpointed rows if the output parquet already exists."""
    if not output_path.exists():
        return []
    frame = pd.read_parquet(output_path)
    return frame.to_dict("records")


def existing_failures(failures_path: Path) -> list[dict[str, object]]:
    """Load checkpointed failures if the failure CSV already exists."""
    if not failures_path.exists():
        return []
    frame = pd.read_csv(failures_path)
    return frame.to_dict("records")


def write_checkpoint(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    output_path: Path,
    failures_path: Path,
) -> None:
    """Persist partial fundamentals and failures so long SEC pulls are resumable."""
    pd.DataFrame(rows, columns=fundamentals_columns()).to_parquet(output_path, index=False)
    pd.DataFrame(failures, columns=["ticker", "cik", "error"]).to_csv(failures_path, index=False)
