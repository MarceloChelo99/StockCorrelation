"""SEC filing metrics plus raw submission-text download helpers."""
from __future__ import annotations

import json
import pathlib
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .networking import RequestRateLimiter, urlopen


_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_RAW_SUBMISSION_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_folder}/{accession_folder}/{submission_file}"
)

GAAP_CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "investing_cash_flow": ["NetCashProvidedByUsedInInvestingActivities"],
    "financing_cash_flow": ["NetCashProvidedByUsedInFinancingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireOtherProductiveAssets",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireOilAndGasPropertyAndEquipment",
        "PaymentsToAcquireOilAndGasProperty",
        "PaymentsToAcquireProjects",
        "PaymentsToAcquireFurnitureAndFixtures",
        "PaymentsToAcquireTimberlands",
        "PropertyPlantAndEquipmentAdditions",
    ],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "accounts_receivable": ["AccountsReceivableNetCurrent"],
    "inventory": ["InventoryNet"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "shareholders_equity": ["StockholdersEquity"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}

DEI_CONCEPT_MAP: dict[str, list[str]] = {
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
}

_FILING_FORMS = {"10-K", "10-Q"}
_METADATA_COLUMNS = {"ticker", "form", "filing_date", "accession_no", "period_start", "period_end"}
RAW_FILING_DATASET_COLUMNS = (
    "ticker",
    "cik",
    "accession_no",
    "form",
    "filing_date",
    "period_end",
    "submission_text",
    "submission_text_length",
    "source_url",
    "downloaded_at",
)


class EdgarCompanyFactsClient:
    """Fetch all XBRL facts for a company from the SEC bulk endpoint."""

    def __init__(
        self,
        identity: str,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        self.identity = identity
        self.rate_limiter = rate_limiter or RequestRateLimiter(max_requests=10, period_seconds=1.0)

    def fetch(self, cik: str) -> dict:
        padded_cik = str(cik).zfill(10)
        url = _COMPANYFACTS_URL.format(cik=padded_cik)
        request = urllib.request.Request(url, headers={"User-Agent": self.identity})
        self.rate_limiter.wait()
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))


class RawFilingSubmissionClient:
    """Download the complete SEC submission text for a filing accession number."""

    def __init__(
        self,
        identity: str,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        self.identity = identity
        self.rate_limiter = rate_limiter or RequestRateLimiter(max_requests=10, period_seconds=1.0)

    def build_url(self, cik: str, accession_no: str) -> str:
        cik_folder = str(int(str(cik).strip()))
        accession_folder = _normalize_accession_no(accession_no)
        submission_file = f"{str(accession_no).strip()}.txt"
        return _RAW_SUBMISSION_URL.format(
            cik_folder=cik_folder,
            accession_folder=accession_folder,
            submission_file=submission_file,
        )

    def fetch(self, cik: str, accession_no: str) -> tuple[str, str]:
        url = self.build_url(cik, accession_no)
        request = urllib.request.Request(url, headers={"User-Agent": self.identity})
        self.rate_limiter.wait()
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("latin-1", errors="replace")
        return text, url


@dataclass(frozen=True)
class _FactEntry:
    metric: str
    concept_priority: int
    val: float
    end: str
    start: str
    accn: str
    form: str
    filed: str


class CompanyFactsParser:
    """Parse EDGAR companyfacts JSON into filing-level rows matching the DB schema."""

    def __init__(
        self,
        gaap_concepts: dict[str, list[str]] | None = None,
        dei_concepts: dict[str, list[str]] | None = None,
    ) -> None:
        self.gaap_concepts = gaap_concepts or GAAP_CONCEPT_MAP
        self.dei_concepts = dei_concepts or DEI_CONCEPT_MAP

    def parse(self, ticker: str, facts_json: dict) -> pd.DataFrame:
        facts = facts_json.get("facts") or {}
        entries = self._collect_entries(facts)
        if not entries:
            return pd.DataFrame()

        filings = self._group_into_filings(ticker, entries)
        if not filings:
            return pd.DataFrame()

        frame = pd.DataFrame(filings)
        numeric_columns = [column for column in frame.columns if column not in _METADATA_COLUMNS]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        frame = self._compute_derived_metrics(frame)
        return frame.sort_values("filing_date", ascending=False, kind="stable").reset_index(drop=True)

    def _collect_entries(self, facts: dict) -> list[_FactEntry]:
        entries: list[_FactEntry] = []

        gaap_facts = facts.get("us-gaap") or {}
        for metric, concept_names in self.gaap_concepts.items():
            for priority, concept_name in enumerate(concept_names):
                concept_data = gaap_facts.get(concept_name)
                if concept_data is None:
                    continue
                entries.extend(self._extract_from_concept(metric, priority, concept_data, unit_filter="USD"))

        dei_facts = facts.get("dei") or {}
        for metric, concept_names in self.dei_concepts.items():
            for priority, concept_name in enumerate(concept_names):
                concept_data = dei_facts.get(concept_name)
                if concept_data is None:
                    continue
                entries.extend(self._extract_from_concept(metric, priority, concept_data, unit_filter="shares"))

        return entries

    @staticmethod
    def _extract_from_concept(
        metric: str,
        priority: int,
        concept_data: dict,
        *,
        unit_filter: str,
    ) -> list[_FactEntry]:
        entries: list[_FactEntry] = []
        units = concept_data.get("units") or {}

        unit_entries = units.get(unit_filter) or []
        if not unit_entries and unit_filter == "USD":
            unit_entries = units.get("USD/shares") or []

        for entry in unit_entries:
            form = entry.get("form", "")
            if form not in _FILING_FORMS:
                continue
            entries.append(
                _FactEntry(
                    metric=metric,
                    concept_priority=priority,
                    val=entry.get("val"),
                    end=entry.get("end", ""),
                    start=entry.get("start", ""),
                    accn=entry.get("accn", ""),
                    form=form,
                    filed=entry.get("filed", ""),
                )
            )
        return entries

    @staticmethod
    def _group_into_filings(ticker: str, entries: list[_FactEntry]) -> list[dict[str, object]]:
        groups: dict[tuple[str, str, str], list[_FactEntry]] = defaultdict(list)
        for entry in entries:
            key = (entry.accn, entry.form, entry.filed)
            groups[key].append(entry)

        filings: list[dict[str, object]] = []
        for (accn, form, filed), group_entries in groups.items():
            row: dict[str, object] = {
                "ticker": ticker.upper(),
                "form": form,
                "filing_date": filed,
                "accession_no": accn,
            }

            best_by_metric: dict[str, _FactEntry] = {}
            for entry in group_entries:
                existing = best_by_metric.get(entry.metric)
                if existing is None:
                    best_by_metric[entry.metric] = entry
                elif entry.concept_priority < existing.concept_priority:
                    best_by_metric[entry.metric] = entry
                elif entry.concept_priority == existing.concept_priority and entry.end > existing.end:
                    best_by_metric[entry.metric] = entry

            period_end = ""
            period_start = ""
            for entry in best_by_metric.values():
                if entry.end and entry.end > period_end:
                    period_end = entry.end
                if entry.start and (not period_start or entry.start < period_start):
                    period_start = entry.start

            row["period_end"] = period_end
            row["period_start"] = period_start

            for metric, entry in best_by_metric.items():
                row[metric] = entry.val

            filings.append(row)

        return filings

    @staticmethod
    def _compute_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        ocf = frame.get("operating_cash_flow")
        capex = frame.get("capex")
        if ocf is not None and capex is not None:
            frame["free_cash_flow"] = ocf - capex

        revenue = frame.get("revenue")
        if revenue is not None:
            safe_revenue = revenue.replace(0, float("nan"))
            gp = frame.get("gross_profit")
            if gp is not None:
                frame["gross_margin"] = gp / safe_revenue
            oi = frame.get("operating_income")
            if oi is not None:
                frame["operating_margin"] = oi / safe_revenue
            ni = frame.get("net_income")
            if ni is not None:
                frame["net_margin"] = ni / safe_revenue

        ltd = frame.get("long_term_debt")
        seq = frame.get("shareholders_equity")
        if ltd is not None and seq is not None:
            safe_equity = seq.replace(0, float("nan"))
            frame["debt_to_equity"] = ltd / safe_equity

        return frame


class FilingMetricsDownloader:
    """Download SEC filing metrics for a ticker using the bulk companyfacts endpoint."""

    def __init__(
        self,
        identity: str | None = None,
        *,
        rate_limiter: RequestRateLimiter | None = None,
        cik_resolver: Callable[[str], str | None] | None = None,
        client: EdgarCompanyFactsClient | None = None,
        parser: CompanyFactsParser | None = None,
    ) -> None:
        if client is None:
            if identity is None:
                raise ValueError("identity is required when client is not provided.")
            client = EdgarCompanyFactsClient(identity=identity, rate_limiter=rate_limiter)

        self.client = client
        self.parser = parser or CompanyFactsParser()
        self.cik_resolver = cik_resolver

    def download(self, ticker: str) -> pd.DataFrame:
        return self.download_for_ticker(ticker)

    def download_for_ticker(
        self,
        ticker: str,
        *,
        since: str | None = None,
        forms: tuple[str, ...] | None = None,
        filings_per_form: int | None = None,
        exclude_accession_numbers: Collection[str] | None = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        cik = self._resolve_cik(ticker)
        if cik is None:
            return pd.DataFrame()

        try:
            facts_json = self.client.fetch(cik)
        except Exception:
            return pd.DataFrame()

        frame = self.parser.parse(ticker, facts_json)
        if frame.empty:
            return frame

        if forms:
            frame = frame[frame["form"].isin(forms)]
        if since:
            frame = frame[frame["filing_date"] >= since]
        if exclude_accession_numbers:
            excluded = {str(accession).strip() for accession in exclude_accession_numbers if accession}
            frame = frame[~frame["accession_no"].isin(excluded)]
        if filings_per_form is not None:
            parts = []
            for form_type in frame["form"].unique():
                subset = frame[frame["form"] == form_type].head(filings_per_form)
                parts.append(subset)
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        return frame.reset_index(drop=True)

    def fetch(self, ticker: str) -> pd.DataFrame:
        return self.download(ticker)

    def _resolve_cik(self, ticker: str) -> str | None:
        if self.cik_resolver is not None:
            return self.cik_resolver(ticker)
        return _resolve_cik_from_cache(ticker)


class RawFilingDownloader:
    """Download full SEC submission text for filings selected by accession number."""

    def __init__(
        self,
        identity: str | None = None,
        *,
        rate_limiter: RequestRateLimiter | None = None,
        cik_resolver: Callable[[str], str | None] | None = None,
        client: RawFilingSubmissionClient | None = None,
        filings_downloader: FilingMetricsDownloader | None = None,
    ) -> None:
        if client is None:
            if identity is None:
                raise ValueError("identity is required when client is not provided.")
            client = RawFilingSubmissionClient(identity=identity, rate_limiter=rate_limiter)
        self.client = client
        self.cik_resolver = cik_resolver
        self.filings_downloader = filings_downloader

    def download_accession(
        self,
        ticker: str,
        accession_no: str,
        *,
        cik: str | None = None,
        form: str | None = None,
        filing_date: str | None = None,
        period_end: str | None = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        resolved_cik = cik or self._resolve_cik(ticker)
        if resolved_cik is None:
            return normalize_raw_filing_dataset(pd.DataFrame())
        try:
            submission_text, source_url = self.client.fetch(resolved_cik, accession_no)
        except Exception:
            return normalize_raw_filing_dataset(pd.DataFrame())

        downloaded_at = datetime.now(timezone.utc).isoformat()
        row = {
            "ticker": ticker,
            "cik": str(resolved_cik).zfill(10),
            "accession_no": accession_no,
            "form": form,
            "filing_date": filing_date,
            "period_end": period_end,
            "submission_text": submission_text,
            "submission_text_length": len(submission_text),
            "source_url": source_url,
            "downloaded_at": downloaded_at,
        }
        return normalize_raw_filing_dataset(pd.DataFrame([row]))

    def download_for_ticker(
        self,
        ticker: str,
        *,
        since: str | None = None,
        forms: tuple[str, ...] | None = None,
        filings_per_form: int | None = None,
        exclude_accession_numbers: Collection[str] | None = None,
        filings_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        source_frame = filings_frame
        if source_frame is None:
            source_frame = self._download_filing_index(
                ticker,
                since=since,
                forms=forms,
                filings_per_form=filings_per_form,
                exclude_accession_numbers=exclude_accession_numbers,
            )
        if source_frame is None or source_frame.empty:
            return normalize_raw_filing_dataset(pd.DataFrame())

        cik = self._resolve_cik(ticker)
        if cik is None:
            return normalize_raw_filing_dataset(pd.DataFrame())

        parts: list[pd.DataFrame] = []
        for _, row in source_frame.iterrows():
            accession_no = row.get("accession_no")
            if not accession_no:
                continue
            raw = self.download_accession(
                ticker,
                str(accession_no),
                cik=cik,
                form=_optional_text(row.get("form")),
                filing_date=_optional_text(row.get("filing_date")),
                period_end=_optional_text(row.get("period_end")),
            )
            if not raw.empty:
                parts.append(raw)
        if not parts:
            return normalize_raw_filing_dataset(pd.DataFrame())
        return normalize_raw_filing_dataset(pd.concat(parts, ignore_index=True))

    def fetch(self, ticker: str, **kwargs) -> pd.DataFrame:
        return self.download_for_ticker(ticker, **kwargs)

    def _download_filing_index(
        self,
        ticker: str,
        *,
        since: str | None,
        forms: tuple[str, ...] | None,
        filings_per_form: int | None,
        exclude_accession_numbers: Collection[str] | None,
    ) -> pd.DataFrame:
        if self.filings_downloader is None:
            raise ValueError(
                "RawFilingDownloader needs either filings_frame input or a filings_downloader "
                "to discover accession numbers."
            )
        return self.filings_downloader.download_for_ticker(
            ticker,
            since=since,
            forms=forms,
            filings_per_form=filings_per_form,
            exclude_accession_numbers=exclude_accession_numbers,
        )

    def _resolve_cik(self, ticker: str) -> str | None:
        if self.cik_resolver is not None:
            return self.cik_resolver(ticker)
        return _resolve_cik_from_cache(ticker)


TickerFinancialFetcher = FilingMetricsDownloader


def normalize_raw_filing_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return raw filing rows with a stable column order and compatible dtypes."""
    if frame.empty and not list(frame.columns):
        normalized = pd.DataFrame(columns=RAW_FILING_DATASET_COLUMNS)
    else:
        normalized = frame.copy()
        for column in RAW_FILING_DATASET_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = pd.NA

    ordered_columns = list(RAW_FILING_DATASET_COLUMNS) + [
        column for column in normalized.columns if column not in RAW_FILING_DATASET_COLUMNS
    ]
    normalized = normalized.loc[:, ordered_columns]

    for column in RAW_FILING_DATASET_COLUMNS:
        if column == "submission_text_length":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
        else:
            normalized[column] = normalized[column].astype("string")

    return normalized.reset_index(drop=True)


def _resolve_cik_from_cache(ticker: str) -> str | None:
    cache_path = pathlib.Path("company_tickers_cache.json")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"CIK cache not found at {cache_path.resolve()}. "
            "Download the SEC ticker directory first, "
            "or provide a cik_resolver when constructing the downloader."
        )
    payload = json.loads(cache_path.read_text())
    data = payload.get("data", payload)
    for entry in data.values():
        if str(entry.get("ticker", "")).upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def _normalize_accession_no(accession_no: str) -> str:
    return str(accession_no).strip().replace("-", "")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
