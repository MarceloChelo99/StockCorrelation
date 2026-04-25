"""Stream historical filing text into topic counts and embeddings without raw storage.

The pipeline order is:

download one raw filing -> parse sections -> compute keyword counts -> embed text -> discard raw text

This is the storage-efficient path for studying business pivots through time.
The output is long-format by section, so adding more parsed sections later only
adds rows; it does not require redesigning the schema.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "libraries" / "market_data_fetcher" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from market_data_fetcher import (  # noqa: E402
    CompanyFactsParser,
    EdgarCompanyFactsClient,
    RequestRateLimiter,
    RawFilingSubmissionClient,
)
from market_data_fetcher.networking import urlopen  # noqa: E402
from src.filings.sections import parse_filing_sections  # noqa: E402
from src.filings.topics import topic_counts_for_text  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import log  # noqa: E402


SECTION_LABELS = {
    "business": "Item 1 Business",
    "risk_factors": "Item 1A Risk Factors",
    "mda": "Item 7 MD&A",
    "item_1c_cybersecurity": "Item 1C Cybersecurity",
    "item_2_properties": "Item 2 Properties",
    "item_3_legal_proceedings": "Item 3 Legal Proceedings",
    "item_7a_market_risk": "Item 7A Market Risk",
    "q_mda": "10-Q Item 2 MD&A",
    "q_market_risk": "10-Q Item 3 Market Risk",
    "q_legal_proceedings": "10-Q Legal Proceedings",
    "q_risk_factors": "10-Q Risk Factors",
    "eightk_item_1_01_material_agreement": "8-K Item 1.01 Material Agreement",
    "eightk_item_2_02_results": "8-K Item 2.02 Results",
    "proxy_governance": "Proxy Corporate Governance",
    "proxy_directors": "Proxy Directors",
    "proxy_compensation": "Proxy Compensation",
    "proxy_pay_vs_performance": "Proxy Pay vs Performance",
    "s1_summary": "S-1 Prospectus Summary",
    "s1_risk_factors": "S-1 Risk Factors",
    "s1_business": "S-1 Business",
    "s1_mda": "S-1 MD&A",
}
COMPANYFACTS_FORMS = {"10-K", "10-Q"}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"


def main() -> None:
    """Stream SEC filings into compact historical section features."""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    state = load_existing_state(output_dir) if args.resume else ExistingState.empty()
    metadata = load_metadata(args.metadata_path, args.limit_tickers)
    existing_accessions = set(state.filing_index["accession_no"].astype(str)) if not state.filing_index.empty else set()

    limiter = RequestRateLimiter(max_requests=args.max_sec_requests_per_second, period_seconds=1.0)
    facts_client = EdgarCompanyFactsClient(args.identity, rate_limiter=limiter)
    raw_client = RawFilingSubmissionClient(args.identity, rate_limiter=limiter)
    facts_parser = CompanyFactsParser()

    section_counts = state.section_counts.to_dict("records")
    embeddings = state.section_embeddings.to_dict("records")
    filing_index = state.filing_index.to_dict("records")
    failures = state.failures.to_dict("records")

    embedder = None
    embed_buffer: list[EmbeddingJob] = []
    forms = tuple(part.upper() for part in parse_csv_arg(args.forms))
    sections_to_keep = set(parse_csv_arg(args.sections))
    use_submissions = args.discovery_source == "submissions" or (
        args.discovery_source == "auto" and any(form not in COMPANYFACTS_FORMS for form in forms)
    )
    if args.embed and not sections_to_keep:
        raise ValueError("--sections must name at least one section when --embed is enabled.")

    log(
        f"Streaming {forms} since {args.since} for {len(metadata)} tickers into {output_dir}.",
        tag="text-stream",
    )

    for ticker_number, record in enumerate(metadata.itertuples(index=False), start=1):
        ticker = str(record.ticker).upper()
        cik = str(record.cik_str).zfill(10)
        if ticker_number == 1 or ticker_number % 10 == 0 or ticker_number == len(metadata):
            log(f"Ticker {ticker_number}/{len(metadata)}: {ticker}", tag="text-stream")

        try:
            if use_submissions:
                filings = fetch_submission_filing_index(
                    ticker,
                    cik,
                    forms=forms,
                    since=args.since,
                    max_per_form=args.max_filings_per_form,
                    user_agent=args.identity,
                    limiter=limiter,
                )
            else:
                facts_payload = facts_client.fetch(cik)
                filings = facts_parser.parse(ticker, facts_payload)
                filings = filter_filings(filings, since=args.since, forms=forms, max_per_form=args.max_filings_per_form)
        except Exception as exc:  # noqa: BLE001
            failures.append({"ticker": ticker, "stage": "filing_index", "error": str(exc)})
            continue

        if filings.empty:
            failures.append({"ticker": ticker, "stage": "filing_index", "error": "No matching filings found."})
            continue

        for filing in filings.itertuples(index=False):
            accession_no = str(filing.accession_no)
            if accession_no in existing_accessions:
                continue
            try:
                submission_text, source_url = raw_client.fetch(cik, accession_no)
                parsed = parse_filing_sections(submission_text, str(filing.form))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "ticker": ticker,
                        "stage": "parse_or_download",
                        "accession_no": accession_no,
                        "error": str(exc),
                    }
                )
                continue

            filing_index.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "accession_no": accession_no,
                    "form": str(filing.form),
                    "filing_date": str(filing.filing_date),
                    "period_end": str(filing.period_end),
                    "source_url": source_url,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            for section_key, span in parsed.sections.items():
                if section_key not in sections_to_keep:
                    continue
                text = span.text or ""
                count_row = {
                    "ticker": ticker,
                    "cik": cik,
                    "accession_no": accession_no,
                    "form": str(filing.form),
                    "filing_date": str(filing.filing_date),
                    "period_end": str(filing.period_end),
                    "section": section_key,
                    "section_label": SECTION_LABELS.get(section_key, section_key),
                    "section_chars": len(text),
                }
                count_row.update(topic_counts_for_text(text))
                section_counts.append(count_row)
                if args.embed and len(text) >= args.min_embed_chars:
                    embed_buffer.append(
                        EmbeddingJob(
                            metadata={
                                key: count_row[key]
                                for key in (
                                    "ticker",
                                    "cik",
                                    "accession_no",
                                    "form",
                                    "filing_date",
                                    "period_end",
                                    "section",
                                    "section_label",
                                    "section_chars",
                                )
                            },
                            text=text[: args.max_chars],
                        )
                    )

            submission_text = ""
            if args.embed and len(embed_buffer) >= args.embedding_batch_size:
                embedder = embedder or load_embedder(args.model_name)
                flush_embeddings(embed_buffer, embeddings, embedder, args.embedding_batch_size)
                embed_buffer = []
                write_outputs(output_dir, filing_index, section_counts, embeddings, failures, args)

            if len(filing_index) % args.checkpoint_every_filings == 0:
                write_outputs(output_dir, filing_index, section_counts, embeddings, failures, args)

    if args.embed and embed_buffer:
        embedder = embedder or load_embedder(args.model_name)
        flush_embeddings(embed_buffer, embeddings, embedder, args.embedding_batch_size)

    write_outputs(output_dir, filing_index, section_counts, embeddings, failures, args)
    log(
        f"Historical text features ready: {len(filing_index)} filings, "
        f"{len(section_counts)} section-count rows, {len(embeddings)} embedding rows.",
        tag="text-stream",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2010-01-01")
    parser.add_argument("--forms", default="10-K")
    parser.add_argument(
        "--sections",
        default=(
            "business,risk_factors,mda,item_1c_cybersecurity,"
            "item_2_properties,item_3_legal_proceedings,item_7a_market_risk,"
            "q_mda,q_market_risk,q_legal_proceedings,q_risk_factors,"
            "eightk_item_1_01_material_agreement,eightk_item_2_02_results,"
            "proxy_governance,proxy_directors,proxy_compensation,proxy_pay_vs_performance,"
            "s1_summary,s1_risk_factors,s1_business,s1_mda"
        ),
        help="Comma-separated parsed sections to keep.",
    )
    parser.add_argument(
        "--discovery-source",
        choices=["auto", "companyfacts", "submissions"],
        default="auto",
        help="Use companyfacts for 10-K/10-Q only, or SEC submissions for forms like 8-K, DEF 14A, and S-1.",
    )
    parser.add_argument("--metadata-path", default="data/processed/metadata/sp500_gics.parquet")
    parser.add_argument("--output-dir", default="data/processed/historical_text")
    parser.add_argument("--identity", default="StockCorrelation research castellanosmarcelo1@gmail.com")
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--max-filings-per-form", type=int, default=None)
    parser.add_argument("--max-sec-requests-per-second", type=int, default=6)
    parser.add_argument("--checkpoint-every-filings", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-embed", dest="embed", action="store_false")
    parser.set_defaults(embed=True)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--min-embed-chars", type=int, default=200)
    return parser.parse_args()


def load_metadata(path: str, limit: int | None) -> pd.DataFrame:
    """Load ticker metadata with CIKs."""
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
    """Parse a comma-separated argument."""
    return [part.strip() for part in value.split(",") if part.strip()]


def filter_filings(
    filings: pd.DataFrame,
    *,
    since: str,
    forms: tuple[str, ...],
    max_per_form: int | None,
) -> pd.DataFrame:
    """Filter filing index rows for the historical text stream."""
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


def fetch_submission_filing_index(
    ticker: str,
    cik: str,
    *,
    forms: tuple[str, ...],
    since: str,
    max_per_form: int | None,
    user_agent: str,
    limiter: RequestRateLimiter,
) -> pd.DataFrame:
    """Fetch filing metadata from SEC submissions JSON, including non-XBRL forms."""
    rows = []
    payload = fetch_submissions_payload(SUBMISSIONS_URL.format(cik=str(cik).zfill(10)), user_agent, limiter)
    rows.extend(submission_rows_from_payload(ticker, cik, payload))

    for file_info in payload.get("filings", {}).get("files", []):
        filing_to = str(file_info.get("filingTo", ""))
        if filing_to and filing_to < since:
            continue
        name = str(file_info.get("name", "")).strip()
        if not name:
            continue
        file_payload = fetch_submissions_payload(SUBMISSIONS_FILE_URL.format(name=name), user_agent, limiter)
        rows.extend(submission_rows_from_payload(ticker, cik, file_payload))

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame[frame["form"].isin(forms)]
    frame = frame[frame["filing_date"] >= since]
    frame = frame.drop_duplicates("accession_no", keep="last")
    frame = frame.sort_values("filing_date", ascending=False, kind="stable")
    if max_per_form is not None:
        parts = [part.head(max_per_form) for _, part in frame.groupby("form", sort=False)]
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return frame.reset_index(drop=True)


def fetch_submissions_payload(url: str, user_agent: str, limiter: RequestRateLimiter) -> dict:
    """Fetch one SEC submissions JSON payload."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    limiter.wait()
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def submission_rows_from_payload(ticker: str, cik: str, payload: dict) -> list[dict[str, object]]:
    """Normalize SEC submissions arrays into filing-index rows."""
    recent = payload.get("filings", {}).get("recent") if "filings" in payload else payload
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])
    items = recent.get("items", [])

    rows = []
    for index, form in enumerate(forms):
        accession_no = value_at(accessions, index)
        filing_date = value_at(filing_dates, index)
        if not accession_no or not filing_date:
            continue
        rows.append(
            {
                "ticker": str(ticker).upper(),
                "cik": str(cik).zfill(10),
                "accession_no": str(accession_no),
                "form": str(form).upper(),
                "filing_date": str(filing_date),
                "period_end": str(value_at(report_dates, index) or ""),
                "primary_document": str(value_at(primary_documents, index) or ""),
                "items": str(value_at(items, index) or ""),
            }
        )
    return rows


def value_at(values: list, index: int) -> object:
    """Return one list value if present."""
    if not isinstance(values, list) or index >= len(values):
        return ""
    return values[index]


class EmbeddingJob:
    """Small holder for one section awaiting embedding."""

    def __init__(self, metadata: dict[str, object], text: str) -> None:
        self.metadata = metadata
        self.text = text


def load_embedder(model_name: str):
    """Load the sentence-transformer model lazily."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Historical text embeddings require sentence-transformers. "
            "Run with --no-embed to collect keyword counts only."
        ) from exc
    log(f"Loading text embedder {model_name}.", tag="text-stream")
    return SentenceTransformer(model_name)


def flush_embeddings(
    jobs: list[EmbeddingJob],
    embeddings: list[dict[str, object]],
    embedder,
    batch_size: int,
) -> None:
    """Embed a buffered batch and append compact embedding rows."""
    texts = [job.text for job in jobs]
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    matrix = np.asarray(vectors, dtype=float)
    for job, vector in zip(jobs, matrix, strict=True):
        row = dict(job.metadata)
        for index, value in enumerate(vector):
            row[f"embedding_{index}"] = float(value)
        embeddings.append(row)
    log(f"Embedded {len(jobs)} section rows.", tag="text-stream")


def write_outputs(
    output_dir: Path,
    filing_index: list[dict[str, object]],
    section_counts: list[dict[str, object]],
    embeddings: list[dict[str, object]],
    failures: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    """Persist compact historical text artifacts."""
    if filing_index:
        pd.DataFrame(filing_index).drop_duplicates("accession_no", keep="last").to_parquet(
            output_dir / "historical_filing_index.parquet",
            index=False,
        )
    if section_counts:
        pd.DataFrame(section_counts).drop_duplicates(["accession_no", "section"], keep="last").to_parquet(
            output_dir / "historical_section_topic_counts.parquet",
            index=False,
        )
    if embeddings:
        pd.DataFrame(embeddings).drop_duplicates(["accession_no", "section"], keep="last").to_parquet(
            output_dir / "historical_section_embeddings.parquet",
            index=False,
        )
    if failures:
        pd.DataFrame(failures).to_json(
            output_dir / "historical_text_failures.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "since": args.since,
        "forms": [part.upper() for part in parse_csv_arg(args.forms)],
        "sections": parse_csv_arg(args.sections),
        "embed": bool(args.embed),
        "model_name": args.model_name if args.embed else None,
        "filings": len({str(row["accession_no"]) for row in filing_index}),
        "section_count_rows": len({(str(row["accession_no"]), str(row["section"])) for row in section_counts}),
        "embedding_rows": len({(str(row["accession_no"]), str(row["section"])) for row in embeddings}),
        "failures": len(failures),
    }
    (output_dir / "historical_text_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class ExistingState:
    """Previously written compact text features for resume mode."""

    def __init__(
        self,
        filing_index: pd.DataFrame,
        section_counts: pd.DataFrame,
        section_embeddings: pd.DataFrame,
        failures: pd.DataFrame,
    ) -> None:
        self.filing_index = filing_index
        self.section_counts = section_counts
        self.section_embeddings = section_embeddings
        self.failures = failures

    @classmethod
    def empty(cls) -> "ExistingState":
        return cls(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def load_existing_state(output_dir: Path) -> ExistingState:
    """Load compact text artifacts for resume mode."""
    filing_path = output_dir / "historical_filing_index.parquet"
    counts_path = output_dir / "historical_section_topic_counts.parquet"
    embeddings_path = output_dir / "historical_section_embeddings.parquet"
    failures_path = output_dir / "historical_text_failures.jsonl"
    filing_index = pd.read_parquet(filing_path) if filing_path.exists() else pd.DataFrame()
    counts = pd.read_parquet(counts_path) if counts_path.exists() else pd.DataFrame()
    embeddings = pd.read_parquet(embeddings_path) if embeddings_path.exists() else pd.DataFrame()
    failures = (
        pd.read_json(failures_path, orient="records", lines=True)
        if failures_path.exists()
        else pd.DataFrame()
    )
    return ExistingState(filing_index, counts, embeddings, failures)


if __name__ == "__main__":
    main()
