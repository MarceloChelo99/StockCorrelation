"""Build compact filing-section snippets from an already-downloaded raw corpus.

This is the local, no-network companion to ``stream_features.py``. It reads raw
filing parquet shards produced by ``fetch_historical_10k_filings.py``, parses the
selected sections, extracts short topic-centered snippets, and writes
``historical_section_snippets.parquet`` for dashboard theme-label evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401

import argparse

import pandas as pd

from scripts.historical_text.stream_features import SECTION_LABELS, parse_csv_arg
from src.filings.sections import parse_filing_sections
from src.filings.topics import topic_evidence_snippets
from src.utils.io import ensure_dir
from src.utils.logging import log


DEFAULT_RAW_CORPUS_DIR = (
    "data/raw_filing_corpora/s_p_500_historical_10_k_10_q_filings_since_2010"
)
DEFAULT_OUTPUT_DIR = "data/processed/historical_text_10k_10q"
DEFAULT_LABEL_SECTIONS = (
    "business,risk_factors,mda,item_1c_cybersecurity,q_mda,q_risk_factors"
)


def main() -> None:
    """Backfill snippet rows from local raw filing shards."""
    args = parse_args()
    raw_corpus_dir = Path(args.raw_corpus_dir)
    output_dir = ensure_dir(args.output_dir)
    raw_index_path = find_raw_index(raw_corpus_dir, args.raw_index_path)
    raw_index = pd.read_json(raw_index_path, lines=True)

    sections_to_keep = set(parse_csv_arg(args.sections))
    forms_to_keep = {form.upper() for form in parse_csv_arg(args.forms)}
    raw_index["form"] = raw_index["form"].astype(str).str.upper()
    raw_index = raw_index[raw_index["form"].isin(forms_to_keep)].copy()
    raw_index = raw_index.sort_values(["ticker", "filing_date", "accession_no"]).reset_index(drop=True)
    if args.limit_filings is not None:
        raw_index = raw_index.head(int(args.limit_filings)).copy()

    snippets_path = output_dir / "historical_section_snippets.parquet"
    existing = load_existing_snippets(snippets_path, args.resume)
    existing_keys = existing_snippet_keys(existing)

    rows = existing.to_dict("records") if not existing.empty else []
    failures: list[dict[str, object]] = []
    current_shard_path: Path | None = None
    current_shard = pd.DataFrame()

    log(
        f"Backfilling snippets for {len(raw_index):,} raw filings from {raw_corpus_dir}.",
        tag="snippet-backfill",
    )

    for position, filing in enumerate(raw_index.itertuples(index=False), start=1):
        if position == 1 or position % int(args.log_every) == 0 or position == len(raw_index):
            log(f"Filing {position:,}/{len(raw_index):,}: {filing.ticker}", tag="snippet-backfill")

        try:
            shard_path = raw_corpus_dir / str(filing.raw_shard_path)
            if current_shard_path != shard_path:
                current_shard = pd.read_parquet(shard_path)
                current_shard_path = shard_path
            shard_row_number = int(filing.raw_shard_row_number)
            submission_text = str(current_shard.iloc[shard_row_number]["submission_text"])
            parsed = parse_filing_sections(submission_text, str(filing.form))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "ticker": getattr(filing, "ticker", ""),
                    "accession_no": getattr(filing, "accession_no", ""),
                    "stage": "parse_raw_shard",
                    "error": str(exc),
                }
            )
            continue

        for section_key, span in parsed.sections.items():
            if section_key not in sections_to_keep:
                continue
            if (str(filing.accession_no), section_key) in existing_keys:
                continue

            text = span.text or ""
            metadata = {
                "ticker": str(filing.ticker).upper(),
                "cik": str(filing.cik).zfill(10),
                "accession_no": str(filing.accession_no),
                "form": str(filing.form),
                "filing_date": str(filing.filing_date),
                "period_end": str(filing.period_end),
                "section": section_key,
                "section_label": SECTION_LABELS.get(section_key, section_key),
                "section_chars": len(text),
            }
            snippets = topic_evidence_snippets(
                text,
                max_snippets=args.snippets_per_section,
                window_chars=args.snippet_window_chars,
            )
            for snippet in snippets:
                row = dict(metadata)
                row.update(snippet)
                rows.append(row)

        if position % int(args.checkpoint_every_filings) == 0:
            write_snippets(snippets_path, rows)

    write_snippets(snippets_path, rows)
    if failures:
        pd.DataFrame(failures).to_json(
            output_dir / "historical_section_snippet_failures.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )

    log(
        f"Wrote {len(rows):,} snippet rows to {snippets_path}. Failures: {len(failures):,}.",
        tag="snippet-backfill",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-corpus-dir", default=DEFAULT_RAW_CORPUS_DIR)
    parser.add_argument("--raw-index-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forms", default="10-K,10-Q")
    parser.add_argument("--sections", default=DEFAULT_LABEL_SECTIONS)
    parser.add_argument("--snippets-per-section", type=int, default=3)
    parser.add_argument("--snippet-window-chars", type=int, default=700)
    parser.add_argument("--checkpoint-every-filings", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--limit-filings", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def find_raw_index(raw_corpus_dir: Path, explicit_path: str | None) -> Path:
    """Return the raw filing index for a downloaded corpus."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw index not found: {path}")
        return path

    candidates = sorted(raw_corpus_dir.glob("*_raw_filing_index.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No *_raw_filing_index.jsonl found in {raw_corpus_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple raw index files found in {raw_corpus_dir}: {candidates}")
    return candidates[0]


def load_existing_snippets(path: Path, should_resume: bool) -> pd.DataFrame:
    """Load existing snippets when resuming a backfill."""
    if should_resume and path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def existing_snippet_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Return accession/section keys already represented in the snippet file."""
    if frame.empty or "accession_no" not in frame.columns or "section" not in frame.columns:
        return set()
    values = frame.loc[:, ["accession_no", "section"]].dropna().drop_duplicates()
    return {(str(row.accession_no), str(row.section)) for row in values.itertuples(index=False)}


def write_snippets(path: Path, rows: list[dict[str, object]]) -> None:
    """Write deduplicated snippet rows."""
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(["accession_no", "section", "snippet_rank"], keep="last")
    frame = frame.sort_values(["ticker", "filing_date", "section", "snippet_rank"])
    frame.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
