from __future__ import annotations

from scripts import _bootstrap  # noqa: F401
REPO_ROOT = _bootstrap.REPO_ROOT

import sys
from pathlib import Path



from market_data_fetcher import DatabaseOperator


def main() -> None:
    operator = DatabaseOperator(
        identity="StockCorrelation/0.1 castellanosmarcelo1@gmail.com",
        storage_dir=REPO_ROOT / "data" / "raw_filing_corpora",
        default_format="jsonl",
    )

    corpus, stored = operator.populate_for_symbols(
        ["AAPL", "MSFT"],
        group_name="Tech Smoke Test",
        format="jsonl",
        filing_options={"since": "2024-01-01", "filings_per_form": 1},
        raw_filing_options={"since": "2024-01-01", "filings_per_form": 1},
    )

    print("summary", corpus.summary())
    print("stored", stored.summary())
    print("latest_structured")
    print(
        operator.latest_filings("Tech Smoke Test", limit=4)[
            ["ticker", "form", "filing_date", "accession_no"]
        ].to_string(index=False)
    )
    print("latest_raw")
    print(
        operator.latest_filings("Tech Smoke Test", raw=True, limit=4)[
            ["ticker", "form", "filing_date", "accession_no", "submission_text_length", "raw_shard_path"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
