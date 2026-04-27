from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.config import load_config
from src.db import FilingsDB
from src.filings.sections import parse_10k_sections
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log
from src.utils.seed import set_seed


def main(config_name: str = "baseline", limit: int | None = None) -> None:
    config = load_config(config_name)
    set_seed(int(config["random_seed"]))

    db = FilingsDB.from_config(config)
    output_dir = ensure_dir(config["paths"]["sections_dir"])
    output_path = Path(output_dir) / "ten_k_sections.parquet"
    manifest_path = Path(output_dir) / "ten_k_sections_manifest.json"

    raw_filings = db.load_raw_filings(form=config["data"]["annual_form"])
    if limit is not None:
        raw_filings = raw_filings.head(limit).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total = len(raw_filings)
    for index, record in raw_filings.iterrows():
        accession_no = str(record["accession_no"])
        ticker = str(record["ticker"])
        if index == 0 or index + 1 == total or (index + 1) % 25 == 0:
            log(f"Parsing sections for {ticker} ({index + 1}/{total}).", tag="sections")

        try:
            submission_text = db.load_raw_submission_text(accession_no)
            if not submission_text:
                raise ValueError("Missing submission text in raw filing shards.")
            parsed = parse_10k_sections(submission_text)
            rows.append(
                {
                    "ticker": ticker,
                    "accession_no": accession_no,
                    "filing_date": record["filing_date"],
                    "business_text": parsed.get("business"),
                    "risk_factors_text": parsed.get("risk_factors"),
                    "mda_text": parsed.get("mda"),
                    "business_chars": len(parsed.get("business") or ""),
                    "risk_factors_chars": len(parsed.get("risk_factors") or ""),
                    "mda_chars": len(parsed.get("mda") or ""),
                }
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "accession_no": accession_no, "error": str(exc)})

    sections = pd.DataFrame(rows)
    sections.to_parquet(output_path, index=False)
    write_json(
        {
            "rows": int(len(sections)),
            "failures": int(len(failures)),
            "output_path": str(output_path),
            "source_group_name": config["data"]["filings_group_name"],
        },
        manifest_path,
    )
    if failures:
        pd.DataFrame(failures).to_json(
            Path(output_dir) / "ten_k_sections_failures.jsonl",
            orient="records",
            lines=True,
        )
    log(f"Wrote parsed sections to {output_path}.", tag="sections")


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    limit_value = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(config_name, limit_value)
