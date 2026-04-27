from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import ssl
import sys
import urllib.request
from urllib.error import URLError

import pandas as pd



from src.config import load_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import log


SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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


def main(config_name: str = "baseline") -> None:
    config = load_config(config_name)
    output_path = Path(config["paths"]["metadata_path"])
    ensure_dir(output_path.parent)

    html_text, ssl_verified = download_html(SOURCE_URL)
    tables = pd.read_html(StringIO(html_text))
    if not tables:
        raise ValueError(f"No tables found at {SOURCE_URL}.")

    raw = tables[0]
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"S&P 500 metadata table is missing columns: {sorted(missing)}")

    metadata = raw.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Headquarters Location": "headquarters_location",
            "Date added": "date_added",
            "CIK": "cik_str",
            "Founded": "founded",
        }
    )
    metadata["ticker"] = metadata["ticker"].astype(str).str.replace(".", "-", regex=False)
    metadata["cik_str"] = metadata["cik_str"].astype(str).str.zfill(10)
    metadata["metadata_source_url"] = SOURCE_URL
    metadata["metadata_fetched_at"] = datetime.now(timezone.utc).isoformat()

    output_columns = [
        "ticker",
        "company_name",
        "gics_sector",
        "gics_sub_industry",
        "headquarters_location",
        "date_added",
        "cik_str",
        "founded",
        "metadata_source_url",
        "metadata_fetched_at",
    ]
    metadata = metadata.loc[:, [column for column in output_columns if column in metadata.columns]]
    metadata.to_parquet(output_path, index=False)
    write_json(
        {
            "rows": int(len(metadata)),
            "source_url": SOURCE_URL,
            "output_path": str(output_path),
            "ssl_verified": bool(ssl_verified),
        },
        output_path.with_suffix(".manifest.json"),
    )
    log(f"Wrote S&P 500 metadata with {len(metadata)} rows to {output_path}.", tag="metadata")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
