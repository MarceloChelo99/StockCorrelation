"""Helpers for loading ticker metadata with external classification fill-ins."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ticker_metadata(db, config: dict) -> pd.DataFrame:
    """Load DB ticker metadata and fill missing fields from the configured metadata parquet.

    The raw filing DB can already contain columns such as ``gics_sector`` or
    ``company_name``, but those values may be null. External metadata should
    fill those gaps rather than being dropped as duplicate columns.
    """
    metadata = db.load_tickers().copy()
    if "ticker" not in metadata.columns:
        raise ValueError("DB ticker metadata must contain a ticker column.")
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()

    metadata_path = Path(config.get("paths", {}).get("metadata_path", ""))
    if not metadata_path.exists():
        return metadata

    external = pd.read_parquet(metadata_path)
    if "ticker" not in external.columns:
        raise ValueError(f"Metadata file {metadata_path} must contain a ticker column.")
    external = external.copy()
    external["ticker"] = external["ticker"].astype(str).str.upper()

    merged = metadata.merge(external, on="ticker", how="left", suffixes=("", "_external"))
    for column in external.columns:
        if column == "ticker":
            continue
        external_column = f"{column}_external"
        if external_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].combine_first(merged[external_column])
            merged = merged.drop(columns=[external_column])
        else:
            merged = merged.rename(columns={external_column: column})
    return merged
