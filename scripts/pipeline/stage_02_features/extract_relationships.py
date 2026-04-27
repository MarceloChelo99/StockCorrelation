from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import pandas as pd



from src.relationships.extract import extract_relationships
from src.utils.io import ensure_dir
from src.utils.logging import log


def main(
    sections_path: str = "data/processed/sections/ten_k_sections.parquet",
    metadata_path: str = "data/processed/metadata/sp500_gics.parquet",
    output_path: str = "data/processed/relationships/relationships.parquet",
) -> None:
    """Extract a sparse company relationship graph from parsed 10-K sections."""
    sections = pd.read_parquet(sections_path)
    metadata = pd.read_parquet(metadata_path)
    relationships = extract_relationships(sections, metadata)
    output = Path(output_path)
    ensure_dir(output.parent)
    relationships.to_parquet(output, index=False)
    relationships.to_csv(output.with_suffix(".csv"), index=False)

    log(f"Wrote {len(relationships):,} relationship rows to {output}.", tag="relationships")
    if not relationships.empty:
        type_counts = relationships["relationship_type"].value_counts().to_dict()
        log(f"Relationship type counts: {type_counts}", tag="relationships")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "data/processed/sections/ten_k_sections.parquet",
        sys.argv[2] if len(sys.argv) > 2 else "data/processed/metadata/sp500_gics.parquet",
        sys.argv[3] if len(sys.argv) > 3 else "data/processed/relationships/relationships.parquet",
    )
