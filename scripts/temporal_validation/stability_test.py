"""Validate whether temporal smoothing reduces movement for stable firms."""
from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from scripts.temporal_validation.common import (
    displacement,
    load_embeddings,
    standardized_embeddings,
    write_markdown,
)


STABLE_FIRMS = ["NEE", "DUK", "SO", "KO", "PG", "CL", "PEP", "KMB", "ED", "WEC"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-embeddings", required=True)
    parser.add_argument("--vanilla-embeddings", required=True)
    parser.add_argument("--output-dir", default="report/temporal_validation")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    temporal = standardized_embeddings(load_embeddings(args.temporal_embeddings))
    vanilla = standardized_embeddings(load_embeddings(args.vanilla_embeddings))
    rows = []
    for ticker in STABLE_FIRMS:
        for year in range(args.start_year, args.end_year):
            start = f"{year}-12-31"
            end = f"{year + 1}-12-31"
            rows.append(
                {
                    "ticker": ticker,
                    "start_date": start,
                    "end_date": end,
                    "temporal_displacement": displacement(temporal, ticker, start, end),
                    "vanilla_displacement": displacement(vanilla, ticker, start, end),
                }
            )
    result = pd.DataFrame(rows)
    result["difference_temporal_minus_vanilla"] = (
        result["temporal_displacement"] - result["vanilla_displacement"]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "stability_yoy_displacements.csv", index=False)

    temporal_median = result["temporal_displacement"].median(skipna=True)
    vanilla_median = result["vanilla_displacement"].median(skipna=True)
    passed = bool(pd.notna(temporal_median) and pd.notna(vanilla_median) and temporal_median < vanilla_median)
    write_markdown(
        output_dir / "stability_test.md",
        "Stability Test",
        [
            f"Temporal embeddings: `{args.temporal_embeddings}`",
            f"Vanilla embeddings: `{args.vanilla_embeddings}`",
            f"Stable-firm median temporal YoY displacement: `{temporal_median:.4f}`",
            f"Stable-firm median vanilla YoY displacement: `{vanilla_median:.4f}`",
            f"Pass condition: temporal median < vanilla median.",
            f"Result: `{'PASS' if passed else 'FAIL'}`",
            "",
            "See `stability_yoy_displacements.csv` for firm/year details.",
        ],
    )


if __name__ == "__main__":
    main()
