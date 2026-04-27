"""Validate whether known AI-language movers travel farther in embedding space."""
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
    write_svg_histogram,
)


AI_MOVERS = ["META", "NVDA", "MSFT", "ADBE", "ORCL", "NOW", "CRM", "GOOGL", "AMZN", "AVGO"]
CONTROL_FIRMS = ["NEE", "DUK", "SO", "KO", "PG", "CL", "PEP", "KMB", "ED", "WEC"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, help="Temporal embeddings parquet.")
    parser.add_argument("--output-dir", default="report/temporal_validation")
    parser.add_argument("--start-date", default="2022-12-31")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()

    embeddings = standardized_embeddings(load_embeddings(args.embeddings))
    rows = []
    for group, tickers in [("ai_mover", AI_MOVERS), ("control", CONTROL_FIRMS)]:
        for ticker in tickers:
            value = displacement(embeddings, ticker, args.start_date, args.end_date)
            rows.append({"ticker": ticker, "group": group, "displacement": value})
    result = pd.DataFrame(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "ai_drift_displacements.csv", index=False)
    write_svg_histogram(result, output_dir / "ai_drift_histogram.svg", "displacement", "group")

    ai_median = result.loc[result["group"] == "ai_mover", "displacement"].median(skipna=True)
    control_median = result.loc[result["group"] == "control", "displacement"].median(skipna=True)
    passed = bool(pd.notna(ai_median) and pd.notna(control_median) and ai_median >= 2.0 * control_median)
    write_markdown(
        output_dir / "ai_drift_test.md",
        "AI Drift Test",
        [
            f"Embeddings: `{args.embeddings}`",
            f"Window: `{args.start_date}` to `{args.end_date}`",
            f"AI-mover median displacement: `{ai_median:.4f}`",
            f"Control median displacement: `{control_median:.4f}`",
            f"Pass condition: AI median >= 2x control median.",
            f"Result: `{'PASS' if passed else 'FAIL'}`",
            "",
            "See `ai_drift_displacements.csv` and `ai_drift_histogram.svg` for details.",
        ],
    )


if __name__ == "__main__":
    main()
