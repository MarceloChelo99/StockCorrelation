"""Validate whether embedding velocity spikes around known strategic events."""
from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from scripts.temporal_validation.common import (
    load_embeddings,
    monthly_velocities,
    standardized_embeddings,
    write_markdown,
)


EVENTS = [
    {"ticker": "META", "event": "Meta metaverse pivot", "event_date": "2021-10-31"},
    {"ticker": "DIS", "event": "Disney Fox acquisition close", "event_date": "2019-03-31"},
    {"ticker": "MSFT", "event": "Microsoft Activision deal close", "event_date": "2023-10-31"},
    {"ticker": "BA", "event": "Boeing 737 MAX grounding", "event_date": "2019-03-31"},
    {"ticker": "PFE", "event": "Pfizer COVID vaccine EUA", "event_date": "2020-12-31"},
    {"ticker": "NVDA", "event": "Generative AI demand inflection", "event_date": "2023-05-31"},
    {"ticker": "ADBE", "event": "Figma acquisition announcement", "event_date": "2022-09-30"},
    {"ticker": "ORCL", "event": "Cerner acquisition close", "event_date": "2022-06-30"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", default="report/temporal_validation")
    parser.add_argument("--event-window-days", type=int, default=45)
    parser.add_argument("--baseline-exclusion-days", type=int, default=180)
    args = parser.parse_args()

    embeddings = standardized_embeddings(load_embeddings(args.embeddings))
    rows = []
    for event in EVENTS:
        velocities = monthly_velocities(embeddings, event["ticker"])
        if velocities.empty:
            rows.append({**event, "event_velocity": None, "baseline_median_velocity": None, "spike_ratio": None})
            continue
        event_date = pd.Timestamp(event["event_date"])
        event_mask = (velocities["end_date"] - event_date).abs().dt.days <= int(args.event_window_days)
        baseline_mask = (velocities["end_date"] - event_date).abs().dt.days > int(args.baseline_exclusion_days)
        event_velocity = velocities.loc[event_mask, "velocity"].max()
        baseline_values = velocities.loc[baseline_mask, "velocity"]
        positive_baseline = baseline_values.loc[baseline_values > 1e-12]
        baseline_median = positive_baseline.median()
        spike_ratio = event_velocity / baseline_median if pd.notna(event_velocity) and baseline_median > 0 else None
        rows.append(
            {
                **event,
                "event_velocity": event_velocity,
                "positive_baseline_median_velocity": baseline_median,
                "n_positive_baseline_updates": int(len(positive_baseline)),
                "spike_ratio": spike_ratio,
                "passed": bool(pd.notna(spike_ratio) and spike_ratio > 1.5),
            }
        )

    result = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "event_alignment.csv", index=False)

    pass_rate = float(result["passed"].fillna(False).mean()) if not result.empty else 0.0
    passed = bool(pass_rate >= 0.60)
    write_markdown(
        output_dir / "event_alignment_test.md",
        "Event Alignment Test",
        [
            f"Embeddings: `{args.embeddings}`",
            f"Event-window days: `{args.event_window_days}`",
            f"Event pass rate: `{pass_rate:.2%}`",
            "Baseline uses the firm's non-zero filing-update velocities, because monthly as-of embeddings are flat between filings.",
            f"Pass condition: at least 60% of events have event-window velocity > 1.5x positive baseline median.",
            f"Result: `{'PASS' if passed else 'FAIL'}`",
            "",
            "See `event_alignment.csv` for event-level details.",
        ],
    )


if __name__ == "__main__":
    main()
