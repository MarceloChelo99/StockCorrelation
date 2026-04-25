from __future__ import annotations

import runpy
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.features.events.item_frequency import EventItemFrequencyProducer, event_count_column


class _FakeDB:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_prices(self, *, date_from=None, date_to=None) -> pd.DataFrame:
        frame = self.prices.copy()
        if date_from is not None:
            frame = frame[frame["date"] >= pd.Timestamp(date_from)]
        if date_to is not None:
            frame = frame[frame["date"] <= pd.Timestamp(date_to)]
        return frame


class EventFeatureTests(unittest.TestCase):
    def test_parse_items_splits_sec_item_string(self) -> None:
        namespace = runpy.run_path("scripts/00_fetch_8k_events.py", run_name="event_fetcher_test")
        parse_items = namespace["parse_items"]

        self.assertEqual(parse_items("5.02,9.01"), ["5.02", "9.01"])
        self.assertEqual(parse_items(""), [])

    def test_event_item_frequency_counts_trailing_windows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            events_path = Path(tmp_dir) / "eight_k_events.parquet"
            events = pd.DataFrame(
                {
                    "ticker": ["AAA", "AAA", "AAA", "BBB"],
                    "filing_date": pd.to_datetime(["2025-01-10", "2025-03-15", "2025-04-01", "2025-02-01"]),
                    "item": ["5.02", "5.02", "9.01", "5.02"],
                }
            )
            events.to_parquet(events_path, index=False)

            dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"])
            prices = pd.DataFrame(
                {
                    "ticker": ["AAA"] * 4 + ["BBB"] * 4,
                    "date": list(dates) + list(dates),
                    "adj_close": [1, 2, 3, 4, 1, 2, 3, 4],
                }
            )
            config = {
                "paths": {"events_path": str(events_path)},
                "data": {"start_date": "2025-01-01", "end_date": None},
                "features": {
                    "price": {"min_history_days": 1},
                    "events": {"items": ["5.02", "9.01"], "windows": [30, 90]},
                },
            }
            producer = EventItemFrequencyProducer(items=["5.02", "9.01"], windows=[30, 90])
            result = producer.compute(_FakeDB(prices), config)

            aaa_april = result[(result["ticker"] == "AAA") & (result["date"] == pd.Timestamp("2025-04-30"))].iloc[0]
            self.assertEqual(aaa_april[event_count_column("5.02", 30)], 0.0)
            self.assertEqual(aaa_april[event_count_column("5.02", 90)], 1.0)
            self.assertEqual(aaa_april[event_count_column("9.01", 30)], 1.0)


if __name__ == "__main__":
    unittest.main()
