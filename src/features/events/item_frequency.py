"""Rolling 8-K item-frequency features sampled at month-end dates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.base import FeatureProducer, FeatureSpec
from src.features.assembly import build_observation_panel


class EventItemFrequencyProducer(FeatureProducer):
    """Compute trailing counts of selected 8-K item types."""

    def __init__(self, items: list[str], windows: list[int]) -> None:
        self.items = [str(item) for item in items]
        self.windows = [int(window) for window in windows]

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="event_item_frequency",
            key_columns=["ticker", "date"],
            columns=event_count_columns(self.items, self.windows),
            source="eight_k_events",
            description="Trailing 8-K item counts over fixed calendar-day windows.",
        )

    def compute(self, db, config: dict) -> pd.DataFrame:
        events_path = Path(config["paths"]["events_path"])
        if not events_path.exists():
            raise FileNotFoundError(
                f"8-K events not found at {events_path}. Run scripts/00_fetch_8k_events.py first."
            )

        panel = build_observation_panel(db, config)
        if panel.empty:
            return pd.DataFrame(columns=self.spec.all_columns)

        events = pd.read_parquet(events_path)
        if events.empty:
            output = panel.copy()
            for column in self.spec.columns:
                output[column] = 0.0
            return output.loc[:, self.spec.all_columns]

        required = {"ticker", "filing_date", "item"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"8-K events are missing columns: {sorted(missing)}")

        events = events.loc[:, ["ticker", "filing_date", "item"]].copy()
        events["ticker"] = events["ticker"].astype(str).str.upper()
        events["date"] = pd.to_datetime(events["filing_date"], errors="coerce")
        events["item"] = events["item"].astype(str)
        events = events.dropna(subset=["date"])
        events = events[events["item"].isin(self.items)]

        parts: list[pd.DataFrame] = []
        events_by_ticker = {ticker: group.sort_values("date") for ticker, group in events.groupby("ticker")}
        for ticker, ticker_panel in panel.groupby("ticker", sort=True):
            ticker_output = ticker_panel.sort_values("date").loc[:, ["ticker", "date"]].copy()
            ticker_events = events_by_ticker.get(str(ticker).upper())
            panel_dates = pd.to_datetime(ticker_output["date"]).to_numpy(dtype="datetime64[ns]")
            for item in self.items:
                item_dates = event_dates_for_item(ticker_events, item)
                for window in self.windows:
                    column = event_count_column(item, window)
                    if len(item_dates) == 0:
                        ticker_output[column] = 0.0
                        continue
                    starts = panel_dates - np.timedelta64(window, "D")
                    right = np.searchsorted(item_dates, panel_dates, side="right")
                    left = np.searchsorted(item_dates, starts, side="right")
                    ticker_output[column] = (right - left).astype(float)
            parts.append(ticker_output)

        output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=self.spec.all_columns)
        return output.loc[:, self.spec.all_columns].reset_index(drop=True)


def event_count_columns(items: list[str], windows: list[int]) -> list[str]:
    """Return event count columns in stable item-window order."""
    return [event_count_column(item, window) for item in items for window in windows]


def event_count_column(item: str, window: int) -> str:
    """Return a safe column name for one 8-K item/window pair."""
    safe_item = str(item).replace(".", "_")
    return f"event_count_{safe_item}_{int(window)}d"


def event_dates_for_item(events: pd.DataFrame | None, item: str) -> np.ndarray:
    """Return sorted event dates for one ticker/item pair."""
    if events is None or events.empty:
        return np.array([], dtype="datetime64[ns]")
    item_events = events[events["item"] == item]
    if item_events.empty:
        return np.array([], dtype="datetime64[ns]")
    return pd.to_datetime(item_events["date"]).to_numpy(dtype="datetime64[ns]")
