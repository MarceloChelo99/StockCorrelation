"""Date utilities for point-in-time merges and month-end sampling."""
from __future__ import annotations

import pandas as pd


def month_end_dates(frame: pd.DataFrame, *, date_column: str = "date") -> pd.Series:
    """Return the observed month-end trading date for each row's month."""
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    return dates.groupby(dates.dt.to_period("M")).transform("max")


def select_month_end_rows(frame: pd.DataFrame, *, date_column: str = "date") -> pd.DataFrame:
    """Keep the last observed row in each ticker-month."""
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    month_key = working[date_column].dt.to_period("M")
    is_month_end = working[date_column].eq(
        working.groupby(["ticker", month_key])[date_column].transform("max")
    )
    return working.loc[is_month_end].reset_index(drop=True)


def as_of_merge(left: pd.DataFrame, right: pd.DataFrame, *, by: list[str] | None = None) -> pd.DataFrame:
    """Left-join each row to the most recent earlier row by key and date."""
    if by is None:
        by = ["ticker"]
    if left.empty:
        return left.copy()
    if right.empty:
        return left.copy()

    left_sorted = left.sort_values(by + ["date"]).reset_index(drop=True)
    right_sorted = right.sort_values(by + ["date"]).reset_index(drop=True)
    left_sorted["_left_order"] = range(len(left_sorted))

    merged_parts: list[pd.DataFrame] = []
    right_groups = {
        group_key: group.copy()
        for group_key, group in right_sorted.groupby(by, sort=False, dropna=False)
    }

    for group_key, left_group in left_sorted.groupby(by, sort=False, dropna=False):
        right_group = right_groups.get(group_key)
        if right_group is None or right_group.empty:
            merged_parts.append(left_group.copy())
            continue

        right_payload = right_group.drop(columns=by, errors="ignore")
        merged = pd.merge_asof(
            left_group.sort_values("date"),
            right_payload.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_parts.append(merged)

    merged_frame = pd.concat(merged_parts, ignore_index=True)
    merged_frame = merged_frame.sort_values("_left_order").drop(columns=["_left_order"]).reset_index(drop=True)
    return merged_frame
