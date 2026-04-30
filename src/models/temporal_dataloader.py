"""Utilities for building consecutive firm-month training pairs.

Temporal autoencoder training needs aligned observations for the same ticker at
adjacent dates. This module keeps that pairing logic separate from the model so
the point-in-time assumptions are easy to inspect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def build_consecutive_pairs(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    max_gap_days: int = 45,
) -> TensorDataset:
    """Return paired tensors ``(features_t, features_t_minus_1)``.

    Rows are paired only within ticker and only when adjacent available dates are
    no more than ``max_gap_days`` apart. Gaps are skipped rather than connecting
    distant observations, which prevents quarterly/annual holes from pretending
    to be smooth monthly transitions.
    """
    required = {"ticker", "date", *feature_cols}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    frame = dataset.loc[:, ["ticker", "date", *feature_cols]].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    current_rows: list[np.ndarray] = []
    previous_rows: list[np.ndarray] = []
    dropped_gap = 0
    dropped_non_finite = 0

    for _, ticker_frame in frame.groupby("ticker", sort=False):
        values = ticker_frame.loc[:, feature_cols].astype(float).to_numpy(dtype=np.float32)
        dates = ticker_frame["date"].to_numpy(dtype="datetime64[ns]")
        for index in range(1, len(ticker_frame)):
            gap_days = int((dates[index] - dates[index - 1]) / np.timedelta64(1, "D"))
            if gap_days > int(max_gap_days):
                dropped_gap += 1
                continue
            current = values[index]
            previous = values[index - 1]
            if not np.isfinite(current).all() or not np.isfinite(previous).all():
                dropped_non_finite += 1
                continue
            current_rows.append(current)
            previous_rows.append(previous)

    if current_rows:
        current_tensor = torch.as_tensor(np.vstack(current_rows), dtype=torch.float32)
        previous_tensor = torch.as_tensor(np.vstack(previous_rows), dtype=torch.float32)
    else:
        width = len(feature_cols)
        current_tensor = torch.empty((0, width), dtype=torch.float32)
        previous_tensor = torch.empty((0, width), dtype=torch.float32)

    pairs = TensorDataset(current_tensor, previous_tensor)
    pairs.n_pairs = int(len(current_tensor))  # type: ignore[attr-defined]
    pairs.n_source_observations = int(len(frame))  # type: ignore[attr-defined]
    pairs.n_observations_dropped_gap = int(dropped_gap)  # type: ignore[attr-defined]
    pairs.n_pairs_dropped_non_finite = int(dropped_non_finite)  # type: ignore[attr-defined]
    return pairs


def split_pair_dataset(
    pairs: TensorDataset,
    train_val_split: float,
    random_seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Split a pair dataset deterministically into train and validation pairs."""
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("Cannot train temporal autoencoder with zero consecutive pairs.")

    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(n_pairs)
    if n_pairs < 3:
        train_index = order
        val_index = order
    else:
        split = int(round(n_pairs * float(train_val_split)))
        split = min(max(split, 2), n_pairs - 1)
        train_index = order[:split]
        val_index = order[split:]

    current, previous = pairs.tensors
    train_pairs = TensorDataset(current[train_index], previous[train_index])
    val_pairs = TensorDataset(current[val_index], previous[val_index])
    train_pairs.n_pairs = int(len(train_pairs))  # type: ignore[attr-defined]
    val_pairs.n_pairs = int(len(val_pairs))  # type: ignore[attr-defined]
    return train_pairs, val_pairs
