"""Shared helpers for temporal autoencoder validation analyses."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_embeddings(path: str | Path) -> pd.DataFrame:
    """Load embeddings and validate the expected ticker/date/embedding schema."""
    frame = pd.read_parquet(path)
    required = {"ticker", "date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Embedding file {path} is missing columns: {missing}")
    columns = embedding_columns(frame)
    if not columns:
        raise ValueError(f"Embedding file {path} has no embedding_ columns.")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return embedding columns in stable numeric order."""
    columns = [column for column in frame.columns if column.startswith("embedding_")]
    return sorted(columns, key=lambda column: int(column.split("_")[-1]))


def standardized_embeddings(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with embedding columns standardized for displacement tests."""
    output = frame.copy()
    columns = embedding_columns(output)
    values = output.loc[:, columns].astype(float)
    scale = values.std(axis=0, ddof=0).replace(0.0, 1.0)
    output.loc[:, columns] = (values - values.mean(axis=0)) / scale
    return output


def vector_near_date(
    frame: pd.DataFrame,
    ticker: str,
    target_date: str,
    tolerance_days: int = 75,
) -> np.ndarray | None:
    """Return the embedding vector nearest to a target date for one ticker."""
    ticker_frame = frame.loc[frame["ticker"] == ticker.upper()].copy()
    if ticker_frame.empty:
        return None
    target = pd.Timestamp(target_date)
    distance = (ticker_frame["date"] - target).abs().dt.days
    best_index = distance.idxmin()
    if int(distance.loc[best_index]) > tolerance_days:
        return None
    return ticker_frame.loc[best_index, embedding_columns(frame)].astype(float).to_numpy()


def displacement(
    frame: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    tolerance_days: int = 75,
) -> float | None:
    """Return embedding displacement between two dates for one ticker."""
    start = vector_near_date(frame, ticker, start_date, tolerance_days=tolerance_days)
    end = vector_near_date(frame, ticker, end_date, tolerance_days=tolerance_days)
    if start is None or end is None:
        return None
    return float(np.linalg.norm(end - start))


def monthly_velocities(
    frame: pd.DataFrame,
    ticker: str,
    max_gap_days: int = 45,
) -> pd.DataFrame:
    """Return month-to-month embedding velocities for one ticker."""
    ticker_frame = frame.loc[frame["ticker"] == ticker.upper()].sort_values("date")
    columns = embedding_columns(frame)
    rows = []
    values = ticker_frame.loc[:, columns].astype(float).to_numpy()
    dates = ticker_frame["date"].to_numpy(dtype="datetime64[ns]")
    for index in range(1, len(ticker_frame)):
        gap_days = int((dates[index] - dates[index - 1]) / np.timedelta64(1, "D"))
        if gap_days > int(max_gap_days):
            continue
        rows.append(
            {
                "ticker": ticker.upper(),
                "start_date": pd.Timestamp(dates[index - 1]),
                "end_date": pd.Timestamp(dates[index]),
                "velocity": float(np.linalg.norm(values[index] - values[index - 1])),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(path: str | Path, title: str, lines: list[str]) -> Path:
    """Write a small Markdown validation report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [f"# {title}", "", *lines]
    output_path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    return output_path


def write_svg_histogram(frame: pd.DataFrame, path: str | Path, value_column: str, group_column: str) -> Path:
    """Write a simple dependency-free SVG histogram for two small groups."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.loc[frame[value_column].notna()].copy()
    if clean.empty:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='240'></svg>\n")
        return output_path
    values = clean[value_column].astype(float)
    bins = np.linspace(float(values.min()), float(values.max()), 12)
    if float(values.min()) == float(values.max()):
        bins = np.linspace(float(values.min()) - 0.5, float(values.max()) + 0.5, 12)
    groups = list(clean[group_column].dropna().astype(str).unique())[:4]
    colors = ["#2f6f73", "#c46a3a", "#5d5f91", "#b3942d"]
    width = 800
    height = 260
    plot_height = 180
    baseline = 220
    bar_width = max(1, int((width - 120) / (len(bins) - 1) / max(1, len(groups))))
    counts = {
        group: np.histogram(clean.loc[clean[group_column] == group, value_column].astype(float), bins=bins)[0]
        for group in groups
    }
    max_count = max([int(values.max()) for values in counts.values()] or [1])
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='260'>",
        "<rect width='800' height='260' fill='#faf7ef'/>",
        "<text x='24' y='30' font-family='serif' font-size='18'>Embedding displacement histogram</text>",
    ]
    for group_index, group in enumerate(groups):
        for bin_index, count in enumerate(counts[group]):
            x = 60 + bin_index * bar_width * max(1, len(groups)) + group_index * bar_width
            bar_height = 0 if max_count == 0 else int(plot_height * int(count) / max_count)
            y = baseline - bar_height
            parts.append(
                f"<rect x='{x}' y='{y}' width='{bar_width - 1}' height='{bar_height}' "
                f"fill='{colors[group_index % len(colors)]}' opacity='0.78'/>"
            )
        legend_y = 52 + group_index * 18
        parts.append(
            f"<rect x='620' y='{legend_y - 10}' width='12' height='12' "
            f"fill='{colors[group_index % len(colors)]}'/>"
        )
        parts.append(
            f"<text x='638' y='{legend_y}' font-family='sans-serif' font-size='12'>{group}</text>"
        )
    parts.extend(
        [
            f"<line x1='50' y1='{baseline}' x2='760' y2='{baseline}' stroke='#333'/>",
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output_path
