from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import numpy as np
import pandas as pd



from src.config import load_config
from src.db import FilingsDB
from src.evaluation.peers import daily_returns_matrix
from src.utils.io import ensure_dir
from src.utils.logging import log
from src.utils.metadata import load_ticker_metadata


DEFAULT_EXAMPLES_PATH = Path("report/semantic_peer_examples.csv")


def main(
    config_name: str,
    examples_path: str,
    output_path: str,
    selected_tickers_csv: str = "CBRE,ETN,SBUX,CMG,META,GOOGL,NFLX,HUM,NKE,VRT",
) -> None:
    """Test whether semantic peer clusters are more than sector proxies."""
    config = load_config(config_name)
    db = FilingsDB.from_config(config)
    metadata = load_metadata(db, config)
    returns = daily_returns_matrix(db.load_prices())
    examples = pd.read_csv(examples_path)
    selected_tickers = [ticker.strip().upper() for ticker in selected_tickers_csv.split(",") if ticker.strip()]

    rows = []
    for ticker in selected_tickers:
        example = examples[examples["ticker"].astype(str).str.upper() == ticker]
        if example.empty:
            continue
        row = example.iloc[0]
        peer_tickers = [value for value in str(row["peer_tickers"]).split("|") if value]
        cluster_tickers = [ticker, *peer_tickers]
        metrics = thematic_cluster_metrics(ticker, cluster_tickers, returns, metadata)
        rows.append({**row.to_dict(), **metrics})

    output = Path(output_path)
    ensure_dir(output.parent)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(markdown_summary(result), encoding="utf-8")
    log(f"Wrote thematic factor tests to {output}.", tag="theme")
    log(f"Wrote thematic factor summary to {markdown_path}.", tag="theme")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load DB metadata and optional external GICS metadata."""
    return load_ticker_metadata(db, config)


def thematic_cluster_metrics(
    seed_ticker: str,
    cluster_tickers: list[str],
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, object]:
    """Regress a semantic cluster return on market and sector returns."""
    tickers = [ticker for ticker in cluster_tickers if ticker in returns.columns]
    if len(tickers) < 3:
        raise ValueError(f"Cluster for {seed_ticker} has fewer than three return series.")

    metadata = metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    seed_meta = metadata[metadata["ticker"] == seed_ticker]
    if seed_meta.empty:
        raise ValueError(f"Missing metadata for {seed_ticker}.")
    seed_sector = str(seed_meta.iloc[0]["gics_sector"])
    sector_tickers = metadata[metadata["gics_sector"] == seed_sector]["ticker"].astype(str).tolist()
    sector_tickers = [ticker for ticker in sector_tickers if ticker in returns.columns]

    frame = pd.DataFrame(
        {
            "cluster": returns.loc[:, tickers].mean(axis=1),
            "market": returns.mean(axis=1),
            "sector": returns.loc[:, sector_tickers].mean(axis=1),
        }
    ).dropna()
    y = frame["cluster"].to_numpy()
    x = frame.loc[:, ["market", "sector"]].to_numpy()
    regression = ordinary_least_squares(y, x)
    residuals = regression["residuals"]
    cluster_vol = float(np.std(y, ddof=1) * np.sqrt(252))
    residual_vol = float(np.std(residuals, ddof=1) * np.sqrt(252))

    return {
        "cluster_tickers": "|".join(tickers),
        "seed_sector": seed_sector,
        "n_cluster_tickers": len(tickers),
        "n_days": int(len(frame)),
        "annual_cluster_vol": cluster_vol,
        "annual_residual_vol_after_market_sector": residual_vol,
        "residual_vol_share": float(residual_vol / cluster_vol) if cluster_vol > 0 else np.nan,
        "market_sector_r2": regression["r2"],
        "corr_with_sector": float(frame["cluster"].corr(frame["sector"])),
        "corr_with_market": float(frame["cluster"].corr(frame["market"])),
    }


def ordinary_least_squares(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    """Fit y on an intercept and x, returning residuals and R-squared."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(design) @ y
    fitted = design @ beta
    residuals = y - fitted
    total = float(np.square(y - y.mean()).sum())
    residual = float(np.square(residuals).sum())
    r2 = 0.0 if total == 0.0 else 1.0 - residual / total
    return {"beta": beta, "residuals": residuals, "r2": float(r2)}


def markdown_summary(result: pd.DataFrame) -> str:
    """Format thematic factor results as a compact markdown table."""
    lines = [
        "# Thematic Factor Tests",
        "",
        "Each semantic peer cluster is equal-weighted and regressed on the market plus the seed ticker's GICS sector basket.",
        "",
        "| Ticker | Cluster | Sector R2 | Residual Vol Share | Corr With Sector |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in result.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ticker"]),
                    str(row["cluster_tickers"]).replace("|", ", "),
                    f"{float(row['market_sector_r2']):.3f}",
                    f"{float(row['residual_vol_share']):.3f}",
                    f"{float(row['corr_with_sector']):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.analysis.thematic_factor_tests "
            "<config_name> <examples_path> <output_path> [selected_tickers_csv]"
        )
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4] if len(sys.argv) > 4 else "CBRE,ETN,SBUX,CMG,META,GOOGL,NFLX,HUM,NKE,VRT",
    )
