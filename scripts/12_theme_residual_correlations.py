from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.db import FilingsDB
from src.evaluation.peers import daily_returns_matrix
from src.utils.io import ensure_dir
from src.utils.logging import log


def main(
    config_name: str,
    examples_path: str,
    output_dir: str,
    selected_tickers_csv: str = "CBRE,ETN,SBUX,CMG,META,GOOGL,NFLX,HUM,NKE,VRT",
) -> None:
    """Compute residual correlations across semantic peer themes."""
    config = load_config(config_name)
    db = FilingsDB.from_config(config)
    metadata = load_metadata(db, config)
    returns = daily_returns_matrix(db.load_prices())
    examples = pd.read_csv(examples_path)
    selected_tickers = [ticker.strip().upper() for ticker in selected_tickers_csv.split(",") if ticker.strip()]

    residuals = {}
    diagnostics = []
    for ticker in selected_tickers:
        example = examples[examples["ticker"].astype(str).str.upper() == ticker]
        if example.empty:
            continue
        peer_tickers = [value for value in str(example.iloc[0]["peer_tickers"]).split("|") if value]
        cluster_tickers = [ticker, *peer_tickers]
        residual, diagnostic = theme_residual(ticker, cluster_tickers, returns, metadata)
        residuals[ticker] = residual
        diagnostics.append(diagnostic)

    residual_frame = pd.DataFrame(residuals).dropna()
    if residual_frame.empty:
        raise ValueError("No overlapping theme residuals were available.")

    corr = residual_frame.corr()
    pairs = residual_correlation_pairs(corr)
    summary = residual_correlation_summary(pairs)

    output_root = ensure_dir(output_dir)
    corr.to_csv(output_root / "theme_residual_correlation_matrix.csv")
    pairs.to_csv(output_root / "theme_residual_correlation_pairs.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_root / "theme_residual_diagnostics.csv", index=False)
    (output_root / "theme_residual_correlation_summary.md").write_text(summary, encoding="utf-8")
    log(f"Wrote theme residual correlations to {output_root}.", tag="theme-resid")


def load_metadata(db: FilingsDB, config: dict) -> pd.DataFrame:
    """Load DB metadata and optional external GICS metadata."""
    metadata = db.load_tickers()
    metadata_path = Path(config["paths"].get("metadata_path", ""))
    if metadata_path.exists():
        external = pd.read_parquet(metadata_path)
        external["ticker"] = external["ticker"].astype(str).str.upper()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        duplicate_columns = [column for column in external.columns if column in metadata.columns and column != "ticker"]
        metadata = metadata.merge(external.drop(columns=duplicate_columns), on="ticker", how="left")
    return metadata


def theme_residual(
    seed_ticker: str,
    cluster_tickers: list[str],
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object]]:
    """Return residual theme returns after removing market and seed-sector returns."""
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
    residuals, r2 = residualize(y, x)
    residual_series = pd.Series(residuals, index=frame.index, name=seed_ticker)
    diagnostic = {
        "ticker": seed_ticker,
        "seed_sector": seed_sector,
        "cluster_tickers": "|".join(tickers),
        "n_days": int(len(frame)),
        "market_sector_r2": float(r2),
        "annual_residual_vol": float(np.std(residuals, ddof=1) * np.sqrt(252)),
    }
    return residual_series, diagnostic


def residualize(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Regress y on an intercept and x, returning residuals and R-squared."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(design) @ y
    fitted = design @ beta
    residuals = y - fitted
    total = float(np.square(y - y.mean()).sum())
    residual = float(np.square(residuals).sum())
    r2 = 0.0 if total == 0.0 else 1.0 - residual / total
    return residuals, r2


def residual_correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Flatten an upper-triangle correlation matrix into pair rows."""
    rows = []
    tickers = corr.columns.tolist()
    for left_index, left in enumerate(tickers):
        for right in tickers[left_index + 1 :]:
            rows.append({"theme_a": left, "theme_b": right, "residual_corr": float(corr.loc[left, right])})
    return pd.DataFrame(rows).sort_values("residual_corr", ascending=False).reset_index(drop=True)


def residual_correlation_summary(pairs: pd.DataFrame) -> str:
    """Write a compact markdown summary of residual correlation structure."""
    abs_corr = pairs["residual_corr"].abs()
    lines = [
        "# Theme Residual Correlations",
        "",
        "Semantic peer-cluster returns are first residualized against the market and the seed ticker's GICS sector basket.",
        "",
        f"- Mean pairwise residual correlation: `{pairs['residual_corr'].mean():.3f}`",
        f"- Median pairwise residual correlation: `{pairs['residual_corr'].median():.3f}`",
        f"- Mean absolute residual correlation: `{abs_corr.mean():.3f}`",
        f"- Max residual correlation: `{pairs['residual_corr'].max():.3f}`",
        f"- Min residual correlation: `{pairs['residual_corr'].min():.3f}`",
        "",
        "Highest residual correlations:",
        "",
        "| Theme A | Theme B | Residual Corr |",
        "| --- | --- | ---: |",
    ]
    for _, row in pairs.head(5).iterrows():
        lines.append(f"| {row['theme_a']} | {row['theme_b']} | {float(row['residual_corr']):.3f} |")
    lines.extend(["", "Lowest residual correlations:", "", "| Theme A | Theme B | Residual Corr |", "| --- | --- | ---: |"])
    for _, row in pairs.tail(5).sort_values("residual_corr").iterrows():
        lines.append(f"| {row['theme_a']} | {row['theme_b']} | {float(row['residual_corr']):.3f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: .venv/bin/python scripts/12_theme_residual_correlations.py "
            "<config_name> <examples_path> <output_dir> [selected_tickers_csv]"
        )
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4] if len(sys.argv) > 4 else "CBRE,ETN,SBUX,CMG,META,GOOGL,NFLX,HUM,NKE,VRT",
    )
