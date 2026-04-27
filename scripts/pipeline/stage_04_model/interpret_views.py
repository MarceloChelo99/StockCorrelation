from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd



from src.clustering.interpret import compare_themes_to_gics, feature_profile_per_theme, top_firms_per_theme
from src.config import load_config
from src.utils.io import ensure_dir
from src.utils.logging import log


SHOWCASE_TICKERS = ["META", "AAPL", "BRK.B", "XOM", "JPM", "SBUX", "PFE", "NEE"]


def main(config_name: str, experiment_dir: str, report_dir: str = "report") -> None:
    """Write per-view theme reports and showcase similarity fingerprints."""
    config = load_config(config_name)
    root = Path(experiment_dir)
    report_root = ensure_dir(report_dir)
    dataset = pd.read_parquet(Path(config["paths"]["dataset_dir"]) / f"{config['assembly']['dataset_name']}.parquet")
    metadata = pd.read_parquet(config["paths"]["metadata_path"])

    embeddings_per_view = {}
    for view_name in config.get("views_enabled", list(config["views"].keys())):
        view_config = config["views"][view_name]
        view_dir = root / "views" / view_name
        loadings = pd.read_parquet(view_dir / "loadings.parquet")
        embeddings = pd.read_parquet(view_dir / "embeddings.parquet")
        embeddings_per_view[view_name] = embeddings
        top = top_firms_per_theme(loadings, k=10)
        profile = feature_profile_per_theme(loadings, dataset, view_config)
        gics = compare_themes_to_gics(loadings, metadata)
        (report_root / f"view_{view_name}_themes.md").write_text(
            theme_markdown(view_name, top, profile, gics),
            encoding="utf-8",
        )
        log(f"Wrote theme report for {view_name}.", tag="interpret")

    (report_root / "similarity_fingerprints.md").write_text(
        fingerprints_markdown(embeddings_per_view, metadata, SHOWCASE_TICKERS),
        encoding="utf-8",
    )
    log("Wrote similarity fingerprints.", tag="interpret")


def theme_markdown(view_name: str, top: pd.DataFrame, profile: pd.DataFrame, gics: pd.DataFrame) -> str:
    """Format one view's themes for manual interpretation."""
    lines = [f"# {view_name.title()} View Themes", "", "Theme labels are intentionally left blank for manual filling.", ""]
    for theme in sorted(top["theme"].unique()):
        lines.extend([f"## {theme}", "", "Manual label: ", "", "Top firms:", ""])
        firms = top[top["theme"] == theme].sort_values("rank")
        lines.append("| Rank | Ticker | Loading |")
        lines.append("| ---: | --- | ---: |")
        for _, row in firms.iterrows():
            lines.append(f"| {int(row['rank'])} | {row['ticker']} | {float(row['loading']):.3f} |")

        gics_rows = gics[gics["theme"] == theme].head(5)
        if not gics_rows.empty:
            lines.extend(["", "GICS mix:", "", "| Sector | Soft Share |", "| --- | ---: |"])
            for _, row in gics_rows.iterrows():
                lines.append(f"| {row['gics_sector']} | {float(row['soft_share']):.3f} |")

        profile_rows = profile[profile["theme"] == theme].copy()
        if not profile_rows.empty:
            profile_rows["abs_value"] = profile_rows["weighted_mean"].abs()
            profile_rows = profile_rows.sort_values("abs_value", ascending=False).head(8)
            lines.extend(["", "Characteristic features:", "", "| Feature | Weighted Mean |", "| --- | ---: |"])
            for _, row in profile_rows.iterrows():
                lines.append(f"| {row['feature']} | {float(row['weighted_mean']):.4f} |")
        lines.append("")
    return "\n".join(lines)


def fingerprints_markdown(embeddings_per_view: dict[str, pd.DataFrame], metadata: pd.DataFrame, tickers: list[str]) -> str:
    """Format top peers per view for showcase tickers."""
    name_map = dict(zip(metadata["ticker"].astype(str).str.upper(), metadata["company_name"].astype(str)))
    lines = ["# Similarity Fingerprints", "", "Top-5 nearest peers per view for selected companies.", ""]
    for ticker in tickers:
        lines.extend([f"## {ticker} - {name_map.get(ticker, '')}", ""])
        lines.append("| View | Top Peers |")
        lines.append("| --- | --- |")
        for view_name, embeddings in embeddings_per_view.items():
            peers = nearest_peers(embeddings, ticker, k=5)
            labels = [f"{peer} ({name_map.get(peer, '')})" for peer in peers]
            lines.append(f"| {view_name} | {', '.join(labels)} |")
        lines.append("")
    return "\n".join(lines)


def nearest_peers(embeddings: pd.DataFrame, ticker: str, k: int) -> list[str]:
    """Return nearest latest embedding peers for one ticker."""
    columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    frame = embeddings.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    latest = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
    if ticker not in set(latest["ticker"]):
        return []
    matrix = latest.loc[:, columns].astype(float).to_numpy()
    index = int(latest.index[latest["ticker"] == ticker][0])
    distances = np.sqrt(np.square(matrix - matrix[index]).sum(axis=1))
    order = np.argsort(distances)
    return [str(latest.iloc[row]["ticker"]) for row in order if int(row) != index][:k]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: .venv/bin/python -m scripts.pipeline.stage_04_model.interpret_views "
            "<config_name> <experiment_dir> [report_dir]"
        )
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "report")
