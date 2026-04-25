from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir
from src.utils.logging import log


DEFAULT_METADATA_PATH = Path("data/processed/metadata/sp500_gics.parquet")
DEFAULT_SELECTED_TICKERS = [
    "CBRE",
    "ETN",
    "SBUX",
    "CMG",
    "META",
    "GOOGL",
    "NFLX",
    "HUM",
    "NKE",
    "VRT",
]


def main(
    embeddings_path: str,
    output_dir: str = "report",
    metadata_path: str = str(DEFAULT_METADATA_PATH),
) -> None:
    """Write nearest-neighbor peer examples from the latest semantic embeddings."""
    embeddings = pd.read_parquet(embeddings_path)
    metadata = pd.read_parquet(metadata_path)
    examples = qualitative_peer_examples(embeddings, metadata, top_k=5)

    output_root = ensure_dir(output_dir)
    csv_path = output_root / "semantic_peer_examples.csv"
    markdown_path = output_root / "semantic_peer_examples_selected.md"
    examples.to_csv(csv_path, index=False)
    selected_markdown = selected_examples_markdown(examples, DEFAULT_SELECTED_TICKERS)
    markdown_path.write_text(selected_markdown, encoding="utf-8")

    log(f"Wrote full qualitative peer table to {csv_path}.", tag="peers")
    log(f"Wrote selected examples to {markdown_path}.", tag="peers")


def qualitative_peer_examples(embeddings: pd.DataFrame, metadata: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """Return one row per ticker with nearest semantic embedding peers."""
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError("Embeddings must contain embedding_* columns.")

    latest = latest_embeddings(embeddings, embedding_columns)
    metadata = metadata.loc[:, ["ticker", "company_name", "gics_sector", "gics_sub_industry"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    latest = latest.merge(metadata, on="ticker", how="left")

    matrix = latest.loc[:, embedding_columns].astype(float).to_numpy()
    distances = pairwise_distances(matrix)
    np.fill_diagonal(distances, np.inf)

    tickers = latest["ticker"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for row_index, ticker in enumerate(tickers):
        peer_indices = np.argsort(distances[row_index])[:top_k]
        own = latest.iloc[row_index]
        peer_rows = latest.iloc[peer_indices]
        peer_descriptions = [
            f"{peer['ticker']} ({peer['company_name']}; {peer['gics_sector']}; {peer['gics_sub_industry']})"
            for _, peer in peer_rows.iterrows()
        ]
        rows.append(
            {
                "ticker": ticker,
                "company_name": own["company_name"],
                "date": pd.Timestamp(own["date"]).strftime("%Y-%m-%d"),
                "gics_sector": own["gics_sector"],
                "gics_sub_industry": own["gics_sub_industry"],
                "peer_tickers": "|".join(peer_rows["ticker"].astype(str).tolist()),
                "peer_descriptions": " | ".join(peer_descriptions),
                "same_sector_count": int((peer_rows["gics_sector"] == own["gics_sector"]).sum()),
                "same_subindustry_count": int((peer_rows["gics_sub_industry"] == own["gics_sub_industry"]).sum()),
                "mean_peer_distance": float(distances[row_index, peer_indices].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["ticker"]).reset_index(drop=True)


def latest_embeddings(embeddings: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    """Return each ticker's latest embedding row."""
    required = {"ticker", "date", *embedding_columns}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(f"Embeddings are missing columns: {sorted(missing)}")

    frame = embeddings.loc[:, ["ticker", "date", *embedding_columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=embedding_columns)
    frame = frame.sort_values(["ticker", "date"])
    return frame.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    difference = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.square(difference).sum(axis=2))


def selected_examples_markdown(examples: pd.DataFrame, tickers: list[str]) -> str:
    """Format selected peer examples as a compact markdown table."""
    selected = examples[examples["ticker"].isin(tickers)].copy()
    selected["ticker"] = pd.Categorical(selected["ticker"], categories=tickers, ordered=True)
    selected = selected.sort_values("ticker")

    lines = [
        "# Semantic Peer Examples",
        "",
        "Nearest neighbors are computed from each ticker's latest semantic embedding.",
        "",
        "| Ticker | Company | GICS label | Embedding-nearest peers | Why it is interesting |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in selected.iterrows():
        interpretation = example_interpretation(str(row["ticker"]))
        gics_label = f"{row['gics_sector']} / {row['gics_sub_industry']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ticker"]),
                    str(row["company_name"]),
                    gics_label,
                    str(row["peer_tickers"]).replace("|", ", "),
                    interpretation,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def example_interpretation(ticker: str) -> str:
    """Return a short hand-written interpretation for selected examples."""
    interpretations = {
        "CBRE": "Real-estate services clusters with asset managers, reflecting capital-markets exposure beyond its GICS real-estate label.",
        "ETN": "Electrical-equipment language clusters with utilities, consistent with grid investment and electrification exposure.",
        "SBUX": "Restaurant label crosses into beverages and packaged foods, matching brand, consumer demand, and distribution language.",
        "CMG": "Restaurant peers include beverage and snack companies, suggesting consumer-brand similarity rather than only restaurant operations.",
        "META": "Communication-services label clusters with enterprise software and app-platform companies.",
        "GOOGL": "Interactive-media label sits near internet infrastructure and software firms, not only communication-services peers.",
        "NFLX": "Entertainment peers include game publishers and cable/media distributors, a sensible content-platform neighborhood.",
        "HUM": "Managed-care label clusters with health-care distributors, reflecting insurance, reimbursement, and care-delivery overlap.",
        "NKE": "Apparel label clusters with consumer packaged brands, highlighting brand and global consumer-product language.",
        "VRT": "Industrial electrical-equipment label clusters with technology infrastructure and power-semiconductor names.",
    }
    return interpretations.get(ticker, "")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: .venv/bin/python scripts/07_qualitative_peers.py "
            "<embeddings_path> [output_dir] [metadata_path]"
        )
    main(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else "report",
        sys.argv[3] if len(sys.argv) > 3 else str(DEFAULT_METADATA_PATH),
    )
