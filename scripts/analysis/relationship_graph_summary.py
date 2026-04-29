from __future__ import annotations

from scripts import _bootstrap  # noqa: F401

import json
import sys
from pathlib import Path

import pandas as pd



from src.utils.io import ensure_dir
from src.utils.logging import log


def main(
    relationships_path: str = "data/processed/relationships/relationships.parquet",
    semantic_metrics_path: str = "experiments/20260423_232517_semantic_embedding/metrics.json",
    output_path: str = "report/relationship_graph_summary.md",
) -> None:
    """Summarize extracted relationship graph coverage and alignment metrics."""
    relationships = pd.read_parquet(relationships_path)
    metrics = json.loads(Path(semantic_metrics_path).read_text(encoding="utf-8"))
    relationship_metrics = metrics.get("relationships", {})
    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text(markdown_summary(relationships, relationship_metrics), encoding="utf-8")
    log(f"Wrote relationship graph summary to {output}.", tag="relationships")


def markdown_summary(relationships: pd.DataFrame, metrics: dict) -> str:
    """Format relationship extraction and evaluation results as markdown."""
    type_counts = relationships["relationship_type"].value_counts()
    has_direction_columns = {"supplier_ticker", "customer_ticker"}.issubset(relationships.columns)
    supply_chain = pd.DataFrame()
    if has_direction_columns:
        supply_chain = relationships[relationships["supplier_ticker"].fillna("").astype(str) != ""]
    lines = [
        "# Relationship Graph Summary",
        "",
        "The graph is extracted from parsed 10-K section text using conservative public-company name matching and rule-based context classification.",
        "Customer/supplier rows are directional: `supplier_ticker` is the firm that provides the product/service, and `customer_ticker` is the firm that buys or depends on it.",
        "",
        f"- Relationship rows: `{len(relationships)}`",
        f"- Source tickers with at least one edge: `{relationships['source_ticker'].nunique()}`",
        f"- Target tickers mentioned: `{relationships['target_ticker'].nunique()}`",
        f"- Directed supply-chain rows: `{len(supply_chain) if has_direction_columns else 0}`",
        f"- Disclosed supplier tickers: `{supply_chain['supplier_ticker'].nunique() if has_direction_columns and not supply_chain.empty else 0}`",
        f"- Disclosed customer tickers: `{supply_chain['customer_ticker'].nunique() if has_direction_columns and not supply_chain.empty else 0}`",
        "",
        "Relationship type counts:",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for relationship_type, count in type_counts.items():
        lines.append(f"| {relationship_type} | {int(count)} |")

    if not supply_chain.empty:
        lines.extend(
            [
                "",
                "Top disclosed suppliers:",
                "",
                "| Supplier | Directed Edges |",
                "| --- | ---: |",
            ]
        )
        for ticker, count in supply_chain["supplier_ticker"].value_counts().head(15).items():
            lines.append(f"| {ticker} | {int(count)} |")

        lines.extend(
            [
                "",
                "Top disclosed customers:",
                "",
                "| Customer | Directed Edges |",
                "| --- | ---: |",
            ]
        )
        for ticker, count in supply_chain["customer_ticker"].value_counts().head(15).items():
            lines.append(f"| {ticker} | {int(count)} |")

        examples = supply_chain.sort_values(["direction_confidence", "confidence"], ascending=False).head(12)
        lines.extend(
            [
                "",
                "High-confidence directed examples:",
                "",
                "| Supplier | Customer | Source Filing Company | Evidence |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in examples.iterrows():
            evidence = str(row["context_snippet"]).replace("|", "\\|")
            if len(evidence) > 160:
                evidence = evidence[:157].rstrip() + "..."
            lines.append(
                f"| {row['supplier_ticker']} | {row['customer_ticker']} | {row['source_ticker']} | {evidence} |"
            )

    if metrics:
        lines.extend(
            [
                "",
                "Semantic embedding graph-alignment evaluation:",
                "",
                "| Peer Set | Direct Link Rate | Neighbor Jaccard |",
                "| --- | ---: | ---: |",
                f"| Semantic embedding peers | {metrics['mean_embedding_direct_rate']:.3f} | {metrics['mean_embedding_jaccard']:.3f} |",
                f"| GICS sub-industry peers | {metrics['mean_benchmark_direct_rate']:.3f} | {metrics['mean_benchmark_jaccard']:.3f} |",
                f"| Random peers | {metrics['mean_random_direct_rate']:.3f} | {metrics['mean_random_jaccard']:.3f} |",
                "",
                "Interpretation: semantic peers are substantially more connected than random peers, but less connected than same-GICS-sub-industry peers.",
                "That means the current text embedding contains relationship-network signal, while GICS remains stronger for direct disclosed links.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "data/processed/relationships/relationships.parquet",
        sys.argv[2] if len(sys.argv) > 2 else "experiments/20260423_232517_semantic_embedding/metrics.json",
        sys.argv[3] if len(sys.argv) > 3 else "report/relationship_graph_summary.md",
    )
