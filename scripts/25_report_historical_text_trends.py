"""Report historical filing-language trends from compact section features."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import log  # noqa: E402


TOPIC_COLUMNS = {
    "ai": "topic_ai_score_per_10k_words",
    "cloud_compute": "topic_cloud_compute_score_per_10k_words",
    "cybersecurity": "topic_cybersecurity_score_per_10k_words",
    "supply_chain": "topic_supply_chain_score_per_10k_words",
    "electrification": "topic_electrification_score_per_10k_words",
}


def main(input_dir: str = "data/processed/historical_text", report_dir: str = "report") -> None:
    """Write historical trend CSVs and markdown summary."""
    input_root = Path(input_dir)
    output_root = ensure_dir(report_dir)
    counts = pd.read_parquet(input_root / "historical_section_topic_counts.parquet")
    metadata = pd.read_parquet("data/processed/metadata/sp500_gics.parquet")

    counts["filing_date"] = pd.to_datetime(counts["filing_date"])
    counts["year"] = counts["filing_date"].dt.year
    metadata = metadata.loc[:, ["ticker", "company_name", "gics_sector", "gics_sub_industry"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()

    yearly = yearly_section_topic_trends(counts)
    company_changes = company_topic_changes(counts, metadata)
    sector_changes = sector_topic_changes(counts, metadata)
    coverage = coverage_summary(counts)

    yearly.to_csv(output_root / "historical_section_topic_trends.csv", index=False)
    company_changes.to_csv(output_root / "historical_company_topic_changes.csv", index=False)
    sector_changes.to_csv(output_root / "historical_sector_topic_changes.csv", index=False)
    coverage.to_csv(output_root / "historical_text_coverage.csv", index=False)
    (output_root / "historical_text_trends.md").write_text(
        trend_markdown(yearly, company_changes, sector_changes, coverage),
        encoding="utf-8",
    )
    log("Wrote historical text trend report.", tag="historical-trends")


def yearly_section_topic_trends(counts: pd.DataFrame) -> pd.DataFrame:
    """Return yearly mean topic scores by section."""
    aggregations = {
        "rows": ("ticker", "size"),
        "tickers": ("ticker", "nunique"),
        "ai_mentions": ("topic_ai_mentions", "sum"),
    }
    for topic, column in TOPIC_COLUMNS.items():
        aggregations[f"{topic}_score"] = (column, "mean")
    return counts.groupby(["section", "year"]).agg(**aggregations).reset_index()


def company_topic_changes(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Compare latest topic intensity to the pre-2019 baseline for each company."""
    sections = counts[counts["section"].isin(["business", "risk_factors"])].copy()
    firm_year = sections.groupby(["ticker", "year"]).agg(
        ai_score=("topic_ai_score_per_10k_words", "mean"),
        ai_mentions=("topic_ai_mentions", "sum"),
        cloud_compute_score=("topic_cloud_compute_score_per_10k_words", "mean"),
        cybersecurity_score=("topic_cybersecurity_score_per_10k_words", "mean"),
        supply_chain_score=("topic_supply_chain_score_per_10k_words", "mean"),
        electrification_score=("topic_electrification_score_per_10k_words", "mean"),
    ).reset_index()

    rows = []
    for ticker, group in firm_year.groupby("ticker"):
        group = group.sort_values("year")
        early = group[group["year"].between(2010, 2018)]
        late = group[group["year"].between(2023, 2026)]
        if len(early) < 3 or late.empty:
            continue
        row = {
            "ticker": ticker,
            "first_year": int(group["year"].min()),
            "latest_year": int(group["year"].max()),
            "late_ai_mentions": int(late["ai_mentions"].sum()),
        }
        for topic in ["ai", "cloud_compute", "cybersecurity", "supply_chain", "electrification"]:
            column = f"{topic}_score"
            row[f"early_{topic}"] = float(early[column].mean())
            row[f"late_{topic}"] = float(late[column].mean())
            row[f"{topic}_change"] = row[f"late_{topic}"] - row[f"early_{topic}"]
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.merge(metadata, on="ticker", how="left")
    return result.sort_values("ai_change", ascending=False)


def sector_topic_changes(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return sector-level topic changes for business and risk sections."""
    frame = counts[counts["section"].isin(["business", "risk_factors"])].merge(metadata, on="ticker", how="left")
    firm_year = frame.groupby(["gics_sector", "ticker", "year"]).agg(
        ai_score=("topic_ai_score_per_10k_words", "mean"),
        cloud_compute_score=("topic_cloud_compute_score_per_10k_words", "mean"),
        cybersecurity_score=("topic_cybersecurity_score_per_10k_words", "mean"),
        supply_chain_score=("topic_supply_chain_score_per_10k_words", "mean"),
        electrification_score=("topic_electrification_score_per_10k_words", "mean"),
    ).reset_index()

    rows = []
    for sector, group in firm_year.dropna(subset=["gics_sector"]).groupby("gics_sector"):
        early = group[group["year"].between(2010, 2018)]
        late = group[group["year"].between(2023, 2026)]
        if early.empty or late.empty:
            continue
        row = {"gics_sector": sector, "tickers": int(group["ticker"].nunique())}
        for topic in ["ai", "cloud_compute", "cybersecurity", "supply_chain", "electrification"]:
            column = f"{topic}_score"
            row[f"early_{topic}"] = float(early[column].mean())
            row[f"late_{topic}"] = float(late[column].mean())
            row[f"{topic}_change"] = row[f"late_{topic}"] - row[f"early_{topic}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ai_change", ascending=False)


def coverage_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Return yearly coverage by section."""
    return counts.groupby(["year", "section"]).agg(
        rows=("ticker", "size"),
        tickers=("ticker", "nunique"),
        median_chars=("section_chars", "median"),
    ).reset_index()


def trend_markdown(
    yearly: pd.DataFrame,
    company_changes: pd.DataFrame,
    sector_changes: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    """Format a compact markdown interpretation report."""
    lines = [
        "# Historical Text Trends",
        "",
        "This report uses compact historical 10-K section features: keyword counts plus MiniLM embeddings. It does not depend on storing full raw SEC filings.",
        "",
        "## Coverage",
        "",
    ]
    latest_coverage = coverage[coverage["year"] >= coverage["year"].max() - 3]
    lines.extend(markdown_table(latest_coverage, ["year", "section", "rows", "tickers", "median_chars"]))

    lines.extend(["", "## AI Trend By Section", ""])
    ai_trend = yearly[yearly["section"].isin(["business", "risk_factors", "mda"])].copy()
    ai_trend = ai_trend[ai_trend["year"] >= 2018]
    lines.extend(markdown_table(ai_trend, ["section", "year", "rows", "tickers", "ai_mentions", "ai_score"]))

    lines.extend(["", "## Largest Company AI Language Increases", ""])
    lines.extend(
        markdown_table(
            company_changes.head(25),
            [
                "ticker",
                "company_name",
                "gics_sector",
                "early_ai",
                "late_ai",
                "ai_change",
                "late_ai_mentions",
                "late_cloud_compute",
                "late_cybersecurity",
            ],
        )
    )

    lines.extend(["", "## Sector AI Language Increases", ""])
    lines.extend(
        markdown_table(
            sector_changes,
            ["gics_sector", "tickers", "early_ai", "late_ai", "ai_change", "late_cloud_compute", "late_cybersecurity"],
        )
    )
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Format dataframe rows as a markdown table."""
    if frame.empty:
        return ["No rows available."]
    usable_columns = [column for column in columns if column in frame.columns]
    lines = ["| " + " | ".join(usable_columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(usable_columns)) + " |")
    for _, row in frame.loc[:, usable_columns].iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in usable_columns) + " |")
    return lines


def format_cell(value: object) -> str:
    """Format one markdown table value."""
    if isinstance(value, float):
        return f"{value:.3f}"
    if pd.isna(value):
        return ""
    return str(value)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "data/processed/historical_text",
        sys.argv[2] if len(sys.argv) > 2 else "report",
    )
