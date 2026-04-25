"""Analyze interpretable theme evolution and topic enrichment in view clusters.

This script overlays human-readable filing topics on the learned soft themes.
It is intentionally diagnostic: it helps answer questions like "which business
themes are AI-heavy?" and "which firms are moving across growth or behavioral
categories?" without pretending one current 10-K filing per firm is a full
multi-year business pivot history.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir
from src.utils.logging import log


from src.filings.topics import TOPICS, phrase_count


def main(experiment_dir: str | None = None, report_dir: str = "report") -> None:
    """Write cluster evolution diagnostics for the selected decomposed run."""
    experiment_root = Path(experiment_dir) if experiment_dir else latest_decomposed_experiment()
    if experiment_root is None:
        raise FileNotFoundError("No decomposed experiment directory found.")
    if not experiment_root.exists():
        raise FileNotFoundError(experiment_root)

    report_root = ensure_dir(report_dir)
    sections = pd.read_parquet("data/processed/sections/ten_k_sections.parquet")
    metadata = pd.read_parquet("data/processed/metadata/sp500_gics.parquet")
    labels = load_theme_labels(report_root / "theme_labels.csv")

    topic_scores = compute_topic_scores(sections)
    topic_scores = topic_scores.merge(metadata_columns(metadata), on="ticker", how="left")
    topic_scores.to_csv(report_root / "topic_scores_by_company.csv", index=False)

    theme_topic_summary = business_theme_topic_summary(experiment_root, topic_scores, metadata, labels)
    theme_topic_summary.to_csv(report_root / "theme_topic_summary.csv", index=False)

    theme_mass_trends = theme_mass_trend_summary(experiment_root, labels)
    theme_mass_trends.to_csv(report_root / "view_theme_mass_trends.csv", index=False)

    migrations = ticker_theme_migrations(experiment_root, labels, metadata)
    migrations.to_csv(report_root / "ticker_theme_migrations.csv", index=False)

    ai_snippets = extract_ai_snippets(sections, topic_scores)
    ai_snippets.to_csv(report_root / "ai_filing_snippets.csv", index=False)

    markdown = evolution_markdown(
        experiment_root,
        topic_scores,
        theme_topic_summary,
        theme_mass_trends,
        migrations,
        ai_snippets,
    )
    (report_root / "theme_evolution_patterns.md").write_text(markdown, encoding="utf-8")
    log(f"Wrote theme evolution diagnostics from {experiment_root}.", tag="theme-evolution")


def latest_decomposed_experiment() -> Path | None:
    """Return the most recent decomposed experiment directory."""
    roots = sorted(Path("experiments").glob("*decomposed*"), key=lambda path: path.name)
    return roots[-1] if roots else None


def load_theme_labels(path: Path) -> pd.DataFrame:
    """Load optional manual theme labels."""
    if not path.exists():
        return pd.DataFrame(columns=["view", "theme", "label", "description"])
    labels = pd.read_csv(path)
    labels["view"] = labels["view"].astype(str)
    labels["theme"] = labels["theme"].astype(str)
    labels["label"] = labels["label"].fillna("").astype(str)
    return labels


def metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return the metadata columns used in interpretation tables."""
    frame = metadata.loc[:, ["ticker", "company_name", "gics_sector", "gics_sub_industry"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame


def compute_topic_scores(sections: pd.DataFrame) -> pd.DataFrame:
    """Score filing text for each interpretable topic, normalized per 10k words."""
    rows = []
    for _, row in sections.iterrows():
        ticker = str(row["ticker"]).upper()
        text = combined_text(row)
        words = max(1, len(re.findall(r"[A-Za-z]+", text)))
        payload = {
            "ticker": ticker,
            "filing_date": pd.to_datetime(row["filing_date"]).date().isoformat(),
            "word_count": words,
        }
        for topic, terms in TOPICS.items():
            raw_count = 0
            weighted_count = 0.0
            for term, weight in terms.items():
                count = phrase_count(text, term)
                raw_count += count
                weighted_count += weight * count
            payload[f"{topic}_mentions"] = raw_count
            payload[f"{topic}_score_per_10k_words"] = 10000.0 * weighted_count / words
        rows.append(payload)
    return pd.DataFrame(rows)


def combined_text(row: pd.Series) -> str:
    """Return business, risk, and MD&A text as lower-case plain text."""
    values = [
        str(row.get("business_text", "") or ""),
        str(row.get("risk_factors_text", "") or ""),
        str(row.get("mda_text", "") or ""),
    ]
    return "\n".join(values).lower()


def business_theme_topic_summary(
    experiment_root: Path,
    topic_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Compute topic enrichment for each business theme."""
    loadings_path = experiment_root / "views" / "business" / "loadings.parquet"
    if not loadings_path.exists():
        return pd.DataFrame()
    loadings = latest_by_ticker(pd.read_parquet(loadings_path))
    theme_columns = theme_cols(loadings)
    frame = loadings.merge(topic_scores, on="ticker", how="inner")
    metadata_frame = metadata_columns(metadata)
    frame = frame.merge(metadata_frame, on="ticker", how="left", suffixes=("", "_meta"))

    rows = []
    for theme in theme_columns:
        weights = frame[theme].astype(float)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            continue
        row = {
            "view": "business",
            "theme": theme,
            "label": theme_label(labels, "business", theme),
            "soft_member_count": total_weight,
            "top_firms": top_loaded_firms(frame, theme, 8),
            "top_sectors": top_loaded_sectors(frame, theme, 3),
        }
        for topic in TOPICS:
            score_col = f"{topic}_score_per_10k_words"
            mention_col = f"{topic}_mentions"
            row[f"{topic}_weighted_score"] = weighted_mean(frame[score_col], weights)
            row[f"{topic}_weighted_mentions"] = weighted_mean(frame[mention_col], weights)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["ai_weighted_score", "cloud_compute_weighted_score"], ascending=False)


def theme_mass_trend_summary(experiment_root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """Summarize how each view's total soft theme mass changes through time."""
    rows = []
    for view_dir in sorted((experiment_root / "views").glob("*")):
        loadings_path = view_dir / "loadings.parquet"
        if not loadings_path.exists():
            continue
        view = view_dir.name
        loadings = pd.read_parquet(loadings_path)
        loadings["date"] = pd.to_datetime(loadings["date"])
        date_count = int(loadings["date"].nunique())
        theme_columns = theme_cols(loadings)
        by_date = loadings.groupby("date")[theme_columns].mean().sort_index()
        for theme in theme_columns:
            values = by_date[theme].astype(float)
            rows.append(
                {
                    "view": view,
                    "theme": theme,
                    "label": theme_label(labels, view, theme),
                    "date_count": date_count,
                    "start_date": by_date.index.min().date().isoformat(),
                    "end_date": by_date.index.max().date().isoformat(),
                    "start_mass": float(values.iloc[0]),
                    "end_mass": float(values.iloc[-1]),
                    "mass_change": float(values.iloc[-1] - values.iloc[0]),
                    "peak_date": values.idxmax().date().isoformat(),
                    "peak_mass": float(values.max()),
                    "trend_slope_per_year": annualized_slope(values),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("trend_slope_per_year", ascending=False)


def ticker_theme_migrations(experiment_root: Path, labels: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Find tickers with the largest dominant-theme shifts in each view."""
    rows = []
    metadata_frame = metadata_columns(metadata)
    for view_dir in sorted((experiment_root / "views").glob("*")):
        loadings_path = view_dir / "loadings.parquet"
        if not loadings_path.exists():
            continue
        view = view_dir.name
        loadings = pd.read_parquet(loadings_path)
        loadings["ticker"] = loadings["ticker"].astype(str).str.upper()
        loadings["date"] = pd.to_datetime(loadings["date"])
        theme_columns = theme_cols(loadings)
        if len(theme_columns) == 0:
            continue
        for ticker, group in loadings.sort_values("date").groupby("ticker"):
            if group["date"].nunique() < 3:
                continue
            dominant = group[theme_columns].astype(float).idxmax(axis=1).tolist()
            first_theme = dominant[0]
            last_theme = dominant[-1]
            switch_count = sum(1 for left, right in zip(dominant, dominant[1:]) if left != right)
            first_vector = group.iloc[0][theme_columns].astype(float).to_numpy()
            last_vector = group.iloc[-1][theme_columns].astype(float).to_numpy()
            rows.append(
                {
                    "view": view,
                    "ticker": ticker,
                    "date_count": int(group["date"].nunique()),
                    "first_date": group["date"].iloc[0].date().isoformat(),
                    "latest_date": group["date"].iloc[-1].date().isoformat(),
                    "first_theme": first_theme,
                    "first_label": theme_label(labels, view, first_theme),
                    "latest_theme": last_theme,
                    "latest_label": theme_label(labels, view, last_theme),
                    "dominant_theme_changed": bool(first_theme != last_theme),
                    "switch_count": int(switch_count),
                    "l1_loading_shift": float(np.abs(last_vector - first_vector).sum()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.merge(metadata_frame, on="ticker", how="left")
    return result.sort_values(["l1_loading_shift", "switch_count"], ascending=False)


def extract_ai_snippets(sections: pd.DataFrame, topic_scores: pd.DataFrame) -> pd.DataFrame:
    """Extract short snippets around AI-language mentions for top-scoring firms."""
    top = topic_scores.sort_values("ai_score_per_10k_words", ascending=False).head(40)
    top_tickers = set(top["ticker"])
    rows = []
    ai_phrases = list(TOPICS["ai"].keys())
    for _, row in sections.iterrows():
        ticker = str(row["ticker"]).upper()
        if ticker not in top_tickers:
            continue
        raw_text = "\n".join(
            [
                str(row.get("business_text", "") or ""),
                str(row.get("risk_factors_text", "") or ""),
                str(row.get("mda_text", "") or ""),
            ]
        )
        snippets = snippets_for_phrases(raw_text, ai_phrases, limit=2)
        for rank, snippet in enumerate(snippets, start=1):
            rows.append({"ticker": ticker, "snippet_rank": rank, "snippet": snippet})
    return pd.DataFrame(rows)


def snippets_for_phrases(text: str, phrases: list[str], limit: int) -> list[str]:
    """Return short snippets around phrase hits."""
    lower = text.lower()
    hits = []
    for phrase in phrases:
        pattern = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
        for match in re.finditer(pattern, lower):
            hits.append((match.start(), match.end()))
    hits = sorted(hits)[:limit]
    snippets = []
    for start, end in hits:
        left = max(0, start - 180)
        right = min(len(text), end + 220)
        snippet = re.sub(r"\s+", " ", text[left:right]).strip()
        snippets.append(snippet)
    return snippets


def latest_by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row for each ticker."""
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).reset_index(drop=True)


def theme_cols(frame: pd.DataFrame) -> list[str]:
    """Return theme loading columns."""
    return [column for column in frame.columns if column.startswith("theme_")]


def theme_label(labels: pd.DataFrame, view: str, theme: str) -> str:
    """Return a manual or seeded label for a view theme."""
    if labels.empty:
        return ""
    match = labels[(labels["view"] == view) & (labels["theme"] == theme)]
    if match.empty:
        return ""
    return str(match.iloc[0]["label"])


def top_loaded_firms(frame: pd.DataFrame, theme: str, k: int) -> str:
    """Return a compact comma-separated top firm list for a theme."""
    tickers = frame.sort_values(theme, ascending=False).head(k)["ticker"].astype(str).tolist()
    return ", ".join(tickers)


def top_loaded_sectors(frame: pd.DataFrame, theme: str, k: int) -> str:
    """Return top sectors by soft loading mass."""
    totals = frame.groupby("gics_sector", dropna=True)[theme].sum().sort_values(ascending=False).head(k)
    return ", ".join(f"{sector} {value:.1f}" for sector, value in totals.items())


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Return a finite weighted mean."""
    clean_values = values.fillna(0.0).astype(float)
    clean_weights = weights.fillna(0.0).astype(float)
    denominator = float(clean_weights.sum())
    if denominator <= 0.0:
        return 0.0
    return float((clean_values * clean_weights).sum() / denominator)


def annualized_slope(values: pd.Series) -> float:
    """Return a simple linear slope in mass units per year."""
    if len(values) < 2:
        return 0.0
    dates = pd.to_datetime(values.index)
    x = (dates - dates.min()).days.to_numpy(dtype=float) / 365.25
    y = values.to_numpy(dtype=float)
    if np.allclose(x.max(), x.min()):
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    if math.isnan(float(slope)):
        return 0.0
    return float(slope)


def evolution_markdown(
    experiment_root: Path,
    topic_scores: pd.DataFrame,
    theme_topic_summary: pd.DataFrame,
    theme_mass_trends: pd.DataFrame,
    migrations: pd.DataFrame,
    ai_snippets: pd.DataFrame,
) -> str:
    """Format the interpretation report."""
    lines = [
        "# Theme Evolution Patterns",
        "",
        f"Experiment: `{experiment_root}`",
        "",
        "## What This Can And Cannot Show",
        "",
        "The business and network views are now point-in-time honest, but the parsed 10-K section table currently has one filing per company. That means this report can identify AI-heavy latest filings and AI-heavy business themes, but it cannot yet prove a multi-year AI pivot for one company. For that, we need older 10-K sections for the same tickers.",
        "",
        "The behavioral and growth views do have long monthly histories, so their migration tables are real time-series movement.",
        "",
    ]

    lines.extend(["## AI-Heavy Business Themes", ""])
    theme_columns = [
        "theme",
        "label",
        "ai_weighted_score",
        "cloud_compute_weighted_score",
        "top_firms",
        "top_sectors",
    ]
    lines.extend(markdown_table(theme_topic_summary.head(12), theme_columns))

    lines.extend(["", "## AI-Heavy Companies", ""])
    company_columns = [
        "ticker",
        "company_name",
        "gics_sector",
        "ai_score_per_10k_words",
        "ai_mentions",
        "cloud_compute_score_per_10k_words",
    ]
    top_companies = topic_scores.sort_values("ai_score_per_10k_words", ascending=False).head(20)
    lines.extend(markdown_table(top_companies, company_columns))

    lines.extend(["", "## Fastest Theme Mass Changes", ""])
    trend_columns = [
        "view",
        "theme",
        "label",
        "date_count",
        "start_date",
        "end_date",
        "mass_change",
        "trend_slope_per_year",
    ]
    long_history = theme_mass_trends[theme_mass_trends["date_count"] >= 24]
    lines.extend(markdown_table(long_history.head(15), trend_columns))

    lines.extend(["", "## Largest Company Theme Migrations", ""])
    migration_columns = [
        "view",
        "ticker",
        "company_name",
        "gics_sector",
        "first_theme",
        "latest_theme",
        "switch_count",
        "l1_loading_shift",
    ]
    long_migrations = migrations[migrations["date_count"] >= 24]
    lines.extend(markdown_table(long_migrations.head(20), migration_columns))

    lines.extend(["", "## Example AI Filing Snippets", ""])
    for _, row in ai_snippets.head(12).iterrows():
        snippet = str(row["snippet"]).replace("|", "\\|")
        lines.extend([f"**{row['ticker']}**: {snippet}", ""])

    lines.extend(
        [
            "## Files Written",
            "",
            "- `report/topic_scores_by_company.csv`",
            "- `report/theme_topic_summary.csv`",
            "- `report/view_theme_mass_trends.csv`",
            "- `report/ticker_theme_migrations.csv`",
            "- `report/ai_filing_snippets.csv`",
        ]
    )
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Format a small dataframe as a markdown table."""
    if frame.empty:
        return ["No rows available."]
    usable_columns = [column for column in columns if column in frame.columns]
    lines = ["| " + " | ".join(usable_columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(usable_columns)) + " |")
    for _, row in frame.loc[:, usable_columns].iterrows():
        values = [format_cell(row[column]) for column in usable_columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def format_cell(value: object) -> str:
    """Format a markdown table cell."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else "report",
    )
