from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
PACKAGE_SRC = REPO_ROOT / "libraries" / "market_data_fetcher" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from market_data_fetcher import DatabaseOperator
from src.applications.sector_relative_outlook import sector_backtest_by_date, sector_outlook_backtest
from src.config import load_config
from src.db import FilingsDB


DEFAULT_STORAGE_DIR = REPO_ROOT / "data" / "raw_filing_corpora"
DEFAULT_IDENTITY = "StockCorrelation/0.1 castellanosmarcelo1@gmail.com"
DEFAULT_HISTORICAL_TEXT_DIR = REPO_ROOT / "data" / "processed" / "historical_text_10k_10q"
FALLBACK_HISTORICAL_TEXT_DIR = REPO_ROOT / "data" / "processed" / "historical_text"
TOPIC_OPTIONS = {
    "AI": "topic_ai_score_per_10k_words",
    "Cloud / Compute": "topic_cloud_compute_score_per_10k_words",
    "Cybersecurity": "topic_cybersecurity_score_per_10k_words",
    "Supply Chain": "topic_supply_chain_score_per_10k_words",
    "Electrification": "topic_electrification_score_per_10k_words",
}
MENTION_OPTIONS = {
    "AI": "topic_ai_mentions",
    "Cloud / Compute": "topic_cloud_compute_mentions",
    "Cybersecurity": "topic_cybersecurity_mentions",
    "Supply Chain": "topic_supply_chain_mentions",
    "Electrification": "topic_electrification_mentions",
}
FRAGMENT_LABEL_SECTIONS = [
    "business",
    "risk_factors",
    "mda",
    "item_1c_cybersecurity",
    "q_mda",
    "q_risk_factors",
]


st.set_page_config(page_title="Stock Embeddings Dashboard", layout="wide")
st.title("Stock Embeddings Dashboard")
st.caption("Browse filings and explore how companies move through similarity categories and filing-language themes.")


@st.cache_resource
def get_operator(storage_dir: str) -> DatabaseOperator:
    return DatabaseOperator(
        identity=DEFAULT_IDENTITY,
        storage_dir=storage_dir,
        default_format="jsonl",
    )


@st.cache_data(show_spinner=False)
def load_loadings(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_projected_market_map(path: str) -> pd.DataFrame:
    """Project all firm-date theme loadings into one stable 2D map."""
    frame = load_loadings(path)
    columns = theme_columns(frame)
    if len(columns) < 2:
        raise ValueError("Market map requires at least two theme columns.")

    matrix = frame.loc[:, columns].astype(float).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    projection = PCA(n_components=2, random_state=7).fit_transform(matrix)
    output = frame.loc[:, ["ticker", "date"]].copy()
    output["x"] = projection[:, 0]
    output["y"] = projection[:, 1]
    output["dominant_theme"] = frame.loc[:, columns].astype(float).idxmax(axis=1)
    output["dominant_loading"] = frame.loc[:, columns].astype(float).max(axis=1)
    return output


@st.cache_data(show_spinner=False)
def load_historical_topic_counts(directory: str) -> pd.DataFrame:
    """Load compact historical section-level topic counts."""
    path = Path(directory) / "historical_section_topic_counts.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    frame["year"] = frame["filing_date"].dt.year
    return frame.sort_values(["ticker", "filing_date", "section"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_historical_filing_index(directory: str) -> pd.DataFrame:
    """Load compact historical filing metadata."""
    path = Path(directory) / "historical_filing_index.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    return frame.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_historical_embeddings(directory: str) -> pd.DataFrame:
    """Load compact historical section embeddings."""
    path = Path(directory) / "historical_section_embeddings.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    return frame.sort_values(["ticker", "filing_date", "section"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_historical_snippets(directory: str) -> pd.DataFrame:
    """Load compact keyword-centered evidence snippets when available."""
    path = Path(directory) / "historical_section_snippets.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    if "period_end" in frame.columns:
        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    return frame.sort_values(["ticker", "filing_date", "section", "snippet_rank"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_historical_manifest(directory: str) -> dict:
    """Load compact historical text manifest when present."""
    path = Path(directory) / "historical_text_manifest.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def default_historical_text_dir() -> Path:
    """Return the richest local historical text artifact directory."""
    if DEFAULT_HISTORICAL_TEXT_DIR.exists():
        return DEFAULT_HISTORICAL_TEXT_DIR
    return FALLBACK_HISTORICAL_TEXT_DIR


def available_historical_text_dirs() -> list[Path]:
    """Return historical text artifact directories that look usable."""
    processed = REPO_ROOT / "data" / "processed"
    candidates = [
        DEFAULT_HISTORICAL_TEXT_DIR,
        FALLBACK_HISTORICAL_TEXT_DIR,
        *(path for path in processed.glob("historical_text*") if path.is_dir()),
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (path / "historical_section_topic_counts.parquet").exists():
            unique.append(path)
    return unique


@st.cache_data(show_spinner=False)
def load_projected_historical_embeddings(directory: str, section: str) -> pd.DataFrame:
    """Project historical section embeddings into one 2D semantic map."""
    embeddings = load_historical_embeddings(directory)
    if embeddings.empty:
        return embeddings
    frame = embeddings[embeddings["section"] == section].copy()
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    if len(embedding_columns) < 2 or len(frame) < 3:
        return pd.DataFrame()
    matrix = frame.loc[:, embedding_columns].astype(float).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    projection = PCA(n_components=2, random_state=7).fit_transform(matrix)
    output = frame.loc[
        :,
        ["ticker", "filing_date", "period_end", "section", "section_label", "section_chars"],
    ].copy()
    output["year"] = output["filing_date"].dt.year
    output["x"] = projection[:, 0]
    output["y"] = projection[:, 1]
    return output


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    metadata_paths = sorted((REPO_ROOT / "data" / "raw_filing_corpora").glob("*/*_tickers.parquet"))
    if metadata_paths:
        frame = pd.read_parquet(metadata_paths[-1])
    else:
        dataset_path = REPO_ROOT / "data" / "processed" / "datasets" / "decomposed.parquet"
        if dataset_path.exists():
            dataset = pd.read_parquet(dataset_path, columns=["ticker", "title", "search_label"])
            frame = dataset.drop_duplicates("ticker")
        else:
            frame = pd.DataFrame(columns=["ticker", "title", "search_label"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "title" not in frame.columns:
        frame["title"] = frame["ticker"]
    if "search_label" not in frame.columns:
        frame["search_label"] = frame["ticker"] + " - " + frame["title"].astype(str)

    gics_path = REPO_ROOT / "data" / "processed" / "metadata" / "sp500_gics.parquet"
    if gics_path.exists():
        gics = pd.read_parquet(gics_path)
        gics["ticker"] = gics["ticker"].astype(str).str.upper()
        keep_columns = [
            column
            for column in ["ticker", "company_name", "gics_sector", "gics_sub_industry"]
            if column in gics.columns
        ]
        frame = frame.merge(gics.loc[:, keep_columns], on="ticker", how="left")
        if "company_name" in frame.columns:
            frame["title"] = frame["title"].fillna(frame["company_name"])
    frame = coalesce_metadata_columns(frame)
    return frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def coalesce_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata columns after optional merges with overlapping names."""
    result = frame.copy()
    for column in ["company_name", "gics_sector", "gics_sub_industry"]:
        variants = [name for name in [column, f"{column}_x", f"{column}_y"] if name in result.columns]
        if not variants:
            result[column] = pd.NA
            continue
        result[column] = result[variants].bfill(axis=1).iloc[:, 0]
    if "title" not in result.columns:
        result["title"] = result["company_name"].fillna(result["ticker"])
    else:
        result["title"] = result["title"].fillna(result["company_name"]).fillna(result["ticker"])
    if "search_label" not in result.columns:
        result["search_label"] = result["ticker"].astype(str) + " - " + result["title"].astype(str)
    else:
        result["search_label"] = result["search_label"].fillna(
            result["ticker"].astype(str) + " - " + result["title"].astype(str)
        )
    drop_columns = [
        name
        for name in result.columns
        if name.endswith("_x") or name.endswith("_y")
    ]
    if drop_columns:
        result = result.drop(columns=drop_columns)
    return result


@st.cache_data(show_spinner=False)
def auto_theme_label_lookup(loadings_path: str, metadata: pd.DataFrame, view_name: str) -> dict[str, str]:
    """Create readable fallback labels from sector mix, without manual labels or ticker lists."""
    loadings = load_loadings(loadings_path)
    latest = loadings.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest = latest.merge(metadata, on="ticker", how="left")
    lookup = {}
    for theme in theme_columns(latest):
        top = latest.sort_values(theme, ascending=False).head(8)
        if "gics_sector" in top.columns and top["gics_sector"].notna().any():
            sector = str(top["gics_sector"].dropna().mode().iloc[0])
            label = f"{sector} {view_label_noun(view_name)} mix"
        else:
            label = f"{view_label_noun(view_name).title()} mix"
        lookup[theme] = label
    return lookup


@st.cache_data(show_spinner=False)
def fragment_theme_label_artifacts(
    loadings_path: str,
    historical_dir: str,
    view_name: str,
    as_of_date: str | None,
    top_tickers: int = 12,
    max_age_days: int = 730,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Label business themes using representative historical filing fragments.

    For each soft theme, we take the firms with the highest loading as of the
    selected date, build a weighted centroid in section-embedding space, then
    find the filing section fragment closest to that centroid. The label uses
    that representative fragment's strongest tracked topics.
    """
    if view_name != "business":
        return {}, pd.DataFrame()

    loadings = load_loadings(loadings_path)
    columns = theme_columns(loadings)
    if not columns:
        return {}, pd.DataFrame()

    as_of_loadings = loadings_as_of(loadings, as_of_date)
    if as_of_loadings.empty:
        return {}, pd.DataFrame()

    embeddings = load_historical_embeddings(historical_dir)
    if embeddings.empty:
        return {}, pd.DataFrame()
    embeddings = embeddings[embeddings["section"].isin(FRAGMENT_LABEL_SECTIONS)].copy()
    embedding_columns = [column for column in embeddings.columns if column.startswith("embedding_")]
    if not embedding_columns:
        return {}, pd.DataFrame()

    topics = load_historical_topic_counts(historical_dir)
    topic_lookup = pd.DataFrame()
    if not topics.empty:
        topic_lookup = topics.set_index(["ticker", "accession_no", "section"], drop=False)
    filing_index = load_historical_filing_index(historical_dir)
    source_lookup = pd.DataFrame()
    if not filing_index.empty:
        source_lookup = filing_index.set_index(["ticker", "accession_no"], drop=False)
    snippets = load_historical_snippets(historical_dir)
    snippet_lookup = pd.DataFrame()
    if not snippets.empty:
        snippet_lookup = snippets.set_index(["ticker", "accession_no", "section"], drop=False)

    date_cutoff = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(as_of_loadings["date"].max())
    rows: list[dict[str, object]] = []
    labels: dict[str, str] = {}
    for theme in columns:
        top = as_of_loadings.sort_values(theme, ascending=False).head(int(top_tickers))
        top = top.loc[top[theme].astype(float) > 0.0].copy()
        if top.empty:
            continue
        weights = dict(zip(top["ticker"], top[theme].astype(float), strict=False))
        candidates = embeddings[embeddings["ticker"].isin(weights)].copy()
        candidates = candidates[candidates["filing_date"] <= date_cutoff]
        recent = candidates[candidates["filing_date"] >= date_cutoff - pd.Timedelta(days=int(max_age_days))]
        if not recent.empty:
            candidates = recent
        if candidates.empty:
            continue

        candidates["theme_weight"] = candidates["ticker"].map(weights).astype(float)
        matrix = candidates.loc[:, embedding_columns].astype(float).to_numpy()
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix_norm = np.linalg.norm(matrix, axis=1)
        valid = matrix_norm > 0.0
        if not valid.any():
            continue
        candidates = candidates.loc[valid].copy()
        matrix = matrix[valid]
        matrix_norm = matrix_norm[valid]
        normalized = matrix / matrix_norm[:, None]
        weight_values = candidates["theme_weight"].astype(float).to_numpy()
        centroid = np.average(normalized, axis=0, weights=weight_values)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0.0:
            continue
        similarity = normalized @ (centroid / centroid_norm)
        best_position = int(np.argmax(similarity))
        representative = candidates.iloc[best_position]
        topic_row = representative_topic_row(topic_lookup, representative)
        filing_row = representative_filing_row(source_lookup, representative)
        snippet_row = representative_snippet_row(snippet_lookup, representative)
        topic_label = fragment_topic_label(topic_row)
        section_label = short_section_label(str(representative.get("section_label", representative["section"])))
        label_core = fragment_display_label(topic_label, section_label)
        top_theme_tickers = ", ".join(top["ticker"].astype(str).head(4).tolist())
        label = label_core
        evidence = topic_evidence_summary(topic_row)
        labels[theme] = label
        row = {
            "theme": theme,
            "dynamic_label": label,
            "label_basis": "nearest filing fragment to weighted theme centroid",
            "representative_ticker": representative["ticker"],
            "representative_filing_date": representative["filing_date"].strftime("%Y-%m-%d"),
            "representative_form": representative.get("form", ""),
            "representative_section": section_label,
            "section_chars": int(representative.get("section_chars", 0)),
            "word_count": int(topic_row.get("word_count", 0)) if not topic_row.empty else None,
            "fragment_similarity": float(similarity[best_position]),
            "fragment_topics": topic_label or "no tracked topic dominates",
            "topic_evidence": evidence,
            "snippet_topic": str(snippet_row.get("snippet_topic", "")) if not snippet_row.empty else "",
            "snippet_terms": str(snippet_row.get("snippet_terms", "")) if not snippet_row.empty else "",
            "evidence_snippet": str(snippet_row.get("evidence_snippet", "")) if not snippet_row.empty else "",
            "top_theme_tickers": top_theme_tickers,
            "source_url": str(filing_row.get("source_url", "")) if not filing_row.empty else "",
        }
        row.update(topic_metric_values(topic_row))
        rows.append(row)

    return labels, pd.DataFrame(rows)


def view_label_noun(view_name: str) -> str:
    """Return the kind of similarity represented by a non-business view."""
    names = {
        "behavioral": "trading-behavior",
        "growth": "growth/lifecycle",
        "network": "relationship-network",
    }
    return names.get(view_name, "similarity")


def fragment_display_label(topic_label: str, section_label: str) -> str:
    """Return a compact semantic label for a representative filing fragment."""
    if topic_label:
        return f"{topic_label} language"
    section_labels = {
        "Business": "Business model language",
        "Risk Factors": "Risk language",
        "MD&A": "Management discussion language",
        "Cybersecurity": "Cybersecurity language",
        "Q MD&A": "Quarterly management discussion",
        "Q Risk Factors": "Quarterly risk language",
    }
    return section_labels.get(section_label, f"{section_label} language")


def loadings_as_of(loadings: pd.DataFrame, as_of_date: str | None) -> pd.DataFrame:
    """Return loadings for a selected date or latest loadings per ticker."""
    frame = loadings.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if as_of_date:
        date = pd.Timestamp(as_of_date)
        exact = frame[frame["date"] == date].copy()
        if not exact.empty:
            return exact
        frame = frame[frame["date"] <= date]
        if frame.empty:
            return frame
    return frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)


def representative_topic_row(topic_lookup: pd.DataFrame, representative: pd.Series) -> pd.Series:
    """Return topic counts for a representative embedding row if available."""
    if topic_lookup.empty:
        return pd.Series(dtype=object)
    key = (
        str(representative["ticker"]),
        str(representative.get("accession_no", "")),
        str(representative["section"]),
    )
    if key not in topic_lookup.index:
        return pd.Series(dtype=object)
    row = topic_lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def representative_filing_row(source_lookup: pd.DataFrame, representative: pd.Series) -> pd.Series:
    """Return filing-index metadata for a representative embedding row if available."""
    if source_lookup.empty:
        return pd.Series(dtype=object)
    key = (
        str(representative["ticker"]),
        str(representative.get("accession_no", "")),
    )
    if key not in source_lookup.index:
        return pd.Series(dtype=object)
    row = source_lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def representative_snippet_row(snippet_lookup: pd.DataFrame, representative: pd.Series) -> pd.Series:
    """Return the strongest stored snippet for a representative embedding row."""
    if snippet_lookup.empty:
        return pd.Series(dtype=object)
    key = (
        str(representative["ticker"]),
        str(representative.get("accession_no", "")),
        str(representative["section"]),
    )
    if key not in snippet_lookup.index:
        return pd.Series(dtype=object)
    row = snippet_lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        return row.sort_values("snippet_rank").iloc[0]
    return row


def fragment_topic_label(topic_row: pd.Series) -> str:
    """Return a short label from the strongest tracked topics in one fragment."""
    if topic_row.empty:
        return ""
    scores = []
    for label, column in TOPIC_OPTIONS.items():
        if column not in topic_row or pd.isna(topic_row[column]):
            continue
        score = float(topic_row[column])
        if score > 0.0:
            scores.append((label, score))
    if not scores:
        return ""
    return " / ".join(label for label, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:2])


def topic_evidence_summary(topic_row: pd.Series) -> str:
    """Return a compact topic-score explanation for a representative fragment."""
    if topic_row.empty:
        return "no topic-count row available"
    parts = []
    for label, score_column in TOPIC_OPTIONS.items():
        if score_column not in topic_row or pd.isna(topic_row[score_column]):
            continue
        score = float(topic_row[score_column])
        if score <= 0.0:
            continue
        mention_column = MENTION_OPTIONS[label]
        mentions = int(topic_row.get(mention_column, 0)) if mention_column in topic_row else 0
        parts.append((label, score, mentions))
    if not parts:
        return "no tracked topic dominates"
    parts = sorted(parts, key=lambda item: item[1], reverse=True)
    return "; ".join(f"{label}: {score:.2f}/10k words ({mentions} mentions)" for label, score, mentions in parts)


def topic_metric_values(topic_row: pd.Series) -> dict[str, float | int | None]:
    """Return topic scores and mentions as flat evidence columns."""
    values: dict[str, float | int | None] = {}
    for label, score_column in TOPIC_OPTIONS.items():
        key = label.lower().replace(" / ", "_").replace(" ", "_")
        mention_column = MENTION_OPTIONS[label]
        values[f"{key}_score_per_10k"] = (
            float(topic_row[score_column])
            if not topic_row.empty and score_column in topic_row and pd.notna(topic_row[score_column])
            else None
        )
        values[f"{key}_mentions"] = (
            int(topic_row[mention_column])
            if not topic_row.empty and mention_column in topic_row and pd.notna(topic_row[mention_column])
            else None
        )
    return values


def short_section_label(label: str) -> str:
    """Shorten verbose filing-section names for compact theme labels."""
    replacements = {
        "10-K Item 1 Business": "Business",
        "10-K Item 1A Risk Factors": "Risk Factors",
        "10-K Item 7 MD&A": "MD&A",
        "10-K Item 1C Cybersecurity": "Cybersecurity",
        "10-Q Item 2 MD&A": "Q MD&A",
        "10-Q Part II Item 1A Risk Factors": "Q Risk Factors",
    }
    return replacements.get(label, label.replace("10-K ", "").replace("10-Q ", ""))


def latest_decomposed_experiment() -> Path | None:
    candidates = []
    for root in sorted((REPO_ROOT / "experiments").glob("*decomposed*"), reverse=True):
        views_dir = root / "views"
        if views_dir.exists() and any(views_dir.glob("*/loadings.parquet")):
            candidates.append(root)
    return candidates[0] if candidates else None


def available_views(experiment_root: Path) -> list[str]:
    views_dir = experiment_root / "views"
    if not views_dir.exists():
        return []
    return sorted(path.name for path in views_dir.iterdir() if (path / "loadings.parquet").exists())


def parse_tickers(raw_value: str) -> list[str]:
    parts = [part.strip().upper() for part in raw_value.replace("\n", ",").split(",")]
    seen: set[str] = set()
    tickers: list[str] = []
    for ticker in parts:
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def safe_frame_subset(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame
    return frame.loc[:, available]


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy that has all requested columns, filling missing ones with NA."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def theme_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("theme_")]


def selected_theme_columns(company_history: pd.DataFrame, k: int, ranking: str) -> list[str]:
    columns = theme_columns(company_history)
    if ranking == "Latest loading":
        scores = company_history.iloc[-1][columns].astype(float)
    elif ranking == "Biggest movement":
        scores = company_history[columns].max(axis=0) - company_history[columns].min(axis=0)
    else:
        scores = company_history[columns].mean(axis=0)
    return scores.sort_values(ascending=False).head(k).index.tolist()


def bounded_state_value(key: str, default: int, minimum: int, maximum: int) -> int:
    """Return a Streamlit integer state value clamped to a valid range."""
    if key not in st.session_state:
        st.session_state[key] = int(default)
    value = int(st.session_state[key])
    value = max(int(minimum), min(int(maximum), value))
    st.session_state[key] = value
    return value


def move_state_value(key: str, delta: int, minimum: int, maximum: int) -> None:
    """Move a Streamlit integer state value by a bounded delta."""
    current = bounded_state_value(key, default=maximum, minimum=minimum, maximum=maximum)
    st.session_state[key] = max(int(minimum), min(int(maximum), current + int(delta)))


def display_theme_name(theme: str, labels: dict[str, str]) -> str:
    """Return a human label when available, otherwise a readable theme id."""
    return labels.get(theme, theme.replace("theme_", "Theme "))


def unique_theme_label_lookup(labels: dict[str, str], columns: list[str]) -> dict[str, str]:
    """Make duplicate dynamic labels distinct without falling back to ticker lists."""
    counts: dict[str, int] = {}
    unique: dict[str, str] = {}
    for column in columns:
        label = labels.get(column, column.replace("theme_", "Theme "))
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1:
            unique[column] = label
        else:
            unique[column] = f"{label} {counts[label]}"
    return unique


def theme_evidence_frame(fragment_explanations: pd.DataFrame, themes: list[str] | set[str]) -> pd.DataFrame:
    """Return a compact evidence table for selected dynamic business labels."""
    if fragment_explanations.empty:
        return fragment_explanations
    theme_set = {str(theme) for theme in themes}
    frame = fragment_explanations[fragment_explanations["theme"].astype(str).isin(theme_set)].copy()
    columns = [
        "theme",
        "dynamic_label",
        "representative_ticker",
        "representative_filing_date",
        "representative_form",
        "representative_section",
        "fragment_similarity",
        "topic_evidence",
        "evidence_snippet",
        "snippet_terms",
        "top_theme_tickers",
        "section_chars",
        "word_count",
        "source_url",
    ]
    frame = ensure_columns(frame, columns)
    return safe_frame_subset(frame, columns).sort_values("theme").reset_index(drop=True)


def loading_bar_frame(row: pd.Series, columns: list[str], labels: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "theme": [display_theme_name(column, labels) for column in columns],
            "loading": [float(row[column]) for column in columns],
        }
    )
    return frame.sort_values("loading", ascending=False).set_index("theme")


def theme_loading_long_frame(
    company_history: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
) -> pd.DataFrame:
    """Return long-form labeled theme loadings for charts."""
    rows = []
    for _, row in company_history.iterrows():
        for column in columns:
            rows.append(
                {
                    "date": row["date"],
                    "date_label": row["date"].strftime("%Y-%m-%d"),
                    "theme_id": column,
                    "theme_label": display_theme_name(column, labels),
                    "loading": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def current_theme_loading_chart(row: pd.Series, columns: list[str], labels: dict[str, str]) -> alt.Chart:
    """Build a labeled current theme-loading bar chart."""
    frame = pd.DataFrame(
        {
            "theme_id": columns,
            "theme_label": [display_theme_name(column, labels) for column in columns],
            "loading": [float(row[column]) for column in columns],
        }
    ).sort_values("loading", ascending=False)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("loading:Q", title="Loading"),
            y=alt.Y("theme_label:N", title="Theme", sort="-x"),
            color=alt.Color("theme_label:N", title="Theme", legend=None),
            tooltip=[
                alt.Tooltip("theme_label:N", title="Theme"),
                alt.Tooltip("theme_id:N", title="Theme id"),
                alt.Tooltip("loading:Q", title="Loading", format=".3f"),
            ],
        )
        .properties(height=max(280, 28 * len(frame)))
    )


def theme_loading_timeline_chart(frame: pd.DataFrame) -> alt.Chart:
    """Build a labeled multi-theme loading timeline chart."""
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("loading:Q", title="Theme loading"),
            color=alt.Color("theme_label:N", title="Theme"),
            tooltip=[
                alt.Tooltip("date_label:N", title="Date"),
                alt.Tooltip("theme_label:N", title="Theme"),
                alt.Tooltip("theme_id:N", title="Theme id"),
                alt.Tooltip("loading:Q", title="Loading", format=".3f"),
            ],
        )
        .properties(height=360)
    )


def movement_frame(
    company_history: pd.DataFrame,
    columns: list[str],
    date_index: int,
    labels: dict[str, str],
) -> pd.DataFrame:
    current = company_history.iloc[date_index]
    if date_index == 0:
        baseline = company_history.iloc[0]
        comparison = "start"
    else:
        baseline = company_history.iloc[date_index - 1]
        comparison = "previous date"
    rows = []
    for column in columns:
        delta = float(current[column] - baseline[column])
        rows.append(
            {
                "theme": display_theme_name(column, labels),
                "loading": float(current[column]),
                f"change_vs_{comparison.replace(' ', '_')}": delta,
            }
        )
    return pd.DataFrame(rows).sort_values("loading", ascending=False)


def market_date_frame(
    projected: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_date: pd.Timestamp,
    labels: dict[str, str],
    k_groups: int,
) -> pd.DataFrame:
    """Return one date of map points with display labels and top-k color groups."""
    date_frame = projected[projected["date"] == selected_date].copy()
    if date_frame.empty:
        return date_frame

    theme_strength = date_frame.groupby("dominant_theme")["dominant_loading"].sum().sort_values(ascending=False)
    top_themes = set(theme_strength.head(k_groups).index.astype(str))
    date_frame["theme_label"] = [
        display_theme_name(theme, labels) if theme in top_themes else "Other themes"
        for theme in date_frame["dominant_theme"].astype(str)
    ]
    metadata_subset = metadata.loc[:, ["ticker", "title", "search_label"]].copy()
    date_frame = date_frame.merge(metadata_subset, on="ticker", how="left")
    date_frame["title"] = date_frame["title"].fillna(date_frame["ticker"])
    date_frame["search_label"] = date_frame["search_label"].fillna(date_frame["ticker"])
    return date_frame


def map_axis_domains(projected: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Return padded global axis domains so the map does not rescale per date."""
    x_min = float(projected["x"].min())
    x_max = float(projected["x"].max())
    y_min = float(projected["y"].min())
    y_max = float(projected["y"].max())
    x_pad = max((x_max - x_min) * 0.06, 0.01)
    y_pad = max((y_max - y_min) * 0.06, 0.01)
    return [x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]


def usable_market_dates(projected: pd.DataFrame, *, hide_collapsed: bool) -> list[pd.Timestamp]:
    """Return dates with enough cross-sectional spread for a readable market map."""
    stats = (
        projected.groupby("date")
        .agg(
            n_tickers=("ticker", "nunique"),
            x_std=("x", "std"),
            y_std=("y", "std"),
        )
        .reset_index()
    )
    stats["spread"] = stats["x_std"].fillna(0.0) + stats["y_std"].fillna(0.0)
    if hide_collapsed:
        global_spread = float(projected["x"].std() + projected["y"].std())
        min_spread = max(global_spread * 0.05, 0.01)
        stats = stats[(stats["spread"] >= min_spread) & (stats["n_tickers"] >= 200)]
    return [pd.Timestamp(date) for date in sorted(stats["date"].tolist())]


def market_map_chart(
    points: pd.DataFrame,
    highlighted: list[str],
    x_domain: list[float],
    y_domain: list[float],
) -> alt.Chart:
    """Build an Altair scatter map for one market date."""
    base = (
        alt.Chart(points)
        .mark_circle(opacity=0.82)
        .encode(
            x=alt.X("x:Q", title="Similarity map X", scale=alt.Scale(domain=x_domain, nice=False)),
            y=alt.Y("y:Q", title="Similarity map Y", scale=alt.Scale(domain=y_domain, nice=False)),
            color=alt.Color("theme_label:N", title="Dominant theme"),
            size=alt.Size("dominant_loading:Q", title="Dominant loading", scale=alt.Scale(range=[35, 210])),
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("title:N", title="Company"),
                alt.Tooltip("theme_label:N", title="Dominant theme"),
                alt.Tooltip("dominant_theme:N", title="Theme id"),
                alt.Tooltip("dominant_loading:Q", title="Loading", format=".3f"),
            ],
        )
    )
    if not highlighted:
        return base.properties(height=620)

    highlight_frame = points[points["ticker"].isin(highlighted)]
    labels = (
        alt.Chart(highlight_frame)
        .mark_text(align="left", dx=8, dy=-8, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_domain, nice=False)),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain, nice=False)),
            text="ticker:N",
            color=alt.value("#111111"),
        )
    )
    rings = (
        alt.Chart(highlight_frame)
        .mark_circle(size=360, opacity=1.0, fillOpacity=0.0, strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_domain, nice=False)),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain, nice=False)),
            color=alt.value("#111111"),
        )
    )
    return (base + rings + labels).properties(height=620)


def market_trail_chart(
    projected: pd.DataFrame,
    selected_date: pd.Timestamp,
    highlighted: list[str],
    months: int,
    labels: dict[str, str],
) -> alt.Chart:
    """Build movement trails for selected tickers up to the selected date."""
    if not highlighted:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
    dates = [pd.Timestamp(date) for date in sorted(projected["date"].unique())]
    date_position = dates.index(pd.Timestamp(selected_date))
    start_position = max(0, date_position - int(months))
    trail_dates = dates[start_position : date_position + 1]
    trail = projected[
        projected["ticker"].isin(highlighted)
        & projected["date"].isin(trail_dates)
    ].copy()
    trail["date_label"] = trail["date"].dt.strftime("%Y-%m-%d")
    trail["theme_label"] = [
        display_theme_name(theme, labels)
        for theme in trail["dominant_theme"].astype(str)
    ]
    return (
        alt.Chart(trail)
        .mark_line(point=True)
        .encode(
            x=alt.X("x:Q", title="Similarity map X"),
            y=alt.Y("y:Q", title="Similarity map Y"),
            color=alt.Color("ticker:N", title="Ticker"),
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("date_label:N", title="Date"),
                alt.Tooltip("theme_label:N", title="Dominant theme"),
                alt.Tooltip("dominant_theme:N", title="Theme id"),
                alt.Tooltip("dominant_loading:Q", title="Loading", format=".3f"),
                alt.Tooltip("x:Q", format=".3f"),
                alt.Tooltip("y:Q", format=".3f"),
            ],
        )
        .properties(height=260)
    )


def topic_yearly_trend(counts: pd.DataFrame, sections: list[str], topic_column: str) -> pd.DataFrame:
    """Return yearly topic scores by section for the whole universe."""
    frame = counts[counts["section"].isin(sections)].copy()
    if frame.empty:
        return frame
    return (
        frame.groupby(["year", "section_label"], as_index=False)
        .agg(
            mean_score=(topic_column, "mean"),
            median_score=(topic_column, "median"),
            rows=("ticker", "size"),
            tickers=("ticker", "nunique"),
        )
        .sort_values(["section_label", "year"])
    )


def company_topic_history(
    counts: pd.DataFrame,
    ticker: str,
    sections: list[str],
    topic_column: str,
    mention_column: str,
) -> pd.DataFrame:
    """Return selected company topic history by filing and section."""
    frame = counts[(counts["ticker"] == ticker) & (counts["section"].isin(sections))].copy()
    if frame.empty:
        return frame
    frame["topic_score"] = frame[topic_column].astype(float)
    frame["topic_mentions"] = frame[mention_column].astype(float) if mention_column in frame.columns else 0.0
    frame["filing_date_label"] = frame["filing_date"].dt.strftime("%Y-%m-%d")
    return frame.sort_values(["filing_date", "section_label"])


def company_topic_change_table(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    sections: list[str],
    topic_column: str,
    mention_column: str,
    early_years: tuple[int, int],
    late_years: tuple[int, int],
    min_early_rows: int,
) -> pd.DataFrame:
    """Rank companies by topic score change between early and late windows."""
    frame = counts[counts["section"].isin(sections)].copy()
    if frame.empty:
        return frame
    yearly = (
        frame.groupby(["ticker", "year"], as_index=False)
        .agg(
            topic_score=(topic_column, "mean"),
            topic_mentions=(mention_column, "sum") if mention_column in frame.columns else (topic_column, "size"),
        )
        .sort_values(["ticker", "year"])
    )
    rows = []
    for ticker, group in yearly.groupby("ticker"):
        early = group[group["year"].between(early_years[0], early_years[1])]
        late = group[group["year"].between(late_years[0], late_years[1])]
        if len(early) < min_early_rows or late.empty:
            continue
        early_score = float(early["topic_score"].mean())
        late_score = float(late["topic_score"].mean())
        rows.append(
            {
                "ticker": ticker,
                "first_year": int(group["year"].min()),
                "latest_year": int(group["year"].max()),
                "early_score": early_score,
                "late_score": late_score,
                "change": late_score - early_score,
                "late_mentions": int(late["topic_mentions"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    keep = ["ticker", "title", "company_name", "gics_sector", "gics_sub_industry"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep]
    result = result.merge(metadata_subset, on="ticker", how="left")
    return result.sort_values("change", ascending=False).reset_index(drop=True)


def company_topic_delta_summary(
    counts: pd.DataFrame,
    ticker: str,
    sections: list[str],
    early_years: tuple[int, int],
    late_years: tuple[int, int],
) -> pd.DataFrame:
    """Summarize all tracked topic changes for one company."""
    frame = counts[(counts["ticker"] == ticker) & (counts["section"].isin(sections))].copy()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for topic_name, score_column in TOPIC_OPTIONS.items():
        mention_column = MENTION_OPTIONS[topic_name]
        early = frame[frame["year"].between(early_years[0], early_years[1])]
        late = frame[frame["year"].between(late_years[0], late_years[1])]
        if early.empty or late.empty or score_column not in frame.columns:
            continue
        rows.append(
            {
                "topic": topic_name,
                "early_score": float(early[score_column].mean()),
                "late_score": float(late[score_column].mean()),
                "change": float(late[score_column].mean() - early[score_column].mean()),
                "late_mentions": int(late[mention_column].sum()) if mention_column in late.columns else 0,
                "early_rows": int(len(early)),
                "late_rows": int(len(late)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("change", ascending=False).reset_index(drop=True)


def historical_embedding_drift_table(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    section: str,
    early_years: tuple[int, int],
    late_years: tuple[int, int],
    min_rows: int = 1,
) -> pd.DataFrame:
    """Rank firms by semantic embedding displacement between two windows."""
    frame = embeddings[embeddings["section"] == section].copy()
    if frame.empty:
        return pd.DataFrame()
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    if not embedding_columns:
        return pd.DataFrame()
    frame["year"] = frame["filing_date"].dt.year
    rows = []
    for ticker, group in frame.groupby("ticker"):
        early = group[group["year"].between(early_years[0], early_years[1])]
        late = group[group["year"].between(late_years[0], late_years[1])]
        if len(early) < min_rows or len(late) < min_rows:
            continue
        early_vector = early.loc[:, embedding_columns].astype(float).mean(axis=0).to_numpy()
        late_vector = late.loc[:, embedding_columns].astype(float).mean(axis=0).to_numpy()
        rows.append(
            {
                "ticker": ticker,
                "embedding_displacement": float(np.linalg.norm(late_vector - early_vector)),
                "early_rows": int(len(early)),
                "late_rows": int(len(late)),
                "first_filing": group["filing_date"].min(),
                "latest_filing": group["filing_date"].max(),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    keep = ["ticker", "title", "company_name", "gics_sector", "gics_sub_industry"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep]
    result = result.merge(metadata_subset, on="ticker", how="left")
    return result.sort_values("embedding_displacement", ascending=False).reset_index(drop=True)


def sector_topic_heatmap(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    sections: list[str],
    topic_column: str,
) -> pd.DataFrame:
    """Return sector-year topic means."""
    metadata_subset = ensure_columns(metadata, ["ticker", "gics_sector"]).loc[:, ["ticker", "gics_sector"]]
    frame = counts[counts["section"].isin(sections)].merge(
        metadata_subset,
        on="ticker",
        how="left",
    )
    frame = frame.dropna(subset=["gics_sector"])
    if frame.empty:
        return frame
    return (
        frame.groupby(["gics_sector", "year"], as_index=False)
        .agg(topic_score=(topic_column, "mean"), tickers=("ticker", "nunique"))
        .sort_values(["gics_sector", "year"])
    )


def historical_embedding_points(
    projected: pd.DataFrame,
    metadata: pd.DataFrame,
    highlighted: list[str],
    color_mode: str,
) -> pd.DataFrame:
    """Attach metadata and display fields to historical embedding map points."""
    points = projected.copy()
    keep = ["ticker", "title", "company_name", "gics_sector", "gics_sub_industry"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep]
    points = points.merge(metadata_subset, on="ticker", how="left")
    points["title"] = points["title"].fillna(points.get("company_name", points["ticker"]))
    points["is_highlighted"] = points["ticker"].isin(highlighted)
    points["filing_date_label"] = points["filing_date"].dt.strftime("%Y-%m-%d")
    if color_mode == "Year":
        points["color_group"] = points["year"].astype(str)
    elif color_mode == "Sector":
        points["color_group"] = points["gics_sector"].fillna("Unknown")
    else:
        points["color_group"] = np.where(points["is_highlighted"], points["ticker"], "Other companies")
    return points


def historical_embedding_map_chart(points: pd.DataFrame, highlighted: list[str]) -> alt.Chart:
    """Build a 2D map of historical section embeddings."""
    base = (
        alt.Chart(points)
        .mark_circle(opacity=0.35)
        .encode(
            x=alt.X("x:Q", title="Historical text embedding X"),
            y=alt.Y("y:Q", title="Historical text embedding Y"),
            color=alt.Color("color_group:N", title="Color"),
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("title:N", title="Company"),
                alt.Tooltip("filing_date_label:N", title="Filing date"),
                alt.Tooltip("section_label:N", title="Section"),
                alt.Tooltip("gics_sector:N", title="Sector"),
            ],
        )
    )
    if not highlighted:
        return base.properties(height=560)

    selected = points[points["ticker"].isin(highlighted)].copy()
    trail = (
        alt.Chart(selected)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("ticker:N", title="Highlighted ticker"),
            order="filing_date:T",
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("filing_date_label:N", title="Filing date"),
                alt.Tooltip("section_label:N", title="Section"),
            ],
        )
    )
    labels = (
        alt.Chart(selected.sort_values("filing_date").groupby("ticker", as_index=False).tail(1))
        .mark_text(align="left", dx=8, dy=-8, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="ticker:N", color=alt.value("#111111"))
    )
    return (base + trail + labels).properties(height=560)


def historical_company_chart(company_history: pd.DataFrame, topic_name: str) -> alt.Chart:
    """Build selected-company section topic timeline."""
    return (
        alt.Chart(company_history)
        .mark_line(point=True)
        .encode(
            x=alt.X("filing_date:T", title="Filing date"),
            y=alt.Y("topic_score:Q", title=f"{topic_name} score per 10k words"),
            color=alt.Color("section_label:N", title="Section"),
            tooltip=[
                alt.Tooltip("filing_date_label:N", title="Filing date"),
                alt.Tooltip("section_label:N", title="Section"),
                alt.Tooltip("topic_score:Q", title="Score", format=".3f"),
                alt.Tooltip("topic_mentions:Q", title="Mentions", format=".0f"),
            ],
        )
        .properties(height=340)
    )


def historical_market_trend_chart(trend: pd.DataFrame, topic_name: str) -> alt.Chart:
    """Build market-wide yearly topic trend by section."""
    return (
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("mean_score:Q", title=f"Mean {topic_name} score per 10k words"),
            color=alt.Color("section_label:N", title="Section"),
            tooltip=[
                "year:O",
                "section_label:N",
                alt.Tooltip("mean_score:Q", title="Mean score", format=".3f"),
                alt.Tooltip("median_score:Q", title="Median score", format=".3f"),
                alt.Tooltip("tickers:Q", title="Tickers", format=".0f"),
            ],
        )
        .properties(height=340)
    )


@st.cache_data(show_spinner=False)
def load_sector_outlook_artifacts(
    config_name: str,
    horizon_days: int,
    min_train_months: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Load inputs and compute sector-relative outlook artifacts."""
    config = load_config(config_name)
    db = FilingsDB.from_config(config)
    prices = db.load_prices(date_from=config["data"].get("start_date"), date_to=config["data"].get("end_date"))
    metadata = load_metadata()
    valuation = optional_feature_frame(REPO_ROOT / "data" / "processed" / "features" / "valuation.parquet")
    growth = optional_feature_frame(REPO_ROOT / "data" / "processed" / "features" / "growth_lifecycle.parquet")
    result = sector_outlook_backtest(
        prices,
        metadata,
        valuation=valuation,
        growth=growth,
        horizon_days=int(horizon_days),
        min_train_months=int(min_train_months),
        ridge_alpha=float(ridge_alpha),
    )
    return result.panel, result.predictions, result.latest, result.metrics, result.coefficients


def optional_feature_frame(path: Path) -> pd.DataFrame:
    """Load an optional feature parquet with normalized ticker/date columns."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def sector_latest_score_chart(latest: pd.DataFrame) -> alt.Chart:
    """Build current sector outlook bar chart."""
    frame = latest.copy()
    frame["direction"] = np.where(frame["predicted_excess_return"] >= 0.0, "Positive", "Negative")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("gics_sector:N", sort="-x", title="Sector"),
            x=alt.X("predicted_excess_return:Q", title="Predicted excess return vs S&P"),
            color=alt.Color("direction:N", scale=alt.Scale(range=["#1f7a4d", "#b8423f"]), legend=None),
            tooltip=[
                alt.Tooltip("gics_sector:N", title="Sector"),
                alt.Tooltip("predicted_excess_return:Q", title="Predicted excess", format=".2%"),
                alt.Tooltip("score_z:Q", title="Score z", format=".2f"),
                alt.Tooltip("prediction_rank:Q", title="Rank", format=".0f"),
            ],
        )
        .properties(height=360)
    )


def sector_backtest_chart(dated: pd.DataFrame) -> alt.Chart:
    """Build date-level sector model backtest chart."""
    frame = dated.copy()
    frame["date_label"] = frame["date"].dt.strftime("%Y-%m-%d")
    return (
        alt.Chart(frame)
        .mark_line(point=False)
        .encode(
            x=alt.X("date:T", title="Prediction date"),
            y=alt.Y("top_minus_bottom:Q", title="Top 3 minus bottom 3 realized excess"),
            tooltip=[
                alt.Tooltip("date_label:N", title="Date"),
                alt.Tooltip("top_minus_bottom:Q", title="Top-bottom", format=".2%"),
                alt.Tooltip("rank_ic:Q", title="Rank IC", format=".3f"),
                alt.Tooltip("top_bucket_excess:Q", title="Top bucket excess", format=".2%"),
            ],
        )
        .properties(height=260)
    )


def latest_coefficient_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Return latest model coefficients sorted by absolute weight."""
    if coefficients.empty:
        return coefficients
    latest_date = coefficients["date"].max()
    frame = coefficients[coefficients["date"] == latest_date].copy()
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def render_sector_outlook() -> None:
    st.subheader("Sector Relative Outlook")
    st.caption(
        "A walk-forward research signal for sector excess returns versus the equal-weight S&P 500 universe. "
        "This is a transparent backtest tool, not a trading recommendation."
    )

    controls = st.columns([1.3, 1.0, 1.0, 1.0])
    config_name = controls[0].text_input("Experiment config", value="decomposed_point_in_time")
    horizon_days = controls[1].selectbox("Forward horizon", [21, 63, 126, 252], index=1)
    min_train_months = controls[2].slider("Min training months", min_value=12, max_value=84, value=36, step=6)
    ridge_alpha = controls[3].select_slider("Ridge regularization", options=[0.1, 1.0, 3.0, 10.0, 30.0, 100.0], value=10.0)

    try:
        panel, predictions, latest, metrics, coefficients = load_sector_outlook_artifacts(
            config_name,
            int(horizon_days),
            int(min_train_months),
            float(ridge_alpha),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Sector outlook failed: {exc}")
        return

    if latest.empty:
        st.warning("Not enough completed history to fit the sector outlook model with these settings.")
        return

    metric_columns = st.columns(5)
    metric_columns[0].metric("Prediction dates", f"{int(metrics.get('n_prediction_dates', 0)):,}")
    metric_columns[1].metric("Mean rank IC", f"{float(metrics.get('mean_rank_ic', np.nan)):.3f}")
    metric_columns[2].metric("Top-bottom avg", f"{float(metrics.get('mean_top_minus_bottom', np.nan)):.2%}")
    metric_columns[3].metric("Top sector hit rate", f"{float(metrics.get('top_sector_hit_rate', np.nan)):.1%}")
    metric_columns[4].metric("Latest date", latest["date"].max().strftime("%Y-%m-%d"))

    st.markdown("**Current Sector Scores**")
    st.altair_chart(sector_latest_score_chart(latest), width="stretch")

    display_columns = [
        "prediction_rank",
        "gics_sector",
        "predicted_excess_return",
        "score_z",
        "sector_excess_momentum_63d",
        "sector_excess_momentum_126d",
        "valuation_sales_yield",
        "valuation_earnings_yield",
        "valuation_book_to_market",
        "growth_revenue_yoy_1y",
        "growth_operating_margin",
    ]
    st.dataframe(
        safe_frame_subset(ensure_columns(latest, display_columns), display_columns),
        width="stretch",
        hide_index=True,
    )

    dated = sector_backtest_by_date(predictions)
    if not dated.empty:
        st.markdown("**Walk-Forward Backtest**")
        st.altair_chart(sector_backtest_chart(dated), width="stretch")
        st.dataframe(
            dated.sort_values("date", ascending=False).head(24),
            width="stretch",
            hide_index=True,
        )

    coefficient_table = latest_coefficient_table(coefficients)
    if not coefficient_table.empty:
        with st.expander("Latest model feature weights"):
            st.dataframe(
                safe_frame_subset(coefficient_table, ["feature", "coefficient", "abs_coefficient"]),
                width="stretch",
                hide_index=True,
            )

    with st.expander("How to read this"):
        st.write(
            "`Predicted excess return` is the model's estimate of future sector return minus the equal-weight S&P 500 "
            "return over the selected horizon. Positive means the sector is scored as likely to outperform the broad "
            "universe; negative means likely to lag."
        )
        st.write(
            "The model is trained walk-forward. For each historical prediction date, it only trains on prior rows whose "
            "forward horizon had already completed. That keeps this closer to point-in-time behavior."
        )
        st.write(
            "The first version uses sector price momentum/volatility plus sector-mean growth and valuation features. "
            "Filing-language topic acceleration can be added next as another feature block."
        )


def render_historical_text() -> None:
    st.subheader("Historical Filing Language")
    st.caption(
        "Explore compact historical filing-language features: keyword counts first, optional evidence snippets, "
        "MiniLM embeddings second, with full raw SEC text discarded after processing."
    )

    artifact_dirs = available_historical_text_dirs()
    default_dir = default_historical_text_dir()
    if artifact_dirs:
        directory_labels = [str(path) for path in artifact_dirs]
        default_index = directory_labels.index(str(default_dir)) if str(default_dir) in directory_labels else 0
        selected_directory = st.selectbox("Historical text artifact directory", directory_labels, index=default_index)
        input_dir = Path(selected_directory)
    else:
        input_dir = Path(st.text_input("Historical text artifact directory", value=str(default_dir)))
    counts = load_historical_topic_counts(str(input_dir))
    if counts.empty:
        st.warning(
            "No historical text topic counts found. Run "
            "`.venv/bin/python -m scripts.historical_text.stream_features --since 2010-01-01 "
            "--forms \"10-K,10-Q\" --output-dir data/processed/historical_text_10k_10q` first."
        )
        return

    manifest = load_historical_manifest(str(input_dir))
    metadata = load_metadata()
    filing_index = load_historical_filing_index(str(input_dir))
    embeddings = load_historical_embeddings(str(input_dir))
    snippets = load_historical_snippets(str(input_dir))

    summary_columns = st.columns(6)
    summary_columns[0].metric("Filings", f"{len(filing_index):,}" if not filing_index.empty else "n/a")
    summary_columns[1].metric("Topic rows", f"{len(counts):,}")
    summary_columns[2].metric("Embedding rows", f"{len(embeddings):,}" if not embeddings.empty else "0")
    summary_columns[3].metric("Snippet rows", f"{len(snippets):,}" if not snippets.empty else "0")
    summary_columns[4].metric("Tickers", f"{counts['ticker'].nunique():,}")
    summary_columns[5].metric("Failures", str(manifest.get("failures", "n/a")))
    if manifest:
        forms = ", ".join(manifest.get("forms", [])) or "n/a"
        sections_configured = len(manifest.get("sections", []))
        st.caption(f"Loaded historical artifact: forms={forms}; configured sections={sections_configured}.")

    sections = sorted(counts["section"].dropna().unique().tolist())
    section_labels = counts.drop_duplicates("section").set_index("section")["section_label"].to_dict()
    default_sections = [section for section in ["business", "risk_factors"] if section in sections]
    controls = st.columns([1.1, 1.8, 1.1, 1.1])
    topic_name = controls[0].selectbox("Topic", list(TOPIC_OPTIONS.keys()), index=0)
    selected_sections = controls[1].multiselect(
        "Sections",
        sections,
        default=default_sections or sections[:2],
        format_func=lambda section: section_labels.get(section, section),
    )
    if not selected_sections:
        st.warning("Select at least one section.")
        return
    topic_column = TOPIC_OPTIONS[topic_name]
    mention_column = MENTION_OPTIONS[topic_name]
    min_year = int(counts["year"].min())
    max_year = int(counts["year"].max())
    early_years = controls[2].slider("Baseline years", min_year, max_year, (min_year, min(2018, max_year)))
    late_years = controls[3].slider("Recent years", min_year, max_year, (max(2023, min_year), max_year))

    trend = topic_yearly_trend(counts, selected_sections, topic_column)
    st.markdown("**Market-Wide Topic Trend**")
    st.altair_chart(historical_market_trend_chart(trend, topic_name), width="stretch")

    changes = company_topic_change_table(
        counts,
        metadata,
        selected_sections,
        topic_column,
        mention_column,
        early_years,
        late_years,
        min_early_rows=2,
    )

    ticker_options = sorted(counts["ticker"].unique().tolist())
    company_labels = metadata[metadata["ticker"].isin(ticker_options)].copy()
    label_lookup = dict(zip(company_labels["search_label"], company_labels["ticker"], strict=False))
    label_options = company_labels["search_label"].tolist()
    if not label_options:
        label_options = ticker_options
        label_lookup = {ticker: ticker for ticker in ticker_options}

    focus_options = ["Manual company selector"]
    focus_lookup = {"Manual company selector": None}
    if not changes.empty:
        for _, row in changes.head(50).iterrows():
            company = str(row.get("company_name", "") or "").strip()
            label = f"{row['ticker']} · +{float(row['change']):.2f}"
            if company:
                label += f" · {company}"
            focus_options.append(label)
            focus_lookup[label] = str(row["ticker"])
    focus_choice = st.selectbox(
        "Timeline focus",
        focus_options,
        help="Pick a top mover to immediately drive the company timeline and semantic-map highlights.",
    )
    focused_ticker = focus_lookup[focus_choice]
    if focused_ticker is None:
        default_label = next((label for label in label_options if label.startswith("NVDA -")), label_options[0])
        selected_label = st.selectbox("Company timeline", label_options, index=label_options.index(default_label))
        selected_ticker = label_lookup[selected_label]
    else:
        selected_ticker = focused_ticker
        st.caption(f"Timeline is focused on top mover: {selected_ticker}")
    company_history = company_topic_history(counts, selected_ticker, selected_sections, topic_column, mention_column)

    timeline_columns = st.columns([1.7, 1.3])
    with timeline_columns[0]:
        st.markdown(f"**{selected_ticker} {topic_name} Timeline**")
        if company_history.empty:
            st.info(f"No selected section history for {selected_ticker}.")
        else:
            st.altair_chart(historical_company_chart(company_history, topic_name), width="stretch")
    with timeline_columns[1]:
        st.markdown("**Selected Company Filing Rows**")
        if not company_history.empty:
            table = company_history.loc[
                :,
                [
                    "filing_date",
                    "section_label",
                    "topic_score",
                    "topic_mentions",
                    "section_chars",
                    "word_count",
                ],
            ].sort_values("filing_date", ascending=False)
            st.dataframe(table, width="stretch", hide_index=True)

    topic_delta = company_topic_delta_summary(counts, selected_ticker, selected_sections, early_years, late_years)
    if not topic_delta.empty:
        st.markdown(f"**What Changed For {selected_ticker}?**")
        st.caption("Topic deltas compare the selected baseline and recent windows across the selected sections.")
        st.dataframe(topic_delta, width="stretch", hide_index=True)

    st.markdown("**Largest Company Topic Increases**")
    if changes.empty:
        st.info("No company change rows available for the selected windows.")
    else:
        display_columns = [
            "ticker",
            "company_name",
            "gics_sector",
            "gics_sub_industry",
            "early_score",
            "late_score",
            "change",
            "late_mentions",
            "first_year",
            "latest_year",
        ]
        st.dataframe(
            safe_frame_subset(ensure_columns(changes, display_columns), display_columns).head(50),
            width="stretch",
            hide_index=True,
        )

    sector_trend = sector_topic_heatmap(counts, metadata, selected_sections, topic_column)
    if not sector_trend.empty:
        with st.expander("Sector-year heatmap"):
            heatmap = (
                alt.Chart(sector_trend)
                .mark_rect()
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("gics_sector:N", title="Sector"),
                    color=alt.Color("topic_score:Q", title=f"{topic_name} score"),
                    tooltip=[
                        "gics_sector:N",
                        "year:O",
                        alt.Tooltip("topic_score:Q", format=".3f"),
                        alt.Tooltip("tickers:Q", format=".0f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(heatmap, width="stretch")

    if embeddings.empty:
        st.info("No historical embeddings found, so the semantic map is unavailable.")
        return

    st.markdown("**Historical Semantic Movement Map**")
    map_controls = st.columns([1.2, 1.2, 2.0])
    map_section = map_controls[0].selectbox(
        "Embedding section",
        sections,
        index=sections.index("business") if "business" in sections else 0,
        format_func=lambda section: section_labels.get(section, section),
    )
    color_mode = map_controls[1].selectbox("Map color", ["Year", "Sector", "Highlighted"], index=0)
    highlight_defaults = [ticker for ticker in ["NVDA", "MSFT", "META", "ADBE", "AMZN", selected_ticker] if ticker in ticker_options]
    highlighted = map_controls[2].multiselect("Highlight filing trails", ticker_options, default=highlight_defaults)
    projected = load_projected_historical_embeddings(str(input_dir), map_section)
    if projected.empty:
        st.info("Not enough embedding rows to project this section.")
        return
    points = historical_embedding_points(projected, metadata, highlighted, color_mode)
    st.altair_chart(historical_embedding_map_chart(points, highlighted), width="stretch")
    st.caption(
        "This PCA map is fit over historical section embeddings. A highlighted line connects a company's filings over time, "
        "so visible drift means the language in that section is moving semantically."
    )

    drift = historical_embedding_drift_table(embeddings, metadata, map_section, early_years, late_years)
    if not drift.empty:
        with st.expander("Semantic drift leaderboard", expanded=True):
            st.write(
                "This ranks companies by embedding displacement between the selected baseline and recent windows. "
                "It is useful for finding firms whose filing language moved the most, even when the keyword topic is not obvious."
            )
            drift_display = ensure_columns(
                drift,
                [
                    "ticker",
                    "company_name",
                    "gics_sector",
                    "gics_sub_industry",
                    "embedding_displacement",
                    "early_rows",
                    "late_rows",
                    "first_filing",
                    "latest_filing",
                ],
            )
            st.dataframe(
                safe_frame_subset(
                    drift_display,
                    [
                        "ticker",
                        "company_name",
                        "gics_sector",
                        "gics_sub_industry",
                        "embedding_displacement",
                        "early_rows",
                        "late_rows",
                        "first_filing",
                        "latest_filing",
                    ],
                ).head(50),
                width="stretch",
                hide_index=True,
            )


def render_filing_browser() -> None:
    storage_dir = st.sidebar.text_input("Storage Directory", value=str(DEFAULT_STORAGE_DIR))
    operator = get_operator(storage_dir)

    st.sidebar.subheader("Build Corpus")
    tickers_input = st.sidebar.text_input("Tickers", value="AAPL, MSFT")
    group_name = st.sidebar.text_input("Group Name", value="Tech Smoke Test")
    since = st.sidebar.text_input("Since", value="2024-01-01")
    filings_per_form = st.sidebar.number_input("Filings Per Form", min_value=1, value=1, step=1)

    if st.sidebar.button("Fetch And Store", width="stretch"):
        tickers = parse_tickers(tickers_input)
        if not tickers:
            st.sidebar.error("Enter at least one ticker.")
        else:
            with st.spinner("Fetching SEC filings and raw submissions..."):
                try:
                    corpus, stored = operator.populate_for_symbols(
                        tickers,
                        group_name=group_name,
                        format="jsonl",
                        filing_options={"since": since, "filings_per_form": int(filings_per_form)},
                        raw_filing_options={"since": since, "filings_per_form": int(filings_per_form)},
                    )
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(str(exc))
                else:
                    st.sidebar.success(
                        f"Stored {corpus.summary()['raw_filing_rows']} raw filings in {stored.raw_filings_path}"
                    )

    stored_corpora = operator.list_corpora()
    group_names = [stored.group_name for stored in stored_corpora]
    if not group_names:
        st.info("No stored corpora yet. Use the sidebar to fetch a small ticker set.")
        return

    selected_group = st.selectbox("Stored Corpus", group_names)
    stored = operator.load_corpus(selected_group)
    description = operator.describe_corpus(selected_group)

    summary_columns = st.columns(4)
    summary_columns[0].metric("Format", str(description["dataset_format"]))
    summary_columns[1].metric("Raw Dataset", "Yes" if description["has_raw_filings"] else "No")
    summary_columns[2].metric("Structured Dataset", "Yes" if description["has_filings"] else "No")
    summary_columns[3].metric("Failures File", "Yes" if description["has_failures"] else "No")

    st.caption(f"Corpus directory: {stored.dataset_dir}")

    tickers = operator.list_tickers(selected_group)
    selected_ticker = st.selectbox("Ticker Filter", ["All"] + tickers)
    ticker_filter = None if selected_ticker == "All" else selected_ticker

    structured = operator.find_structured_filings(selected_group, ticker=ticker_filter)
    raw = operator.find_raw_filings(selected_group, ticker=ticker_filter)

    structured_tab, raw_tab, metadata_tab = st.tabs(["Structured Filings", "Raw Filings", "Metadata"])

    with structured_tab:
        st.dataframe(
            safe_frame_subset(
                structured,
                ["ticker", "form", "filing_date", "accession_no", "period_end", "revenue", "net_income"],
            ),
            width="stretch",
            hide_index=True,
        )

    with raw_tab:
        st.dataframe(
            safe_frame_subset(
                raw,
                [
                    "ticker",
                    "form",
                    "filing_date",
                    "accession_no",
                    "submission_text_length",
                    "source_url",
                    "raw_shard_path",
                ],
            ),
            width="stretch",
            hide_index=True,
        )
        if not raw.empty:
            accession_options = raw["accession_no"].astype(str).tolist()
            selected_accession = st.selectbox("Preview Submission", accession_options)
            preview_text = operator.load_raw_submission_text(selected_group, selected_accession)
            st.text_area(
                "Submission Text Preview",
                value=(preview_text or "")[:12000],
                height=320,
            )

    with metadata_tab:
        st.json(
            {
                "stored": {key: str(value) for key, value in description.items()},
                "raw_filings_path": str(stored.raw_filings_path),
                "filings_path": str(stored.filings_path) if stored.filings_path else None,
                "tickers_path": str(stored.tickers_path) if stored.tickers_path else None,
                "manifest_path": str(stored.manifest_path),
            }
        )


def render_similarity_shifts() -> None:
    st.subheader("Similarity Shifts")
    st.caption(
        "Each view assigns a company soft memberships across themes. "
        "Use this tab to watch those mixed category loadings move through time."
    )

    default_experiment = latest_decomposed_experiment()
    default_value = str(default_experiment) if default_experiment else ""
    experiment_value = st.text_input("Decomposed experiment directory", value=default_value)
    if not experiment_value:
        st.warning("No decomposed experiment with view loadings was found.")
        return

    experiment_root = Path(experiment_value)
    if not experiment_root.exists():
        st.error(f"Experiment directory does not exist: {experiment_root}")
        return

    views = available_views(experiment_root)
    if not views:
        st.error(f"No view loadings found under {experiment_root / 'views'}")
        return

    metadata = load_metadata()
    controls = st.columns([1.2, 1.5, 1.0, 1.0])
    selected_view = controls[0].selectbox("Similarity view", views, index=views.index("business") if "business" in views else 0)
    loadings_path = experiment_root / "views" / selected_view / "loadings.parquet"
    loadings = load_loadings(str(loadings_path))
    ticker_options = sorted(loadings["ticker"].unique().tolist())
    company_labels = metadata[metadata["ticker"].isin(ticker_options)].copy()
    label_lookup = dict(zip(company_labels["search_label"], company_labels["ticker"], strict=False))
    label_options = company_labels["search_label"].tolist()
    if not label_options:
        label_options = ticker_options
        label_lookup = {ticker: ticker for ticker in ticker_options}
    default_label = next((label for label in label_options if label.startswith("META -")), label_options[0])
    selected_label = controls[1].selectbox("Company", label_options, index=label_options.index(default_label))
    selected_ticker = label_lookup[selected_label]

    company_history = loadings[loadings["ticker"] == selected_ticker].copy()
    if company_history.empty:
        st.warning(f"No loadings found for {selected_ticker}.")
        return

    all_theme_columns = theme_columns(company_history)
    max_themes = len(all_theme_columns)
    preset = controls[2].selectbox("Granularity", ["Coarse", "Medium", "Fine", "Custom"], index=1)
    default_k = {"Coarse": 5, "Medium": 10, "Fine": min(20, max_themes), "Custom": min(12, max_themes)}[preset]
    selected_k = controls[3].slider("k groups", min_value=3, max_value=max_themes, value=min(default_k, max_themes), step=1)
    ranking = st.radio(
        "Which themes should be shown?",
        ["Average loading", "Latest loading", "Biggest movement"],
        horizontal=True,
    )
    selected_columns = selected_theme_columns(company_history, selected_k, ranking)

    dates = company_history["date"].dt.strftime("%Y-%m-%d").tolist()
    date_state_key = f"company_shift_date_index_{selected_view}_{selected_ticker}"
    bounded_state_value(date_state_key, default=len(dates) - 1, minimum=0, maximum=len(dates) - 1)
    step_controls = st.columns([0.9, 0.9, 0.9, 0.9, 1.0, 2.4])
    shift_step = step_controls[0].slider("Step", min_value=1, max_value=24, value=3, step=1, key=f"{date_state_key}_step")
    if step_controls[1].button("Back", width="stretch", key=f"{date_state_key}_back"):
        move_state_value(date_state_key, -int(shift_step), 0, len(dates) - 1)
    if step_controls[2].button("Forward", width="stretch", key=f"{date_state_key}_forward"):
        move_state_value(date_state_key, int(shift_step), 0, len(dates) - 1)
    if step_controls[3].button("Start", width="stretch", key=f"{date_state_key}_start"):
        st.session_state[date_state_key] = 0
    if step_controls[4].button("Latest", width="stretch", key=f"{date_state_key}_latest"):
        st.session_state[date_state_key] = len(dates) - 1
    date_index = step_controls[5].slider(
        "Simulation date",
        min_value=0,
        max_value=len(dates) - 1,
        key=date_state_key,
        format="%d",
    )
    selected_date = dates[date_index]
    label_lookup_for_view = auto_theme_label_lookup(str(loadings_path), metadata, selected_view)
    fragment_label_lookup, fragment_explanations = fragment_theme_label_artifacts(
        str(loadings_path),
        str(default_historical_text_dir()),
        selected_view,
        None,
    )
    label_lookup_for_view.update(fragment_label_lookup)
    label_lookup_for_view = unique_theme_label_lookup(label_lookup_for_view, all_theme_columns)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected ticker", selected_ticker)
    metric_columns[1].metric("View", selected_view)
    metric_columns[2].metric("Date", selected_date)
    metric_columns[3].metric("Groups shown", selected_k)

    current_row = company_history.iloc[date_index]
    st.caption(f"{selected_ticker} category mixture on {selected_date}")
    st.altair_chart(
        current_theme_loading_chart(current_row, selected_columns, label_lookup_for_view),
        width="stretch",
    )

    timeline_frame = theme_loading_long_frame(company_history, selected_columns, label_lookup_for_view)
    st.altair_chart(theme_loading_timeline_chart(timeline_frame), width="stretch")

    table_columns = st.columns(2)
    with table_columns[0]:
        st.markdown("**Current Mixed Category Loadings**")
        st.dataframe(
            movement_frame(company_history, selected_columns, date_index, label_lookup_for_view),
            width="stretch",
            hide_index=True,
        )
    with table_columns[1]:
        st.markdown("**Largest Full-Period Shifts**")
        movement = []
        first = company_history.iloc[0]
        last = company_history.iloc[-1]
        for column in all_theme_columns:
            movement.append(
                {
                    "theme": display_theme_name(column, label_lookup_for_view),
                    "start_loading": float(first[column]),
                    "latest_loading": float(last[column]),
                    "change": float(last[column] - first[column]),
                    "absolute_change": abs(float(last[column] - first[column])),
                }
            )
        st.dataframe(
            pd.DataFrame(movement).sort_values("absolute_change", ascending=False).head(12),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Dynamic theme labels", expanded=True):
        st.write(
            "This dashboard ignores the manual label CSV. Business labels are generated from stable representative "
            "filing fragments for each theme. Other views use computed sector-mix labels as a fallback."
        )
        if selected_view == "business" and not fragment_explanations.empty:
            st.markdown("**Evidence for visible dynamic labels**")
            shown = theme_evidence_frame(fragment_explanations, selected_columns)
            st.dataframe(
                shown,
                width="stretch",
                hide_index=True,
                column_config={
                    "source_url": st.column_config.LinkColumn("SEC source filing"),
                    "fragment_similarity": st.column_config.NumberColumn("fragment_similarity", format="%.3f"),
                },
            )
            st.caption(
                "The representative fragment is chosen by cosine similarity in MiniLM section-embedding space. "
                "Compact snippets appear when `historical_section_snippets.parquet` has been generated; older artifacts "
                "show topic counts and SEC source links only."
            )
        elif selected_view != "business":
            st.info("Fragment-derived dynamic labels are currently available for the business view only.")
        label_preview = pd.DataFrame(
            [{"theme": theme, "dynamic_label": display_theme_name(theme, label_lookup_for_view)} for theme in all_theme_columns]
        )
        st.dataframe(label_preview, width="stretch", hide_index=True)

    with st.expander("What am I looking at?"):
        st.write(
            "These are soft GMM category loadings from the selected view. "
            "A hard cluster would force one company into one bucket. Here, a company can belong partly to many themes, "
            "which is why the chart can show gradual shifts rather than one sudden label change."
        )
        st.write(
            "Granularity controls how many themes are visible. Higher k shows more detail, but can get noisier. "
            "The underlying GMM was already fit in the decomposed experiment; this dashboard is an exploratory viewer."
        )


def render_market_map() -> None:
    st.subheader("S&P 500 Similarity Map")
    st.caption(
        "Every point is a stock. The 2D plane is a PCA projection of soft theme loadings for the selected view, "
        "fit once across the full history so movement through time is visible."
    )

    default_experiment = latest_decomposed_experiment()
    default_value = str(default_experiment) if default_experiment else ""
    experiment_value = st.text_input("Market map experiment directory", value=default_value)
    if not experiment_value:
        st.warning("No decomposed experiment with view loadings was found.")
        return

    experiment_root = Path(experiment_value)
    if not experiment_root.exists():
        st.error(f"Experiment directory does not exist: {experiment_root}")
        return

    views = available_views(experiment_root)
    if not views:
        st.error(f"No view loadings found under {experiment_root / 'views'}")
        return

    metadata = load_metadata()
    controls = st.columns([1.0, 1.0, 1.0, 1.4])
    selected_view = controls[0].selectbox(
        "Map view",
        views,
        index=views.index("behavioral") if "behavioral" in views else 0,
        key="market_map_view",
    )
    loadings_path = experiment_root / "views" / selected_view / "loadings.parquet"
    loadings = load_loadings(str(loadings_path))
    projected = load_projected_market_map(str(loadings_path))

    hide_collapsed = st.checkbox(
        "Hide collapsed / uninformative dates",
        value=True,
        help="Skips dates where nearly all stocks project to the same place because the view has little information yet.",
    )
    available_dates = usable_market_dates(projected, hide_collapsed=hide_collapsed)
    if not available_dates:
        st.warning("No usable map dates are available for this view with the current filter.")
        return
    date_labels = [date.strftime("%Y-%m-%d") for date in available_dates]
    date_state_key = f"market_map_date_index_{selected_view}_{int(hide_collapsed)}"
    if date_state_key not in st.session_state:
        st.session_state[date_state_key] = len(date_labels) - 1
    st.session_state[date_state_key] = int(
        max(0, min(len(date_labels) - 1, st.session_state[date_state_key]))
    )
    step_size = controls[1].slider("Step increment", min_value=1, max_value=24, value=3, step=1)
    k_groups = controls[2].slider("Color groups", min_value=3, max_value=len(theme_columns(loadings)), value=12, step=1)

    step_columns = st.columns([0.9, 0.9, 1.0, 1.0, 2.0])
    if step_columns[0].button("Back", width="stretch"):
        st.session_state[date_state_key] = max(0, st.session_state[date_state_key] - int(step_size))
    if step_columns[1].button("Forward", width="stretch"):
        st.session_state[date_state_key] = min(len(date_labels) - 1, st.session_state[date_state_key] + int(step_size))
    if step_columns[2].button("Start", width="stretch"):
        st.session_state[date_state_key] = 0
    if step_columns[3].button("Latest", width="stretch"):
        st.session_state[date_state_key] = len(date_labels) - 1
    date_index = step_columns[4].slider(
        "Map date",
        min_value=0,
        max_value=len(date_labels) - 1,
        key=date_state_key,
        format="%d",
    )
    st.caption(f"Selected map date: {date_labels[int(date_index)]}")
    selected_date = pd.Timestamp(available_dates[date_index])
    label_lookup_for_view = auto_theme_label_lookup(str(loadings_path), metadata, selected_view)
    fragment_label_lookup, fragment_explanations = fragment_theme_label_artifacts(
        str(loadings_path),
        str(default_historical_text_dir()),
        selected_view,
        None,
    )
    label_lookup_for_view.update(fragment_label_lookup)
    label_lookup_for_view = unique_theme_label_lookup(label_lookup_for_view, theme_columns(loadings))

    ticker_options = sorted(projected["ticker"].unique().tolist())
    default_highlights = [ticker for ticker in ["META", "AAPL", "MSFT", "XOM", "JPM", "SBUX"] if ticker in ticker_options]
    highlighted = controls[3].multiselect("Highlight / trace tickers", ticker_options, default=default_highlights)
    trail_months = st.slider("Movement trail length, in monthly steps", min_value=3, max_value=60, value=18, step=3)

    points = market_date_frame(projected, metadata, selected_date, label_lookup_for_view, k_groups)
    if points.empty:
        st.warning(f"No map points found for {selected_date.strftime('%Y-%m-%d')}.")
        return

    metrics = st.columns(4)
    metrics[0].metric("Stocks shown", int(points["ticker"].nunique()))
    metrics[1].metric("View", selected_view)
    metrics[2].metric("Date", selected_date.strftime("%Y-%m-%d"))
    metrics[3].metric("Color groups", k_groups)

    x_domain, y_domain = map_axis_domains(projected)
    st.caption(f"S&P 500 similarity map on {selected_date.strftime('%Y-%m-%d')}")
    st.altair_chart(market_map_chart(points, highlighted, x_domain, y_domain), width="stretch")

    if highlighted:
        st.markdown("**Highlighted Ticker Trails**")
        st.altair_chart(
            market_trail_chart(projected, selected_date, highlighted, trail_months, label_lookup_for_view),
            width="stretch",
        )

    table_columns = st.columns(2)
    with table_columns[0]:
        st.markdown("**Dominant Theme Counts**")
        counts = points.groupby("theme_label").size().reset_index(name="stock_count")
        st.dataframe(counts.sort_values("stock_count", ascending=False), width="stretch", hide_index=True)
    with table_columns[1]:
        st.markdown("**Largest Dominant Loadings**")
        st.dataframe(
            points.sort_values("dominant_loading", ascending=False)
            .loc[:, ["ticker", "title", "theme_label", "dominant_loading"]]
            .head(20),
            width="stretch",
            hide_index=True,
        )

    if selected_view == "business" and not fragment_explanations.empty:
        with st.expander("Dynamic Labels: Representative Filing Fragments"):
            shown_themes = set(points["dominant_theme"].astype(str).unique())
            shown = theme_evidence_frame(fragment_explanations, shown_themes)
            st.dataframe(
                shown,
                width="stretch",
                hide_index=True,
                column_config={
                    "source_url": st.column_config.LinkColumn("SEC source filing"),
                    "fragment_similarity": st.column_config.NumberColumn("fragment_similarity", format="%.3f"),
                },
            )
            st.caption(
                "Stable labels are generated by comparing candidate filing-section fragments to each theme centroid "
                "in MiniLM embedding space, then using the nearest fragment's strongest tracked topics. "
                "Compact snippets appear when the snippet artifact has been generated; otherwise use the source filing "
                "link for the underlying text."
            )

    with st.expander("How to read this map"):
        st.write(
            "Distance means similarity inside the selected view. If two stocks move in the same direction over time, "
            "their soft theme memberships are changing in similar ways."
        )
        st.write(
            "Color is the stock's dominant soft theme on the selected date. The `Color groups` slider decides how many "
            "of the biggest themes get their own colors; smaller themes are grouped into `Other themes`."
        )
        st.write(
            "This uses PCA for speed and stability. It is an exploratory map, not a trading signal."
        )


filing_tab, historical_tab, shifts_tab, map_tab, sector_tab = st.tabs(
    ["Filing Browser", "Historical Text", "Similarity Shifts", "Market Map", "Sector Outlook"]
)

with filing_tab:
    render_filing_browser()

with historical_tab:
    render_historical_text()

with shifts_tab:
    render_similarity_shifts()

with map_tab:
    render_market_map()

with sector_tab:
    render_sector_outlook()
