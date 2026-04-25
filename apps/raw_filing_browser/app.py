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
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from market_data_fetcher import DatabaseOperator


DEFAULT_STORAGE_DIR = REPO_ROOT / "data" / "raw_filing_corpora"
DEFAULT_IDENTITY = "StockCorrelation/0.1 castellanosmarcelo1@gmail.com"
DEFAULT_THEME_LABELS_PATH = REPO_ROOT / "report" / "theme_labels.csv"
DEFAULT_HISTORICAL_TEXT_DIR = REPO_ROOT / "data" / "processed" / "historical_text"
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
def load_historical_manifest(directory: str) -> dict:
    """Load compact historical text manifest when present."""
    path = Path(directory) / "historical_text_manifest.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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


def load_theme_labels(path: Path) -> pd.DataFrame:
    """Load optional manual theme labels from CSV."""
    columns = ["view", "theme", "label", "description"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame["view"] = frame["view"].astype(str)
    frame["theme"] = frame["theme"].astype(str)
    frame["label"] = frame["label"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    return frame.loc[:, columns]


def theme_label_lookup(labels: pd.DataFrame, view_name: str) -> dict[str, str]:
    """Return display names for one view's themes."""
    if labels.empty:
        return {}
    view_labels = labels[labels["view"] == view_name]
    lookup = {}
    for _, row in view_labels.iterrows():
        label = str(row["label"]).strip()
        if label:
            lookup[str(row["theme"])] = f"{row['theme']} · {label}"
    return lookup


@st.cache_data(show_spinner=False)
def auto_theme_label_lookup(loadings_path: str, metadata: pd.DataFrame, view_name: str) -> dict[str, str]:
    """Create readable fallback labels from top firms and sector mix."""
    loadings = load_loadings(loadings_path)
    latest = loadings.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest = latest.merge(metadata, on="ticker", how="left")
    lookup = {}
    for theme in theme_columns(latest):
        top = latest.sort_values(theme, ascending=False).head(8)
        top_tickers = ", ".join(top["ticker"].astype(str).head(3).tolist())
        if "gics_sector" in top.columns and top["gics_sector"].notna().any():
            sector = str(top["gics_sector"].dropna().mode().iloc[0])
            label = f"{theme} · {sector} · {top_tickers}"
        else:
            label = f"{theme} · {top_tickers}"
        lookup[theme] = label
    return lookup


def ensure_label_rows(labels: pd.DataFrame, view_name: str, columns: list[str]) -> pd.DataFrame:
    """Ensure the editable label table has one row per theme."""
    existing = labels.copy()
    existing_keys = set(zip(existing["view"], existing["theme"], strict=False)) if not existing.empty else set()
    rows = []
    for theme in columns:
        key = (view_name, theme)
        if key not in existing_keys:
            rows.append({"view": view_name, "theme": theme, "label": "", "description": ""})
    if rows:
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    return existing.sort_values(["view", "theme"]).reset_index(drop=True)


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


def combined_theme_labels(manual_labels: dict[str, str], auto_labels: dict[str, str]) -> dict[str, str]:
    """Use manual theme names when available and auto names for the rest."""
    combined = dict(auto_labels)
    combined.update(manual_labels)
    return combined


def loading_bar_frame(row: pd.Series, columns: list[str], labels: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "theme": [display_theme_name(column, labels) for column in columns],
            "loading": [float(row[column]) for column in columns],
        }
    )
    return frame.sort_values("loading", ascending=False).set_index("theme")


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


def market_trail_chart(projected: pd.DataFrame, selected_date: pd.Timestamp, highlighted: list[str], months: int) -> alt.Chart:
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
    return (
        alt.Chart(trail)
        .mark_line(point=True)
        .encode(
            x=alt.X("x:Q", title="Similarity map X"),
            y=alt.Y("y:Q", title="Similarity map Y"),
            color=alt.Color("ticker:N", title="Ticker"),
            tooltip=["ticker:N", "date_label:N", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")],
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


def render_historical_text() -> None:
    st.subheader("Historical Filing Language")
    st.caption(
        "Explore compact historical 10-K section features: keyword counts first, MiniLM embeddings second, "
        "with raw SEC text discarded after processing."
    )

    input_dir = Path(st.text_input("Historical text artifact directory", value=str(DEFAULT_HISTORICAL_TEXT_DIR)))
    counts = load_historical_topic_counts(str(input_dir))
    if counts.empty:
        st.warning(
            "No historical text topic counts found. Run "
            "`.venv/bin/python scripts/24_stream_historical_text_features.py --since 2010-01-01` first."
        )
        return

    manifest = load_historical_manifest(str(input_dir))
    metadata = load_metadata()
    filing_index = load_historical_filing_index(str(input_dir))
    embeddings = load_historical_embeddings(str(input_dir))

    summary_columns = st.columns(5)
    summary_columns[0].metric("Filings", f"{len(filing_index):,}" if not filing_index.empty else "n/a")
    summary_columns[1].metric("Topic rows", f"{len(counts):,}")
    summary_columns[2].metric("Embedding rows", f"{len(embeddings):,}" if not embeddings.empty else "0")
    summary_columns[3].metric("Tickers", f"{counts['ticker'].nunique():,}")
    summary_columns[4].metric("Failures", str(manifest.get("failures", "n/a")))

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
    labels_path = Path(st.text_input("Theme labels CSV", value=str(DEFAULT_THEME_LABELS_PATH)))
    theme_labels = load_theme_labels(labels_path)
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
    theme_labels = ensure_label_rows(theme_labels, selected_view, all_theme_columns)
    manual_label_lookup = theme_label_lookup(theme_labels, selected_view)
    auto_label_lookup = auto_theme_label_lookup(str(loadings_path), metadata, selected_view)
    label_lookup_for_view = combined_theme_labels(manual_label_lookup, auto_label_lookup)
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

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected ticker", selected_ticker)
    metric_columns[1].metric("View", selected_view)
    metric_columns[2].metric("Date", selected_date)
    metric_columns[3].metric("Groups shown", selected_k)

    line_frame = company_history.set_index("date").loc[:, selected_columns]
    line_frame = line_frame.rename(
        columns={column: display_theme_name(column, label_lookup_for_view) for column in selected_columns}
    )

    current_row = company_history.iloc[date_index]
    st.caption(f"{selected_ticker} category mixture on {selected_date}")
    st.bar_chart(loading_bar_frame(current_row, selected_columns, label_lookup_for_view))

    st.line_chart(line_frame, width="stretch")

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

    with st.expander("Label themes"):
        st.write(
            "Theme labels are manual on purpose. Inspect top firms and movement, then give a theme a short name."
        )
        editable = theme_labels[theme_labels["view"] == selected_view].copy()
        edited = st.data_editor(
            editable,
            width="stretch",
            hide_index=True,
            disabled=["view", "theme"],
            column_config={
                "label": st.column_config.TextColumn("Label", help="Short human-readable name."),
                "description": st.column_config.TextColumn("Description", help="Optional interpretation notes."),
            },
        )
        if st.button("Save theme labels", width="stretch"):
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            other_views = theme_labels[theme_labels["view"] != selected_view]
            output = pd.concat([other_views, edited], ignore_index=True)
            output = output.sort_values(["view", "theme"]).reset_index(drop=True)
            output.to_csv(labels_path, index=False)
            st.success(f"Saved labels to {labels_path}. Refresh or rerun controls to see labels everywhere.")

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
    labels_path = Path(st.text_input("Market map theme labels CSV", value=str(DEFAULT_THEME_LABELS_PATH)))
    theme_labels = ensure_label_rows(load_theme_labels(labels_path), selected_view, theme_columns(loadings))
    manual_label_lookup = theme_label_lookup(theme_labels, selected_view)
    auto_label_lookup = auto_theme_label_lookup(str(loadings_path), metadata, selected_view)
    label_lookup_for_view = combined_theme_labels(manual_label_lookup, auto_label_lookup)

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
            market_trail_chart(projected, selected_date, highlighted, trail_months),
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


filing_tab, historical_tab, shifts_tab, map_tab = st.tabs(
    ["Filing Browser", "Historical Text", "Similarity Shifts", "Market Map"]
)

with filing_tab:
    render_filing_browser()

with historical_tab:
    render_historical_text()

with shifts_tab:
    render_similarity_shifts()

with map_tab:
    render_market_map()
