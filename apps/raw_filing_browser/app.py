from __future__ import annotations

import importlib
import inspect
import re
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
from src.applications.cluster_model_comparison import cluster_cross_section, embedding_columns
from src.applications import sector_relative_outlook as sector_outlook
from src.config import load_config
from src.db import FilingsDB


sector_outlook = importlib.reload(sector_outlook)
SECTOR_MODEL_LABELS = sector_outlook.SECTOR_MODEL_LABELS
THEME_ASSIGNMENT_LABELS = getattr(
    sector_outlook,
    "THEME_ASSIGNMENT_LABELS",
    {
        "gics": "GICS sectors",
        "soft": "Mixed soft loadings",
        "hard_top1": "Hard top-1 theme",
        "auto": "Auto by view",
    },
)
completed_sector_predictions = sector_outlook.completed_sector_predictions
rotation_simulation_metrics = sector_outlook.rotation_simulation_metrics
sector_backtest_by_date = sector_outlook.sector_backtest_by_date
sector_outlook_backtest = sector_outlook.sector_outlook_backtest
sector_prediction_audit = sector_outlook.sector_prediction_audit
simulate_group_rotation = sector_outlook.simulate_group_rotation


def default_theme_assignment_strategy(view_name: str | None) -> str:
    """Return the dashboard's view-specific theme assignment policy."""
    if hasattr(sector_outlook, "default_theme_assignment_strategy"):
        return sector_outlook.default_theme_assignment_strategy(view_name)
    return "hard_top1" if str(view_name or "").strip().lower() == "behavioral" else "soft"


DEFAULT_STORAGE_DIR = REPO_ROOT / "data" / "raw_filing_corpora"
DEFAULT_IDENTITY = "StockCorrelation/0.1 castellanosmarcelo1@gmail.com"
DEFAULT_HISTORICAL_TEXT_DIR = REPO_ROOT / "data" / "processed" / "historical_text_10k_10q"
FALLBACK_HISTORICAL_TEXT_DIR = REPO_ROOT / "data" / "processed" / "historical_text"
SP500_GICS_PATH = REPO_ROOT / "data" / "processed" / "metadata" / "sp500_gics.parquet"
SP500_MEMBERSHIP_PATH = REPO_ROOT / "data" / "processed" / "metadata" / "sp500_membership_history.parquet"
SP500_DELETED_PRICES_PATH = REPO_ROOT / "data" / "processed" / "prices" / "sp500_deleted_constituents.parquet"
VALUATION_FEATURE_PATH = REPO_ROOT / "data" / "processed" / "features" / "valuation.parquet"
GROWTH_FEATURE_PATH = REPO_ROOT / "data" / "processed" / "features" / "growth_lifecycle.parquet"
RELATIONSHIPS_PATH = REPO_ROOT / "data" / "processed" / "relationships" / "relationships.parquet"
DEFAULT_SP500_BENCHMARK_PATH = REPO_ROOT / "data" / "processed" / "benchmarks" / "spy_benchmark.parquet"
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
INTERPRETABLE_HISTORICAL_SECTIONS = [
    "business",
    "risk_factors",
    "mda",
]
FRAGMENT_LABEL_SECTIONS = INTERPRETABLE_HISTORICAL_SECTIONS
DEFAULT_SECTOR_CONFIG = "decomposed_point_in_time"
DEFAULT_SECTOR_MODEL = "ridge"
DEFAULT_SECTOR_HORIZON_DAYS = 63
DEFAULT_SECTOR_MIN_TRAIN_MONTHS = 36
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_MIN_THEME_EFFECTIVE_MEMBERS = 5.0
DEFAULT_SHIFT_THEME_COUNT = 10
DEFAULT_MARKET_TRAIL_MONTHS = 18
DEFAULT_SIMULATION_CAPITAL = 10_000.0
MAP_EXCLUDED_VIEWS = {"growth"}
MAX_DYNAMIC_LABEL_WORDS = 8
MAX_DYNAMIC_LABEL_CHARS = 68
METADATA_CACHE_VERSION = 2


def inject_neumorphic_theme() -> None:
    """Apply the dashboard's soft neumorphic visual system."""
    st.markdown(
        """
        <style>
        :root {
            --neu-bg: #edf2ee;
            --neu-bg-warm: #f6f1e8;
            --neu-surface: rgba(248, 250, 246, 0.88);
            --neu-surface-solid: #f7f9f4;
            --neu-surface-deep: #e4ebe6;
            --neu-text: #24312f;
            --neu-muted: #65746d;
            --neu-accent: #2f6f64;
            --neu-accent-soft: #dcebe5;
            --neu-gold: #c28645;
            --neu-border: rgba(255, 255, 255, 0.78);
            --neu-shadow-dark: rgba(139, 153, 145, 0.55);
            --neu-shadow-light: rgba(255, 255, 255, 0.96);
            --neu-radius: 26px;
            --neu-radius-sm: 16px;
        }

        .stApp {
            color: var(--neu-text);
            font-family: "Avenir Next", "Nunito Sans", "SF Pro Display", "Helvetica Neue", sans-serif;
            background:
                radial-gradient(circle at 12% 8%, rgba(255, 255, 255, 0.95), transparent 27rem),
                radial-gradient(circle at 88% 2%, rgba(215, 231, 224, 0.9), transparent 31rem),
                radial-gradient(circle at 58% 96%, rgba(243, 225, 198, 0.58), transparent 26rem),
                linear-gradient(135deg, var(--neu-bg) 0%, #e7eee9 48%, var(--neu-bg-warm) 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(237, 242, 238, 0.72);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.62);
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        .block-container {
            max-width: 1520px;
            padding-top: 4.35rem;
            padding-bottom: 4rem;
        }

        .neu-hero {
            margin: 0 0 1.5rem;
            padding: 2.1rem 2.25rem;
            border-radius: 34px;
            border: 1px solid var(--neu-border);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.76), rgba(239, 246, 241, 0.78)),
                radial-gradient(circle at 88% 18%, rgba(194, 134, 69, 0.17), transparent 18rem);
            box-shadow:
                18px 18px 38px var(--neu-shadow-dark),
                -18px -18px 38px var(--neu-shadow-light);
        }

        .neu-kicker {
            margin: 0 0 0.45rem;
            color: var(--neu-accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }

        .neu-hero h1 {
            margin: 0;
            color: var(--neu-text);
            font-size: clamp(2.2rem, 4.6vw, 4.7rem);
            font-weight: 850;
            letter-spacing: -0.055em;
            line-height: 0.96;
        }

        .neu-hero p:last-child {
            max-width: 820px;
            margin: 1rem 0 0;
            color: var(--neu-muted);
            font-size: 1.04rem;
            line-height: 1.62;
        }

        h1, h2, h3, h4 {
            color: var(--neu-text);
            letter-spacing: -0.025em;
        }

        p, label, [data-testid="stMarkdownContainer"] li {
            color: var(--neu-text);
        }

        small, .caption, [data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {
            color: var(--neu-muted) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding: 0.48rem;
            border-radius: 999px;
            background: rgba(225, 234, 228, 0.82);
            flex-wrap: wrap;
            overflow: visible;
            box-shadow:
                inset 7px 7px 15px rgba(151, 163, 154, 0.38),
                inset -7px -7px 15px rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.7);
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 2.65rem;
            padding: 0.45rem 1rem;
            border-radius: 999px;
            color: var(--neu-muted);
            font-weight: 750;
            transition: all 140ms ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--neu-accent);
            background: rgba(255, 255, 255, 0.38);
        }

        .stTabs [aria-selected="true"] {
            color: var(--neu-accent) !important;
            background: var(--neu-surface-solid);
            box-shadow:
                8px 8px 16px rgba(154, 166, 158, 0.38),
                -8px -8px 16px rgba(255, 255, 255, 0.95);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background: transparent;
        }

        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"],
        [data-testid="stMetric"],
        div[data-testid="stAltairChart"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        .stPlotlyChart,
        .stVegaLiteChart {
            border-radius: var(--neu-radius) !important;
            border: 1px solid var(--neu-border) !important;
            background: var(--neu-surface) !important;
            box-shadow:
                13px 13px 28px rgba(147, 160, 152, 0.35),
                -13px -13px 28px rgba(255, 255, 255, 0.86) !important;
        }

        [data-testid="stMetric"] {
            padding: 1.05rem 1.15rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--neu-muted) !important;
            font-weight: 750;
        }

        [data-testid="stMetricValue"] {
            color: var(--neu-text);
            font-size: clamp(1.16rem, 1.85vw, 1.9rem);
            font-weight: 850;
            line-height: 1.05;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
        }

        [data-testid="stExpander"] details summary {
            font-weight: 800;
            color: var(--neu-text);
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            border: 1px solid rgba(255, 255, 255, 0.78);
            border-radius: 999px;
            background: linear-gradient(145deg, #fbfcf8, #dfe8e2);
            color: var(--neu-text);
            font-weight: 800;
            box-shadow:
                8px 8px 17px rgba(141, 153, 146, 0.48),
                -8px -8px 17px rgba(255, 255, 255, 0.95);
            transition: transform 130ms ease, box-shadow 130ms ease, color 130ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            color: var(--neu-accent);
            transform: translateY(-1px);
            box-shadow:
                10px 10px 20px rgba(131, 146, 138, 0.52),
                -10px -10px 20px rgba(255, 255, 255, 0.98);
        }

        .stButton > button:active,
        .stDownloadButton > button:active,
        [data-testid="stFormSubmitButton"] button:active {
            transform: translateY(1px);
            box-shadow:
                inset 6px 6px 12px rgba(137, 151, 143, 0.42),
                inset -6px -6px 12px rgba(255, 255, 255, 0.92);
        }

        .stSelectbox [data-baseweb="select"],
        .stMultiSelect [data-baseweb="select"],
        [data-baseweb="input"],
        [data-baseweb="textarea"],
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stDateInput input {
            border: 1px solid rgba(255, 255, 255, 0.72) !important;
            border-radius: var(--neu-radius-sm) !important;
            background: rgba(231, 238, 233, 0.82) !important;
            color: var(--neu-text) !important;
            outline: none !important;
            box-shadow:
                inset 6px 6px 12px rgba(147, 160, 152, 0.34),
                inset -6px -6px 12px rgba(255, 255, 255, 0.88);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            border-color: transparent !important;
            outline: none !important;
            box-shadow: none !important;
        }

        [data-baseweb="input"] *,
        [data-baseweb="textarea"] *,
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            border-color: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"]:focus-within,
        .stSelectbox [data-baseweb="select"]:focus-within,
        .stMultiSelect [data-baseweb="select"]:focus-within {
            border-color: rgba(47, 111, 100, 0.46) !important;
            box-shadow:
                inset 5px 5px 10px rgba(130, 145, 137, 0.3),
                inset -5px -5px 10px rgba(255, 255, 255, 0.9),
                0 0 0 3px rgba(47, 111, 100, 0.12) !important;
        }

        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] * {
            color: var(--neu-text) !important;
            background-color: transparent !important;
        }

        [data-baseweb="popover"] [role="listbox"] {
            border-radius: 18px;
            background: var(--neu-surface-solid);
            box-shadow:
                10px 10px 24px rgba(114, 128, 120, 0.34),
                -10px -10px 24px rgba(255, 255, 255, 0.9);
        }

        .stSlider [data-baseweb="slider"] > div {
            color: var(--neu-accent);
        }

        .stSlider [role="slider"] {
            background-color: var(--neu-accent) !important;
            box-shadow:
                4px 4px 10px rgba(119, 134, 126, 0.42),
                -4px -4px 10px rgba(255, 255, 255, 0.9);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            overflow: hidden;
        }

        [data-testid="stAlert"] {
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.72);
            box-shadow:
                8px 8px 18px rgba(147, 160, 152, 0.28),
                -8px -8px 18px rgba(255, 255, 255, 0.75);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(239, 245, 240, 0.96), rgba(231, 238, 233, 0.96));
            border-right: 1px solid rgba(255, 255, 255, 0.7);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: var(--neu-text) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            box-shadow:
                inset -8px 0 18px rgba(145, 158, 150, 0.14),
                inset 8px 0 18px rgba(255, 255, 255, 0.55);
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
        }

        button[title="Scroll tabs left"],
        button[title="Scroll tabs right"] {
            display: none !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 3.4rem;
            }

            .neu-hero {
                padding: 1.5rem;
                border-radius: 26px;
            }

            .neu-hero h1 {
                font-size: 2.35rem;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 0.35rem 0.7rem;
                min-height: 2.25rem;
            }
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(70, 92, 84, 0.22), transparent);
        }

        ::selection {
            background: rgba(47, 111, 100, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header() -> None:
    """Render the dashboard title as a reusable visual header."""
    st.markdown(
        """
        <section class="neu-hero">
            <p class="neu-kicker">Multimodal market research dashboard</p>
            <h1>Stock Embeddings Dashboard</h1>
            <p>
                Browse filings, study learned company themes, inspect historical
                language shifts, and test sector rotation ideas with point-in-time
                guardrails.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Stock Embeddings Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_neumorphic_theme()
render_dashboard_header()


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
def load_embeddings(path: str) -> pd.DataFrame:
    """Load view embeddings with normalized ticker/date columns."""
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_projected_embeddings(path: str) -> pd.DataFrame:
    """Project view embeddings into one stable 2D plane for model comparison."""
    frame = load_embeddings(path)
    columns = embedding_columns(frame)
    if len(columns) < 2 or len(frame) < 2:
        result = frame.loc[:, ["ticker", "date"]].copy()
        result["x"] = 0.0
        result["y"] = 0.0
        return result
    matrix = (
        frame.loc[:, columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    projected = PCA(n_components=2, random_state=0).fit_transform((matrix - mean) / std)
    result = frame.loc[:, ["ticker", "date"]].copy()
    result["x"] = projected[:, 0]
    result["y"] = projected[:, 1]
    return result


@st.cache_data(show_spinner=False)
def load_projected_market_map(path: str, projection_method: str = "PCA") -> pd.DataFrame:
    """Project all firm-date theme loadings into one stable 2D map."""
    frame = load_loadings(path)
    columns = theme_columns(frame)
    if len(columns) < 2:
        raise ValueError("Market map requires at least two theme columns.")

    matrix = frame.loc[:, columns].astype(float).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    method = str(projection_method).strip().upper()
    if len(frame) < 2:
        projection = np.zeros((len(frame), 2), dtype=float)
        method = "CONSTANT"
    elif method == "UMAP":
        projection = project_with_umap(matrix)
    else:
        method = "PCA"
        projection = PCA(n_components=2, random_state=7).fit_transform(matrix)
    output = frame.loc[:, ["ticker", "date"]].copy()
    output["x"] = projection[:, 0]
    output["y"] = projection[:, 1]
    output["projection_method"] = method
    output["dominant_theme"] = frame.loc[:, columns].astype(float).idxmax(axis=1)
    output["dominant_loading"] = frame.loc[:, columns].astype(float).max(axis=1)
    return output


def project_with_umap(matrix: np.ndarray) -> np.ndarray:
    """Return a UMAP projection when the optional dependency is installed."""
    n_rows = int(matrix.shape[0])
    if n_rows < 3:
        raise RuntimeError("UMAP projection requires at least three rows.")
    try:
        from umap import UMAP
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("UMAP projection requires the optional `umap-learn` package.") from exc

    return UMAP(
        n_components=2,
        n_neighbors=min(25, n_rows - 1),
        min_dist=0.15,
        metric="cosine",
        random_state=7,
        low_memory=True,
    ).fit_transform(matrix)


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
def load_metadata(
    artifact_version: tuple[tuple[str, int, int], ...] | None = None,
    cache_version: int = METADATA_CACHE_VERSION,
) -> pd.DataFrame:
    _ = (artifact_version, cache_version)
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

    if SP500_GICS_PATH.exists():
        gics = pd.read_parquet(SP500_GICS_PATH)
        gics["ticker"] = gics["ticker"].astype(str).str.upper()
        keep_columns = [
            column
            for column in ["ticker", "company_name", "gics_sector", "gics_sub_industry", "date_added"]
            if column in gics.columns
        ]
        frame = frame.merge(gics.loc[:, keep_columns], on="ticker", how="left")
        if "company_name" in frame.columns:
            frame["title"] = frame["title"].fillna(frame["company_name"])
    frame = coalesce_metadata_columns(frame)
    return frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sp500_membership_history(
    artifact_version: tuple[tuple[str, int, int], ...] | None = None,
) -> pd.DataFrame:
    """Load historical S&P 500 membership intervals when available."""
    _ = artifact_version
    path = SP500_MEMBERSHIP_PATH
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    return frame.sort_values(["ticker", "start_date", "end_date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_relationships(
    artifact_version: tuple[tuple[str, int, int], ...] | None = None,
) -> pd.DataFrame:
    """Load filing-derived relationship edges when available."""
    _ = artifact_version
    path = RELATIONSHIPS_PATH
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    for column in ["source_ticker", "target_ticker", "supplier_ticker", "customer_ticker"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.upper()
    if "filing_date" in frame.columns:
        frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    for column in ["confidence", "direction_confidence"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sp500_historical_constituent_prices(
    artifact_version: tuple[tuple[str, int, int], ...] | None = None,
) -> pd.DataFrame:
    """Load legacy sidecar prices for deleted S&P 500 constituents if present."""
    _ = artifact_version
    path = SP500_DELETED_PRICES_PATH
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    returns = frame.groupby("ticker")["adj_close"].pct_change()
    bad_tickers = set(frame.loc[returns.abs() > 3.0, "ticker"])
    if bad_tickers:
        frame = frame[~frame["ticker"].isin(bad_tickers)].copy()
    return frame.reset_index(drop=True)


def append_missing_legacy_historical_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Append legacy sidecar prices only for tickers absent from the DB group."""
    sidecar = load_sp500_historical_constituent_prices(artifact_signature(SP500_DELETED_PRICES_PATH))
    if sidecar.empty:
        return prices
    existing_tickers = set(prices["ticker"].astype(str).str.upper())
    sidecar = sidecar[~sidecar["ticker"].isin(existing_tickers)].copy()
    if sidecar.empty:
        return prices
    combined = pd.concat([prices, sidecar], ignore_index=True, sort=False)
    combined["ticker"] = combined["ticker"].astype(str).str.upper()
    combined["date"] = pd.to_datetime(combined["date"])
    return combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")


@st.cache_data(show_spinner=False)
def load_sp500_benchmark_returns(path: str) -> pd.Series:
    """Load regular S&P 500 benchmark returns from a local SPY proxy parquet."""
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_parquet(benchmark_path)
    if "date" not in frame.columns or "adj_close" not in frame.columns:
        return pd.Series(dtype=float)
    frame = frame.loc[:, ["date", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["date", "adj_close"]).sort_values("date")
    returns = frame.set_index("date")["adj_close"].pct_change().dropna()
    returns.name = "sp500_benchmark_return"
    return returns


@st.cache_data(show_spinner=False)
def load_report_csv(filename: str) -> pd.DataFrame:
    """Load a report CSV if it exists."""
    path = REPO_ROOT / "report" / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def model_decision_frame() -> pd.DataFrame:
    """Return the dashboard's current default model choices and why."""
    return pd.DataFrame(
        [
            {
                "pipeline_step": "Filing text representation",
                "default_model": "MiniLM section embeddings",
                "why_this_is_default": "Small, fast sentence embeddings that work well on SEC section text.",
                "comparison_or_fallback": "Older hashed text features remain available for reproduction.",
            },
            {
                "pipeline_step": "Company embedding",
                "default_model": "Autoencoder over view features",
                "why_this_is_default": "Compresses text, price, fundamentals, or network features into a low-dimensional company vector.",
                "comparison_or_fallback": "PCA and temporal autoencoder experiments are tracked in reports.",
            },
            {
                "pipeline_step": "Theme/category membership",
                "default_model": "GMM soft loadings",
                "why_this_is_default": "Companies can be partly in multiple economic themes; soft loadings preserve that mixed identity.",
                "comparison_or_fallback": "k-means and DBSCAN are kept in the comparison tab as classroom baselines.",
            },
            {
                "pipeline_step": "Behavioral theme returns",
                "default_model": "Hard top-1 assignment",
                "why_this_is_default": "Behavioral clusters behaved more like discrete return groups than blended semantic themes.",
                "comparison_or_fallback": "Business and network views stay soft by default.",
            },
            {
                "pipeline_step": "2D dashboard map",
                "default_model": "PCA projection",
                "why_this_is_default": "Stable and fast enough for time navigation; UMAP is optional for local-neighborhood exploration.",
                "comparison_or_fallback": "UMAP can be selected in the Similarity Explorer.",
            },
            {
                "pipeline_step": "Group excess-return prediction",
                "default_model": "Ridge regression",
                "why_this_is_default": "Few monthly group observations; ridge is transparent and less likely to overfit than flexible trees.",
                "comparison_or_fallback": "Elastic Net, Huber, Random Forest, Extra Trees, and Gradient Boosting can be compared here.",
            },
            {
                "pipeline_step": "Covariance benchmark",
                "default_model": "Ledoit-Wolf",
                "why_this_is_default": "Standard statistical benchmark; our direct multi-view covariance model did not beat it full-universe.",
                "comparison_or_fallback": "Embedding covariance remains useful as a negative result and slice diagnostic.",
            },
        ]
    )


def sector_model_display_name(model_key: str) -> str:
    """Return a selector label for one sector predictor."""
    label = SECTOR_MODEL_LABELS.get(model_key, str(model_key))
    if model_key == DEFAULT_SECTOR_MODEL:
        return f"{label} (default)"
    return label


def sector_prediction_hard_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Return row-level prediction accuracy metrics for completed horizons."""
    completed = completed_sector_predictions(predictions)
    if completed.empty:
        return {
            "prediction_mae": np.nan,
            "prediction_rmse": np.nan,
            "prediction_correlation": np.nan,
            "directional_accuracy": np.nan,
            "positive_precision": np.nan,
            "positive_recall": np.nan,
            "n_completed_rows": 0,
        }

    frame = completed.loc[:, ["predicted_excess_return", "future_excess_return"]].copy()
    frame["predicted_excess_return"] = pd.to_numeric(frame["predicted_excess_return"], errors="coerce")
    frame["future_excess_return"] = pd.to_numeric(frame["future_excess_return"], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        return {
            "prediction_mae": np.nan,
            "prediction_rmse": np.nan,
            "prediction_correlation": np.nan,
            "directional_accuracy": np.nan,
            "positive_precision": np.nan,
            "positive_recall": np.nan,
            "n_completed_rows": 0,
        }

    error = frame["predicted_excess_return"] - frame["future_excess_return"]
    predicted_positive = frame["predicted_excess_return"] > 0.0
    actual_positive = frame["future_excess_return"] > 0.0
    true_positive = predicted_positive & actual_positive
    return {
        "prediction_mae": float(error.abs().mean()),
        "prediction_rmse": float(np.sqrt(np.square(error).mean())),
        "prediction_correlation": float(frame["predicted_excess_return"].corr(frame["future_excess_return"])),
        "directional_accuracy": float((predicted_positive == actual_positive).mean()),
        "positive_precision": float(true_positive.sum() / max(predicted_positive.sum(), 1)),
        "positive_recall": float(true_positive.sum() / max(actual_positive.sum(), 1)),
        "n_completed_rows": int(len(frame)),
    }


def rotation_extra_metrics(simulation: pd.DataFrame) -> dict[str, float]:
    """Return extra simulation risk metrics not covered by the base summary."""
    if simulation.empty:
        return {
            "annualized_volatility": np.nan,
            "sharpe_like": np.nan,
            "avg_period_excess": np.nan,
            "excess_win_rate_vs_spy": np.nan,
            "turnover_rate": np.nan,
        }
    frame = simulation.copy()
    returns = pd.to_numeric(frame.get("net_period_return"), errors="coerce").dropna()
    excess = pd.to_numeric(frame.get("period_excess_return"), errors="coerce").dropna()
    periods_per_year = 252.0 / max(float(DEFAULT_SECTOR_HORIZON_DAYS), 1.0)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else np.nan
    annualized_mean = float(returns.mean() * periods_per_year) if len(returns) else np.nan
    if not np.isfinite(annualized_volatility) or annualized_volatility == 0.0:
        sharpe_like = np.nan
    else:
        sharpe_like = annualized_mean / annualized_volatility

    excess_win_rate_vs_spy = np.nan
    if "period_benchmark_return" in frame.columns:
        benchmark = pd.to_numeric(frame["period_benchmark_return"], errors="coerce")
        comparable = pd.DataFrame({"strategy": returns, "benchmark": benchmark}).dropna()
        if not comparable.empty:
            excess_win_rate_vs_spy = float((comparable["strategy"] > comparable["benchmark"]).mean())

    turnover_rate = np.nan
    if "turnover" in frame.columns:
        turnover_rate = float(pd.to_numeric(frame["turnover"], errors="coerce").mean())

    return {
        "annualized_volatility": annualized_volatility,
        "sharpe_like": float(sharpe_like) if np.isfinite(sharpe_like) else np.nan,
        "avg_period_excess": float(excess.mean()) if len(excess) else np.nan,
        "excess_win_rate_vs_spy": excess_win_rate_vs_spy,
        "turnover_rate": turnover_rate,
    }


def sector_model_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact model leaderboard for the comparison tab."""
    columns = [
        "model",
        "ending_capital",
        "regular_sp500_capital",
        "equal_weight_capital",
        "excess_return_vs_sp500",
        "excess_return_vs_equal_weight",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "period_win_rate",
        "mean_rank_ic",
        "directional_accuracy",
        "status",
    ]
    if frame.empty:
        return frame
    return safe_frame_subset(ensure_columns(frame, columns), columns).sort_values(
        "ending_capital",
        ascending=False,
        na_position="last",
    )


def sector_prediction_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ML-style hard prediction metrics for all sector models."""
    columns = [
        "model",
        "completed_rows",
        "completed_dates",
        "mean_rank_ic",
        "median_rank_ic",
        "top_sector_hit_rate",
        "mean_top_minus_bottom",
        "mean_top_bucket_excess",
        "prediction_mae",
        "prediction_rmse",
        "prediction_correlation",
        "directional_accuracy",
        "positive_precision",
        "positive_recall",
        "status",
    ]
    if frame.empty:
        return frame
    return safe_frame_subset(ensure_columns(frame, columns), columns).sort_values(
        ["mean_rank_ic", "directional_accuracy"],
        ascending=[False, False],
        na_position="last",
    )


def sector_simulation_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return finance/risk simulation metrics for all sector models."""
    columns = [
        "model",
        "n_rebalances",
        "ending_capital",
        "regular_sp500_capital",
        "equal_weight_capital",
        "total_return",
        "excess_return_vs_sp500",
        "excess_return_vs_equal_weight",
        "annualized_return",
        "annualized_volatility",
        "sharpe_like",
        "period_win_rate",
        "excess_win_rate_vs_spy",
        "avg_period_return",
        "avg_period_excess",
        "max_drawdown",
        "turnover_rate",
        "status",
    ]
    if frame.empty:
        return frame
    return safe_frame_subset(ensure_columns(frame, columns), columns).sort_values(
        "ending_capital",
        ascending=False,
        na_position="last",
    )


def compact_experiment_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the most useful embedding experiment columns."""
    columns = [
        "experiment_name",
        "model_name",
        "input_dim",
        "embedding_dim",
        "final_reconstruction_mse",
        "clustering_nmi",
        "clustering_ari",
        "peer_corr_diff",
        "cov_embedding_annual_variance",
        "cov_ledoit_wolf_annual_variance",
        "path",
    ]
    if frame.empty:
        return frame
    return safe_frame_subset(ensure_columns(frame, columns), columns).sort_values(
        "clustering_nmi",
        ascending=False,
        na_position="last",
    )


def compact_sector_autoencoder_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the most useful sector-AE comparison columns."""
    columns = [
        "feature_set",
        "fit_mode",
        "n_features",
        "n_prediction_dates",
        "mean_rank_ic",
        "mean_top_minus_bottom",
        "top_sector_hit_rate",
        "ending_capital",
        "spy_ending_capital",
        "excess_total_return_vs_spy",
        "max_drawdown",
        "selected_embedding_summary",
    ]
    if frame.empty:
        return frame
    return safe_frame_subset(ensure_columns(frame, columns), columns).sort_values(
        "ending_capital",
        ascending=False,
        na_position="last",
    )


def coalesce_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata columns after optional merges with overlapping names."""
    result = frame.copy()
    for column in ["company_name", "gics_sector", "gics_sub_industry"]:
        variants = [name for name in [column, f"{column}_x", f"{column}_y"] if name in result.columns]
        if not variants:
            result[column] = pd.NA
            continue
        result[column] = result[variants].bfill(axis=1).iloc[:, 0]
        if column == "company_name" and "ticker" in result.columns:
            ticker_only = result[column].astype(str).str.upper().eq(result["ticker"].astype(str).str.upper())
            for variant in variants:
                candidate = result[variant]
                valid_candidate = (
                    ticker_only
                    & candidate.notna()
                    & candidate.astype(str).str.strip().ne("")
                    & ~candidate.astype(str).str.upper().eq(result["ticker"].astype(str).str.upper())
                )
                result.loc[valid_candidate, column] = candidate.loc[valid_candidate]
                ticker_only = result[column].astype(str).str.upper().eq(result["ticker"].astype(str).str.upper())
    if "title" not in result.columns:
        result["title"] = result["company_name"].fillna(result["ticker"])
    else:
        result["title"] = result["title"].fillna(result["company_name"]).fillna(result["ticker"])
        ticker_only = result["title"].astype(str).str.upper().eq(result["ticker"].astype(str).str.upper())
        has_company_name = result["company_name"].notna() & result["company_name"].astype(str).str.strip().ne("")
        result.loc[ticker_only & has_company_name, "title"] = result.loc[ticker_only & has_company_name, "company_name"]
    if "search_label" not in result.columns:
        result["search_label"] = result["ticker"].astype(str) + " - " + result["title"].astype(str)
    else:
        result["search_label"] = result["search_label"].fillna(
            result["ticker"].astype(str) + " - " + result["title"].astype(str)
        )
        ticker_label = result["search_label"].astype(str).str.upper().eq(
            result["ticker"].astype(str).str.upper() + " - " + result["ticker"].astype(str).str.upper()
        )
        result.loc[ticker_label, "search_label"] = (
            result.loc[ticker_label, "ticker"].astype(str) + " - " + result.loc[ticker_label, "title"].astype(str)
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
    """Create readable fallback labels from weighted sector/sub-industry mix."""
    loadings = load_loadings(loadings_path)
    latest = loadings.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest = latest.merge(metadata, on="ticker", how="left")
    lookup = {}
    for theme in theme_columns(latest):
        lookup[theme] = composition_theme_label(latest, theme, view_name)
    return lookup


def composition_theme_label(frame: pd.DataFrame, theme: str, view_name: str, top_n: int = 20) -> str:
    """Return a dynamic label from the weighted composition of a theme."""
    top = frame.sort_values(theme, ascending=False).head(int(top_n)).copy()
    top = top.loc[pd.to_numeric(top[theme], errors="coerce") > 0.0]
    noun = view_label_noun(view_name)
    if top.empty:
        return f"Mixed {noun}"

    subindustry_label = dominant_weighted_label(top, theme, "gics_sub_industry", minimum_share=0.38)
    if subindustry_label:
        return f"{subindustry_label} {noun}"

    sector_label = dominant_weighted_label(top, theme, "gics_sector", minimum_share=0.34)
    if sector_label:
        return f"{sector_label} {noun}"

    blended = blended_weighted_label(top, theme, "gics_sector", k=2)
    if blended:
        return f"{blended} {noun}"
    return f"Mixed {noun}"


def dominant_weighted_label(frame: pd.DataFrame, weight_column: str, label_column: str, minimum_share: float) -> str:
    """Return the dominant weighted label if it clears a minimum share."""
    if label_column not in frame.columns:
        return ""
    values = weighted_label_shares(frame, weight_column, label_column)
    if values.empty:
        return ""
    label = str(values.index[0])
    share = float(values.iloc[0])
    return label if share >= float(minimum_share) else ""


def blended_weighted_label(frame: pd.DataFrame, weight_column: str, label_column: str, k: int = 2) -> str:
    """Return a compact multi-label composition summary."""
    if label_column not in frame.columns:
        return ""
    values = weighted_label_shares(frame, weight_column, label_column)
    if values.empty:
        return ""
    labels = [str(label) for label in values.head(int(k)).index]
    return " / ".join(labels)


def weighted_label_shares(frame: pd.DataFrame, weight_column: str, label_column: str) -> pd.Series:
    """Return normalized loading shares by metadata label."""
    if label_column not in frame.columns or weight_column not in frame.columns:
        return pd.Series(dtype=float)
    clean = frame.loc[:, [weight_column, label_column]].dropna().copy()
    if clean.empty:
        return pd.Series(dtype=float)
    clean[weight_column] = pd.to_numeric(clean[weight_column], errors="coerce")
    clean = clean.dropna(subset=[weight_column])
    clean = clean[clean[weight_column] > 0.0]
    if clean.empty:
        return pd.Series(dtype=float)
    shares = clean.groupby(label_column)[weight_column].sum().sort_values(ascending=False)
    total = float(shares.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return shares / total


@st.cache_data(show_spinner=False)
def fragment_theme_label_artifacts(
    loadings_path: str,
    historical_dir: str,
    view_name: str,
    as_of_date: str | None,
    metadata: pd.DataFrame | None = None,
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
        topic_lookup = topics.set_index(["ticker", "accession_no", "section"], drop=False).sort_index()
    filing_index = load_historical_filing_index(historical_dir)
    source_lookup = pd.DataFrame()
    if not filing_index.empty:
        source_lookup = filing_index.set_index(["ticker", "accession_no"], drop=False).sort_index()
    snippets = load_historical_snippets(historical_dir)
    snippet_lookup = pd.DataFrame()
    if not snippets.empty:
        snippet_lookup = snippets.set_index(["ticker", "accession_no", "section"], drop=False).sort_index()

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
        label_phrase = snippet_label_phrase(
            str(snippet_row.get("evidence_snippet", "")) if not snippet_row.empty else "",
            str(snippet_row.get("snippet_terms", "")) if not snippet_row.empty else "",
        )
        label_core = fragment_display_label(topic_label, section_label, top, theme, metadata, label_phrase)
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
            "label_phrase": label_phrase,
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


def fragment_display_label(
    topic_label: str,
    section_label: str,
    top_loadings: pd.DataFrame | None = None,
    theme: str | None = None,
    metadata: pd.DataFrame | None = None,
    label_phrase: str = "",
) -> str:
    """Return a compact semantic label for a representative filing fragment."""
    composition = ""
    if top_loadings is not None and theme is not None and metadata is not None and not metadata.empty:
        enriched = top_loadings.merge(metadata, on="ticker", how="left")
        composition = dominant_weighted_label(enriched, theme, "gics_sub_industry", minimum_share=0.38)
        if not composition:
            composition = dominant_weighted_label(enriched, theme, "gics_sector", minimum_share=0.34)

    section_context = {
        "Business": "business",
        "Risk Factors": "risk",
        "MD&A": "management discussion",
        "Cybersecurity": "cybersecurity",
        "Q MD&A": "quarterly management discussion",
        "Q Risk Factors": "quarterly risk",
    }.get(section_label, section_label.lower())

    if label_phrase:
        if topic_label:
            return f"{label_phrase} - {topic_label}"
        if composition:
            return f"{label_phrase} - {composition}"
        return label_phrase
    if topic_label:
        if composition:
            return f"{composition} {topic_label} language"
        return f"{topic_label} {section_context} language"
    if composition:
        return f"{composition} {section_context} language"
    section_labels = {
        "Business": "Business model language",
        "Risk Factors": "Risk language",
        "MD&A": "Management discussion language",
        "Cybersecurity": "Cybersecurity language",
        "Q MD&A": "Quarterly management discussion",
        "Q Risk Factors": "Quarterly risk language",
    }
    return section_labels.get(section_label, f"{section_label} language")


def snippet_label_phrase(snippet: str, terms: str, max_words: int = MAX_DYNAMIC_LABEL_WORDS) -> str:
    """Return a short phrase from the representative evidence snippet.

    The full snippet remains in the evidence table. This phrase is intentionally
    short enough for chart legends while still coming from the central filing
    fragment instead of from a manual theme dictionary.
    """
    text = compact_label_text(snippet)
    if not text:
        return ""

    term_values = [
        compact_label_text(value).lower()
        for value in re.split(r"[,;/|]", str(terms))
        if compact_label_text(value)
    ]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", text) if part.strip()]
    if not sentences:
        sentences = [text]

    sentence = sentences[0]
    for candidate in sentences:
        lower = candidate.lower()
        if any(term and term in lower for term in term_values):
            sentence = candidate
            break

    words = sentence.split()
    if not words:
        return ""

    start = 0
    lower_words = [word.strip(".,;:()[]{}\"'").lower() for word in words]
    for term in term_values:
        term_first_word = term.split()[0] if term else ""
        if term_first_word in lower_words:
            center = lower_words.index(term_first_word)
            start = max(0, center - 3)
            break

    phrase = " ".join(words[start : start + max_words])
    phrase = compact_label_text(phrase).strip(" .,:;")
    phrase = strip_label_lead_in(phrase)
    if len(phrase) > MAX_DYNAMIC_LABEL_CHARS:
        phrase = phrase[:MAX_DYNAMIC_LABEL_CHARS].rsplit(" ", 1)[0].strip(" .,:;")
    if not phrase or is_weak_dynamic_label_phrase(phrase):
        return ""
    return phrase[0].upper() + phrase[1:]


def is_weak_dynamic_label_phrase(phrase: str) -> bool:
    """Return true when a snippet phrase is only filing boilerplate."""
    cleaned = compact_label_text(phrase).strip(" .,:;").lower()
    if not cleaned:
        return True
    if re.fullmatch(r"(item|part)\s+[0-9ivxlc]+[a-z]?", cleaned):
        return True
    if re.fullmatch(r"(item|part)\s+[0-9ivxlc]+[a-z]?\s+[-–—]?\s*(business|risk factors|md&a|management discussion)?", cleaned):
        return True
    boilerplate = {
        "table of contents",
        "management's discussion and analysis",
        "management discussion and analysis",
        "risk factors",
        "business",
    }
    if cleaned in boilerplate:
        return True
    alpha_words = re.findall(r"[a-z]{3,}", cleaned)
    return len(alpha_words) < 2


def compact_label_text(value: str) -> str:
    """Normalize whitespace for dynamic labels and snippets."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def strip_label_lead_in(phrase: str) -> str:
    """Remove common filing boilerplate from the front of a short label."""
    cleaned = phrase.strip()
    patterns = [
        r"^(we|our|the company|the corporation)\s+(believe|expect|intend|continue|may|are|have|use)\s+",
        r"^(including|such as)\s+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .,:;")


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
        candidates = row.sort_values("snippet_rank").copy()
        candidates["_label_phrase"] = candidates.apply(
            lambda item: snippet_label_phrase(
                str(item.get("evidence_snippet", "") or ""),
                str(item.get("snippet_terms", "") or ""),
            ),
            axis=1,
        )
        usable = candidates[candidates["_label_phrase"].astype(str).ne("")]
        topical = usable[~usable["snippet_topic"].astype(str).eq("section_start")]
        if not topical.empty:
            return topical.drop(columns=["_label_phrase"]).iloc[0]
        if not usable.empty:
            return usable.drop(columns=["_label_phrase"]).iloc[0]
        non_start = candidates[~candidates["snippet_topic"].astype(str).eq("section_start")]
        if not non_start.empty:
            return non_start.drop(columns=["_label_phrase"]).iloc[0]
        return candidates.drop(columns=["_label_phrase"]).iloc[0]
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
        "Item 1 Business": "Business",
        "10-K Item 1A Risk Factors": "Risk Factors",
        "Item 1A Risk Factors": "Risk Factors",
        "10-K Item 7 MD&A": "MD&A",
        "Item 7 MD&A": "MD&A",
        "10-K Item 1C Cybersecurity": "Cybersecurity",
        "Item 1C Cybersecurity": "Cybersecurity",
        "10-Q Item 2 MD&A": "Q MD&A",
        "Item 2 MD&A": "Q MD&A",
        "10-Q Part II Item 1A Risk Factors": "Q Risk Factors",
        "Part II Item 1A Risk Factors": "Q Risk Factors",
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


def available_embedding_views(experiment_root: Path) -> list[str]:
    """Return decomposed views with saved embeddings."""
    views_dir = experiment_root / "views"
    if not views_dir.exists():
        return []
    return sorted(path.name for path in views_dir.iterdir() if (path / "embeddings.parquet").exists())


def map_visible_views(views: list[str]) -> list[str]:
    """Hide views that are useful numerically but misleading as cluster maps."""
    return [view for view in views if view not in MAP_EXCLUDED_VIEWS]


def parse_tickers(raw_value: str) -> list[str]:
    parts = [part.strip().upper() for part in raw_value.replace("\n", ",").split(",")]
    seen: set[str] = set()
    tickers: list[str] = []
    for ticker in parts:
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def artifact_signature(*paths: Path | str) -> tuple[tuple[str, int, int], ...]:
    """Return a stable cache-key fragment for local artifact freshness."""
    signature = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            stat = path.stat()
            signature.append((str(path), int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            signature.append((str(path), 0, 0))
    return tuple(signature)


def sector_outlook_artifact_signature(
    *,
    group_mode: str,
    loadings_path: str,
    embeddings_path: str,
    membership_mode: str,
    include_embedding_features: bool,
) -> tuple[tuple[str, int, int], ...]:
    """Return artifact versions that should invalidate cached sector runs."""
    paths: list[Path | str] = [
        SP500_GICS_PATH,
        VALUATION_FEATURE_PATH,
        GROWTH_FEATURE_PATH,
    ]
    if membership_mode == "historical":
        paths.extend([SP500_MEMBERSHIP_PATH, SP500_DELETED_PRICES_PATH])
    if group_mode == "theme" and loadings_path:
        paths.append(loadings_path)
    if include_embedding_features and embeddings_path:
        paths.append(embeddings_path)
    return artifact_signature(*paths)


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
        "label_phrase",
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
) -> pd.DataFrame:
    """Return one date of map points with one label per actual dominant theme."""
    date_frame = projected[projected["date"] == selected_date].copy()
    if date_frame.empty:
        return date_frame

    date_frame["theme_label"] = [
        display_theme_name(theme, labels)
        for theme in date_frame["dominant_theme"].astype(str)
    ]
    metadata_subset = ensure_columns(
        metadata,
        ["ticker", "title", "search_label", "gics_sector", "gics_sub_industry"],
    ).loc[:, ["ticker", "title", "search_label", "gics_sector", "gics_sub_industry"]].copy()
    date_frame = date_frame.merge(metadata_subset, on="ticker", how="left")
    date_frame["title"] = date_frame["title"].fillna(date_frame["ticker"])
    date_frame["search_label"] = date_frame["search_label"].fillna(date_frame["ticker"])
    date_frame["gics_sector"] = date_frame["gics_sector"].fillna("Unknown")
    date_frame["gics_sub_industry"] = date_frame["gics_sub_industry"].fillna("Unknown")
    date_frame["chart_label"] = date_frame.apply(
        lambda row: compact_company_name(row.get("title", ""), row.get("ticker", ""), max_chars=28),
        axis=1,
    )
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
                alt.Tooltip("chart_label:N", title="Company"),
                alt.Tooltip("ticker:N", title="Ticker"),
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
            text="chart_label:N",
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


def theme_interpretation_frame(
    points: pd.DataFrame,
    relationships: pd.DataFrame,
    max_rows: int = 12,
) -> pd.DataFrame:
    """Summarize visible map themes in human terms."""
    if points.empty:
        return pd.DataFrame()
    rows = []
    supply_chain = pd.DataFrame()
    if not relationships.empty and {"supplier_ticker", "customer_ticker"}.issubset(relationships.columns):
        supply_chain = relationships[relationships["supplier_ticker"].astype(str).ne("")].copy()

    for theme_label, group in points.groupby("theme_label"):
        group = group.sort_values("dominant_loading", ascending=False)
        tickers = set(group["ticker"].astype(str))
        sector_counts = group["gics_sector"].fillna("Unknown").value_counts(normalize=True)
        top_sector = ""
        if not sector_counts.empty:
            top_sector = f"{sector_counts.index[0]} ({sector_counts.iloc[0]:.0%})"
        top_stocks = ", ".join(group["ticker"].head(8).astype(str).tolist())
        examples = relationship_examples_for_tickers(supply_chain, tickers, limit=3)
        rows.append(
            {
                "theme": theme_label,
                "stocks": int(group["ticker"].nunique()),
                "median_loading": float(group["dominant_loading"].median()),
                "dominant_sector": top_sector,
                "top_stocks": top_stocks,
                "supply_chain_examples": examples,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["stocks", "median_loading"], ascending=False).head(int(max_rows)).reset_index(drop=True)


def relationship_examples_for_tickers(supply_chain: pd.DataFrame, tickers: set[str], limit: int = 3) -> str:
    """Return compact supplier-customer examples touching a ticker set."""
    if supply_chain.empty or not tickers:
        return ""
    internal = supply_chain[
        supply_chain["supplier_ticker"].isin(tickers)
        & supply_chain["customer_ticker"].isin(tickers)
    ].copy()
    touching = supply_chain[
        supply_chain["supplier_ticker"].isin(tickers)
        | supply_chain["customer_ticker"].isin(tickers)
    ].copy()
    candidates = internal if not internal.empty else touching
    if candidates.empty:
        return ""
    if "direction_confidence" in candidates.columns:
        candidates = candidates.sort_values("direction_confidence", ascending=False)
    pairs = []
    seen = set()
    for _, row in candidates.iterrows():
        supplier = str(row["supplier_ticker"])
        customer = str(row["customer_ticker"])
        key = (supplier, customer)
        if not supplier or not customer or key in seen:
            continue
        seen.add(key)
        pairs.append(f"{supplier}->{customer}")
        if len(pairs) >= int(limit):
            break
    return ", ".join(pairs)


def focus_theme_loading_frame(
    loadings: pd.DataFrame,
    selected_date: pd.Timestamp,
    focus_ticker: str,
    labels: dict[str, str],
    top_n: int = 8,
) -> pd.DataFrame:
    """Return the selected company's strongest theme memberships."""
    columns = theme_columns(loadings)
    row = loadings[
        loadings["date"].eq(pd.Timestamp(selected_date))
        & loadings["ticker"].eq(str(focus_ticker).upper())
    ]
    if row.empty or not columns:
        return pd.DataFrame()
    row = row.iloc[0]
    frame = pd.DataFrame(
        {
            "theme": [display_theme_name(column, labels) for column in columns],
            "loading": [float(row[column]) for column in columns],
        }
    )
    return frame.sort_values("loading", ascending=False).head(int(top_n)).reset_index(drop=True)


def nearest_theme_loading_peers(
    loadings: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_date: pd.Timestamp,
    focus_ticker: str,
    labels: dict[str, str],
    top_n: int = 12,
) -> pd.DataFrame:
    """Find nearest peers using the actual theme-loading vector, not the 2D map."""
    date_frame = loadings[loadings["date"].eq(pd.Timestamp(selected_date))].copy()
    columns = theme_columns(date_frame)
    focus_ticker = str(focus_ticker).upper()
    if date_frame.empty or focus_ticker not in set(date_frame["ticker"]) or not columns:
        return pd.DataFrame()

    matrix = date_frame.loc[:, columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    tickers = date_frame["ticker"].astype(str).to_numpy()
    focus_index = int(np.where(tickers == focus_ticker)[0][0])
    focus_vector = matrix[focus_index]
    norms = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(focus_vector)), 1e-12)
    similarities = np.divide(matrix @ focus_vector, norms, out=np.zeros(len(matrix)), where=norms > 0)
    distances = np.linalg.norm(matrix - focus_vector, axis=1)
    dominant_columns = date_frame.loc[:, columns].astype(float).idxmax(axis=1)

    peers = pd.DataFrame(
        {
            "ticker": tickers,
            "loading_similarity": similarities,
            "loading_distance": distances,
            "dominant_theme": [display_theme_name(theme, labels) for theme in dominant_columns.astype(str)],
        }
    )
    peers = peers[peers["ticker"] != focus_ticker].sort_values(
        ["loading_similarity", "loading_distance"],
        ascending=[False, True],
    )
    keep = ["ticker", "title", "gics_sector", "gics_sub_industry"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep].drop_duplicates("ticker")
    peers = peers.merge(metadata_subset, on="ticker", how="left")
    peers["title"] = peers["title"].fillna(peers["ticker"])
    return peers.head(int(top_n)).reset_index(drop=True)


def focus_relationship_frame(
    relationships: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_ticker: str,
    max_rows: int = 12,
) -> pd.DataFrame:
    """Return directed supplier/customer links around one company."""
    if relationships.empty:
        return pd.DataFrame()
    focus_ticker = str(focus_ticker).upper()
    frame = relationships[
        relationships["supplier_ticker"].eq(focus_ticker)
        | relationships["customer_ticker"].eq(focus_ticker)
    ].copy()
    if frame.empty:
        return frame
    frame["direction"] = np.where(
        frame["supplier_ticker"].eq(focus_ticker),
        "supplies",
        "buys from",
    )
    frame["counterparty"] = np.where(
        frame["supplier_ticker"].eq(focus_ticker),
        frame["customer_ticker"],
        frame["supplier_ticker"],
    )
    keep = ["ticker", "title", "gics_sector"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep].drop_duplicates("ticker")
    frame = frame.merge(metadata_subset, left_on="counterparty", right_on="ticker", how="left")
    frame["counterparty_name"] = frame["title"].fillna(frame["counterparty"])
    frame["counterparty_sector"] = frame["gics_sector"].fillna("Unknown")
    if "direction_confidence" in frame.columns:
        frame = frame.sort_values("direction_confidence", ascending=False)
    columns = [
        "direction",
        "counterparty",
        "counterparty_name",
        "counterparty_sector",
        "relationship_type",
        "filing_date",
        "context_snippet",
    ]
    return safe_frame_subset(ensure_columns(frame, columns), columns).head(int(max_rows)).reset_index(drop=True)


def clean_relationship_value(value: object) -> str:
    """Return a clean string for relationship-table values."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def compact_company_name(name: object, ticker: object, max_chars: int = 30) -> str:
    """Return a compact company label that is readable inside charts."""
    ticker_text = clean_relationship_value(ticker).upper()
    name_text = clean_relationship_value(name)
    if not name_text or name_text.upper() == ticker_text:
        return ticker_text
    name_text = re.sub(r"\s+", " ", name_text)
    name_text = re.sub(
        r",?\s+(inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|plc|class [a-z])$",
        "",
        name_text,
        flags=re.IGNORECASE,
    ).strip(" ,")
    if len(name_text) > max_chars:
        name_text = name_text[: max_chars - 1].rstrip() + "..."
    return f"{name_text} ({ticker_text})"


def relationship_strength(row: pd.Series) -> float:
    """Return the best available confidence score for one relationship row."""
    values = []
    for column in ["direction_confidence", "confidence"]:
        value = pd.to_numeric(row.get(column, np.nan), errors="coerce")
        if np.isfinite(value):
            values.append(float(value))
    return max(values) if values else 0.0


def relationship_side_for_focus(row: pd.Series, focus_ticker: str) -> dict[str, str | float]:
    """Return focal-company relationship semantics for a raw edge row."""
    focus = str(focus_ticker).upper()
    supplier = clean_relationship_value(row.get("supplier_ticker", "")).upper()
    customer = clean_relationship_value(row.get("customer_ticker", "")).upper()
    source = clean_relationship_value(row.get("source_ticker", "")).upper()
    target = clean_relationship_value(row.get("target_ticker", "")).upper()
    relationship_type = clean_relationship_value(row.get("relationship_type", "")).lower() or "mention"

    if supplier and customer:
        if supplier == focus and customer != focus:
            return {
                "relationship_side": "Customer",
                "counterparty": customer,
                "direction": f"{focus} supplies {customer}",
                "relationship_label": "customer / buyer",
            }
        if customer == focus and supplier != focus:
            return {
                "relationship_side": "Supplier",
                "counterparty": supplier,
                "direction": f"{supplier} supplies {focus}",
                "relationship_label": "supplier",
            }

    counterparty = ""
    if source == focus and target != focus:
        counterparty = target
    elif target == focus and source != focus:
        counterparty = source
    if not counterparty:
        return {
            "relationship_side": "Other",
            "counterparty": "",
            "direction": "",
            "relationship_label": relationship_type,
        }

    if relationship_type == "competitor":
        side = "Competitor"
        label = "competitor"
    elif relationship_type in {"partner", "agreement"}:
        side = "Partner / agreement"
        label = relationship_type
    else:
        side = "Other"
        label = relationship_type

    return {
        "relationship_side": side,
        "counterparty": counterparty,
        "direction": f"{focus} linked with {counterparty}",
        "relationship_label": label,
    }


def current_network_edges(
    relationships: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_ticker: str,
    max_rows: int = 50,
) -> pd.DataFrame:
    """Return current-state relationship evidence around one focal company."""
    if relationships.empty:
        return pd.DataFrame()
    focus = str(focus_ticker).upper()
    frame = ensure_columns(
        relationships,
        [
            "source_ticker",
            "target_ticker",
            "relationship_type",
            "filing_date",
            "form",
            "confidence",
            "matched_alias",
            "source_role",
            "target_role",
            "supplier_ticker",
            "customer_ticker",
            "direction_confidence",
            "context_snippet",
        ],
    )
    touch_mask = (
        frame["source_ticker"].eq(focus)
        | frame["target_ticker"].eq(focus)
        | frame["supplier_ticker"].eq(focus)
        | frame["customer_ticker"].eq(focus)
    )
    frame = frame.loc[touch_mask].copy()
    if frame.empty:
        return frame

    semantic_rows = []
    for _, row in frame.iterrows():
        semantics = relationship_side_for_focus(row, focus)
        if not semantics["counterparty"]:
            continue
        semantic_rows.append(semantics)
    if not semantic_rows:
        return pd.DataFrame()

    semantic_frame = pd.DataFrame(semantic_rows, index=frame.index)
    frame = pd.concat([frame, semantic_frame], axis=1)
    frame["strength"] = frame.apply(relationship_strength, axis=1)
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")

    keep = ["ticker", "title", "gics_sector", "gics_sub_industry"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep].drop_duplicates("ticker")
    frame = frame.merge(metadata_subset, left_on="counterparty", right_on="ticker", how="left")
    frame["counterparty_name"] = frame["title"].fillna(frame["counterparty"])
    frame["counterparty_sector"] = frame["gics_sector"].fillna("Unknown")
    frame["counterparty_industry"] = frame["gics_sub_industry"].fillna("Unknown")

    frame = frame.sort_values(["strength", "filing_date"], ascending=[False, False], na_position="last")
    frame = frame.drop_duplicates(["relationship_side", "counterparty"], keep="first")
    columns = [
        "relationship_side",
        "direction",
        "counterparty",
        "counterparty_name",
        "counterparty_sector",
        "counterparty_industry",
        "relationship_label",
        "relationship_type",
        "strength",
        "confidence",
        "direction_confidence",
        "filing_date",
        "form",
        "matched_alias",
        "context_snippet",
    ]
    return safe_frame_subset(ensure_columns(frame, columns), columns).head(int(max_rows)).reset_index(drop=True)


def network_company_overview(relationships: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return a current-state company-level summary of relationship graph coverage."""
    if relationships.empty:
        return pd.DataFrame()
    count_rows = []
    for column in ["source_ticker", "target_ticker", "supplier_ticker", "customer_ticker"]:
        if column not in relationships.columns:
            continue
        values = relationships[column].fillna("").astype(str).str.upper()
        values = values[values.ne("")]
        if values.empty:
            continue
        count_rows.append(values.rename("ticker").to_frame())
    if not count_rows:
        return pd.DataFrame()
    counts = (
        pd.concat(count_rows, ignore_index=True)["ticker"]
        .value_counts()
        .rename_axis("ticker")
        .reset_index(name="link_rows")
    )

    edge_rows = []
    frame = ensure_columns(
        relationships,
        ["source_ticker", "target_ticker", "relationship_type", "supplier_ticker", "customer_ticker"],
    )
    for _, row in frame.iterrows():
        supplier = clean_relationship_value(row.get("supplier_ticker", "")).upper()
        customer = clean_relationship_value(row.get("customer_ticker", "")).upper()
        source = clean_relationship_value(row.get("source_ticker", "")).upper()
        target = clean_relationship_value(row.get("target_ticker", "")).upper()
        relationship_type = clean_relationship_value(row.get("relationship_type", "")).lower()
        if supplier and customer and supplier != customer:
            edge_rows.append({"ticker": supplier, "counterparty": customer, "relationship_side": "Customer"})
            edge_rows.append({"ticker": customer, "counterparty": supplier, "relationship_side": "Supplier"})
            continue
        if source and target and source != target:
            if relationship_type == "competitor":
                side = "Competitor"
            elif relationship_type in {"partner", "agreement"}:
                side = "Partner / agreement"
            else:
                side = "Other"
            edge_rows.append({"ticker": source, "counterparty": target, "relationship_side": side})
            edge_rows.append({"ticker": target, "counterparty": source, "relationship_side": side})

    if edge_rows:
        edge_frame = pd.DataFrame(edge_rows).drop_duplicates(["ticker", "counterparty", "relationship_side"])
        direct = edge_frame.groupby("ticker")["counterparty"].nunique().reset_index(name="direct_counterparties")
        side_counts = pd.crosstab(edge_frame["ticker"], edge_frame["relationship_side"]).reset_index()
        overview = counts.merge(direct, on="ticker", how="left").merge(side_counts, on="ticker", how="left")
    else:
        overview = counts.copy()
        overview["direct_counterparties"] = 0
    rename_columns = {
        "Supplier": "suppliers",
        "Customer": "customers",
        "Competitor": "competitors",
        "Partner / agreement": "partners_or_agreements",
    }
    overview = overview.rename(columns=rename_columns)
    for column in ["direct_counterparties", "suppliers", "customers", "competitors", "partners_or_agreements"]:
        if column not in overview.columns:
            overview[column] = 0
        overview[column] = overview[column].fillna(0).astype(int)
    keep = ["ticker", "title", "gics_sector", "gics_sub_industry", "search_label"]
    metadata_subset = ensure_columns(metadata, keep).loc[:, keep].drop_duplicates("ticker")
    overview = overview.merge(metadata_subset, on="ticker", how="left")
    overview["title"] = overview["title"].fillna(overview["ticker"])
    overview["search_label"] = overview["search_label"].fillna(overview["ticker"] + " - " + overview["title"])
    return overview.sort_values(["direct_counterparties", "link_rows"], ascending=False).reset_index(drop=True)


def network_relationship_plot_data(
    edges: pd.DataFrame,
    focus_ticker: str,
    focus_name: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return node and line data for the focal relationship chart."""
    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    side_x = {
        "Supplier": -1.35,
        "Customer": 1.35,
        "Competitor": 0.0,
        "Partner / agreement": 0.75,
        "Other": -0.75,
    }
    side_y_offset = {
        "Supplier": 0.0,
        "Customer": 0.0,
        "Competitor": -1.2,
        "Partner / agreement": 1.15,
        "Other": 1.15,
    }
    rows = [
        {
            "ticker": focus_ticker,
            "label": compact_company_name(focus_name, focus_ticker, max_chars=34),
            "x": 0.0,
            "y": 0.0,
            "relationship_side": "Focus",
            "strength": 1.0,
            "counterparty_name": focus_name or focus_ticker,
        }
    ]
    line_rows = []
    for side, group in edges.groupby("relationship_side", sort=False):
        group = group.sort_values(["strength", "counterparty"], ascending=[False, True]).reset_index(drop=True)
        n = len(group)
        for index, row in group.iterrows():
            x = side_x.get(str(side), -0.75)
            spacing = 0.38 if n > 10 else 0.48
            y = side_y_offset.get(str(side), 0.0) + (index - (n - 1) / 2) * spacing
            edge_id = f"{side}_{row['counterparty']}_{index}"
            counterparty_name = row.get("counterparty_name", row["counterparty"])
            rows.append(
                {
                    "ticker": row["counterparty"],
                    "label": compact_company_name(counterparty_name, row["counterparty"]),
                    "x": x,
                    "y": y,
                    "relationship_side": side,
                    "strength": float(row.get("strength", 0.0) or 0.0),
                    "counterparty_name": counterparty_name,
                }
            )
            line_rows.extend(
                [
                    {
                        "edge_id": edge_id,
                        "x": 0.0,
                        "y": 0.0,
                        "relationship_side": side,
                        "strength": float(row.get("strength", 0.0) or 0.0),
                    },
                    {
                        "edge_id": edge_id,
                        "x": x,
                        "y": y,
                        "relationship_side": side,
                        "strength": float(row.get("strength", 0.0) or 0.0),
                    },
                ]
            )
    return pd.DataFrame(rows), pd.DataFrame(line_rows)


def network_relationship_chart(edges: pd.DataFrame, focus_ticker: str, focus_name: str = "") -> alt.Chart:
    """Render a focal-company current-state relationship map."""
    nodes, lines = network_relationship_plot_data(edges, focus_ticker, focus_name)
    side_counts = nodes[nodes["relationship_side"].ne("Focus")]["relationship_side"].value_counts()
    chart_height = int(min(900, max(540, 180 + 42 * int(side_counts.max() if not side_counts.empty else 1))))
    color = alt.Scale(
        domain=["Focus", "Supplier", "Customer", "Competitor", "Partner / agreement", "Other"],
        range=["#24312f", "#2f6f64", "#c28645", "#8c4f3d", "#4f6f9f", "#8a8f86"],
    )
    edge_layer = (
        alt.Chart(lines)
        .mark_line(strokeWidth=2.2, opacity=0.52)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.9, 1.9])),
            y=alt.Y("y:Q", axis=None),
            detail="edge_id:N",
            color=alt.Color("relationship_side:N", scale=color, legend=alt.Legend(title="Relationship")),
            opacity=alt.Opacity("strength:Q", scale=alt.Scale(domain=[0, 1], range=[0.25, 0.8]), legend=None),
        )
    )
    node_layer = (
        alt.Chart(nodes)
        .mark_circle(stroke="#ffffff", strokeWidth=2)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.9, 1.9])),
            y=alt.Y("y:Q", axis=None),
            size=alt.condition(
                alt.datum.relationship_side == "Focus",
                alt.value(1150),
                alt.Size("strength:Q", scale=alt.Scale(domain=[0, 1], range=[220, 680]), legend=None),
            ),
            color=alt.Color("relationship_side:N", scale=color, legend=None),
            tooltip=[
                "ticker:N",
                alt.Tooltip("counterparty_name:N", title="company"),
                "relationship_side:N",
                alt.Tooltip("strength:Q", format=".2f"),
            ],
        )
    )
    left_labels = (
        alt.Chart(nodes[nodes["x"].le(0.0) & nodes["relationship_side"].ne("Focus")])
        .mark_text(dx=12, dy=-1, align="left", fontSize=13, fontWeight="bold", color="#24312f")
        .encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.9, 1.9])), y="y:Q", text="label:N")
    )
    right_labels = (
        alt.Chart(nodes[nodes["x"].gt(0.0)])
        .mark_text(dx=-12, dy=-1, align="right", fontSize=13, fontWeight="bold", color="#24312f")
        .encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.9, 1.9])), y="y:Q", text="label:N")
    )
    focus_label = (
        alt.Chart(nodes[nodes["relationship_side"].eq("Focus")])
        .mark_text(dy=-28, align="center", fontSize=15, fontWeight="bold", color="#24312f")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )
    return (edge_layer + node_layer + left_labels + right_labels + focus_label).properties(height=chart_height)


def network_sector_chart(edges: pd.DataFrame) -> alt.Chart:
    """Render counterparty sector composition for a focal network."""
    sectors = (
        edges["counterparty_sector"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("sector")
        .reset_index(name="links")
        .head(10)
    )
    return (
        alt.Chart(sectors)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, color="#2f6f64")
        .encode(
            x=alt.X("links:Q", title="Links"),
            y=alt.Y("sector:N", sort="-x", title=None),
            tooltip=["sector:N", "links:Q"],
        )
        .properties(height=320)
    )


def network_strength_chart(edges: pd.DataFrame) -> alt.Chart:
    """Render a readable ranked chart of focal-company relationship links."""
    frame = edges.copy()
    if "counterparty_display" not in frame.columns:
        frame["counterparty_display"] = frame.apply(
            lambda row: compact_company_name(row.get("counterparty_name", ""), row.get("counterparty", ""), max_chars=46),
            axis=1,
        )
    frame["snippet_short"] = frame["context_snippet"].fillna("").astype(str).str.slice(0, 220)
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X(
                "strength:Q",
                title="Evidence strength",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "counterparty_display:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=310),
            ),
            color=alt.Color(
                "relationship_side:N",
                title="Relationship",
                scale=alt.Scale(
                    domain=["Supplier", "Customer", "Competitor", "Partner / agreement", "Other"],
                    range=["#2f6f64", "#c28645", "#8c4f3d", "#4f6f9f", "#8a8f86"],
                ),
            ),
            tooltip=[
                alt.Tooltip("counterparty_display:N", title="Company"),
                alt.Tooltip("relationship_side:N", title="Relationship"),
                alt.Tooltip("relationship_label:N", title="Label"),
                alt.Tooltip("strength:Q", title="Strength", format=".2f"),
                alt.Tooltip("snippet_short:N", title="Evidence"),
            ],
        )
        .properties(height=int(min(620, max(320, 28 * len(frame)))))
    )


def focus_dominant_theme_members(points: pd.DataFrame, focus_ticker: str, max_rows: int = 15) -> pd.DataFrame:
    """Return companies sharing the focus ticker's dominant map theme."""
    if points.empty:
        return pd.DataFrame()
    focus = points[points["ticker"].eq(str(focus_ticker).upper())]
    if focus.empty:
        return pd.DataFrame()
    theme = str(focus.iloc[0]["theme_label"])
    members = points[points["theme_label"].eq(theme)].copy()
    columns = ["ticker", "title", "gics_sector", "gics_sub_industry", "dominant_loading"]
    return (
        safe_frame_subset(ensure_columns(members, columns), columns)
        .sort_values("dominant_loading", ascending=False)
        .head(int(max_rows))
        .reset_index(drop=True)
    )


def cluster_model_comparison_points(
    assignments: pd.DataFrame,
    projected: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> pd.DataFrame:
    """Attach 2D coordinates and metadata to clustering assignments."""
    coordinates = projected[projected["date"].eq(pd.Timestamp(selected_date))].copy()
    points = assignments.merge(coordinates, on=["ticker", "date"], how="left")
    meta_columns = [
        column for column in ["ticker", "title", "company_name", "gics_sector", "gics_sub_industry"] if column in metadata.columns
    ]
    if meta_columns:
        meta = metadata.loc[:, meta_columns].drop_duplicates("ticker").copy()
        points = points.merge(meta, on="ticker", how="left")
    points["title"] = points.get("title", points["ticker"]).fillna(points["ticker"])
    points["gics_sector"] = points.get("gics_sector", pd.Series(index=points.index, dtype=object)).fillna("Unknown")
    points["cluster_label"] = np.where(points["is_noise"], "Noise / outlier", points["cluster_label"])
    return points.dropna(subset=["x", "y"]).reset_index(drop=True)


def cluster_model_comparison_chart(points: pd.DataFrame) -> alt.Chart:
    """Show GMM, k-means, and DBSCAN assignments on the same embedding map."""
    return (
        alt.Chart(points)
        .mark_circle(opacity=0.78, size=58)
        .encode(
            x=alt.X("x:Q", title="Embedding map X"),
            y=alt.Y("y:Q", title="Embedding map Y"),
            color=alt.Color("cluster_label:N", title="Cluster"),
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("title:N", title="Company"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("gics_sector:N", title="GICS sector"),
            ],
        )
        .properties(width=290, height=390)
        .facet(column=alt.Column("model:N", title=None))
        .resolve_scale(color="independent")
    )


def cluster_model_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return model-comparison metrics in a stable teaching order."""
    order = {"GMM": 0, "k-means": 1, "DBSCAN": 2}
    frame = metrics.copy()
    frame["_order"] = frame["model"].map(order).fillna(99)
    display_columns = [
        "model",
        "clusters_found",
        "noise_share",
        "largest_cluster_share",
        "silhouette",
        "nmi_vs_gics",
        "ari_vs_gics",
        "interpretation",
    ]
    return frame.sort_values("_order").loc[:, display_columns].reset_index(drop=True)


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


def interpretable_historical_sections(sections: list[str]) -> list[str]:
    """Return the section subset used for dashboard interpretation."""
    selected = [section for section in INTERPRETABLE_HISTORICAL_SECTIONS if section in sections]
    if selected:
        return selected
    fallback = [section for section in ["business", "risk_factors"] if section in sections]
    return fallback or sections[:2]


def section_scope_caption(selected_sections: list[str], section_labels: dict[str, str]) -> str:
    """Return a readable summary of the hidden section filter."""
    labels = [section_labels.get(section, section) for section in selected_sections]
    return "Using interpretation-friendly sections: " + ", ".join(labels) + "."


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


def company_language_story_frame(
    counts: pd.DataFrame,
    snippets: pd.DataFrame,
    filing_index: pd.DataFrame,
    ticker: str,
    sections: list[str],
    early_years: tuple[int, int],
    late_years: tuple[int, int],
) -> pd.DataFrame:
    """Return one compact row per tracked topic for a selected company."""
    frame = counts[(counts["ticker"].eq(str(ticker).upper())) & (counts["section"].isin(sections))].copy()
    if frame.empty:
        return pd.DataFrame()

    rows = []
    source_columns = [column for column in ["ticker", "accession_no", "source_url"] if column in filing_index.columns]
    sources = pd.DataFrame(columns=["ticker", "accession_no", "source_url"])
    if "source_url" in source_columns:
        sources = filing_index.loc[:, source_columns].drop_duplicates(["ticker", "accession_no"]).copy()
        sources["ticker"] = sources["ticker"].astype(str).str.upper()

    for topic_name, score_column in TOPIC_OPTIONS.items():
        if score_column not in frame.columns:
            continue
        mention_column = MENTION_OPTIONS[topic_name]
        early = frame[frame["year"].between(early_years[0], early_years[1])].copy()
        late = frame[frame["year"].between(late_years[0], late_years[1])].copy()
        if early.empty or late.empty:
            continue

        late["topic_score"] = pd.to_numeric(late[score_column], errors="coerce")
        late["topic_mentions"] = (
            pd.to_numeric(late[mention_column], errors="coerce") if mention_column in late.columns else 0.0
        )
        best = late.sort_values(["topic_score", "filing_date"], ascending=[False, False]).iloc[0]
        snippet_topic = topic_slug_from_score_column(score_column)
        evidence_snippet = ""
        snippet_terms = ""
        if not snippets.empty:
            snippet_frame = ensure_columns(
                snippets,
                ["ticker", "accession_no", "section", "snippet_topic", "snippet_rank", "snippet_terms", "evidence_snippet"],
            )
            evidence = snippet_frame[
                snippet_frame["ticker"].eq(str(ticker).upper())
                & snippet_frame["accession_no"].eq(best["accession_no"])
                & snippet_frame["section"].eq(best["section"])
                & snippet_frame["snippet_topic"].isin([snippet_topic, "section_start"])
            ].copy()
            if not evidence.empty:
                evidence = evidence.sort_values("snippet_rank")
                snippet_terms = str(evidence.iloc[0].get("snippet_terms", "") or "")
                evidence_snippet = str(evidence.iloc[0].get("evidence_snippet", "") or "")

        source_url = ""
        if not sources.empty:
            source_match = sources[
                sources["ticker"].eq(str(ticker).upper()) & sources["accession_no"].eq(best["accession_no"])
            ]
            if not source_match.empty:
                source_url = str(source_match.iloc[0].get("source_url", "") or "")

        early_score = float(pd.to_numeric(early[score_column], errors="coerce").mean())
        recent_score = float(pd.to_numeric(late[score_column], errors="coerce").mean())
        rows.append(
            {
                "topic": topic_name,
                "early_score": early_score,
                "recent_score": recent_score,
                "change": recent_score - early_score,
                "recent_mentions": int(late["topic_mentions"].sum()),
                "evidence_date": best["filing_date"],
                "form": best.get("form", ""),
                "section": best.get("section_label", ""),
                "snippet_terms": snippet_terms,
                "evidence_snippet": evidence_snippet,
                "source_url": source_url,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("change", ascending=False).reset_index(drop=True)


def company_language_shift_chart(story: pd.DataFrame) -> alt.Chart:
    """Show a compact all-topic shift profile for one company."""
    frame = story.copy()
    frame["direction"] = np.where(frame["change"] >= 0, "Increased", "Decreased")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("change:Q", title="Recent score minus baseline score"),
            y=alt.Y("topic:N", sort="-x", title="Topic"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Increased", "Decreased"], range=["#4f7cff", "#a8afc4"]),
                legend=None,
            ),
            tooltip=[
                "topic:N",
                alt.Tooltip("early_score:Q", title="Baseline", format=".3f"),
                alt.Tooltip("recent_score:Q", title="Recent", format=".3f"),
                alt.Tooltip("change:Q", title="Change", format=".3f"),
                alt.Tooltip("recent_mentions:Q", title="Recent mentions", format=".0f"),
            ],
        )
        .properties(height=250)
    )


def topic_slug_from_score_column(topic_column: str) -> str:
    """Return the snippet topic slug that corresponds to a topic score column."""
    value = str(topic_column)
    value = value.removeprefix("topic_")
    return value.removesuffix("_score_per_10k_words")


def add_source_urls(frame: pd.DataFrame, filing_index: pd.DataFrame) -> pd.DataFrame:
    """Attach SEC source URLs to filing-section rows."""
    result = frame.copy()
    if filing_index.empty or "accession_no" not in result.columns or "source_url" in result.columns:
        return result
    columns = [column for column in ["ticker", "accession_no", "source_url"] if column in filing_index.columns]
    if "source_url" not in columns:
        return result
    sources = filing_index.loc[:, columns].drop_duplicates(["ticker", "accession_no"])
    result["ticker"] = result["ticker"].astype(str).str.upper()
    sources["ticker"] = sources["ticker"].astype(str).str.upper()
    return result.merge(sources, on=["ticker", "accession_no"], how="left")


def company_topic_evidence_frame(
    counts: pd.DataFrame,
    snippets: pd.DataFrame,
    filing_index: pd.DataFrame,
    ticker: str,
    sections: list[str],
    topic_column: str,
    mention_column: str,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Return filing rows plus snippets that explain a selected company's topic movement."""
    base = counts[(counts["ticker"].eq(str(ticker).upper())) & (counts["section"].isin(sections))].copy()
    if base.empty:
        return pd.DataFrame()
    base["topic_score"] = pd.to_numeric(base[topic_column], errors="coerce")
    base["topic_mentions"] = pd.to_numeric(base[mention_column], errors="coerce") if mention_column in base.columns else 0.0
    base = base.sort_values(["topic_score", "filing_date"], ascending=[False, False])

    if snippets.empty:
        fallback = add_source_urls(base, filing_index)
        fallback["snippet_terms"] = ""
        fallback["evidence_snippet"] = ""
        columns = [
            "filing_date",
            "form",
            "section_label",
            "topic_score",
            "topic_mentions",
            "snippet_terms",
            "evidence_snippet",
            "source_url",
        ]
        return safe_frame_subset(ensure_columns(fallback, columns), columns).head(int(max_rows)).reset_index(drop=True)

    topic_slug = topic_slug_from_score_column(topic_column)
    snippets = ensure_columns(snippets, ["ticker", "section", "snippet_topic"])
    evidence = snippets[
        snippets["ticker"].eq(str(ticker).upper())
        & snippets["section"].isin(sections)
        & snippets["snippet_topic"].isin([topic_slug, "section_start"])
    ].copy()
    if evidence.empty:
        evidence = snippets[
            snippets["ticker"].eq(str(ticker).upper()) & snippets["section"].isin(sections)
        ].copy()
    if evidence.empty:
        return company_topic_evidence_frame(
            counts,
            pd.DataFrame(),
            filing_index,
            ticker,
            sections,
            topic_column,
            mention_column,
            max_rows,
        )

    keys = ["ticker", "accession_no", "section"]
    metric_columns = [column for column in [topic_column, mention_column] if column in base.columns]
    base_metrics = base.loc[:, [*keys, *metric_columns]].copy()
    evidence = evidence.merge(base_metrics, on=keys, how="left")
    evidence["topic_score"] = pd.to_numeric(evidence[topic_column], errors="coerce")
    evidence["topic_mentions"] = (
        pd.to_numeric(evidence[mention_column], errors="coerce") if mention_column in evidence.columns else 0.0
    )
    evidence = add_source_urls(evidence, filing_index)
    columns = [
        "filing_date",
        "form",
        "section_label",
        "snippet_topic",
        "snippet_terms",
        "topic_score",
        "topic_mentions",
        "evidence_snippet",
        "source_url",
    ]
    return (
        safe_frame_subset(ensure_columns(evidence, columns), columns)
        .sort_values(["topic_score", "filing_date"], ascending=[False, False])
        .head(int(max_rows))
        .reset_index(drop=True)
    )


def topic_change_evidence_frame(
    changes: pd.DataFrame,
    counts: pd.DataFrame,
    snippets: pd.DataFrame,
    filing_index: pd.DataFrame,
    sections: list[str],
    topic_column: str,
    mention_column: str,
    late_years: tuple[int, int],
    max_companies: int = 12,
) -> pd.DataFrame:
    """Return one late-window evidence row for each top company topic increase."""
    if changes.empty:
        return pd.DataFrame()
    rows = []
    for _, change in changes.head(int(max_companies)).iterrows():
        ticker = str(change["ticker"]).upper()
        late = counts[
            counts["ticker"].eq(ticker)
            & counts["section"].isin(sections)
            & counts["year"].between(late_years[0], late_years[1])
        ].copy()
        if late.empty:
            continue
        late["topic_score"] = pd.to_numeric(late[topic_column], errors="coerce")
        late["topic_mentions"] = pd.to_numeric(late[mention_column], errors="coerce") if mention_column in late.columns else 0.0
        best = late.sort_values(["topic_score", "filing_date"], ascending=[False, False]).iloc[0]
        evidence = company_topic_evidence_frame(
            counts,
            snippets,
            filing_index,
            ticker,
            [str(best["section"])],
            topic_column,
            mention_column,
            max_rows=1,
        )
        snippet_terms = ""
        evidence_snippet = ""
        source_url = ""
        if not evidence.empty:
            snippet_terms = str(evidence.iloc[0].get("snippet_terms", "") or "")
            evidence_snippet = str(evidence.iloc[0].get("evidence_snippet", "") or "")
            source_url = str(evidence.iloc[0].get("source_url", "") or "")
        rows.append(
            {
                "ticker": ticker,
                "company_name": change.get("company_name", ""),
                "gics_sector": change.get("gics_sector", ""),
                "change": float(change.get("change", np.nan)),
                "late_score": float(change.get("late_score", np.nan)),
                "filing_date": best["filing_date"],
                "form": best.get("form", ""),
                "section_label": best.get("section_label", ""),
                "snippet_terms": snippet_terms,
                "evidence_snippet": evidence_snippet,
                "source_url": source_url,
            }
        )
    return pd.DataFrame(rows)


def emerging_topic_evidence_frame(
    counts: pd.DataFrame,
    snippets: pd.DataFrame,
    filing_index: pd.DataFrame,
    metadata: pd.DataFrame,
    sections: list[str],
    early_years: tuple[int, int],
    late_years: tuple[int, int],
    companies_per_topic: int = 8,
) -> pd.DataFrame:
    """Return top company/topic language increases across all tracked topics."""
    rows = []
    for topic_name, topic_column in TOPIC_OPTIONS.items():
        mention_column = MENTION_OPTIONS[topic_name]
        changes = company_topic_change_table(
            counts,
            metadata,
            sections,
            topic_column,
            mention_column,
            early_years,
            late_years,
            min_early_rows=2,
        )
        if changes.empty:
            continue
        evidence = topic_change_evidence_frame(
            changes,
            counts,
            snippets,
            filing_index,
            sections,
            topic_column,
            mention_column,
            late_years,
            max_companies=int(companies_per_topic),
        )
        if evidence.empty:
            evidence = changes.head(int(companies_per_topic)).copy()
            evidence["filing_date"] = pd.NaT
            evidence["form"] = ""
            evidence["section_label"] = ""
            evidence["snippet_terms"] = ""
            evidence["evidence_snippet"] = ""
            evidence["source_url"] = ""
        evidence = evidence.copy()
        evidence.insert(0, "topic", topic_name)
        rows.append(evidence)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["change"] = pd.to_numeric(result["change"], errors="coerce")
    result["late_score"] = pd.to_numeric(result.get("late_score", np.nan), errors="coerce")
    result["trend_label"] = result["ticker"].astype(str) + " · " + result["topic"].astype(str)
    return result.sort_values("change", ascending=False).reset_index(drop=True)


def emerging_trends_chart(emerging: pd.DataFrame, max_rows: int = 20) -> alt.Chart:
    """Show the strongest emerging language trends as one ranked chart."""
    frame = emerging.head(int(max_rows)).copy()
    frame["company_label"] = frame.apply(
        lambda row: f"{row.get('ticker', '')} · {row.get('topic', '')}",
        axis=1,
    )
    height = max(280, min(620, 26 * len(frame)))
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("change:Q", title="Recent score minus baseline score"),
            y=alt.Y("company_label:N", title="", sort="-x"),
            color=alt.Color("topic:N", title="Theme"),
            tooltip=[
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("company_name:N", title="Company"),
                alt.Tooltip("gics_sector:N", title="Sector"),
                alt.Tooltip("topic:N", title="Theme"),
                alt.Tooltip("change:Q", title="Change", format=".3f"),
                alt.Tooltip("late_score:Q", title="Recent score", format=".3f"),
                alt.Tooltip("snippet_terms:N", title="Evidence terms"),
            ],
        )
        .properties(height=height)
    )


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


def sector_topic_insight_frame(sector_trend: pd.DataFrame, topic_name: str) -> pd.DataFrame:
    """Return a small table of interpretable sector/topic changes."""
    if sector_trend.empty:
        return pd.DataFrame()
    frame = sector_trend.copy()
    latest = frame.sort_values("year").groupby("gics_sector", as_index=False).tail(1)
    earliest = frame.sort_values("year").groupby("gics_sector", as_index=False).head(1)
    change = latest.merge(
        earliest.loc[:, ["gics_sector", "year", "topic_score"]],
        on="gics_sector",
        how="left",
        suffixes=("_latest", "_earliest"),
    )
    change["score_change"] = change["topic_score_latest"] - change["topic_score_earliest"]

    rows = []
    for _, row in latest.sort_values("topic_score", ascending=False).head(5).iterrows():
        rows.append(
            {
                "insight": f"Highest recent {topic_name}",
                "sector": row["gics_sector"],
                "period": str(int(row["year"])),
                "score": float(row["topic_score"]),
                "change": pd.NA,
            }
        )
    for _, row in change.sort_values("score_change", ascending=False).head(5).iterrows():
        rows.append(
            {
                "insight": f"Largest {topic_name} increase",
                "sector": row["gics_sector"],
                "period": f"{int(row['year_earliest'])}-{int(row['year_latest'])}",
                "score": float(row["topic_score_latest"]),
                "change": float(row["score_change"]),
            }
        )
    return pd.DataFrame(rows)


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
    points["chart_label"] = points.apply(
        lambda row: compact_company_name(row.get("title", ""), row.get("ticker", ""), max_chars=28),
        axis=1,
    )
    if color_mode == "Year":
        points["color_group"] = points["year"].astype(str)
    elif color_mode == "Sector":
        points["color_group"] = points["gics_sector"].fillna("Unknown")
    else:
        points["color_group"] = np.where(points["is_highlighted"], points["chart_label"], "Other companies")
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
                alt.Tooltip("chart_label:N", title="Company"),
                alt.Tooltip("ticker:N", title="Ticker"),
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
            color=alt.Color("chart_label:N", title="Highlighted company"),
            order="filing_date:T",
            tooltip=[
                alt.Tooltip("chart_label:N", title="Company"),
                alt.Tooltip("ticker:N", title="Ticker"),
                alt.Tooltip("filing_date_label:N", title="Filing date"),
                alt.Tooltip("section_label:N", title="Section"),
            ],
        )
    )
    labels = (
        alt.Chart(selected.sort_values("filing_date").groupby("ticker", as_index=False).tail(1))
        .mark_text(align="left", dx=8, dy=-8, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="chart_label:N", color=alt.value("#111111"))
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
    model_type: str,
    group_mode: str,
    group_view: str,
    loadings_path: str,
    embeddings_path: str,
    membership_mode: str,
    theme_assignment: str,
    include_embedding_features: bool,
    min_theme_effective_members: float = DEFAULT_MIN_THEME_EFFECTIVE_MEMBERS,
    artifact_version: tuple[tuple[str, int, int], ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Load inputs and compute sector-relative outlook artifacts."""
    _ = artifact_version
    config = load_config(config_name)
    db = FilingsDB.from_config(config)
    prices = db.load_prices(date_from=config["data"].get("start_date"), date_to=config["data"].get("end_date"))
    metadata = load_metadata(artifact_signature(SP500_GICS_PATH))
    valuation = optional_feature_frame(VALUATION_FEATURE_PATH)
    growth = optional_feature_frame(GROWTH_FEATURE_PATH)
    group_loadings = load_loadings(loadings_path) if group_mode == "theme" and loadings_path else None
    group_embeddings = load_embeddings(embeddings_path) if include_embedding_features and embeddings_path else None
    membership = (
        load_sp500_membership_history(artifact_signature(SP500_MEMBERSHIP_PATH))
        if membership_mode == "historical"
        else None
    )
    if membership_mode == "historical":
        prices = append_missing_legacy_historical_prices(prices)
    keyword_args = {
        "valuation": valuation,
        "growth": growth,
        "group_loadings": group_loadings,
        "group_embeddings": group_embeddings,
        "membership": membership,
        "group_mode": group_mode,
        "group_view": group_view or None,
        "horizon_days": int(horizon_days),
        "min_train_months": int(min_train_months),
        "ridge_alpha": float(ridge_alpha),
        "model_type": model_type,
        "membership_mode": membership_mode,
        "theme_assignment": theme_assignment,
        "include_embedding_features": include_embedding_features,
        "min_theme_effective_members": float(min_theme_effective_members),
    }
    supported_args = set(inspect.signature(sector_outlook_backtest).parameters)
    keyword_args = {key: value for key, value in keyword_args.items() if key in supported_args}
    result = sector_outlook_backtest(prices, metadata, **keyword_args)
    return result.panel, result.predictions, result.latest, result.metrics, result.coefficients


@st.cache_data(show_spinner=False)
def load_sector_predictor_comparison(
    config_name: str,
    horizon_days: int,
    min_train_months: int,
    ridge_alpha: float,
    group_mode: str,
    group_view: str,
    loadings_path: str,
    embeddings_path: str,
    membership_mode: str,
    theme_assignment: str,
    include_embedding_features: bool,
    min_theme_effective_members: float = DEFAULT_MIN_THEME_EFFECTIVE_MEMBERS,
    artifact_version: tuple[tuple[str, int, int], ...] | None = None,
) -> pd.DataFrame:
    """Compare all supported sector/group prediction models with the same inputs."""
    _ = artifact_version
    rows = []
    benchmark_returns = load_sp500_benchmark_returns(str(DEFAULT_SP500_BENCHMARK_PATH))
    for model_type, model_label in SECTOR_MODEL_LABELS.items():
        try:
            _, predictions, _, metrics, _ = load_sector_outlook_artifacts(
                config_name,
                int(horizon_days),
                int(min_train_months),
                float(ridge_alpha),
                str(model_type),
                group_mode,
                group_view,
                loadings_path,
                embeddings_path,
                membership_mode,
                theme_assignment,
                bool(include_embedding_features),
                float(min_theme_effective_members),
                artifact_version,
            )
            completed = completed_sector_predictions(predictions)
            simulation = simulate_group_rotation(
                completed,
                starting_capital=DEFAULT_SIMULATION_CAPITAL,
                top_n=1,
                transaction_cost_bps=0.0,
                benchmark_returns=benchmark_returns if not benchmark_returns.empty else None,
                benchmark_label="Regular S&P 500 (SPY)",
            )
            sim_metrics = rotation_simulation_metrics(simulation, DEFAULT_SIMULATION_CAPITAL) if not simulation.empty else {}
            hard_metrics = sector_prediction_hard_metrics(predictions)
            risk_metrics = rotation_extra_metrics(simulation)
            rows.append(
                {
                    "model": model_label,
                    "model_key": model_type,
                    "completed_dates": int(metrics.get("n_prediction_dates", completed["date"].nunique() if not completed.empty else 0)),
                    "completed_rows": hard_metrics.get("n_completed_rows", 0),
                    "mean_rank_ic": metrics.get("mean_rank_ic", np.nan),
                    "median_rank_ic": metrics.get("median_rank_ic", np.nan),
                    "mean_top_minus_bottom": metrics.get("mean_top_minus_bottom", np.nan),
                    "mean_top_bucket_excess": metrics.get("mean_top_bucket_excess", np.nan),
                    "top_sector_hit_rate": metrics.get("top_sector_hit_rate", np.nan),
                    "prediction_mae": hard_metrics.get("prediction_mae", np.nan),
                    "prediction_rmse": hard_metrics.get("prediction_rmse", np.nan),
                    "prediction_correlation": hard_metrics.get("prediction_correlation", np.nan),
                    "directional_accuracy": hard_metrics.get("directional_accuracy", np.nan),
                    "positive_precision": hard_metrics.get("positive_precision", np.nan),
                    "positive_recall": hard_metrics.get("positive_recall", np.nan),
                    "n_rebalances": sim_metrics.get("n_rebalances", 0),
                    "ending_capital": sim_metrics.get("ending_capital", np.nan),
                    "regular_sp500_capital": sim_metrics.get("benchmark_ending_capital", np.nan),
                    "equal_weight_capital": sim_metrics.get("market_ending_capital", np.nan),
                    "total_return": sim_metrics.get("total_return", np.nan),
                    "annualized_return": sim_metrics.get("annualized_return", np.nan),
                    "annualized_volatility": risk_metrics.get("annualized_volatility", np.nan),
                    "sharpe_like": risk_metrics.get("sharpe_like", np.nan),
                    "period_win_rate": sim_metrics.get("period_win_rate", np.nan),
                    "excess_win_rate_vs_spy": risk_metrics.get("excess_win_rate_vs_spy", np.nan),
                    "avg_period_return": sim_metrics.get("average_period_return", np.nan),
                    "avg_period_excess": risk_metrics.get("avg_period_excess", np.nan),
                    "excess_return_vs_sp500": sim_metrics.get("excess_total_return_vs_benchmark", np.nan),
                    "excess_return_vs_equal_weight": sim_metrics.get("excess_total_return", np.nan),
                    "max_drawdown": sim_metrics.get("max_drawdown", np.nan),
                    "turnover_rate": risk_metrics.get("turnover_rate", np.nan),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "model": model_label,
                    "model_key": model_type,
                    "completed_dates": 0,
                    "completed_rows": 0,
                    "mean_rank_ic": np.nan,
                    "median_rank_ic": np.nan,
                    "mean_top_minus_bottom": np.nan,
                    "mean_top_bucket_excess": np.nan,
                    "top_sector_hit_rate": np.nan,
                    "prediction_mae": np.nan,
                    "prediction_rmse": np.nan,
                    "prediction_correlation": np.nan,
                    "directional_accuracy": np.nan,
                    "positive_precision": np.nan,
                    "positive_recall": np.nan,
                    "n_rebalances": 0,
                    "ending_capital": np.nan,
                    "regular_sp500_capital": np.nan,
                    "equal_weight_capital": np.nan,
                    "total_return": np.nan,
                    "annualized_return": np.nan,
                    "annualized_volatility": np.nan,
                    "sharpe_like": np.nan,
                    "period_win_rate": np.nan,
                    "excess_win_rate_vs_spy": np.nan,
                    "avg_period_return": np.nan,
                    "avg_period_excess": np.nan,
                    "excess_return_vs_sp500": np.nan,
                    "excess_return_vs_equal_weight": np.nan,
                    "max_drawdown": np.nan,
                    "turnover_rate": np.nan,
                    "status": str(exc),
                }
            )
    return pd.DataFrame(rows).sort_values("ending_capital", ascending=False, na_position="last").reset_index(drop=True)


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
    group_column = "group_label" if "group_label" in frame.columns else "gics_sector"
    frame["predicted_excess_return"] = pd.to_numeric(frame["predicted_excess_return"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["predicted_excess_return"])
    frame["direction"] = np.where(frame["predicted_excess_return"] >= 0.0, "Positive", "Negative")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y(f"{group_column}:N", sort="-x", title="Group"),
            x=alt.X("predicted_excess_return:Q", title="Predicted excess return vs S&P"),
            color=alt.Color("direction:N", scale=alt.Scale(range=["#1f7a4d", "#b8423f"]), legend=None),
            tooltip=[
                alt.Tooltip(f"{group_column}:N", title="Group"),
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
    frame["top_minus_bottom"] = pd.to_numeric(frame["top_minus_bottom"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "top_minus_bottom"])
    frame["date_label"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["target_end_label"] = frame["target_end_date"].dt.strftime("%Y-%m-%d")
    return (
        alt.Chart(frame)
        .mark_line(point=False)
        .encode(
            x=alt.X("date:T", title="Prediction date"),
            y=alt.Y("top_minus_bottom:Q", title="Top 3 minus bottom 3 realized excess"),
            tooltip=[
                alt.Tooltip("date_label:N", title="Date"),
                alt.Tooltip("target_end_label:N", title="Horizon ended"),
                alt.Tooltip("top_minus_bottom:Q", title="Top-bottom", format=".2%"),
                alt.Tooltip("rank_ic:Q", title="Rank IC", format=".3f"),
                alt.Tooltip("top_bucket_excess:Q", title="Top bucket excess", format=".2%"),
                alt.Tooltip("predicted_top_sector:N", title="Predicted top"),
                alt.Tooltip("realized_best_sector:N", title="Realized best"),
            ],
        )
        .properties(height=260)
    )


def sector_rank_ic_chart(dated: pd.DataFrame) -> alt.Chart:
    """Build date-level rank-correlation chart for historical sector predictions."""
    frame = dated.copy()
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "rank_ic"])
    frame["date_label"] = frame["date"].dt.strftime("%Y-%m-%d")
    return (
        alt.Chart(frame)
        .mark_line(point=False, color="#2f6f9d")
        .encode(
            x=alt.X("date:T", title="Prediction date"),
            y=alt.Y("rank_ic:Q", title="Spearman rank IC", scale=alt.Scale(domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("date_label:N", title="Date"),
                alt.Tooltip("rank_ic:Q", title="Rank IC", format=".3f"),
                alt.Tooltip("predicted_top_sector:N", title="Predicted top"),
                alt.Tooltip("realized_best_sector:N", title="Realized best"),
            ],
        )
        .properties(height=260)
    )


def sector_prediction_scatter(frame: pd.DataFrame) -> alt.Chart:
    """Build predicted-versus-realized scatter for one historical prediction date."""
    plot = frame.copy()
    group_column = "group_label" if "group_label" in plot.columns else "gics_sector"
    plot["predicted_excess_return"] = pd.to_numeric(plot["predicted_excess_return"], errors="coerce")
    plot["future_excess_return"] = pd.to_numeric(plot["future_excess_return"], errors="coerce")
    plot = plot.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["predicted_excess_return", "future_excess_return"]
    )
    plot["date_label"] = plot["date"].dt.strftime("%Y-%m-%d")
    plot["target_end_label"] = plot["target_end_date"].dt.strftime("%Y-%m-%d")
    return (
        alt.Chart(plot)
        .mark_circle(size=130, opacity=0.85)
        .encode(
            x=alt.X("predicted_excess_return:Q", title="Predicted excess return", axis=alt.Axis(format="%")),
            y=alt.Y("future_excess_return:Q", title="Realized excess return", axis=alt.Axis(format="%")),
            color=alt.Color(f"{group_column}:N", legend=None),
            tooltip=[
                alt.Tooltip(f"{group_column}:N", title="Group"),
                alt.Tooltip("date_label:N", title="Prediction date"),
                alt.Tooltip("target_end_label:N", title="Horizon ended"),
                alt.Tooltip("prediction_rank:Q", title="Predicted rank", format=".0f"),
                alt.Tooltip("predicted_excess_return:Q", title="Predicted", format=".2%"),
                alt.Tooltip("future_excess_return:Q", title="Realized", format=".2%"),
            ],
        )
        .properties(height=300)
    )


def sector_rotation_equity_chart(simulation: pd.DataFrame) -> alt.Chart:
    """Build a capital curve for the walk-forward rotation simulation."""
    frame = simulation.copy()
    if frame.empty:
        return alt.Chart(pd.DataFrame({"step_point": [], "portfolio": [], "capital": []})).mark_line()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_end_date"] = pd.to_datetime(frame["target_end_date"])
    frame["date_label"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["target_end_label"] = frame["target_end_date"].dt.strftime("%Y-%m-%d")
    if "step" not in frame.columns:
        frame["step"] = np.arange(1, len(frame) + 1)
    if "start_capital" not in frame.columns:
        frame["start_capital"] = frame["capital"] / (1.0 + frame["net_period_return"].replace(-1.0, np.nan))
    if "start_market_capital" not in frame.columns:
        frame["start_market_capital"] = frame["market_capital"] / (1.0 + frame["period_market_return"].replace(-1.0, np.nan))
    has_benchmark = "benchmark_capital" in frame.columns and frame["benchmark_capital"].notna().any()
    if has_benchmark and "start_benchmark_capital" not in frame.columns:
        frame["start_benchmark_capital"] = frame["benchmark_capital"] / (
            1.0 + frame["period_benchmark_return"].replace(-1.0, np.nan)
        )

    curve_rows = []
    for _, row in frame.iterrows():
        step = int(row["step"])
        portfolio_specs = [
            ("strategy", "Predicted-group rotation", "start_capital", "capital", "net_period_return"),
            ("market", "Equal-weight S&P universe", "start_market_capital", "market_capital", "period_market_return"),
        ]
        if has_benchmark:
            benchmark_label = str(row.get("benchmark_label", "S&P 500 benchmark"))
            portfolio_specs.append(
                (
                    "benchmark",
                    benchmark_label,
                    "start_benchmark_capital",
                    "benchmark_capital",
                    "period_benchmark_return",
                )
            )
        for portfolio_key, portfolio_label, start_column, end_column, return_column in portfolio_specs:
            if pd.isna(row.get(start_column)) or pd.isna(row.get(end_column)):
                continue
            curve_rows.append(
                {
                    "step_point": step - 1,
                    "step_label": f"Step {step - 1}",
                    "portfolio_key": portfolio_key,
                    "portfolio": portfolio_label,
                    "date": row["date"],
                    "date_label": row["date_label"],
                    "target_end_label": row["target_end_label"],
                    "selected_group_label": row["selected_group_label"],
                    "portfolio_value": float(row[start_column]),
                    "period_return": 0.0,
                }
            )
            curve_rows.append(
                {
                    "step_point": step,
                    "step_label": f"Step {step}",
                    "portfolio_key": portfolio_key,
                    "portfolio": portfolio_label,
                    "date": row["target_end_date"],
                    "date_label": row["date_label"],
                    "target_end_label": row["target_end_label"],
                    "selected_group_label": row["selected_group_label"],
                    "portfolio_value": float(row[end_column]),
                    "period_return": float(row[return_column]),
                }
            )
    long = (
        pd.DataFrame(curve_rows)
        .sort_values(["portfolio_key", "step_point", "date"])
        .drop_duplicates(["portfolio_key", "step_point"], keep="last")
    )
    return (
        alt.Chart(long)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("step_point:Q", title="Rebalance step", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y("portfolio_value:Q", title="Portfolio value", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("portfolio:N", title=None),
            tooltip=[
                alt.Tooltip("portfolio:N", title="Portfolio"),
                alt.Tooltip("step_label:N", title="Step"),
                alt.Tooltip("date:T", title="Capital date"),
                alt.Tooltip("date_label:N", title="Entered on"),
                alt.Tooltip("target_end_label:N", title="Exit date"),
                alt.Tooltip("selected_group_label:N", title="Selected group"),
                alt.Tooltip("portfolio_value:Q", title="Capital", format="$,.2f"),
                alt.Tooltip("period_return:Q", title="Period return", format=".2%"),
            ],
        )
        .properties(height=380)
    )


def sector_rotation_step_table(simulation: pd.DataFrame) -> pd.DataFrame:
    """Return a readable ledger showing how capital changes at each step."""
    frame = simulation.copy()
    if frame.empty:
        return frame
    if "step" not in frame.columns:
        frame["step"] = np.arange(1, len(frame) + 1)
    if "start_capital" not in frame.columns:
        frame["start_capital"] = frame["capital"] / (1.0 + frame["net_period_return"].replace(-1.0, np.nan))
    if "capital_change" not in frame.columns:
        frame["capital_change"] = frame["capital"] - frame["start_capital"]
    if "start_market_capital" not in frame.columns:
        frame["start_market_capital"] = frame["market_capital"] / (1.0 + frame["period_market_return"].replace(-1.0, np.nan))
    if "market_capital_change" not in frame.columns:
        frame["market_capital_change"] = frame["market_capital"] - frame["start_market_capital"]
    columns = [
        "step",
        "date",
        "target_end_date",
        "selected_group_label",
        "predicted_excess_return",
        "start_capital",
        "net_period_return",
        "capital_change",
        "capital",
        "start_market_capital",
        "period_market_return",
        "market_capital_change",
        "market_capital",
    ]
    has_benchmark = "benchmark_capital" in frame.columns and frame["benchmark_capital"].notna().any()
    if has_benchmark:
        columns.extend(
            [
                "start_benchmark_capital",
                "period_benchmark_return",
                "benchmark_capital_change",
                "benchmark_capital",
            ]
        )
    output = frame.loc[:, columns].copy()
    output = output.rename(
        columns={
            "date": "entry_date",
            "target_end_date": "exit_date",
            "selected_group_label": "selected_group",
            "predicted_excess_return": "predicted_excess",
            "start_capital": "strategy_start",
            "net_period_return": "strategy_return",
            "capital_change": "strategy_dollar_change",
            "capital": "strategy_end",
            "start_market_capital": "market_start",
            "period_market_return": "market_return",
            "market_capital_change": "market_dollar_change",
            "market_capital": "market_end",
            "start_benchmark_capital": "sp500_start",
            "period_benchmark_return": "sp500_return",
            "benchmark_capital_change": "sp500_dollar_change",
            "benchmark_capital": "sp500_end",
        }
    )
    return output


def latest_coefficient_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Return latest model coefficients sorted by absolute weight."""
    if coefficients.empty:
        return coefficients
    latest_date = coefficients["date"].max()
    frame = coefficients[coefficients["date"] == latest_date].copy()
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def apply_theme_group_labels(
    frame: pd.DataFrame,
    labels: dict[str, str],
    *,
    group_column: str = "gics_sector",
) -> pd.DataFrame:
    """Attach readable dynamic labels to learned-theme prediction groups."""
    if frame.empty:
        return frame
    result = frame.copy()
    if group_column in frame.columns:
        result["group_label"] = [
            display_theme_name(str(value), labels) if str(value).startswith("theme_") else str(value)
            for value in result[group_column]
        ]
    for column in ["predicted_top_sector", "realized_best_sector"]:
        if column in result.columns:
            result[column] = [
                display_theme_name(str(value), labels) if str(value).startswith("theme_") else str(value)
                for value in result[column]
            ]
    return result


def membership_mode_display(membership_mode: str) -> str:
    """Return a readable dashboard label for the selected universe mode."""
    labels = {
        "historical": "historical constituents",
        "date_added": "date-added current members",
        "current": "current roster",
    }
    return labels.get(str(membership_mode), str(membership_mode))


def render_sector_outlook() -> None:
    st.subheader("Group Outlook")
    st.caption(
        "Walk-forward group excess-return predictions versus the equal-weight S&P 500 universe. "
        "Current scores are separated from completed historical predictions."
    )

    config_name = DEFAULT_SECTOR_CONFIG
    model_type = DEFAULT_SECTOR_MODEL
    horizon_days = DEFAULT_SECTOR_HORIZON_DAYS
    min_train_months = DEFAULT_SECTOR_MIN_TRAIN_MONTHS
    ridge_alpha = DEFAULT_RIDGE_ALPHA

    controls = st.columns([1.05, 1.15, 1.05])
    group_choice = controls[0].selectbox("Prediction groups", ["GICS sectors", "Learned themes"])
    membership_choice = controls[1].selectbox(
        "Universe",
        ["Historical constituents", "Date-added current members", "Current roster (diagnostic only)"],
        help=(
            "Historical mode uses add/remove intervals when available. It only includes deleted companies "
            "if their prices and labels are present locally. Date-added mode is the safer fallback for the "
            "current roster. Current-roster mode is not a valid historical backtest."
        ),
    )
    model_type = controls[2].selectbox(
        "Prediction model",
        list(SECTOR_MODEL_LABELS.keys()),
        index=list(SECTOR_MODEL_LABELS.keys()).index(DEFAULT_SECTOR_MODEL),
        format_func=sector_model_display_name,
        help=(
            "Choose the model used for the live Sector Outlook run. The Model Comparison tab keeps the broader "
            "side-by-side benchmark table."
        ),
    )
    membership_mode_lookup = {
        "Historical constituents": "historical",
        "Date-added current members": "date_added",
        "Current roster (diagnostic only)": "current",
    }
    membership_mode = membership_mode_lookup[membership_choice]
    if membership_mode == "historical":
        membership = load_sp500_membership_history(artifact_signature(SP500_MEMBERSHIP_PATH))
        if membership.empty:
            st.warning(
                "Historical membership intervals were not found. Run "
                "`python scripts/data_setup/fetch_sp500_membership_history.py` first."
            )
    if membership_mode == "current":
        st.error(
            "Current roster mode is diagnostic only, not a valid historical backtest. It can create "
            "survivorship/look-ahead bias by letting today's S&P 500 winners appear in earlier history before "
            "they were actually index members."
        )

    with st.expander("Advanced model settings", expanded=False):
        advanced = st.columns([1.25, 1.0, 1.0, 1.0])
        config_name = advanced[0].text_input("Experiment config", value=config_name)
        horizon_days = advanced[1].selectbox("Forward horizon", [21, 63, 126, 252], index=1)
        min_train_months = advanced[2].slider(
            "Min training months",
            min_value=12,
            max_value=84,
            value=DEFAULT_SECTOR_MIN_TRAIN_MONTHS,
            step=6,
        )
        ridge_alpha = advanced[3].select_slider(
            "Linear regularization",
            options=[0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
            value=DEFAULT_RIDGE_ALPHA,
            help="Used by Ridge directly and Huber as a scaled regularization value. Tree models ignore this.",
        )

    group_mode = "theme" if group_choice == "Learned themes" else "gics"
    theme_view = ""
    theme_loadings_path = ""
    theme_embeddings_path = ""
    theme_assignment = "soft"
    include_embedding_features = False
    experiment_root = latest_decomposed_experiment()
    if group_mode == "theme":
        if experiment_root is None:
            st.warning("No decomposed experiment with learned theme loadings was found.")
        else:
            views = available_views(experiment_root)
            if views:
                theme_view = st.selectbox(
                    "Learned theme view",
                    views,
                    index=views.index("business") if "business" in views else 0,
                )
                theme_loadings_path = str(experiment_root / "views" / theme_view / "loadings.parquet")
                candidate_embeddings_path = experiment_root / "views" / theme_view / "embeddings.parquet"
                if candidate_embeddings_path.exists():
                    theme_embeddings_path = str(candidate_embeddings_path)
                    include_embedding_features = True
                theme_assignment = default_theme_assignment_strategy(theme_view)
                st.caption(
                    "Theme assignment policy: "
                    f"{THEME_ASSIGNMENT_LABELS.get(theme_assignment, theme_assignment)} "
                    f"for the {theme_view} view. "
                    f"Embedding predictor features: {'on' if include_embedding_features else 'off'}. "
                    f"Thin learned themes require at least {DEFAULT_MIN_THEME_EFFECTIVE_MEMBERS:g} effective members."
                )
                if membership_mode == "historical":
                    st.info(
                        "Historical learned-theme mode uses point-in-time S&P membership and deleted-constituent "
                        "prices where available, but it can only form learned-theme groups for ticker-dates that "
                        "have saved theme loadings. This is a valid covered-universe backtest, not a complete "
                        "all-historical-constituents theme backtest until deleted constituents also have embeddings."
                    )
    current_artifact_version = sector_outlook_artifact_signature(
        group_mode=group_mode,
        loadings_path=theme_loadings_path,
        embeddings_path=theme_embeddings_path,
        membership_mode=membership_mode,
        include_embedding_features=include_embedding_features,
    )

    try:
        panel, predictions, latest, metrics, coefficients = load_sector_outlook_artifacts(
            config_name,
            int(horizon_days),
            int(min_train_months),
            float(ridge_alpha),
            str(model_type),
            group_mode,
            theme_view,
            theme_loadings_path,
            theme_embeddings_path,
            membership_mode,
            theme_assignment,
            include_embedding_features,
            DEFAULT_MIN_THEME_EFFECTIVE_MEMBERS,
            current_artifact_version,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Sector outlook failed: {exc}")
        return

    if latest.empty:
        st.warning("Not enough completed history to fit the sector outlook model with these settings.")
        return

    if group_mode == "theme" and theme_loadings_path:
        metadata = load_metadata()
        label_lookup_for_view = auto_theme_label_lookup(theme_loadings_path, metadata, theme_view)
        fragment_label_lookup, fragment_explanations = fragment_theme_label_artifacts(
            theme_loadings_path,
            str(default_historical_text_dir()),
            theme_view,
            as_of_date=str(latest["date"].max().date()),
            metadata=metadata,
        )
        label_lookup_for_view.update(fragment_label_lookup)
        label_lookup_for_view = unique_theme_label_lookup(label_lookup_for_view, theme_columns(load_loadings(theme_loadings_path)))
        latest = apply_theme_group_labels(latest, label_lookup_for_view)
        predictions = apply_theme_group_labels(predictions, label_lookup_for_view)
    else:
        fragment_explanations = pd.DataFrame()

    audit = sector_prediction_audit(predictions)
    completed_predictions = completed_sector_predictions(predictions)
    dated = sector_backtest_by_date(predictions)
    if group_mode == "theme" and theme_loadings_path:
        dated = apply_theme_group_labels(dated, label_lookup_for_view)

    display_columns = [
        "prediction_rank",
        "group_label",
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

    metric_columns = st.columns(4)
    metric_columns[0].metric("Completed dates", f"{int(audit.get('n_completed_dates', 0)):,}")
    metric_columns[1].metric("Mean rank IC", f"{float(metrics.get('mean_rank_ic', np.nan)):.3f}")
    metric_columns[2].metric("Top-bottom avg", f"{float(metrics.get('mean_top_minus_bottom', np.nan)):.2%}")
    metric_columns[3].metric("Leakage flags", f"{int(audit.get('n_leakage_violations', 0)):,}")
    if int(audit.get("n_leakage_violations", 0)) > 0:
        st.error("Audit found prediction rows whose training outcomes ended on or after the prediction date.")
    else:
        st.caption(
            f"Historical audit uses completed horizons only: {audit.get('first_completed_date')} to "
            f"{audit.get('latest_completed_date')}. Current scores through "
            f"{audit.get('latest_prediction_date')} remain separate until their forward horizon completes. "
            f"Predictor: {SECTOR_MODEL_LABELS.get(str(metrics.get('model_type')), str(metrics.get('model_type')))}. "
            f"Groups: {'learned ' + theme_view + ' themes' if group_mode == 'theme' else 'GICS sectors'}. "
            f"Assignment: {THEME_ASSIGNMENT_LABELS.get(str(metrics.get('theme_assignment')), str(metrics.get('theme_assignment')))}. "
            f"Embedding features: {'on' if bool(metrics.get('include_embedding_features')) else 'off'}. "
            f"Min theme effective members: {float(metrics.get('min_theme_effective_members', 1.0)):.0f}. "
            f"Universe: {membership_mode_display(str(metrics.get('membership_mode')))}. "
            f"Backtest status: {str(metrics.get('backtest_validity', 'unknown')).replace('_', ' ')}."
        )

    current_tab, historical_tab, simulation_tab, weights_tab, notes_tab = st.tabs(
        ["Current scores", "Historical walk-forward", "Rotation simulation", "Feature weights", "How to read"]
    )

    with current_tab:
        latest_date = latest["date"].max().strftime("%Y-%m-%d")
        st.caption(
            f"Current scores are as of {latest_date}. They are live/unrealized rows, not historical backtest results."
        )
        st.altair_chart(sector_latest_score_chart(latest), width="stretch")
        with st.expander("Show current sector input table", expanded=False):
            st.dataframe(
                safe_frame_subset(ensure_columns(latest, display_columns), display_columns),
                width="stretch",
                hide_index=True,
            )

    with historical_tab:
        if completed_predictions.empty or dated.empty:
            st.warning("No completed historical prediction horizons are available for these settings yet.")
        else:
            chart_columns = st.columns(2)
            chart_columns[0].altair_chart(sector_backtest_chart(dated), width="stretch")
            chart_columns[1].altair_chart(sector_rank_ic_chart(dated), width="stretch")

            completed_dates = sorted(completed_predictions["date"].dropna().unique())
            selected_date = st.selectbox(
                "Historical prediction date",
                completed_dates,
                index=len(completed_dates) - 1,
                format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            )
            selected = completed_predictions[completed_predictions["date"].eq(pd.Timestamp(selected_date))].copy()
            selected = selected.sort_values("prediction_rank")
            st.caption(
                "This table is what the walk-forward model had predicted on that historical date, "
                "paired with the realized sector excess return after the forward horizon completed."
            )
            historical_columns = [
                "prediction_rank",
                "group_label",
                "gics_sector",
                "predicted_excess_return",
                "future_excess_return",
                "target_end_date",
                "future_days_available",
                "training_dates",
                "training_latest_target_end_date",
            ]
            st.altair_chart(sector_prediction_scatter(selected), width="stretch")
            st.dataframe(
                safe_frame_subset(ensure_columns(selected, historical_columns), historical_columns),
                width="stretch",
                hide_index=True,
            )
            with st.expander("Show date-level historical diagnostics", expanded=False):
                st.dataframe(
                    dated.sort_values("date", ascending=False),
                    width="stretch",
                    hide_index=True,
                )

    with simulation_tab:
        if completed_predictions.empty:
            st.warning("No completed historical prediction horizons are available for simulation.")
        else:
            starting_capital = DEFAULT_SIMULATION_CAPITAL
            top_n = 1
            transaction_cost_bps = 0.0
            st.caption(
                "Simulation defaults: start with $10,000, rotate into the single best predicted group, "
                "and assume zero transaction costs. This keeps the demo focused on the walk-forward signal."
            )
            benchmark_returns = load_sp500_benchmark_returns(str(DEFAULT_SP500_BENCHMARK_PATH))
            if benchmark_returns.empty:
                st.info(
                    "Regular S&P benchmark prices were not found locally yet. Run "
                    "`.venv/bin/python scripts/data_setup/fetch_sp500_benchmark.py` to add the SPY benchmark line."
                )
            simulation = simulate_group_rotation(
                completed_predictions,
                starting_capital=float(starting_capital),
                top_n=int(top_n),
                transaction_cost_bps=float(transaction_cost_bps),
                benchmark_returns=benchmark_returns if not benchmark_returns.empty else None,
                benchmark_label="Regular S&P 500 (SPY)",
            )
            if simulation.empty:
                st.warning("The selected settings did not produce any non-overlapping completed holding periods.")
            else:
                sim_metrics = rotation_simulation_metrics(simulation, float(starting_capital))
                sim_columns = st.columns(6)
                sim_columns[0].metric("Ending capital", f"${float(sim_metrics['ending_capital']):,.0f}")
                sim_columns[1].metric("Equal-weight S&P", f"${float(sim_metrics['market_ending_capital']):,.0f}")
                if pd.notna(sim_metrics.get("benchmark_ending_capital")):
                    sim_columns[2].metric(
                        "Regular S&P",
                        f"${float(sim_metrics['benchmark_ending_capital']):,.0f}",
                    )
                    sim_columns[4].metric(
                        "Excess vs S&P",
                        f"{float(sim_metrics['excess_total_return_vs_benchmark']):.1%}",
                    )
                else:
                    sim_columns[2].metric("Regular S&P", "missing")
                    sim_columns[4].metric("Excess vs S&P", "n/a")
                sim_columns[3].metric("Strategy return", f"{float(sim_metrics['total_return']):.1%}")
                sim_columns[5].metric("Max drawdown", f"{float(sim_metrics['max_drawdown']):.1%}")
                st.caption(
                    "This is a historical walk-forward simulation. At each rebalance date, the model chooses the "
                    "best predicted group using only information available then, holds through the selected forward "
                    "horizon, then waits for the next non-overlapping rebalance date."
                )
                if membership_mode == "current":
                    st.error(
                        "This specific simulation is not a valid historical backtest because the universe is today's "
                        "current S&P roster. Switch to Historical constituents for the defensible version."
                    )
                st.altair_chart(sector_rotation_equity_chart(simulation), width="stretch")
                st.markdown("**How the $10,000 changes at each step**")
                st.dataframe(
                    sector_rotation_step_table(simulation),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "predicted_excess": st.column_config.NumberColumn("predicted_excess", format="percent"),
                        "strategy_start": st.column_config.NumberColumn("strategy_start", format="$%.2f"),
                        "strategy_return": st.column_config.NumberColumn("strategy_return", format="percent"),
                        "strategy_dollar_change": st.column_config.NumberColumn(
                            "strategy_dollar_change",
                            format="$%.2f",
                        ),
                        "strategy_end": st.column_config.NumberColumn("strategy_end", format="$%.2f"),
                        "market_start": st.column_config.NumberColumn("market_start", format="$%.2f"),
                        "market_return": st.column_config.NumberColumn("market_return", format="percent"),
                        "market_dollar_change": st.column_config.NumberColumn(
                            "market_dollar_change",
                            format="$%.2f",
                        ),
                        "market_end": st.column_config.NumberColumn("market_end", format="$%.2f"),
                        "sp500_start": st.column_config.NumberColumn("sp500_start", format="$%.2f"),
                        "sp500_return": st.column_config.NumberColumn("sp500_return", format="percent"),
                        "sp500_dollar_change": st.column_config.NumberColumn(
                            "sp500_dollar_change",
                            format="$%.2f",
                        ),
                        "sp500_end": st.column_config.NumberColumn("sp500_end", format="$%.2f"),
                    },
                )
                with st.expander("Show rotation log", expanded=False):
                    rotation_columns = [
                        "step",
                        "date",
                        "target_end_date",
                        "selected_group_label",
                        "predicted_excess_return",
                        "start_capital",
                        "period_group_return",
                        "period_market_return",
                        "period_excess_return",
                        "transaction_cost",
                        "capital_change",
                        "capital",
                        "market_capital",
                        "period_benchmark_return",
                        "benchmark_capital",
                    ]
                    st.dataframe(
                        safe_frame_subset(ensure_columns(simulation, rotation_columns), rotation_columns),
                        width="stretch",
                        hide_index=True,
                    )

    with weights_tab:
        coefficient_table = latest_coefficient_table(coefficients)
        if coefficient_table.empty:
            st.info("No coefficient history is available for the current settings.")
        else:
            st.caption(
                "Latest fitted feature weights. Linear models show signed coefficients; tree models show "
                "nonnegative feature importances."
            )
            st.dataframe(
                safe_frame_subset(coefficient_table, ["feature", "coefficient", "abs_coefficient", "weight_type"]),
                width="stretch",
                hide_index=True,
            )

    with notes_tab:
        st.write(
            "`Predicted excess return` is the model's estimate of future group return minus the equal-weight S&P 500 "
            "return over the selected horizon. Positive means the group is scored as likely to outperform the broad "
            "universe; negative means likely to lag."
        )
        st.write(
            "The historical tab only includes rows where the full forward horizon has completed. Rows near the latest "
            "available price date are shown only as current/unrealized scores."
        )
        st.write(
            "The walk-forward audit checks that every prediction date trains only on rows whose target end date is "
            "strictly before that prediction date."
        )
        st.write(
            "In GICS mode, groups are official sectors. In learned-theme mode, each group's returns and fundamentals "
            "are loading-weighted averages from our soft GMM themes, so a company can contribute partially to multiple "
            "groups. The dashboard now uses hard top-1 assignment for the behavioral view because that performed better "
            "in the soft-vs-hard comparison, while business and growth stay mixed."
        )
        st.write(
            "Learned-theme returns require a minimum effective member count and compare against the same covered "
            "ticker universe used by the theme loadings. This prevents a tiny one-stock theme from producing a "
            "spectacular but misleading historical simulation."
        )
        st.write(
            "When learned-theme mode has a matching view embeddings parquet, the predictor also receives direct "
            "`group_embedding_*` features. These are loading-weighted averages of the company embedding coordinates, "
            "so the return model can test whether the learned representation itself adds predictive information beyond "
            "the hand-built momentum, valuation, and growth features."
        )
        st.write(
            "The historical universe uses add/remove intervals when the membership file is present. The date-added "
            "fallback only removes pre-entry rows for current constituents, so it is less complete than true historical "
            "membership."
        )
        st.write(
            "The rotation simulation is not a look-ahead oracle. It uses each historical prediction, buys the top-ranked "
            "group or equal-weights the top few groups, holds through the selected forward horizon, and only rebalances "
            "again after that holding window ends."
        )
        if group_mode == "theme" and theme_view == "business" and not fragment_explanations.empty:
            with st.expander("Learned theme label evidence", expanded=False):
                st.dataframe(
                    theme_evidence_frame(fragment_explanations, set(latest["gics_sector"].astype(str))),
                    width="stretch",
                    hide_index=True,
                )


def render_historical_text() -> None:
    st.subheader("Historical Filing Language")
    st.caption(
        "A compact workflow for asking: what changed in a company's filings, who else is changing, "
        "and whether the language moved semantically."
    )

    artifact_dirs = available_historical_text_dirs()
    default_dir = default_historical_text_dir()
    with st.expander("Choose historical artifact", expanded=False):
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

    sections = sorted(counts["section"].dropna().unique().tolist())
    section_labels = counts.drop_duplicates("section").set_index("section")["section_label"].to_dict()
    selected_sections = interpretable_historical_sections(sections)

    with st.expander("Loaded artifact summary", expanded=False):
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

    st.caption(
        section_scope_caption(selected_sections, section_labels)
        + " Rule-driven Item 1C cybersecurity disclosures and noisier sections such as legal proceedings, "
        + "properties, market risk, and quarterly item headings are hidden."
    )
    if not selected_sections:
        st.warning("No interpretation-friendly filing sections were found in this artifact.")
        return
    controls = st.columns([1.0, 1.0])
    min_year = int(counts["year"].min())
    max_year = int(counts["year"].max())
    early_years = controls[0].slider("Baseline years", min_year, max_year, (min_year, min(2018, max_year)))
    late_years = controls[1].slider("Recent years", min_year, max_year, (max(2023, min_year), max_year))

    ticker_options = sorted(counts["ticker"].unique().tolist())
    company_labels = metadata[metadata["ticker"].isin(ticker_options)].copy()
    label_lookup = dict(zip(company_labels["search_label"], company_labels["ticker"], strict=False))
    label_options = company_labels["search_label"].tolist()
    if not label_options:
        label_options = ticker_options
        label_lookup = {ticker: ticker for ticker in ticker_options}

    default_label = next((label for label in label_options if label.startswith("NVDA -")), label_options[0])
    selected_label = st.selectbox("Company story", label_options, index=label_options.index(default_label))
    selected_ticker = label_lookup[selected_label]

    story_tab, market_tab, map_tab = st.tabs(["Company Story", "Emerging Trends", "Semantic Map"])

    with story_tab:
        story = company_language_story_frame(
            counts,
            snippets,
            filing_index,
            selected_ticker,
            selected_sections,
            early_years,
            late_years,
        )
        if story.empty:
            st.info(f"No historical language story available for {selected_ticker}.")
        else:
            strongest = story.iloc[0]
            metric_columns = st.columns(4)
            metric_columns[0].metric("Largest topic increase", str(strongest["topic"]))
            metric_columns[1].metric("Change", f"{float(strongest['change']):+.3f}")
            metric_columns[2].metric("Recent mentions", f"{int(strongest['recent_mentions']):,}")
            metric_columns[3].metric("Evidence date", pd.Timestamp(strongest["evidence_date"]).strftime("%Y-%m-%d"))
            chart_col, evidence_col = st.columns([1.05, 1.35])
            with chart_col:
                st.markdown(f"**{selected_ticker} Language Shift Profile**")
                st.altair_chart(company_language_shift_chart(story), width="stretch")
            with evidence_col:
                st.markdown("**What changed, in plain evidence**")
                story_display = safe_frame_subset(
                    ensure_columns(
                        story,
                        [
                            "topic",
                            "change",
                            "recent_score",
                            "evidence_date",
                            "section",
                            "snippet_terms",
                            "evidence_snippet",
                            "source_url",
                        ],
                    ),
                    [
                        "topic",
                        "change",
                        "recent_score",
                        "evidence_date",
                        "section",
                        "snippet_terms",
                        "evidence_snippet",
                        "source_url",
                    ],
                )
                st.dataframe(
                    story_display,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "change": st.column_config.NumberColumn("change", format="%+.3f"),
                        "recent_score": st.column_config.NumberColumn("recent_score", format="%.3f"),
                        "source_url": st.column_config.LinkColumn("SEC filing"),
                    },
                )

            story_topic_name = str(strongest["topic"])
            story_topic_column = TOPIC_OPTIONS.get(story_topic_name, TOPIC_OPTIONS["AI"])
            story_mention_column = MENTION_OPTIONS.get(story_topic_name, MENTION_OPTIONS["AI"])
            company_history = company_topic_history(
                counts,
                selected_ticker,
                selected_sections,
                story_topic_column,
                story_mention_column,
            )
            if not company_history.empty:
                with st.expander(f"Optional {story_topic_name} timeline and filing rows", expanded=False):
                    st.altair_chart(historical_company_chart(company_history, story_topic_name), width="stretch")
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

    with market_tab:
        emerging = emerging_topic_evidence_frame(
            counts,
            snippets,
            filing_index,
            metadata,
            selected_sections,
            early_years,
            late_years,
            companies_per_topic=8,
        )
        st.markdown("**Emerging Themes Across Filings**")
        st.caption(
            "This scans all tracked themes across the interpretable filing sections and ranks the strongest "
            "company-level language increases. Use the evidence table to see what the filings actually said."
        )
        if emerging.empty:
            st.info("No emerging language trends were found for the selected baseline and recent windows.")
        else:
            chart_col, table_col = st.columns([1.0, 1.45])
            with chart_col:
                st.altair_chart(emerging_trends_chart(emerging, max_rows=18), width="stretch")
            with table_col:
                emerging_columns = [
                    "topic",
                    "ticker",
                    "company_name",
                    "gics_sector",
                    "change",
                    "late_score",
                    "filing_date",
                    "section_label",
                    "snippet_terms",
                    "evidence_snippet",
                    "source_url",
                ]
                st.dataframe(
                    safe_frame_subset(ensure_columns(emerging, emerging_columns), emerging_columns).head(30),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "change": st.column_config.NumberColumn("change", format="%.3f"),
                        "late_score": st.column_config.NumberColumn("recent_score", format="%.3f"),
                        "source_url": st.column_config.LinkColumn("SEC filing"),
                    },
                )

        with st.expander("Deep dive into one theme", expanded=False):
            st.caption(
                "Use this diagnostic section when you want to focus on one specific theme after spotting it "
                "in the emerging-trends table."
            )
            topic_name = st.selectbox(
                "Deep-dive theme",
                list(TOPIC_OPTIONS.keys()),
                index=0,
                key="historical_text_deep_dive_theme",
            )
            topic_column = TOPIC_OPTIONS[topic_name]
            mention_column = MENTION_OPTIONS[topic_name]
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
            trend = topic_yearly_trend(counts, selected_sections, topic_column)
            trend_col, sector_col = st.columns([1.3, 1.0])
            with trend_col:
                st.markdown(f"**Market-wide {topic_name} trend**")
                st.altair_chart(historical_market_trend_chart(trend, topic_name), width="stretch")
            sector_trend = sector_topic_heatmap(counts, metadata, selected_sections, topic_column)
            with sector_col:
                st.markdown("**Sector takeaways**")
                if sector_trend.empty:
                    st.info("No sector trend rows available.")
                else:
                    st.dataframe(
                        sector_topic_insight_frame(sector_trend, topic_name),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "score": st.column_config.NumberColumn("score", format="%.3f"),
                            "change": st.column_config.NumberColumn("change", format="%.3f"),
                        },
                    )

            st.markdown(f"**Companies whose {topic_name} language rose most**")
            if changes.empty:
                st.info("No company change rows available for the selected windows.")
            else:
                mover_evidence = topic_change_evidence_frame(
                    changes,
                    counts,
                    snippets,
                    filing_index,
                    selected_sections,
                    topic_column,
                    mention_column,
                    late_years,
                    max_companies=18,
                )
                if mover_evidence.empty:
                    display_columns = [
                        "ticker",
                        "company_name",
                        "gics_sector",
                        "change",
                        "late_score",
                        "late_mentions",
                        "first_year",
                        "latest_year",
                    ]
                    st.dataframe(
                        safe_frame_subset(ensure_columns(changes, display_columns), display_columns).head(50),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        mover_evidence,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "change": st.column_config.NumberColumn("change", format="%.3f"),
                            "late_score": st.column_config.NumberColumn("late_score", format="%.3f"),
                            "source_url": st.column_config.LinkColumn("SEC filing"),
                        },
                    )
                with st.expander("Optional sector heatmap", expanded=False):
                    if not sector_trend.empty:
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
    with map_tab:
        if embeddings.empty:
            st.info("No historical embeddings found, so the semantic map is unavailable.")
            return

        st.markdown("**Historical semantic movement**")
        map_controls = st.columns([1.2, 1.2, 2.0])
        map_sections = [section for section in selected_sections if section in sections]
        map_section = map_controls[0].selectbox(
            "Semantic section",
            map_sections,
            index=map_sections.index("business") if "business" in map_sections else 0,
            format_func=lambda section: section_labels.get(section, section),
        )
        color_mode = map_controls[1].selectbox("Map color", ["Year", "Sector", "Highlighted"], index=0)
        highlight_defaults = [
            ticker
            for ticker in ["NVDA", "MSFT", "META", "ADBE", "AMZN", selected_ticker]
            if ticker in ticker_options
        ]
        highlighted = map_controls[2].multiselect("Highlight filing trails", ticker_options, default=highlight_defaults)
        projected = load_projected_historical_embeddings(str(input_dir), map_section)
        if projected.empty:
            st.info("Not enough embedding rows to project this section.")
            return
        points = historical_embedding_points(projected, metadata, highlighted, color_mode)
        st.altair_chart(historical_embedding_map_chart(points, highlighted), width="stretch")
        st.caption(
            "The map uses historical MiniLM section embeddings. A line connects each highlighted company's filings over time, "
            "so the main thing to read is direction and size of movement, not the absolute x/y coordinates."
        )

        drift = historical_embedding_drift_table(embeddings, metadata, map_section, early_years, late_years)
        if not drift.empty:
            with st.expander("Semantic drift leaderboard", expanded=False):
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
    experiment_value = default_value
    with st.expander("Data source", expanded=False):
        experiment_value = st.text_input("Decomposed experiment directory", value=experiment_value)
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
    controls = st.columns([1.2, 1.8])
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
    if max_themes == 0:
        st.warning("This view has no theme-loading columns to display.")
        return
    selected_k = min(DEFAULT_SHIFT_THEME_COUNT, max_themes)
    selected_columns = selected_theme_columns(company_history, selected_k, "Biggest movement")

    dates = company_history["date"].dt.strftime("%Y-%m-%d").tolist()
    date_state_key = f"company_shift_date_index_{selected_view}_{selected_ticker}"
    bounded_state_value(date_state_key, default=len(dates) - 1, minimum=0, maximum=len(dates) - 1)
    step_controls = st.columns([0.9, 0.9, 0.9, 0.9, 2.4])
    if step_controls[0].button("Back", width="stretch", key=f"{date_state_key}_back"):
        move_state_value(date_state_key, -1, 0, len(dates) - 1)
    if step_controls[1].button("Forward", width="stretch", key=f"{date_state_key}_forward"):
        move_state_value(date_state_key, 1, 0, len(dates) - 1)
    if step_controls[2].button("Start", width="stretch", key=f"{date_state_key}_start"):
        st.session_state[date_state_key] = 0
    if step_controls[3].button("Latest", width="stretch", key=f"{date_state_key}_latest"):
        st.session_state[date_state_key] = len(dates) - 1
    date_index = step_controls[4].slider(
        "Available filing/view date",
        min_value=0,
        max_value=len(dates) - 1,
        key=date_state_key,
        format="%d",
    )
    st.caption("Back and Forward move one available company observation at a time.")
    selected_date = dates[date_index]
    label_lookup_for_view = auto_theme_label_lookup(str(loadings_path), metadata, selected_view)
    fragment_label_lookup, fragment_explanations = fragment_theme_label_artifacts(
        str(loadings_path),
        str(default_historical_text_dir()),
        selected_view,
        selected_date,
        metadata=metadata,
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
            f"The chart shows the {DEFAULT_SHIFT_THEME_COUNT} themes with the largest movement for the selected company. "
            "The underlying GMM was already fit in the decomposed experiment; this dashboard is an exploratory viewer."
        )


def render_similarity_explorer() -> None:
    st.subheader("Similarity Explorer")
    st.caption(
        "Pick a company and read its learned identity: strongest themes, closest peers, and optional market context."
    )

    default_experiment = latest_decomposed_experiment()
    default_value = str(default_experiment) if default_experiment else ""
    experiment_value = default_value
    with st.expander("Data source", expanded=False):
        experiment_value = st.text_input("Similarity experiment directory", value=experiment_value)
    if not experiment_value:
        st.warning("No decomposed experiment with view loadings was found.")
        return

    experiment_root = Path(experiment_value)
    if not experiment_root.exists():
        st.error(f"Experiment directory does not exist: {experiment_root}")
        return

    all_views = available_views(experiment_root)
    views = map_visible_views(all_views)
    if not views:
        st.error(f"No map-suitable view loadings found under {experiment_root / 'views'}")
        return
    hidden_views = sorted(set(all_views).difference(views))
    if hidden_views:
        st.caption(
            "Hidden from this explorer because they are better used as prediction features than visual clusters: "
            + ", ".join(hidden_views)
        )

    metadata = load_metadata()
    selected_view = st.selectbox(
        "Similarity view",
        views,
        index=views.index("business") if "business" in views else 0,
        key="market_map_view",
    )
    projection_method = "PCA"
    loadings_path = experiment_root / "views" / selected_view / "loadings.parquet"
    loadings = load_loadings(str(loadings_path))
    try:
        projected = load_projected_market_map(str(loadings_path), projection_method)
    except RuntimeError as exc:
        st.warning(f"{exc} Falling back to PCA.")
        projection_method = "PCA"
        projected = load_projected_market_map(str(loadings_path), projection_method)

    hide_collapsed = True
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
    if not theme_columns(loadings):
        st.warning("This view has no theme-loading columns to display.")
        return

    with st.expander("Time controls", expanded=False):
        st.caption("Default is the latest usable date. Open this only when you want to step through history.")
        step_columns = st.columns([0.9, 0.9, 1.0, 1.0, 2.0])
        if step_columns[0].button("Back", width="stretch"):
            st.session_state[date_state_key] = max(0, st.session_state[date_state_key] - 1)
        if step_columns[1].button("Forward", width="stretch"):
            st.session_state[date_state_key] = min(len(date_labels) - 1, st.session_state[date_state_key] + 1)
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
        st.caption(
            f"Selected map date: {date_labels[int(date_index)]}. "
            "Back and Forward move one usable map date at a time."
        )
    date_index = int(st.session_state[date_state_key])
    selected_date = pd.Timestamp(available_dates[date_index])
    label_lookup_for_view = auto_theme_label_lookup(str(loadings_path), metadata, selected_view)
    fragment_label_lookup, fragment_explanations = fragment_theme_label_artifacts(
        str(loadings_path),
        str(default_historical_text_dir()),
        selected_view,
        str(selected_date.date()),
        metadata=metadata,
    )
    label_lookup_for_view.update(fragment_label_lookup)
    label_lookup_for_view = unique_theme_label_lookup(label_lookup_for_view, theme_columns(loadings))
    trail_months = DEFAULT_MARKET_TRAIL_MONTHS

    points = market_date_frame(projected, metadata, selected_date, label_lookup_for_view)
    if points.empty:
        st.warning(f"No map points found for {selected_date.strftime('%Y-%m-%d')}.")
        return

    point_options = points.sort_values("ticker")["search_label"].astype(str).tolist()
    focus_lookup = dict(zip(point_options, points.sort_values("ticker")["ticker"], strict=False))
    default_focus = next(
        (label for label in point_options if label.startswith("MSFT -")),
        point_options[0],
    )
    focus_label = st.selectbox(
        "Focus company",
        point_options,
        index=point_options.index(default_focus),
        help="The map will highlight this company and its nearest theme-loading peers.",
    )
    focus_ticker = str(focus_lookup[focus_label]).upper()
    company_history = loadings[loadings["ticker"].eq(focus_ticker)].copy()
    company_history = company_history.sort_values("date").reset_index(drop=True)
    all_theme_columns = theme_columns(company_history)
    if not all_theme_columns:
        st.warning(f"No theme-loading history found for {focus_ticker}.")
        return
    selected_columns = selected_theme_columns(
        company_history,
        min(DEFAULT_SHIFT_THEME_COUNT, len(all_theme_columns)),
        "Biggest movement",
    )
    exact_dates = company_history["date"].eq(selected_date)
    if exact_dates.any():
        company_date_index = int(np.flatnonzero(exact_dates.to_numpy())[0])
    else:
        prior = company_history[company_history["date"] <= selected_date]
        company_date_index = int(prior.index[-1]) if not prior.empty else 0
    current_row = company_history.iloc[company_date_index]
    current_theme_columns = (
        current_row[all_theme_columns]
        .astype(float)
        .sort_values(ascending=False)
        .head(min(DEFAULT_SHIFT_THEME_COUNT, len(all_theme_columns)))
        .index.tolist()
    )

    relationships = load_relationships(artifact_signature(RELATIONSHIPS_PATH))
    peer_frame = nearest_theme_loading_peers(
        loadings,
        metadata,
        selected_date,
        focus_ticker,
        label_lookup_for_view,
        top_n=12,
    )
    highlighted = [focus_ticker, *peer_frame["ticker"].head(8).astype(str).tolist()] if not peer_frame.empty else [focus_ticker]

    metrics = st.columns(4)
    metrics[0].metric("Stocks shown", int(points["ticker"].nunique()))
    metrics[1].metric("View", selected_view)
    metrics[2].metric("Date", selected_date.strftime("%Y-%m-%d"))
    metrics[3].metric("Focus", focus_ticker)
    st.info(
        "Read this tab from top to bottom: the bar chart explains the selected company's theme mix, "
        "the peer table shows the closest companies in the full embedding space, and the 2D map is optional context."
    )

    snapshot_columns = st.columns([1.0, 1.25])
    with snapshot_columns[0]:
        st.markdown(f"**{focus_ticker} Theme Mix**")
        st.caption(
            f"Top memberships for {focus_ticker} on {selected_date.strftime('%Y-%m-%d')}. "
            "This is the most interpretable part of the similarity model."
        )
        st.altair_chart(
            current_theme_loading_chart(current_row, current_theme_columns, label_lookup_for_view),
            width="stretch",
        )
    with snapshot_columns[1]:
        st.markdown("**Closest Peers**")
        st.caption(
            "Computed in the full theme-loading vector, not from the 2D map. Use this table as the trusted peer view."
        )
        st.dataframe(
            safe_frame_subset(
                ensure_columns(
                    peer_frame,
                    [
                        "ticker",
                        "title",
                        "gics_sector",
                        "dominant_theme",
                        "loading_similarity",
                        "loading_distance",
                    ],
                ),
                [
                    "ticker",
                    "title",
                    "gics_sector",
                    "dominant_theme",
                    "loading_similarity",
                    "loading_distance",
                ],
            ).head(10),
            width="stretch",
            hide_index=True,
            column_config={
                "loading_similarity": st.column_config.NumberColumn("similarity", format="%.3f"),
                "loading_distance": st.column_config.NumberColumn("distance", format="%.3f"),
            },
        )

    with st.expander(f"{focus_ticker} theme details and recent movement", expanded=False):
        detail_columns = st.columns(2)
        with detail_columns[0]:
            st.markdown("Current theme memberships")
            st.dataframe(
                focus_theme_loading_frame(loadings, selected_date, focus_ticker, label_lookup_for_view),
                width="stretch",
                hide_index=True,
                column_config={
                    "loading": st.column_config.NumberColumn("loading", format="%.3f"),
                },
            )
        with detail_columns[1]:
            st.markdown("Movement at this date")
            st.dataframe(
                movement_frame(company_history, selected_columns, company_date_index, label_lookup_for_view),
                width="stretch",
                hide_index=True,
                column_config={
                    "loading": st.column_config.NumberColumn("loading", format="%.3f"),
                },
            )

    with st.expander(f"{focus_ticker} biggest full-period theme shifts", expanded=False):
        st.caption(
            "The old all-themes-over-time chart was too noisy, so this table keeps the useful part: "
            "which theme memberships changed most from the first available observation to the latest."
        )
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
            column_config={
                "start_loading": st.column_config.NumberColumn("start", format="%.3f"),
                "latest_loading": st.column_config.NumberColumn("latest", format="%.3f"),
                "change": st.column_config.NumberColumn("change", format="%.3f"),
                "absolute_change": st.column_config.NumberColumn("absolute_change", format="%.3f"),
            },
        )

    x_domain, y_domain = map_axis_domains(projected)
    with st.expander("Optional 2D market map", expanded=False):
        st.caption(
            f"S&P 500 similarity map on {selected_date.strftime('%Y-%m-%d')}. "
            f"Highlighted points are {focus_ticker} and its nearest peers from the full loading vector. "
            "Use this for spatial intuition only; trust the peer table above for exact similarity."
        )
        st.altair_chart(market_map_chart(points, highlighted, x_domain, y_domain), width="stretch")

    with st.expander(f"Companies sharing {focus_ticker}'s dominant theme", expanded=False):
        st.dataframe(
            focus_dominant_theme_members(points, focus_ticker),
            width="stretch",
            hide_index=True,
            column_config={
                "dominant_loading": st.column_config.NumberColumn("dominant_loading", format="%.3f"),
            },
        )

    with st.expander("All visible group summaries", expanded=False):
        st.caption(
            "This is a diagnostic overview of the whole map. Use the focused peer table above for company-specific evidence."
        )
        interpretation = theme_interpretation_frame(points, relationships, max_rows=12)
        st.dataframe(
            interpretation,
            width="stretch",
            hide_index=True,
            column_config={
                "median_loading": st.column_config.NumberColumn("median_loading", format="%.3f"),
            },
        )

    if highlighted:
        with st.expander("Optional highlighted ticker trails", expanded=False):
            st.caption(
                "This is useful for exploration, but it is intentionally hidden by default because trajectory charts "
                "can look noisy in a 2D projection."
            )
            st.altair_chart(
                market_trail_chart(projected, selected_date, highlighted, trail_months, label_lookup_for_view),
                width="stretch",
            )

    with st.expander("General map diagnostics", expanded=False):
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
            "The focus tables and trajectory use the same theme-loading data as the market map. The map is a 2D projection "
            "for context; the nearest-peer table uses the full loading vector."
        )
        st.write(
            "Color is the stock's actual dominant soft theme on the selected date. "
            "The map no longer pools small groups into `Other themes`, because that fake bucket can become visually dominant."
        )
        st.write(
            "PCA is the default for speed and stability; UMAP is optional for local-neighborhood exploration. "
            "Either projection is an exploratory map, not a trading signal."
        )


def render_network_explorer() -> None:
    """Render current-state filing-derived relationship analysis."""
    st.subheader("Network Explorer")
    st.caption(
        "Current-state relationship evidence from SEC filings. This tab focuses on who a company is linked to now, "
        "rather than trying to animate a sparse historical network."
    )

    relationships = load_relationships(artifact_signature(RELATIONSHIPS_PATH))
    if relationships.empty:
        st.info("No relationship graph artifact found at `data/processed/relationships/relationships.parquet`.")
        return

    metadata = load_metadata(artifact_signature(SP500_GICS_PATH))
    overview = network_company_overview(relationships, metadata)
    if overview.empty:
        st.info("The relationship graph exists, but no company tickers were found in the edge columns.")
        return

    overview["option_label"] = overview.apply(
        lambda row: (
            f"{row['ticker']} - {row['title']} "
            f"({int(row.get('direct_counterparties', 0) or 0)} links)"
        ),
        axis=1,
    )
    option_labels = overview["option_label"].tolist()
    option_lookup = dict(zip(option_labels, overview["ticker"], strict=False))
    default_ticker = str(overview.iloc[0]["ticker"])
    default_label = next(label for label, ticker in option_lookup.items() if ticker == default_ticker)

    controls = st.columns([1.8, 1.0, 0.9])
    selected_label = controls[0].selectbox(
        "Focus company",
        option_labels,
        index=option_labels.index(default_label),
        key="network_focus_company",
    )
    relationship_filter = controls[1].selectbox(
        "Relationship view",
        ["All links", "Supply chain", "Competitors", "Partners / agreements"],
        key="network_relationship_filter",
    )
    max_links = controls[2].selectbox("Links shown", [12, 20, 35, 50], index=1, key="network_max_links")
    focus_ticker = str(option_lookup[selected_label]).upper()

    all_edges = current_network_edges(relationships, metadata, focus_ticker, max_rows=200)
    if all_edges.empty:
        st.warning(f"No relationship edges found around {focus_ticker}.")
        return

    if relationship_filter == "Supply chain":
        visible_edges = all_edges[all_edges["relationship_side"].isin(["Supplier", "Customer"])].copy()
    elif relationship_filter == "Competitors":
        visible_edges = all_edges[all_edges["relationship_side"].eq("Competitor")].copy()
    elif relationship_filter == "Partners / agreements":
        visible_edges = all_edges[all_edges["relationship_side"].eq("Partner / agreement")].copy()
    else:
        visible_edges = all_edges.copy()
    visible_edges = visible_edges.sort_values(["strength", "filing_date"], ascending=[False, False]).head(int(max_links))

    if visible_edges.empty:
        st.warning(f"No {relationship_filter.lower()} edges found for {focus_ticker}.")
        return

    focus_row = overview[overview["ticker"].eq(focus_ticker)].head(1)
    focus_name = str(focus_row.iloc[0]["title"]) if not focus_row.empty else focus_ticker
    focus_display = compact_company_name(focus_name, focus_ticker, max_chars=42)
    visible_edges["counterparty_display"] = visible_edges.apply(
        lambda row: compact_company_name(row.get("counterparty_name", ""), row.get("counterparty", ""), max_chars=46),
        axis=1,
    )
    visible_edges["direction_readable"] = visible_edges.apply(
        lambda row: (
            f"{row['counterparty_display']} supplies {focus_display}"
            if row["relationship_side"] == "Supplier"
            else f"{focus_display} supplies {row['counterparty_display']}"
            if row["relationship_side"] == "Customer"
            else f"{focus_display} competes with {row['counterparty_display']}"
            if row["relationship_side"] == "Competitor"
            else f"{focus_display} linked with {row['counterparty_display']}"
        ),
        axis=1,
    )

    metrics = st.columns(5)
    metrics[0].metric("Counterparties", int(visible_edges["counterparty"].nunique()))
    metrics[1].metric("Suppliers", int(visible_edges["relationship_side"].eq("Supplier").sum()))
    metrics[2].metric("Customers", int(visible_edges["relationship_side"].eq("Customer").sum()))
    metrics[3].metric("Competitors", int(visible_edges["relationship_side"].eq("Competitor").sum()))
    metrics[4].metric("Evidence rows", int(len(visible_edges)))

    chart_columns = st.columns([1.6, 1.0])
    with chart_columns[0]:
        st.markdown(f"**{focus_display} Current Relationship Map**")
        st.altair_chart(network_relationship_chart(visible_edges, focus_ticker, focus_name), width="stretch")
    with chart_columns[1]:
        st.markdown("**Readable Relationship Ranking**")
        st.altair_chart(network_strength_chart(visible_edges), width="stretch")
        st.caption(
            "Use this chart when the node map gets dense. It keeps company names visible and ranks links by evidence strength."
        )

    with st.expander("Counterparty sector mix", expanded=False):
        st.caption(
            "This is not a complete supply-chain database. It is a filing-derived network, so every link should be "
            "read together with its evidence snippet."
        )
        st.altair_chart(network_sector_chart(visible_edges), width="stretch")

    st.markdown("**Relationship Evidence**")
    evidence_columns = [
        "relationship_side",
        "direction_readable",
        "counterparty",
        "counterparty_display",
        "counterparty_name",
        "counterparty_sector",
        "relationship_label",
        "strength",
        "filing_date",
        "form",
        "matched_alias",
        "context_snippet",
    ]
    st.dataframe(
        safe_frame_subset(ensure_columns(visible_edges, evidence_columns), evidence_columns),
        width="stretch",
        hide_index=True,
        column_config={
            "strength": st.column_config.NumberColumn("strength", format="%.2f"),
            "filing_date": st.column_config.DateColumn("filing_date"),
            "context_snippet": st.column_config.TextColumn("filing evidence", width="large"),
        },
    )

    with st.expander("Most connected companies in the current relationship graph", expanded=False):
        overview_columns = [
            "ticker",
            "title",
            "gics_sector",
            "direct_counterparties",
            "suppliers",
            "customers",
            "competitors",
            "partners_or_agreements",
            "link_rows",
        ]
        st.dataframe(
            safe_frame_subset(ensure_columns(overview, overview_columns), overview_columns).head(40),
            width="stretch",
            hide_index=True,
        )

    with st.expander("How to read this network", expanded=False):
        st.write(
            "The center node is the selected company. Supplier nodes are companies that filings suggest supply the "
            "focus company; customer nodes are companies the focus company appears to supply. Competitors and partners "
            "come from named relationship language in filings."
        )
        st.write(
            "The extraction is intentionally evidence-first. A link is useful when the snippet makes economic sense; "
            "if the snippet looks generic or noisy, treat that edge as weak."
        )


def render_model_comparison() -> None:
    st.subheader("Model Comparison")
    st.caption(
        "A presentation-style summary of the modeling choices. This tab explains what each model did, "
        "why the default was chosen, and what the saved results say."
    )

    st.markdown("### 1. Current Defaults")
    st.write(
        "These are the choices used by the main dashboard. The goal is not to claim every default is universally best, "
        "but to keep the demo honest, interpretable, and reproducible."
    )
    st.dataframe(model_decision_frame(), width="stretch", hide_index=True)

    st.markdown("### 2. Embedding Models")
    experiments = compact_experiment_comparison(load_report_csv("experiment_comparison.csv"))
    ablation = load_report_csv("feature_ablation_summary.csv")
    if experiments.empty:
        st.info("No `report/experiment_comparison.csv` file found yet.")
    else:
        best_nmi = experiments.dropna(subset=["clustering_nmi"]).head(1)
        best_peer = experiments.dropna(subset=["peer_corr_diff"]).sort_values("peer_corr_diff", ascending=False).head(1)
        covariance_rows = experiments.dropna(subset=["cov_embedding_annual_variance", "cov_ledoit_wolf_annual_variance"])
        metric_cols = st.columns(4)
        if not best_nmi.empty:
            metric_cols[0].metric(
                "Best semantic NMI",
                f"{float(best_nmi.iloc[0]['clustering_nmi']):.3f}",
                str(best_nmi.iloc[0]["experiment_name"]),
            )
        if not best_peer.empty:
            metric_cols[1].metric(
                "Best peer gap",
                f"{float(best_peer.iloc[0]['peer_corr_diff']):.3f}",
                str(best_peer.iloc[0]["experiment_name"]),
            )
        if not covariance_rows.empty:
            row = covariance_rows.iloc[0]
            diff = float(row["cov_embedding_annual_variance"] - row["cov_ledoit_wolf_annual_variance"])
            metric_cols[2].metric("Embedding vs LW variance", f"{diff:+.5f}", str(row["experiment_name"]))
        metric_cols[3].metric("Experiments compared", f"{len(experiments):,}")
        st.write(
            "Headline reading: text embeddings gave the clearest business structure, while price/risk embeddings were "
            "better for behavioral similarity. The full-universe peer and covariance tests did not beat the strongest "
            "benchmarks, which became part of the project's thesis: different similarity views answer different questions."
        )
        st.dataframe(
            experiments.head(8),
            width="stretch",
            hide_index=True,
            column_config={
                "final_reconstruction_mse": st.column_config.NumberColumn("recon_mse", format="%.4f"),
                "clustering_nmi": st.column_config.NumberColumn("clustering_nmi", format="%.3f"),
                "clustering_ari": st.column_config.NumberColumn("clustering_ari", format="%.3f"),
                "peer_corr_diff": st.column_config.NumberColumn("peer_corr_diff", format="%.3f"),
                "cov_embedding_annual_variance": st.column_config.NumberColumn("embedding_cov_var", format="%.5f"),
                "cov_ledoit_wolf_annual_variance": st.column_config.NumberColumn("lw_cov_var", format="%.5f"),
            },
        )

    if not ablation.empty:
        st.markdown("**Feature ablation takeaway**")
        best_ablation = ablation.sort_values("clustering_nmi", ascending=False, na_position="last").head(1)
        if not best_ablation.empty:
            row = best_ablation.iloc[0]
            st.write(
                f"The strongest GICS-alignment ablation was **{row['label']}** "
                f"(NMI {float(row['clustering_nmi']):.3f}, ARI {float(row['clustering_ari']):.3f}). "
                "That supports the interpretation that filing text carries most of the semantic company-identity signal."
            )
        st.dataframe(
            safe_frame_subset(
                ensure_columns(
                    ablation,
                    ["label", "clustering_nmi", "clustering_ari", "peer_corr_diff", "features_enabled"],
                ),
                ["label", "clustering_nmi", "clustering_ari", "peer_corr_diff", "features_enabled"],
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "clustering_nmi": st.column_config.NumberColumn("NMI", format="%.3f"),
                "clustering_ari": st.column_config.NumberColumn("ARI", format="%.3f"),
                "peer_corr_diff": st.column_config.NumberColumn("peer gap", format="%.3f"),
            },
        )

    st.markdown("### 3. Clustering Choice")
    st.write(
        "The dashboard uses **Gaussian Mixture Models** for theme membership because public companies rarely belong to "
        "one clean category. A company can be partly cloud infrastructure, partly advertising, partly AI platform, and "
        "partly enterprise software. Soft GMM loadings preserve that mixed identity."
    )
    clustering_summary = pd.DataFrame(
        [
            {
                "method": "Gaussian Mixture Model",
                "role": "Default theme model",
                "what_it_outputs": "Soft probabilities across themes",
                "why_it_matters": "Matches the idea that companies can belong to several economic themes at once.",
            },
            {
                "method": "k-means",
                "role": "Simple classroom baseline",
                "what_it_outputs": "One hard label per company",
                "why_it_matters": "Easy to explain, but too rigid for mixed business models.",
            },
            {
                "method": "DBSCAN",
                "role": "Outlier/density diagnostic",
                "what_it_outputs": "Dense clusters plus noise points",
                "why_it_matters": "Useful for anomaly discovery, but unstable for high-dimensional market embeddings.",
            },
        ]
    )
    st.dataframe(clustering_summary, width="stretch", hide_index=True)

    st.markdown("### 4. Sector / Group Prediction Models")
    st.write(
        "The prediction task is walk-forward: at each date, use only information available up to that date to predict "
        "future group excess return. Ridge regression remains the default because the sample is small, features are "
        "correlated, and interpretability matters more than raw flexibility."
    )
    sector_ae = compact_sector_autoencoder_comparison(load_report_csv("sector_autoencoder_outlook.csv"))
    embedding_corr = load_report_csv("sector_autoencoder_embedding_correlations.csv")
    if sector_ae.empty:
        st.info("No `report/sector_autoencoder_outlook.csv` file found yet.")
    else:
        best_sector = sector_ae.sort_values("ending_capital", ascending=False, na_position="last").iloc[0]
        raw_baseline = sector_ae[sector_ae["feature_set"].eq("raw_baseline")]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Best sector feature set", str(best_sector["feature_set"]))
        metric_cols[1].metric("Ending capital", f"${float(best_sector['ending_capital']):,.0f}")
        metric_cols[2].metric("SPY benchmark", f"${float(best_sector['spy_ending_capital']):,.0f}")
        metric_cols[3].metric("Max drawdown", f"{float(best_sector['max_drawdown']):.1%}")
        if not raw_baseline.empty:
            raw = raw_baseline.iloc[0]
            st.write(
                f"The raw structured-feature baseline ended at USD {float(raw['ending_capital']):,.0f}. "
                f"The best diagnostic sector-autoencoder feature set ended at USD {float(best_sector['ending_capital']):,.0f}. "
                "This suggests the numerical embedding can add useful structure, but this result should be presented "
                "as a backtest finding with survivorship and benchmark caveats, not as a trading strategy."
            )
        st.dataframe(
            sector_ae,
            width="stretch",
            hide_index=True,
            column_config={
                "mean_rank_ic": st.column_config.NumberColumn("rank IC", format="%.3f"),
                "mean_top_minus_bottom": st.column_config.NumberColumn("top-bottom", format="percent"),
                "top_sector_hit_rate": st.column_config.NumberColumn("hit rate", format="percent"),
                "ending_capital": st.column_config.NumberColumn("ending capital", format="$%.0f"),
                "spy_ending_capital": st.column_config.NumberColumn("SPY capital", format="$%.0f"),
                "excess_total_return_vs_spy": st.column_config.NumberColumn("excess vs SPY", format="percent"),
                "max_drawdown": st.column_config.NumberColumn("max drawdown", format="percent"),
            },
        )

    if not embedding_corr.empty:
        with st.expander("Numerical embedding feature correlations", expanded=False):
            st.write(
                "These correlations were used as a sanity check for which sector-state embedding dimensions carried "
                "predictive information."
            )
            st.dataframe(
                embedding_corr,
                width="stretch",
                hide_index=True,
                column_config={
                    "pearson_corr": st.column_config.NumberColumn("pearson", format="%.3f"),
                    "spearman_corr": st.column_config.NumberColumn("spearman", format="%.3f"),
                    "abs_spearman_corr": st.column_config.NumberColumn("|spearman|", format="%.3f"),
                },
            )

    st.markdown("### 5. Final Caveats")
    caveats = pd.DataFrame(
        [
            {
                "topic": "Benchmarks",
                "caveat": "GICS peers and Ledoit-Wolf covariance are strong specialized baselines; losing to them is informative.",
            },
            {
                "topic": "Backtests",
                "caveat": "Sector rotation results are historical simulations, not evidence of a deployable trading strategy.",
            },
            {
                "topic": "S&P membership",
                "caveat": "The dashboard now favors historical membership data, but older deleted constituents can still have incomplete filings/prices.",
            },
            {
                "topic": "Theme labels",
                "caveat": "Theme labels are generated dynamically from representative filing fragments; they are aids for interpretation, not supervised truth.",
            },
        ]
    )
    st.dataframe(caveats, width="stretch", hide_index=True)


filing_tab, historical_tab, similarity_tab, network_tab, sector_tab, comparison_tab = st.tabs(
    ["Filing Browser", "Historical Text", "Similarity Explorer", "Network", "Sector Outlook", "Model Comparison"]
)

with filing_tab:
    render_filing_browser()

with historical_tab:
    render_historical_text()

with similarity_tab:
    render_similarity_explorer()

with network_tab:
    render_network_explorer()

with sector_tab:
    render_sector_outlook()

with comparison_tab:
    render_model_comparison()
