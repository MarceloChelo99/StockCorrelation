"""Test whether sector-level autoencoder states help predict sector outperformance.

This analysis compresses the cross-section of stocks inside each GICS sector
into a sector-month embedding, then compares three walk-forward predictors:
raw sector features, sector-autoencoder embeddings only, and the hybrid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401

from src.applications.sector_relative_outlook import (
    GROUP_EMBEDDING_PREFIX,
    build_sector_feature_panel,
    fit_sector_model,
    filter_frame_by_membership,
    observed_month_ends,
    predict_sector_model,
    rotation_simulation_metrics,
    sector_and_market_returns,
    sector_prediction_metrics,
    simulate_group_rotation,
    ticker_label_frame,
    usable_feature_columns,
    walk_forward_model_predictions,
)
from src.config import load_config
from src.db import FilingsDB
from src.models.autoencoder import Autoencoder
from src.utils.io import ensure_dir
from src.utils.logging import log


FEATURE_GROUPS = [
    "price_volatility",
    "price_momentum",
    "price_liquidity",
    "valuation",
    "growth_lifecycle",
]
FEATURE_PREFIXES = ("price_", "valuation_", "growth_")


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="decomposed_point_in_time")
    parser.add_argument("--horizon-days", type=int, default=63)
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--fit-mode",
        choices=["fixed_initial", "expanding", "full_sample"],
        default="fixed_initial",
        help=(
            "fixed_initial trains one point-in-time encoder on the first training window. "
            "expanding refits monthly and is diagnostic only because embedding axes can drift. "
            "full_sample is diagnostic only because it uses future data."
        ),
    )
    parser.add_argument("--embedding-corr-threshold", type=float, default=0.05)
    parser.add_argument("--max-selected-embeddings", type=int, default=4)
    parser.add_argument("--output-csv", default="report/sector_autoencoder_outlook.csv")
    parser.add_argument("--output-md", default="report/sector_autoencoder_outlook.md")
    parser.add_argument(
        "--correlation-csv",
        default="report/sector_autoencoder_embedding_correlations.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run the sector-autoencoder predictor ablation."""
    args = parse_args()
    config = load_config(args.config)
    metadata = load_metadata()
    membership = load_membership()
    prices = load_prices(config)
    valuation = load_feature_frame("valuation")
    growth = load_feature_frame("growth_lifecycle")
    benchmark_returns = load_spy_returns()

    sector_returns, market_returns = sector_and_market_returns(
        prices,
        metadata,
        membership_mode="historical",
        membership=membership,
    )
    month_ends = observed_month_ends(sector_returns)
    base_panel = build_sector_feature_panel(
        sector_returns,
        market_returns,
        month_ends,
        metadata,
        valuation=valuation,
        growth=growth,
        membership=membership,
        membership_mode="historical",
        horizon_days=int(args.horizon_days),
    )

    sector_states = build_sector_state_features(metadata, membership)
    embedding_frame = sector_autoencoder_embeddings(
        sector_states,
        embedding_dim=int(args.embedding_dim),
        epochs=int(args.epochs),
        fit_mode=str(args.fit_mode),
        initial_train_months=int(args.min_train_months),
    )
    panel = base_panel.merge(embedding_frame, on=["date", "gics_sector"], how="left")

    raw_columns = usable_feature_columns(base_panel)
    ae_columns = [column for column in panel.columns if column.startswith(GROUP_EMBEDDING_PREFIX)]
    comparisons = {
        "raw_baseline": raw_columns,
        f"{args.fit_mode}_sector_ae_only": ae_columns,
        f"raw_plus_{args.fit_mode}_sector_ae": [*raw_columns, *ae_columns],
    }
    rows = []
    for feature_set, columns in comparisons.items():
        predictions, _ = walk_forward_model_predictions(
            panel,
            columns,
            min_train_months=int(args.min_train_months),
            ridge_alpha=float(args.ridge_alpha),
            model_type="ridge",
        )
        metrics = sector_prediction_metrics(predictions, int(args.horizon_days))
        simulation = simulate_group_rotation(
            predictions,
            starting_capital=10_000.0,
            top_n=1,
            benchmark_returns=benchmark_returns,
            benchmark_label="Regular S&P 500 (SPY)",
        )
        simulation_metrics = rotation_simulation_metrics(simulation, 10_000.0)
        rows.append(
            {
                "feature_set": feature_set,
                "fit_mode": args.fit_mode,
                "n_features": int(len(columns)),
                "n_prediction_dates": int(metrics.get("n_prediction_dates", 0)),
                "mean_rank_ic": metrics.get("mean_rank_ic"),
                "mean_top_minus_bottom": metrics.get("mean_top_minus_bottom"),
                "top_sector_hit_rate": metrics.get("top_sector_hit_rate"),
                "ending_capital": simulation_metrics.get("ending_capital"),
                "spy_ending_capital": simulation_metrics.get("benchmark_ending_capital"),
                "excess_total_return_vs_spy": simulation_metrics.get("excess_total_return_vs_benchmark"),
                "max_drawdown": simulation_metrics.get("max_drawdown"),
                "first_rebalance_date": simulation_metrics.get("first_rebalance_date"),
                "latest_exit_date": simulation_metrics.get("latest_exit_date"),
                "mean_selected_embeddings": np.nan,
                "selected_embedding_summary": "",
            }
        )

    filtered_runs = {
        f"corr_filtered_{args.fit_mode}_sector_ae_only": False,
        f"raw_plus_corr_filtered_{args.fit_mode}_sector_ae": True,
    }
    for feature_set, include_raw in filtered_runs.items():
        predictions, selection_history = walk_forward_filtered_embedding_predictions(
            panel,
            raw_columns=raw_columns,
            embedding_columns=ae_columns,
            include_raw=include_raw,
            min_train_months=int(args.min_train_months),
            ridge_alpha=float(args.ridge_alpha),
            corr_threshold=float(args.embedding_corr_threshold),
            max_selected_embeddings=int(args.max_selected_embeddings),
        )
        metrics = sector_prediction_metrics(predictions, int(args.horizon_days))
        simulation = simulate_group_rotation(
            predictions,
            starting_capital=10_000.0,
            top_n=1,
            benchmark_returns=benchmark_returns,
            benchmark_label="Regular S&P 500 (SPY)",
        )
        simulation_metrics = rotation_simulation_metrics(simulation, 10_000.0)
        rows.append(
            {
                "feature_set": feature_set,
                "fit_mode": args.fit_mode,
                "n_features": int((len(raw_columns) if include_raw else 0) + min(len(ae_columns), args.max_selected_embeddings)),
                "n_prediction_dates": int(metrics.get("n_prediction_dates", 0)),
                "mean_rank_ic": metrics.get("mean_rank_ic"),
                "mean_top_minus_bottom": metrics.get("mean_top_minus_bottom"),
                "top_sector_hit_rate": metrics.get("top_sector_hit_rate"),
                "ending_capital": simulation_metrics.get("ending_capital"),
                "spy_ending_capital": simulation_metrics.get("benchmark_ending_capital"),
                "excess_total_return_vs_spy": simulation_metrics.get("excess_total_return_vs_benchmark"),
                "max_drawdown": simulation_metrics.get("max_drawdown"),
                "first_rebalance_date": simulation_metrics.get("first_rebalance_date"),
                "latest_exit_date": simulation_metrics.get("latest_exit_date"),
                "mean_selected_embeddings": mean_selected_embeddings(selection_history),
                "selected_embedding_summary": selected_embedding_summary(selection_history),
            }
        )

    result = pd.DataFrame(rows)
    output_csv = resolved_path(args.output_csv)
    ensure_dir(output_csv.parent)
    result.to_csv(output_csv, index=False)
    correlation_summary = embedding_correlation_summary(panel, ae_columns)
    correlation_csv = resolved_path(args.correlation_csv)
    correlation_summary.to_csv(correlation_csv, index=False)
    write_report(
        result,
        resolved_path(args.output_md),
        fit_mode=str(args.fit_mode),
        n_sector_state_rows=len(sector_states),
        n_sector_state_features=count_state_features(sector_states),
        n_embedding_rows=len(embedding_frame),
        correlation_summary=correlation_summary,
        corr_threshold=float(args.embedding_corr_threshold),
        max_selected_embeddings=int(args.max_selected_embeddings),
    )
    log(f"Wrote comparison CSV to {output_csv}.", tag="sector-ae")
    log(f"Wrote embedding correlation CSV to {correlation_csv}.", tag="sector-ae")
    log(f"Wrote report to {resolved_path(args.output_md)}.", tag="sector-ae")
    print(result.to_string(index=False), flush=True)


def resolved_path(path: str | Path) -> Path:
    """Resolve a repo-relative path."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_metadata() -> pd.DataFrame:
    """Load ticker metadata with normalized tickers."""
    frame = pd.read_parquet(REPO_ROOT / "data" / "processed" / "metadata" / "sp500_gics.parquet")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date_added" in frame.columns:
        frame["date_added"] = pd.to_datetime(frame["date_added"], errors="coerce")
    return frame


def load_membership() -> pd.DataFrame:
    """Load historical S&P membership intervals."""
    frame = pd.read_parquet(REPO_ROOT / "data" / "processed" / "metadata" / "sp500_membership_history.parquet")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    return frame


def load_prices(config: dict) -> pd.DataFrame:
    """Load canonical project prices from the configured DB group."""
    db = FilingsDB.from_config(config)
    frame = db.load_prices(date_from=config["data"].get("start_date"), date_to=config["data"].get("end_date"))
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_feature_frame(name: str) -> pd.DataFrame:
    """Load one processed feature parquet."""
    frame = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features" / f"{name}.parquet")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_spy_returns() -> pd.Series:
    """Load regular S&P proxy returns."""
    path = REPO_ROOT / "data" / "processed" / "benchmarks" / "spy_benchmark.parquet"
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    returns = frame.sort_values("date").set_index("date")["adj_close"].pct_change().dropna()
    returns.name = "spy_return"
    return returns


def build_sector_state_features(metadata: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Aggregate company-level numerical features into sector-month states."""
    company = combined_company_features()
    company = filter_frame_by_membership(company, metadata, "historical", membership=membership)
    labels = ticker_label_frame(metadata, membership=membership).dropna(subset=["gics_sector"])
    labels = labels.loc[:, ["ticker", "gics_sector"]].drop_duplicates("ticker")
    company = company.merge(labels, on="ticker", how="inner")
    feature_columns = [
        column
        for column in company.columns
        if column.startswith(FEATURE_PREFIXES)
        and not column.endswith("_has_full_history")
        and not column.endswith("_has_full_valuation")
    ]
    grouped = company.groupby(["date", "gics_sector"])[feature_columns].agg(["mean", "median", "std"]).reset_index()
    grouped.columns = flattened_columns(grouped.columns)
    grouped = grouped.rename(columns={"date_": "date", "gics_sector_": "gics_sector"})
    grouped["date"] = pd.to_datetime(grouped["date"])
    state_columns = state_feature_columns(grouped)
    return grouped.loc[:, ["date", "gics_sector", *state_columns]].sort_values(["date", "gics_sector"]).reset_index(drop=True)


def combined_company_features() -> pd.DataFrame:
    """Merge all company-level numerical feature groups on ticker/date."""
    frames = []
    for name in FEATURE_GROUPS:
        frame = load_feature_frame(name)
        keep = [
            "ticker",
            "date",
            *[
                column
                for column in frame.columns
                if column.startswith(FEATURE_PREFIXES)
                and not column.endswith("_has_full_history")
                and not column.endswith("_has_full_valuation")
            ],
        ]
        frames.append(frame.loc[:, keep])
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=["ticker", "date"], how="outer")
    return combined


def flattened_columns(columns: pd.Index) -> list[str]:
    """Flatten MultiIndex aggregation columns."""
    result = []
    for column in columns:
        if isinstance(column, tuple):
            result.append("_".join(str(part) for part in column if str(part)))
        else:
            result.append(str(column))
    return result


def state_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return sector-state feature columns with enough data to train on."""
    columns = [column for column in frame.columns if column not in {"date", "gics_sector"}]
    usable = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() >= 0.15 and values.nunique(dropna=True) > 1:
            usable.append(column)
    return usable


def sector_autoencoder_embeddings(
    sector_states: pd.DataFrame,
    *,
    embedding_dim: int,
    epochs: int,
    fit_mode: str,
    initial_train_months: int,
) -> pd.DataFrame:
    """Return sector-month AE embeddings.

    ``fixed_initial`` is the safe predictive mode: it trains one encoder before
    the first walk-forward prediction window, so embedding coordinates remain
    comparable across time without using future data.
    """
    state_columns = state_feature_columns(sector_states)
    if not state_columns:
        raise ValueError("No sector-state feature columns were available for the autoencoder.")
    if fit_mode == "fixed_initial":
        return fixed_initial_embeddings(
            sector_states,
            state_columns,
            embedding_dim=embedding_dim,
            epochs=epochs,
            initial_train_months=initial_train_months,
        )
    if fit_mode == "full_sample":
        return full_sample_embeddings(sector_states, state_columns, embedding_dim=embedding_dim, epochs=epochs)
    return expanding_embeddings(sector_states, state_columns, embedding_dim=embedding_dim, epochs=epochs)


def fixed_initial_embeddings(
    sector_states: pd.DataFrame,
    state_columns: list[str],
    *,
    embedding_dim: int,
    epochs: int,
    initial_train_months: int,
) -> pd.DataFrame:
    """Train one early-window autoencoder and encode all sector-month states."""
    dates = sorted(sector_states["date"].dropna().unique())
    if len(dates) < int(initial_train_months):
        raise ValueError(
            f"Need at least {int(initial_train_months)} sector-state months for fixed_initial mode; got {len(dates)}."
        )
    cutoff = pd.Timestamp(dates[int(initial_train_months) - 1])
    train = sector_states[sector_states["date"] <= cutoff].copy()
    if len(train) < 100:
        raise ValueError("Not enough early sector-state rows to train a fixed initial sector autoencoder.")

    train_matrix = clean_matrix(train, state_columns)
    all_matrix = clean_matrix(sector_states, state_columns)
    model = Autoencoder(
        input_dim=train_matrix.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=[32, 16],
        epochs=epochs,
        batch_size=128,
        early_stopping_patience=5,
        random_seed=7,
    )
    model.fit(train_matrix)
    return embedding_frame(sector_states.loc[:, ["date", "gics_sector"]], model.encode(all_matrix))


def full_sample_embeddings(
    sector_states: pd.DataFrame,
    state_columns: list[str],
    *,
    embedding_dim: int,
    epochs: int,
) -> pd.DataFrame:
    """Fit one full-sample sector autoencoder and encode every sector-month."""
    matrix = clean_matrix(sector_states, state_columns)
    model = Autoencoder(
        input_dim=matrix.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=[64, 32],
        epochs=epochs,
        batch_size=128,
        early_stopping_patience=8,
        random_seed=7,
    )
    model.fit(matrix)
    return embedding_frame(sector_states.loc[:, ["date", "gics_sector"]], model.encode(matrix))


def expanding_embeddings(
    sector_states: pd.DataFrame,
    state_columns: list[str],
    *,
    embedding_dim: int,
    epochs: int,
) -> pd.DataFrame:
    """Fit one sector autoencoder per date using only current and prior states."""
    rows = []
    dates = sorted(sector_states["date"].dropna().unique())
    for index, date in enumerate(dates, start=1):
        history = sector_states[sector_states["date"] <= date].copy()
        current = sector_states[sector_states["date"].eq(date)].copy()
        if history["date"].nunique() < 24 or len(history) < 100 or current.empty:
            continue
        train_matrix = clean_matrix(history, state_columns)
        current_matrix = clean_matrix(current, state_columns)
        model = Autoencoder(
            input_dim=train_matrix.shape[1],
            embedding_dim=embedding_dim,
            hidden_dims=[32, 16],
            epochs=epochs,
            batch_size=128,
            early_stopping_patience=5,
            random_seed=7,
        )
        model.fit(train_matrix)
        rows.append(embedding_frame(current.loc[:, ["date", "gics_sector"]], model.encode(current_matrix)))
        if index % 50 == 0:
            log(f"Fit expanding sector AE through {pd.Timestamp(date).date()}.", tag="sector-ae")
    if not rows:
        return pd.DataFrame(columns=["date", "gics_sector"])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "gics_sector"]).reset_index(drop=True)


def clean_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return a finite matrix for autoencoder fitting/inference."""
    return (
        frame.loc[:, columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )


def embedding_frame(keys: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """Attach stable embedding column names to key columns."""
    frame = keys.copy()
    for index in range(matrix.shape[1]):
        frame[f"{GROUP_EMBEDDING_PREFIX}{index}"] = matrix[:, index]
    return frame


def count_state_features(sector_states: pd.DataFrame) -> int:
    """Return count of usable sector-state features."""
    return len(state_feature_columns(sector_states))


def walk_forward_filtered_embedding_predictions(
    panel: pd.DataFrame,
    *,
    raw_columns: list[str],
    embedding_columns: list[str],
    include_raw: bool,
    min_train_months: int,
    ridge_alpha: float,
    corr_threshold: float,
    max_selected_embeddings: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict with embedding columns selected from training-period correlations.

    The filter is point-in-time: for each prediction date, correlations are
    computed only on completed training rows whose target horizon ended before
    the prediction date.
    """
    rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    dates = sorted(panel["date"].dropna().unique())
    for date in dates:
        current_date = pd.Timestamp(date)
        train = panel[
            panel["is_horizon_complete"].fillna(False)
            & (panel["target_end_date"] < current_date)
            & panel["future_excess_return"].notna()
        ].copy()
        if train["date"].nunique() < int(min_train_months):
            continue
        current = panel[panel["date"].eq(current_date)].copy()
        if current.empty:
            continue

        selected_embeddings = selected_embedding_columns(
            train,
            embedding_columns,
            corr_threshold=corr_threshold,
            max_selected_embeddings=max_selected_embeddings,
        )
        feature_columns = [*raw_columns, *selected_embeddings] if include_raw else selected_embeddings
        selection_rows.append(
            {
                "date": current_date,
                "n_selected_embeddings": int(len(selected_embeddings)),
                "selected_embeddings": ",".join(selected_embeddings),
                "include_raw": bool(include_raw),
            }
        )
        if not feature_columns:
            continue

        fitted = fit_sector_model(train, feature_columns, "ridge", ridge_alpha)
        current["predicted_excess_return"] = predict_sector_model(current, fitted)
        current["prediction_rank"] = current["predicted_excess_return"].rank(ascending=False, method="first")
        current["training_rows"] = int(len(train))
        current["training_dates"] = int(train["date"].nunique())
        current["training_latest_target_end_date"] = pd.Timestamp(train["target_end_date"].max())
        current["model_type"] = "ridge_corr_filtered"
        rows.append(current)

    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    selection_history = pd.DataFrame(selection_rows)
    return predictions, selection_history


def selected_embedding_columns(
    train: pd.DataFrame,
    embedding_columns: list[str],
    *,
    corr_threshold: float,
    max_selected_embeddings: int,
) -> list[str]:
    """Return embedding dimensions whose training correlation clears a threshold."""
    scored = []
    target = pd.to_numeric(train["future_excess_return"], errors="coerce")
    for column in embedding_columns:
        values = pd.to_numeric(train[column], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 30 or values.loc[valid].nunique(dropna=True) < 2:
            continue
        corr = values.loc[valid].corr(target.loc[valid], method="spearman")
        if pd.isna(corr):
            continue
        if abs(float(corr)) >= float(corr_threshold):
            scored.append((column, abs(float(corr)), float(corr)))
    scored = sorted(scored, key=lambda item: item[1], reverse=True)
    return [column for column, _, _ in scored[: int(max_selected_embeddings)]]


def embedding_correlation_summary(panel: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    """Return full-sample diagnostics for each sector-AE embedding dimension.

    These correlations are diagnostic only; the predictive filter uses
    walk-forward training windows to avoid peeking at future outcomes.
    """
    completed = panel[
        panel["is_horizon_complete"].fillna(False)
        & panel["future_excess_return"].notna()
    ].copy()
    rows = []
    target = pd.to_numeric(completed["future_excess_return"], errors="coerce")
    for column in embedding_columns:
        values = pd.to_numeric(completed[column], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 30 or values.loc[valid].nunique(dropna=True) < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = values.loc[valid].corr(target.loc[valid], method="pearson")
            spearman = values.loc[valid].corr(target.loc[valid], method="spearman")
        rows.append(
            {
                "embedding": column,
                "n": int(valid.sum()),
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "abs_spearman_corr": abs(float(spearman)) if pd.notna(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman_corr", ascending=False).reset_index(drop=True)


def mean_selected_embeddings(selection_history: pd.DataFrame) -> float:
    """Return average number of selected embedding dimensions."""
    if selection_history.empty:
        return float("nan")
    return float(pd.to_numeric(selection_history["n_selected_embeddings"], errors="coerce").mean())


def selected_embedding_summary(selection_history: pd.DataFrame) -> str:
    """Return a compact count summary of selected embedding dimensions."""
    if selection_history.empty:
        return ""
    counts: dict[str, int] = {}
    for value in selection_history["selected_embeddings"].dropna():
        for column in str(value).split(","):
            if not column:
                continue
            counts[column] = counts.get(column, 0) + 1
    if not counts:
        return ""
    parts = [f"{column}:{count}" for column, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return ", ".join(parts)


def write_report(
    result: pd.DataFrame,
    path: Path,
    *,
    fit_mode: str,
    n_sector_state_rows: int,
    n_sector_state_features: int,
    n_embedding_rows: int,
    correlation_summary: pd.DataFrame,
    corr_threshold: float,
    max_selected_embeddings: int,
) -> None:
    """Write a compact markdown interpretation of the ablation."""
    ensure_dir(path.parent)
    table = result.copy()
    for column in ["mean_rank_ic", "mean_top_minus_bottom", "top_sector_hit_rate", "excess_total_return_vs_spy", "max_drawdown"]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    if "mean_selected_embeddings" in table.columns:
        table["mean_selected_embeddings"] = table["mean_selected_embeddings"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
        )
    for column in ["ending_capital", "spy_ending_capital"]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"${float(value):,.0f}")
    best_rank = result.sort_values("mean_rank_ic", ascending=False).iloc[0]["feature_set"]
    best_capital = result.sort_values("ending_capital", ascending=False).iloc[0]["feature_set"]
    corr_table = correlation_summary.head(8).copy()
    for column in ["pearson_corr", "spearman_corr", "abs_spearman_corr"]:
        corr_table[column] = corr_table[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    if fit_mode == "fixed_initial":
        fit_mode_note = (
            "The encoder is trained once on the initial walk-forward training window and then reused, "
            "so embedding coordinates stay comparable through time without using future data."
        )
    elif fit_mode == "expanding":
        fit_mode_note = (
            "Diagnostic only: this mode refits a new autoencoder each month, so embedding axes can drift. "
            "Do not treat these coordinates as a clean predictive feature set."
        )
    else:
        fit_mode_note = (
            "Diagnostic only: this mode fits one encoder on the full sample, so it uses future information."
        )
    markdown = [
        "# Sector Autoencoder Outlook Ablation",
        "",
        f"Fit mode: `{fit_mode}`",
        "",
        fit_mode_note,
        "",
        "This analysis tests whether a sector-level autoencoder can compress the stocks inside each GICS sector into a useful sector-state embedding for predicting sector excess returns.",
        "",
        "Construction:",
        "",
        "- Aggregate company price, valuation, and growth/lifecycle features within each `(date, GICS sector)` using mean, median, and standard deviation.",
        "- Train an autoencoder on those sector-month states.",
        "- Compare Ridge walk-forward sector prediction using raw features, sector-AE embeddings only, and raw features plus sector-AE embeddings.",
        "",
        f"Sector-state rows: `{n_sector_state_rows:,}`",
        f"Sector-state features: `{n_sector_state_features:,}`",
        f"Embedding rows: `{n_embedding_rows:,}`",
        "",
        "## Results",
        "",
        markdown_table(table),
        "",
        "## Embedding Correlation Diagnostics",
        "",
        "This table is a full-sample diagnostic ranking of sector-AE dimensions by absolute Spearman correlation with future sector excess return. It is not used directly for the point-in-time filter.",
        "",
        markdown_table(corr_table),
        "",
        "Point-in-time filter settings:",
        "",
        f"- Minimum absolute training-window Spearman correlation: `{corr_threshold:.3f}`",
        f"- Maximum selected embedding dimensions per prediction date: `{max_selected_embeddings}`",
        "",
        "## Interpretation",
        "",
        f"Best mean rank IC: `{best_rank}`.",
        f"Best ending capital: `{best_capital}`.",
        "",
        "The sector-AE concept has some signal if the embedding-only model produces positive rank IC and a competitive rotation simulation. The key question is whether it improves the raw-feature baseline. The correlation-filtered rows test whether weak embedding dimensions should be dropped rather than handed to Ridge.",
        "",
        "Current recommendation: treat this as an exploratory numerical-embedding ablation, not a presentation headline, unless the filtered or hybrid version consistently improves rank IC/top-bottom spread under point-in-time settings.",
        "",
    ]
    path.write_text("\n".join(markdown), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional deps."""
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
